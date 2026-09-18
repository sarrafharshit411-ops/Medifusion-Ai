"""
MediFusion AI - Model Loader
Loads all trained models once at startup for the API.
"""

import os

# Fix macOS deadlock: PyTorch initializes OpenMP/MKL threads, then joblib's
# loky/fork backend deadlocks on macOS. Force single-threaded mode before
# any heavy imports to prevent this.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

import json
import logging
from typing import Dict, Optional

import joblib
import torch

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Singleton registry for all trained models.
    Models are loaded once and reused across API requests.
    """
    
    def __init__(self):
        self.image_model = None
        self.symptom_model = None
        self.clinical_model = None
        self.clinical_scaler = None
        self.fusion_model = None
        self.general_disease_model = None
        self.general_disease_kb = None
        self.general_disease_symptoms = None
        self.symptom_label_encoder = None
        self.clinical_label_encoder = None
        self.device = torch.device("cpu")
        self._loaded: Dict[str, bool] = {
            "image": False,
            "symptom": False,
            "clinical": False,
            "fusion": False,
            "general_disease": False,
        }

        # ── Advanced Image Models ──
        self._image_models: Dict[str, object] = {}
        self._image_models_loaded: Dict[str, bool] = {
            "brain_tumor": False,
            "skin_cancer": False,
            "retinopathy": False,
            "blood_cell": False,
        }
        self.image_models_metadata: Dict = {}
    
    @property
    def models_loaded(self) -> Dict[str, bool]:
        combined = self._loaded.copy()
        for key, loaded in self._image_models_loaded.items():
            combined[f"image_{key}"] = loaded
        return combined
    
    def get_image_model(self, dataset_key: str):
        """Get a loaded image model by dataset key."""
        if dataset_key == "chest_xray":
            return self.image_model
        return self._image_models.get(dataset_key)
    
    def is_image_model_loaded(self, dataset_key: str) -> bool:
        """Check if a specific image model is loaded."""
        if dataset_key == "chest_xray":
            return self._loaded.get("image", False)
        return self._image_models_loaded.get(dataset_key, False)
    
    def load_all(self, models_dir: str):
        """Load all available models from the models directory.
        
        IMPORTANT: joblib (XGBoost/sklearn) models are loaded BEFORE PyTorch
        models to avoid a macOS deadlock where PyTorch's OpenMP thread pool
        blocks joblib's loky/fork backend from spawning workers.
        """
        logger.info(f"Loading models from {models_dir}...")
        
        # Load joblib-based models first (before PyTorch inits its thread pool)
        self._load_symptom_model(models_dir)
        self._load_clinical_model(models_dir)
        self._load_general_disease_model(models_dir)
        
        # Load PyTorch-based models after
        self._load_image_model(models_dir)
        self._load_fusion_model(models_dir)
        self._load_advanced_image_models(models_dir)
        self._load_image_models_metadata(models_dir)
        
        loaded_count = sum(self._loaded.values()) + sum(self._image_models_loaded.values())
        total_count = len(self._loaded) + len(self._image_models_loaded)
        logger.info(f"Model loading complete: {loaded_count}/{total_count} models loaded")
    
    def _load_image_model(self, models_dir: str):
        """Load the ResNet18 image model."""
        path = os.path.join(models_dir, "image_model.pth")
        if not os.path.exists(path):
            logger.warning(f"Image model not found at {path}")
            return
        
        try:
            from src.models.image_model import ChestXRayModel
            self.image_model = ChestXRayModel(num_classes=2, pretrained=False)
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            self.image_model.load_state_dict(state_dict)
            self.image_model.to(self.device)
            self.image_model.eval()
            self._loaded["image"] = True
            logger.info("✓ Image model loaded")
        except Exception as e:
            logger.error(f"✗ Failed to load image model: {e}")
    
    def _load_symptom_model(self, models_dir: str):
        """Load the XGBoost symptom model."""
        model_path = os.path.join(models_dir, "symptom_model.joblib")
        encoder_path = os.path.join(models_dir, "symptom_label_encoder.joblib")
        
        if not os.path.exists(model_path):
            logger.warning(f"Symptom model not found at {model_path}")
            return
        
        try:
            self.symptom_model = joblib.load(model_path)
            if os.path.exists(encoder_path):
                self.symptom_label_encoder = joblib.load(encoder_path)
            self._loaded["symptom"] = True
            logger.info("✓ Symptom model loaded")
        except Exception as e:
            logger.error(f"✗ Failed to load symptom model: {e}")
    
    def _load_clinical_model(self, models_dir: str):
        """Load the XGBoost clinical model."""
        model_path = os.path.join(models_dir, "clinical_model.joblib")
        scaler_path = os.path.join(models_dir, "clinical_scaler.joblib")
        encoder_path = os.path.join(models_dir, "clinical_label_encoder.joblib")
        
        if not os.path.exists(model_path):
            logger.warning(f"Clinical model not found at {model_path}")
            return
        
        try:
            self.clinical_model = joblib.load(model_path)
            if os.path.exists(scaler_path):
                self.clinical_scaler = joblib.load(scaler_path)
            if os.path.exists(encoder_path):
                self.clinical_label_encoder = joblib.load(encoder_path)
            self._loaded["clinical"] = True
            logger.info("✓ Clinical model loaded")
        except Exception as e:
            logger.error(f"✗ Failed to load clinical model: {e}")
    
    def _load_fusion_model(self, models_dir: str):
        """Load the fusion model (prefer gated over concat)."""
        gated_path = os.path.join(models_dir, "fusion_gated.pth")
        concat_path = os.path.join(models_dir, "fusion_concat.pth")
        
        path = gated_path if os.path.exists(gated_path) else concat_path
        model_type = "gated" if os.path.exists(gated_path) else "concat"
        
        if not os.path.exists(path):
            logger.warning("No fusion model found")
            return
        
        try:
            from src.models.fusion_model import GatedAttentionFusion, ConcatFusion
            if model_type == "gated":
                self.fusion_model = GatedAttentionFusion()
            else:
                self.fusion_model = ConcatFusion()
            
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            self.fusion_model.load_state_dict(state_dict)
            self.fusion_model.to(self.device)
            self.fusion_model.eval()
            self._loaded["fusion"] = True
            logger.info(f"✓ Fusion model ({model_type}) loaded")
        except Exception as e:
            logger.error(f"Failed to load fusion model: {e}")

    def _load_general_disease_model(self, models_dir: str):
        """Load the 41-disease ensemble classifier, knowledge base, and symptom metadata."""
        model_path = os.path.join(models_dir, "general_disease_ensemble.joblib")
        kb_path = os.path.join(models_dir, "disease_knowledge_base.json")
        meta_path = os.path.join(models_dir, "symptoms_132_metadata.json")

        if not os.path.exists(model_path):
            logger.warning(f"General disease model not found at {model_path}")
            return

        try:
            import json
            from src.models.general_disease_model import GeneralDiseaseEnsemble
            self.general_disease_model = joblib.load(model_path)
            
            if os.path.exists(kb_path):
                with open(kb_path, "r", encoding="utf-8") as f:
                    self.general_disease_kb = json.load(f)
                    
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.general_disease_symptoms = json.load(f)

            self._loaded["general_disease"] = True
            logger.info("✓ General disease 41-class ensemble & pharmacology loaded")
        except Exception as e:
            logger.error(f"Failed to load general disease model: {e}")

    def _load_advanced_image_models(self, models_dir: str):
        """Load all advanced image models (brain tumor, skin cancer, retinopathy, blood cell)."""
        from src.models.multi_image_model import MedicalImageClassifier, IMAGE_MODEL_CONFIGS

        for dataset_key in self._image_models_loaded.keys():
            filename = f"{dataset_key}_image_model.pth"
            path = os.path.join(models_dir, filename)

            if not os.path.exists(path):
                logger.info(f"  ○ {dataset_key} image model not found at {path} (skipped)")
                continue

            try:
                config = IMAGE_MODEL_CONFIGS.get(dataset_key, {})
                num_classes = config.get("num_classes", 2)

                model = MedicalImageClassifier(
                    dataset_key=dataset_key,
                    num_classes=num_classes,
                    pretrained=False,
                )
                state_dict = torch.load(path, map_location=self.device, weights_only=True)
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()

                self._image_models[dataset_key] = model
                self._image_models_loaded[dataset_key] = True
                logger.info(f"✓ {dataset_key} image model loaded ({num_classes} classes)")
            except Exception as e:
                logger.error(f"✗ Failed to load {dataset_key} image model: {e}")

    def _load_image_models_metadata(self, models_dir: str):
        """Load clinical metadata JSON for all image models."""
        meta_path = os.path.join(models_dir, "image_models_metadata.json")
        if not os.path.exists(meta_path):
            logger.info("  ○ image_models_metadata.json not found (skipped)")
            return

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                self.image_models_metadata = json.load(f)
            logger.info(f"✓ Image models metadata loaded ({len(self.image_models_metadata)} entries)")
        except Exception as e:
            logger.error(f"Failed to load image models metadata: {e}")


# Global singleton
registry = ModelRegistry()


