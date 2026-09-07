"""
MediFusion AI - Symptom Classification Model
XGBoost-based classifier for symptom-to-disease prediction.

Dataset: 15 binary symptom features → 3 disease classes (Malaria, Pneumonia, Typhoid)
"""

import logging
from typing import Dict, Optional

import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class SymptomModel:
    """
    XGBoost symptom-based disease classifier.
    
    Predicts disease probabilities from 15 binary symptom features.
    Also provides leaf-index embeddings for fusion.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        xgb_params = config.get("xgboost", {}) if config else {}
        self.model = XGBClassifier(
            max_depth=xgb_params.get("max_depth", 6),
            n_estimators=xgb_params.get("n_estimators", 200),
            learning_rate=xgb_params.get("learning_rate", 0.1),
            eval_metric=xgb_params.get("eval_metric", "mlogloss"),
            objective="multi:softprob",
            num_class=3,
            use_label_encoder=False,
            random_state=42,
            verbosity=0,
        )
        self.label_encoder: Optional[LabelEncoder] = None
        self.feature_names = None
        self.is_fitted = False
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        feature_names: list = None,
        label_encoder: LabelEncoder = None,
    ):
        """Train the XGBoost classifier."""
        self.feature_names = feature_names
        self.label_encoder = label_encoder
        
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
        )
        self.is_fitted = True
        logger.info("Symptom model training complete")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities. Returns (n_samples, 3)."""
        return self.model.predict_proba(X)
    
    def get_embedding(self, X: np.ndarray) -> np.ndarray:
        """
        Get symptom feature representation for fusion.
        Returns the raw symptom features (15-dim) since they are already
        meaningful binary features.
        """
        return X.astype(np.float32)
    
    def get_leaf_embedding(self, X: np.ndarray) -> np.ndarray:
        """
        Get XGBoost leaf indices as an alternative embedding.
        Each sample gets a vector of leaf indices across all trees.
        """
        return self.model.apply(X).astype(np.float32)
    
    def save(self, model_path: str, encoder_path: str):
        """Save model and label encoder."""
        joblib.dump(self.model, model_path)
        if self.label_encoder is not None:
            joblib.dump(self.label_encoder, encoder_path)
        logger.info(f"Symptom model saved to {model_path}")
    
    def load(self, model_path: str, encoder_path: str):
        """Load model and label encoder."""
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(encoder_path)
        self.is_fitted = True
        logger.info(f"Symptom model loaded from {model_path}")


def load_symptom_model(model_path: str, encoder_path: str) -> SymptomModel:
    """Convenience function to load a trained SymptomModel."""
    sm = SymptomModel()
    sm.load(model_path, encoder_path)
    return sm
