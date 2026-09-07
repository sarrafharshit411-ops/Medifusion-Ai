"""
MediFusion AI - FastAPI Application
Main API server with all prediction and explanation endpoints.

DISCLAIMER: This is an educational/research clinical decision-support prototype.
It is NOT a definitive medical diagnostic tool.
"""

import os
import sys
import io
import logging
from contextlib import asynccontextmanager

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from api.schemas import (
    SymptomRequest, ClinicalRequest, MultimodalRequest,
    ImagePredictionResponse, SymptomPredictionResponse,
    ClinicalPredictionResponse, MultimodalPredictionResponse,
    ExplainImageResponse, ExplainClinicalResponse,
    HealthResponse, PredictionResult,
)
from api.model_loader import registry
from src.data.xray_dataset import get_eval_transforms
from src.data.symptom_dataset import (
    preprocess_symptoms, preprocess_clinical,
    generate_synthetic_clinical_features,
    SYMPTOM_FEATURES, SYMPTOM_LABELS,
)
from src.models.fusion_model import UNIFIED_CLASSES
from src.utils.image_validator import validate_chest_radiograph
from api.general_disease import router as general_disease_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IMAGE_CLASSES = ["NORMAL", "PNEUMONIA"]
CLINICAL_FEATURES = [
    "age", "temperature", "heart_rate",
    "bp_systolic", "bp_diastolic",
    "respiratory_rate", "oxygen_saturation",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup."""
    models_dir = os.path.join(project_root, "models")
    registry.load_all(models_dir)
    yield
    logger.info("Shutting down MediFusion AI API")


app = FastAPI(
    title="MediFusion AI",
    description=(
        "Multimodal Healthcare Disease-Assessment System. "
        "EDUCATIONAL/RESEARCH PROTOTYPE - NOT for clinical use."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_cache_control_header(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

app.include_router(general_disease_router)

# Serve frontend static files and pages
frontend_dir = os.path.join(project_root, "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/")
async def serve_index():
    """Serve the landing page."""
    return FileResponse(os.path.join(frontend_dir, "index.html"))


@app.get("/image-analysis")
@app.get("/image-analysis.html")
async def serve_image_analysis():
    """Serve the image analysis workspace."""
    return FileResponse(os.path.join(frontend_dir, "image-analysis.html"))


@app.get("/symptom-analysis")
@app.get("/symptom-analysis.html")
async def serve_symptom_analysis():
    """Serve the symptom analysis workspace."""
    return FileResponse(os.path.join(frontend_dir, "symptom-analysis.html"))


@app.get("/multimodal")
@app.get("/multimodal.html")
async def serve_multimodal():
    """Serve the multimodal assessment workspace."""
    return FileResponse(os.path.join(frontend_dir, "multimodal.html"))


@app.get("/insights")
@app.get("/insights.html")
async def serve_insights():
    """Serve the model insights & explainability workspace."""
    return FileResponse(os.path.join(frontend_dir, "insights.html"))


@app.get("/about")
@app.get("/about.html")
async def serve_about():
    """Serve the research & documentation page."""
    return FileResponse(os.path.join(frontend_dir, "about.html"))


# ──────────────────────────────────────────────────────────────
# Health Check
# ──────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model loading status."""
    return HealthResponse(
        status="healthy",
        models_loaded=registry.models_loaded,
    )


# ──────────────────────────────────────────────────────────────
# Image Prediction
# ──────────────────────────────────────────────────────────────

@app.post("/api/predict/image", response_model=ImagePredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """Predict disease from chest X-ray image with strict radiograph validation."""
    if not registry._loaded["image"]:
        raise HTTPException(status_code=503, detail="Image model not loaded")
    
    try:
        contents = await file.read()
        
        # Enforce strict medical radiograph validation
        is_valid, err_code, err_msg = validate_chest_radiograph(contents)
        if not is_valid:
            logger.warning(f"Rejected invalid image upload ({err_code}): {err_msg}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid Image: {err_msg}"
            )
        
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        transform = get_eval_transforms(224)
        tensor = transform(image).unsqueeze(0).to(registry.device)
        
        with torch.no_grad():
            logits, _ = registry.image_model(tensor)
            probs = F.softmax(logits, dim=1)[0]
        
        predictions = [
            PredictionResult(disease=cls, probability=float(probs[i]))
            for i, cls in enumerate(IMAGE_CLASSES)
        ]
        pred_idx = probs.argmax().item()
        
        return ImagePredictionResponse(
            predictions=predictions,
            predicted_class=IMAGE_CLASSES[pred_idx],
            confidence=float(probs[pred_idx]),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────
# Symptom Prediction
# ──────────────────────────────────────────────────────────────

@app.post("/api/predict/symptoms", response_model=SymptomPredictionResponse)
async def predict_symptoms(request: SymptomRequest):
    """Predict disease from symptoms."""
    if not registry._loaded["symptom"]:
        raise HTTPException(status_code=503, detail="Symptom model not loaded")
    
    try:
        X = preprocess_symptoms(request.symptoms)
        proba = registry.symptom_model.predict_proba(X)[0]
        
        classes = list(registry.symptom_label_encoder.classes_)
        predictions = [
            PredictionResult(disease=cls, probability=float(proba[i]))
            for i, cls in enumerate(classes)
        ]
        pred_idx = int(np.argmax(proba))
        
        return SymptomPredictionResponse(
            predictions=predictions,
            predicted_class=classes[pred_idx],
            confidence=float(proba[pred_idx]),
        )
    except Exception as e:
        logger.error(f"Symptom prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────
# Clinical Prediction
# ──────────────────────────────────────────────────────────────

@app.post("/api/predict/clinical", response_model=ClinicalPredictionResponse)
async def predict_clinical(request: ClinicalRequest):
    """Predict disease risk from clinical data + symptoms."""
    if not registry._loaded["clinical"]:
        raise HTTPException(status_code=503, detail="Clinical model not loaded")
    
    try:
        # Build combined feature vector (15 symptoms + 7 clinical = 22)
        sym_vec = preprocess_symptoms(request.symptoms)  # (1, 15)
        
        clin_values = np.array([[
            request.clinical_data.get("age", 45),
            request.clinical_data.get("temperature", 37.0),
            request.clinical_data.get("heart_rate", 75),
            request.clinical_data.get("bp_systolic", 120),
            request.clinical_data.get("bp_diastolic", 80),
            request.clinical_data.get("respiratory_rate", 16),
            request.clinical_data.get("oxygen_saturation", 98),
        ]], dtype=np.float32)
        
        X = np.hstack([sym_vec, clin_values])  # (1, 22)
        X_scaled = registry.clinical_scaler.transform(X)
        proba = registry.clinical_model.predict_proba(X_scaled)[0]
        
        classes = list(registry.clinical_label_encoder.classes_)
        predictions = [
            PredictionResult(disease=cls, probability=float(proba[i]))
            for i, cls in enumerate(classes)
        ]
        pred_idx = int(np.argmax(proba))
        
        return ClinicalPredictionResponse(
            predictions=predictions,
            predicted_class=classes[pred_idx],
            confidence=float(proba[pred_idx]),
        )
    except Exception as e:
        logger.error(f"Clinical prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────
# Multimodal Prediction
# ──────────────────────────────────────────────────────────────

@app.post("/api/predict/multimodal", response_model=MultimodalPredictionResponse)
async def predict_multimodal(
    file: UploadFile = File(None),
    symptoms: str = Form(None),
    clinical_data: str = Form(None),
):
    """
    Multimodal prediction combining available modalities.
    Accepts any combination of image, symptoms, and clinical data.
    """
    if not registry._loaded["fusion"]:
        raise HTTPException(status_code=503, detail="Fusion model not loaded")
    
    import json
    
    try:
        modalities_used = []
        individual_predictions = {}
        
        # Modality mask: [image, symptom, clinical]
        mask = [0.0, 0.0, 0.0]
        
        # --- Image embedding ---
        image_emb = torch.zeros(1, 512)
        if file is not None and registry._loaded["image"]:
            contents = await file.read()
            if len(contents) > 0:
                is_valid, err_code, err_msg = validate_chest_radiograph(contents)
                if not is_valid:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid Image in Radiology Stream: {err_msg}"
                    )
                image = Image.open(io.BytesIO(contents)).convert("RGB")
                transform = get_eval_transforms(224)
                tensor = transform(image).unsqueeze(0).to(registry.device)
                
                with torch.no_grad():
                    logits, emb = registry.image_model(tensor)
                    probs = F.softmax(logits, dim=1)[0]
                
                image_emb = emb
                mask[0] = 1.0
                modalities_used.append("image")
                
                individual_predictions["image"] = [
                    PredictionResult(disease=cls, probability=float(probs[i]))
                    for i, cls in enumerate(IMAGE_CLASSES)
                ]
        
        # --- Symptom embedding ---
        symptom_emb = torch.zeros(1, 15)
        if symptoms is not None and registry._loaded["symptom"]:
            sym_dict = json.loads(symptoms)
            sym_vec = preprocess_symptoms(sym_dict)
            proba = registry.symptom_model.predict_proba(sym_vec)[0]
            
            symptom_emb = torch.tensor(sym_vec, dtype=torch.float32)
            mask[1] = 1.0
            modalities_used.append("symptoms")
            
            classes = list(registry.symptom_label_encoder.classes_)
            individual_predictions["symptoms"] = [
                PredictionResult(disease=cls, probability=float(proba[i]))
                for i, cls in enumerate(classes)
            ]
        
        # --- Clinical embedding ---
        clinical_emb = torch.zeros(1, 22)
        if clinical_data is not None and symptoms is not None and registry._loaded["clinical"]:
            clin_dict = json.loads(clinical_data)
            sym_dict = json.loads(symptoms)
            sym_vec = preprocess_symptoms(sym_dict)
            
            clin_values = np.array([[
                clin_dict.get("age", 45),
                clin_dict.get("temperature", 37.0),
                clin_dict.get("heart_rate", 75),
                clin_dict.get("bp_systolic", 120),
                clin_dict.get("bp_diastolic", 80),
                clin_dict.get("respiratory_rate", 16),
                clin_dict.get("oxygen_saturation", 98),
            ]], dtype=np.float32)
            
            X = np.hstack([sym_vec, clin_values])
            X_scaled = registry.clinical_scaler.transform(X)
            
            clinical_emb = torch.tensor(X_scaled, dtype=torch.float32)
            mask[2] = 1.0
            modalities_used.append("clinical")
            
            clin_proba = registry.clinical_model.predict_proba(X_scaled)[0]
            clin_classes = list(registry.clinical_label_encoder.classes_)
            individual_predictions["clinical"] = [
                PredictionResult(disease=cls, probability=float(clin_proba[i]))
                for i, cls in enumerate(clin_classes)
            ]
        
        if not any(m > 0 for m in mask):
            raise HTTPException(status_code=400, detail="At least one modality must be provided")
        
        # Run fusion
        mask_tensor = torch.tensor([mask], dtype=torch.float32).to(registry.device)
        
        with torch.no_grad():
            logits = registry.fusion_model(
                image_emb.to(registry.device),
                symptom_emb.to(registry.device),
                clinical_emb.to(registry.device),
                mask_tensor,
            )
            probs = F.softmax(logits, dim=1)[0]
        
        predictions = [
            PredictionResult(disease=cls, probability=float(probs[i]))
            for i, cls in enumerate(UNIFIED_CLASSES)
        ]
        pred_idx = probs.argmax().item()
        
        return MultimodalPredictionResponse(
            unified_predictions=predictions,
            predicted_class=UNIFIED_CLASSES[pred_idx],
            confidence=float(probs[pred_idx]),
            modalities_used=modalities_used,
            individual_predictions=individual_predictions,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multimodal prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────
# Explainability Endpoints
# ──────────────────────────────────────────────────────────────

@app.post("/api/explain/image", response_model=ExplainImageResponse)
async def explain_image(file: UploadFile = File(...)):
    """Generate Grad-CAM explanation for a chest X-ray."""
    if not registry._loaded["image"]:
        raise HTTPException(status_code=503, detail="Image model not loaded")
    
    try:
        from src.explainability.gradcam import GradCAM
        
        contents = await file.read()
        
        # Enforce strict medical radiograph validation
        is_valid, err_code, err_msg = validate_chest_radiograph(contents)
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid Image: {err_msg}"
            )
        
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        transform = get_eval_transforms(224)
        tensor = transform(image).unsqueeze(0).to(registry.device)
        
        gradcam = GradCAM(registry.image_model)
        heatmap, pred_class, confidence = gradcam.generate(tensor)
        
        overlay_b64 = gradcam.create_overlay(image, heatmap)
        heatmap_b64 = gradcam.heatmap_to_base64(heatmap)
        
        return ExplainImageResponse(
            predicted_class=IMAGE_CLASSES[pred_class],
            confidence=confidence,
            gradcam_overlay_base64=overlay_b64,
            gradcam_heatmap_base64=heatmap_b64,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image explanation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/explain/clinical", response_model=ExplainClinicalResponse)
async def explain_clinical(request: ClinicalRequest):
    """Generate SHAP explanation for clinical prediction."""
    if not registry._loaded["clinical"]:
        raise HTTPException(status_code=503, detail="Clinical model not loaded")
    
    try:
        from src.explainability.shap_explain import SHAPExplainer
        
        sym_vec = preprocess_symptoms(request.symptoms)
        clin_values = np.array([[
            request.clinical_data.get("age", 45),
            request.clinical_data.get("temperature", 37.0),
            request.clinical_data.get("heart_rate", 75),
            request.clinical_data.get("bp_systolic", 120),
            request.clinical_data.get("bp_diastolic", 80),
            request.clinical_data.get("respiratory_rate", 16),
            request.clinical_data.get("oxygen_saturation", 98),
        ]], dtype=np.float32)
        
        X = np.hstack([sym_vec, clin_values])
        X_scaled = registry.clinical_scaler.transform(X)
        
        feature_names = SYMPTOM_FEATURES + CLINICAL_FEATURES
        explainer = SHAPExplainer(registry.clinical_model, feature_names)
        result = explainer.explain(X_scaled, class_names=SYMPTOM_LABELS)
        
        return ExplainClinicalResponse(
            feature_importance=result["feature_importance"],
            plot_base64=result["plot_base64"],
            per_class_importance=result.get("per_class_importance"),
        )
    except Exception as e:
        logger.error(f"Clinical explanation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
