"""
MediFusion AI — Advanced Image Analysis Router
API endpoints for multi-dataset medical image classification.

Supports 5 imaging pipelines:
    POST /api/predict/image/chest-xray     — Chest X-Ray Pneumonia (2-class)
    POST /api/predict/image/brain-tumor    — Brain Tumor MRI (4-class)
    POST /api/predict/image/skin-cancer    — Skin Lesion (7-class)
    POST /api/predict/image/retinopathy    — Diabetic Retinopathy (5-class)
    POST /api/predict/image/blood-cell     — Blood Cell Type (4-class)
    GET  /api/image-models/status          — Model availability status
    GET  /api/image-models/metadata        — Clinical metadata for all models
    POST /api/explain/image/advanced       — Grad-CAM for any image model
"""

import io
import json
import os
import logging
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel, Field
from typing import List, Dict, Any

from api.model_loader import registry

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Advanced Image Analysis"])


# ─── Response Schemas ───────────────────────────────────────────────

class ClassPrediction(BaseModel):
    class_id: str
    display_name: str
    probability: float
    severity: str = ""
    color: str = ""
    description: str = ""
    action: str = ""


class AdvancedImagePredictionResponse(BaseModel):
    model_key: str
    model_title: str
    backbone: str
    predicted_class: str
    predicted_display_name: str
    confidence: float
    severity: str
    predictions: List[ClassPrediction]
    clinical_context: str = ""
    recommended_action: str = ""
    disclaimer: str = (
        "DISCLAIMER: MediFusion AI is an educational/research clinical decision-support "
        "prototype. It is NOT a definitive medical diagnostic tool. Always consult a "
        "qualified healthcare professional for medical advice."
    )


class ImageModelStatus(BaseModel):
    model_key: str
    title: str
    loaded: bool
    num_classes: int
    image_type: str


class ImageModelsStatusResponse(BaseModel):
    total_models: int
    loaded_models: int
    models: List[ImageModelStatus]


# ─── Helpers ────────────────────────────────────────────────────────

def _get_eval_transforms(input_size: int = 224):
    """Get the standard eval transforms for inference."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def _get_metadata():
    """Load image models metadata."""
    return registry.image_models_metadata or {}


def _run_inference(model, image_bytes: bytes, model_key: str) -> AdvancedImagePredictionResponse:
    """Run inference on a loaded model and return formatted response."""
    metadata = _get_metadata()
    model_meta = metadata.get(model_key, {})

    # Decode image
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Unable to decode image. Please upload a valid PNG or JPEG.")

    transform = _get_eval_transforms(224)
    tensor = transform(image).unsqueeze(0).to(registry.device)

    # Inference
    with torch.no_grad():
        logits, _ = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    # Build class predictions using metadata
    classes_meta = model_meta.get("classes", {})
    from src.models.multi_image_model import IMAGE_MODEL_CONFIGS
    model_config = IMAGE_MODEL_CONFIGS.get(model_key, {})
    class_names = model_config.get("class_names", [])
    display_names = model_config.get("display_names", class_names)

    predictions = []
    for i, cls_id in enumerate(class_names):
        cls_meta = classes_meta.get(cls_id, {})
        predictions.append(ClassPrediction(
            class_id=cls_id,
            display_name=cls_meta.get("display_name", display_names[i] if i < len(display_names) else cls_id),
            probability=float(probs[i]),
            severity=cls_meta.get("severity", ""),
            color=cls_meta.get("color", "#3b82f6"),
            description=cls_meta.get("description", ""),
            action=cls_meta.get("action", ""),
        ))

    # Sort by probability descending for display
    predictions.sort(key=lambda p: p.probability, reverse=True)

    pred_idx = probs.argmax().item()
    pred_class_id = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
    pred_meta = classes_meta.get(pred_class_id, {})

    return AdvancedImagePredictionResponse(
        model_key=model_key,
        model_title=model_meta.get("title", model_config.get("description", model_key)),
        backbone=model_meta.get("backbone", model_config.get("backbone", "ResNet-18")),
        predicted_class=pred_class_id,
        predicted_display_name=pred_meta.get("display_name", display_names[pred_idx] if pred_idx < len(display_names) else pred_class_id),
        confidence=float(probs[pred_idx]),
        severity=pred_meta.get("severity", "unknown"),
        predictions=predictions,
        clinical_context=pred_meta.get("description", ""),
        recommended_action=pred_meta.get("action", "Consult a healthcare professional."),
    )


# ─── Endpoints ──────────────────────────────────────────────────────

@router.get("/api/image-models/status", response_model=ImageModelsStatusResponse)
async def get_image_models_status():
    """Return the loading status of all image models."""
    metadata = _get_metadata()
    from src.models.multi_image_model import IMAGE_MODEL_CONFIGS

    model_keys = ["chest_xray", "brain_tumor", "skin_cancer", "retinopathy", "blood_cell"]
    models_status = []
    loaded_count = 0

    for key in model_keys:
        is_loaded = registry.is_image_model_loaded(key)
        if is_loaded:
            loaded_count += 1

        meta = metadata.get(key, {})
        config = IMAGE_MODEL_CONFIGS.get(key, {})

        models_status.append(ImageModelStatus(
            model_key=key,
            title=meta.get("title", config.get("description", key)),
            loaded=is_loaded,
            num_classes=config.get("num_classes", 0),
            image_type=config.get("image_type", "unknown"),
        ))

    return ImageModelsStatusResponse(
        total_models=len(model_keys),
        loaded_models=loaded_count,
        models=models_status,
    )


@router.get("/api/image-models/metadata")
async def get_image_models_metadata():
    """Return clinical metadata for all image models."""
    metadata = _get_metadata()
    if not metadata:
        raise HTTPException(status_code=503, detail="Image models metadata not loaded.")
    return metadata


@router.post("/api/predict/image/brain-tumor", response_model=AdvancedImagePredictionResponse)
async def predict_brain_tumor(file: UploadFile = File(...)):
    """Classify brain tumor type from MRI image."""
    model = registry.get_image_model("brain_tumor")
    if model is None:
        raise HTTPException(status_code=503, detail="Brain tumor model not loaded. Train with: python -m src.training.train_multi_image --dataset brain_tumor")

    try:
        contents = await file.read()
        return _run_inference(model, contents, "brain_tumor")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Brain tumor prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/predict/image/skin-cancer", response_model=AdvancedImagePredictionResponse)
async def predict_skin_cancer(file: UploadFile = File(...)):
    """Classify skin lesion type from dermoscopic image."""
    model = registry.get_image_model("skin_cancer")
    if model is None:
        raise HTTPException(status_code=503, detail="Skin cancer model not loaded. Train with: python -m src.training.train_multi_image --dataset skin_cancer")

    try:
        contents = await file.read()
        return _run_inference(model, contents, "skin_cancer")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Skin cancer prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/predict/image/retinopathy", response_model=AdvancedImagePredictionResponse)
async def predict_retinopathy(file: UploadFile = File(...)):
    """Grade diabetic retinopathy severity from retinal fundus image."""
    model = registry.get_image_model("retinopathy")
    if model is None:
        raise HTTPException(status_code=503, detail="Retinopathy model not loaded. Train with: python -m src.training.train_multi_image --dataset retinopathy")

    try:
        contents = await file.read()
        return _run_inference(model, contents, "retinopathy")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Retinopathy prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/predict/image/blood-cell", response_model=AdvancedImagePredictionResponse)
async def predict_blood_cell(file: UploadFile = File(...)):
    """Classify blood cell type from microscopy image."""
    model = registry.get_image_model("blood_cell")
    if model is None:
        raise HTTPException(status_code=503, detail="Blood cell model not loaded. Train with: python -m src.training.train_multi_image --dataset blood_cell")

    try:
        contents = await file.read()
        return _run_inference(model, contents, "blood_cell")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Blood cell prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/explain/image/advanced")
async def explain_image_advanced(
    file: UploadFile = File(...),
    model_key: str = Form("chest_xray"),
):
    """
    Generate Grad-CAM explanation for any loaded image model.
    Returns the same Grad-CAM overlay + heatmap as the original endpoint.
    """
    # For chest_xray, use the original model from registry
    if model_key == "chest_xray":
        model = registry.image_model
        if model is None:
            raise HTTPException(status_code=503, detail="Chest X-Ray model not loaded")
    else:
        model = registry.get_image_model(model_key)
        if model is None:
            raise HTTPException(
                status_code=503,
                detail=f"{model_key} model not loaded."
            )

    try:
        from src.explainability.gradcam import GradCAM
        from src.models.multi_image_model import IMAGE_MODEL_CONFIGS

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        transform = _get_eval_transforms(224)
        tensor = transform(image).unsqueeze(0).to(registry.device)

        gradcam = GradCAM(model)
        heatmap, pred_class, confidence = gradcam.generate(tensor)

        overlay_b64 = gradcam.create_overlay(image, heatmap)
        heatmap_b64 = gradcam.heatmap_to_base64(heatmap)

        config = IMAGE_MODEL_CONFIGS.get(model_key, {})
        class_names = config.get("display_names", config.get("class_names", []))
        pred_name = class_names[pred_class] if pred_class < len(class_names) else str(pred_class)

        return {
            "model_key": model_key,
            "predicted_class": pred_name,
            "confidence": confidence,
            "gradcam_overlay_base64": overlay_b64,
            "gradcam_heatmap_base64": heatmap_b64,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Advanced image explanation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
