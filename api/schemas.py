"""
MediFusion AI - Pydantic Request/Response Schemas
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────
# Request Schemas
# ──────────────────────────────────────────────────────────────

class SymptomRequest(BaseModel):
    """Request body for symptom-based prediction."""
    symptoms: Dict[str, int] = Field(
        ...,
        description="Dictionary of symptom_name -> 0/1 values",
        examples=[{"fever": 1, "cough": 1, "headache": 0, "fatigue": 1}],
    )


class ClinicalRequest(BaseModel):
    """Request body for clinical data prediction."""
    symptoms: Dict[str, int] = Field(
        ...,
        description="Dictionary of symptom_name -> 0/1 values",
    )
    clinical_data: Dict[str, float] = Field(
        ...,
        description="Dictionary of clinical measurements",
        examples=[{
            "age": 45, "temperature": 38.5, "heart_rate": 95,
            "bp_systolic": 130, "bp_diastolic": 85,
            "respiratory_rate": 22, "oxygen_saturation": 95,
        }],
    )


class MultimodalRequest(BaseModel):
    """Request body for multimodal prediction. All fields are optional."""
    symptoms: Optional[Dict[str, int]] = None
    clinical_data: Optional[Dict[str, float]] = None
    # Image is sent as file upload, not in JSON body


# ──────────────────────────────────────────────────────────────
# Response Schemas
# ──────────────────────────────────────────────────────────────

class PredictionResult(BaseModel):
    """Single prediction result."""
    disease: str
    probability: float
    

class DisclaimerMixin(BaseModel):
    """Mixin adding the required educational disclaimer."""
    disclaimer: str = (
        "DISCLAIMER: MediFusion AI is an educational/research clinical decision-support "
        "prototype. It is NOT a definitive medical diagnostic tool. Always consult a "
        "qualified healthcare professional for medical advice."
    )


class ImagePredictionResponse(DisclaimerMixin):
    """Response for image-based prediction."""
    predictions: List[PredictionResult]
    predicted_class: str
    confidence: float
    model: str = "ResNet18"


class SymptomPredictionResponse(DisclaimerMixin):
    """Response for symptom-based prediction."""
    predictions: List[PredictionResult]
    predicted_class: str
    confidence: float
    model: str = "XGBoost"


class ClinicalPredictionResponse(DisclaimerMixin):
    """Response for clinical data prediction."""
    predictions: List[PredictionResult]
    predicted_class: str
    confidence: float
    model: str = "XGBoost"
    synthetic_data_note: str = "Clinical features include synthetic data for demonstration purposes"


class MultimodalPredictionResponse(DisclaimerMixin):
    """Response for multimodal prediction."""
    unified_predictions: List[PredictionResult]
    predicted_class: str
    confidence: float
    modalities_used: List[str]
    fusion_model: str = "GatedAttentionFusion"
    individual_predictions: Optional[Dict[str, List[PredictionResult]]] = None
    data_note: str = (
        "The initial datasets are NOT patient-paired multimodal data. "
        "This fusion demonstrates the architecture for combining modalities."
    )


class ExplainImageResponse(DisclaimerMixin):
    """Response for Grad-CAM explanation."""
    predicted_class: str
    confidence: float
    gradcam_overlay_base64: str
    gradcam_heatmap_base64: str


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float


class ExplainClinicalResponse(DisclaimerMixin):
    """Response for SHAP explanation."""
    feature_importance: List[FeatureImportanceItem]
    plot_base64: str
    per_class_importance: Optional[Dict[str, List[FeatureImportanceItem]]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: Dict[str, bool]
    version: str = "1.0.0"
