"""
MediFusion AI - API Integration Tests
"""

import os
import sys
import json
import pytest

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def test_imports():
    """Test that all core modules can be imported."""
    from src.data.xray_dataset import load_xray_data, get_eval_transforms
    from src.data.symptom_dataset import load_symptom_data, preprocess_symptoms, SYMPTOM_FEATURES
    from src.models.image_model import ChestXRayModel
    from src.models.symptom_model import SymptomModel
    from src.models.clinical_model import ClinicalModel
    from src.models.fusion_model import ConcatFusion, GatedAttentionFusion, UNIFIED_CLASSES
    from src.explainability.gradcam import GradCAM
    from src.explainability.shap_explain import SHAPExplainer
    from src.utils.helpers import load_config, set_seed, get_device
    print("All imports successful")


def test_symptom_preprocessing():
    """Test symptom preprocessing."""
    from src.data.symptom_dataset import preprocess_symptoms, SYMPTOM_FEATURES
    import numpy as np
    
    symptoms = {"fever": 1, "cough": 1, "headache": 0}
    result = preprocess_symptoms(symptoms)
    
    assert result.shape == (1, 15)
    assert result[0, 0] == 1.0  # fever
    assert result[0, 1] == 1.0  # cough
    assert result[0, 2] == 0.0  # headache
    print("Symptom preprocessing: PASSED")


def test_model_creation():
    """Test that models can be instantiated."""
    from src.models.image_model import ChestXRayModel
    from src.models.symptom_model import SymptomModel
    from src.models.clinical_model import ClinicalModel
    from src.models.fusion_model import ConcatFusion, GatedAttentionFusion
    import torch
    
    # Image model
    img = ChestXRayModel(num_classes=2, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    logits, emb = img(x)
    assert logits.shape == (2, 2)
    assert emb.shape == (2, 512)
    
    # Fusion models
    concat = ConcatFusion()
    gated = GatedAttentionFusion()
    
    img_emb = torch.randn(2, 512)
    sym_emb = torch.randn(2, 15)
    cli_emb = torch.randn(2, 22)
    mask = torch.ones(2, 3)
    
    out_c = concat(img_emb, sym_emb, cli_emb, mask)
    out_g = gated(img_emb, sym_emb, cli_emb, mask)
    
    assert out_c.shape == (2, 4)
    assert out_g.shape == (2, 4)
    print("Model creation: PASSED")


def test_config_loading():
    """Test config can be loaded."""
    from src.utils.helpers import load_config
    
    config = load_config()
    assert "dataset" in config
    assert "image_model" in config
    assert "symptom_model" in config
    assert "fusion_model" in config
    print("Config loading: PASSED")


if __name__ == "__main__":
    test_imports()
    test_symptom_preprocessing()
    test_model_creation()
    test_config_loading()
    print("\n[OK] All tests passed!")
