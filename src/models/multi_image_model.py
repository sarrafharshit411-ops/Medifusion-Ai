"""
MediFusion AI - Multi-Disease Medical Image Classifier
Generic, configurable ResNet-based transfer learning model for all medical image datasets.

Supports multiple dataset-specific configurations via IMAGE_MODEL_CONFIGS registry.
Each model returns both classification logits and feature embeddings (for fusion).

Supported datasets:
    - chest_xray:   2-class (Normal, Pneumonia)
    - brain_tumor:  4-class (Glioma, Meningioma, No Tumor, Pituitary)
    - skin_cancer:  7-class (Actinic Keratoses, BCC, BKL, DF, Melanoma, Nevi, Vascular)
    - retinopathy:  5-class (No DR, Mild, Moderate, Severe, Proliferative DR)
    - blood_cell:   4-class (Eosinophil, Lymphocyte, Monocyte, Neutrophil)
"""

import logging
from typing import Tuple, Optional, Dict, List

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights

logger = logging.getLogger(__name__)


# ─── Model Configuration Registry ───────────────────────────────────
# Each entry defines the architecture and class configuration for a dataset.

IMAGE_MODEL_CONFIGS: Dict[str, dict] = {
    "chest_xray": {
        "backbone": "resnet18",
        "num_classes": 2,
        "class_names": ["NORMAL", "PNEUMONIA"],
        "display_names": ["Normal", "Pneumonia"],
        "input_size": 224,
        "dropout": 0.3,
        "freeze_layers": ["conv1", "bn1", "layer1", "layer2"],
        "image_type": "grayscale",
        "description": "Chest X-Ray Pneumonia Detection",
    },
    "brain_tumor": {
        "backbone": "resnet18",
        "num_classes": 4,
        "class_names": ["glioma", "meningioma", "notumor", "pituitary"],
        "display_names": ["Glioma", "Meningioma", "No Tumor", "Pituitary"],
        "input_size": 224,
        "dropout": 0.3,
        "freeze_layers": ["conv1", "bn1", "layer1", "layer2"],
        "image_type": "grayscale",
        "description": "Brain Tumor MRI Classification",
    },
    "skin_cancer": {
        "backbone": "resnet18",
        "num_classes": 7,
        "class_names": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
        "display_names": [
            "Actinic Keratoses", "Basal Cell Carcinoma", "Benign Keratosis",
            "Dermatofibroma", "Melanoma", "Melanocytic Nevi", "Vascular Lesions",
        ],
        "input_size": 224,
        "dropout": 0.4,
        "freeze_layers": ["conv1", "bn1", "layer1"],
        "image_type": "color",
        "description": "Skin Lesion Classification (HAM10000)",
    },
    "retinopathy": {
        "backbone": "resnet18",
        "num_classes": 5,
        "class_names": ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"],
        "display_names": ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"],
        "input_size": 224,
        "dropout": 0.35,
        "freeze_layers": ["conv1", "bn1", "layer1", "layer2"],
        "image_type": "color",
        "description": "Diabetic Retinopathy Severity Grading",
    },
    "blood_cell": {
        "backbone": "resnet18",
        "num_classes": 4,
        "class_names": ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"],
        "display_names": ["Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"],
        "input_size": 224,
        "dropout": 0.3,
        "freeze_layers": ["conv1", "bn1", "layer1", "layer2"],
        "image_type": "color",
        "description": "Blood Cell Type Classification",
    },
}


class MedicalImageClassifier(nn.Module):
    """
    Generic medical image classifier using ResNet transfer learning.

    Supports ResNet18 and ResNet50 backbones with configurable class counts,
    dropout, and layer freezing strategies.

    Outputs:
        logits: (batch, num_classes) — classification logits
        embedding: (batch, embedding_dim) — feature embedding from avgpool layer
    """

    def __init__(
        self,
        dataset_key: str = "chest_xray",
        num_classes: Optional[int] = None,
        backbone: Optional[str] = None,
        dropout: Optional[float] = None,
        freeze_layers: Optional[List[str]] = None,
        pretrained: bool = True,
    ):
        super().__init__()

        # Resolve config from registry
        if dataset_key in IMAGE_MODEL_CONFIGS:
            config = IMAGE_MODEL_CONFIGS[dataset_key]
        else:
            config = {
                "backbone": backbone or "resnet18",
                "num_classes": num_classes or 2,
                "dropout": dropout or 0.3,
                "freeze_layers": freeze_layers or ["conv1", "bn1", "layer1", "layer2"],
            }

        self.dataset_key = dataset_key
        self.num_classes = num_classes or config["num_classes"]
        backbone_name = backbone or config["backbone"]
        self.dropout_rate = dropout or config["dropout"]
        self.freeze_layer_names = freeze_layers or config["freeze_layers"]

        # Load backbone
        if backbone_name == "resnet50":
            if pretrained:
                self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
                logger.info(f"[{dataset_key}] Loaded ResNet50 with ImageNet pretrained weights")
            else:
                self.backbone = models.resnet50(weights=None)
        else:
            if pretrained:
                self.backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
                logger.info(f"[{dataset_key}] Loaded ResNet18 with ImageNet pretrained weights")
            else:
                self.backbone = models.resnet18(weights=None)

        # Store the embedding dimension (512 for ResNet18, 2048 for ResNet50)
        self.embedding_dim = self.backbone.fc.in_features

        # Replace the final FC layer with custom head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.embedding_dim, self.num_classes),
        )

        # Freeze early layers
        if pretrained:
            self._freeze_early_layers()

    def _freeze_early_layers(self):
        """Freeze specified layers. Fine-tune the rest + FC head."""
        for name, param in self.backbone.named_parameters():
            if any(name.startswith(mod) for mod in self.freeze_layer_names):
                param.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"[{self.dataset_key}] Parameters: "
            f"{trainable:,} trainable / {total:,} total "
            f"({100 * trainable / total:.1f}% trainable)"
        )

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embedding without classification head."""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            logits: (batch, num_classes)
            embedding: (batch, embedding_dim)
        """
        embedding = self.get_embedding(x)
        logits = self.backbone.fc(embedding)
        return logits, embedding


def load_medical_image_model(
    dataset_key: str,
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> MedicalImageClassifier:
    """Load a trained MedicalImageClassifier from checkpoint."""
    if device is None:
        device = torch.device("cpu")

    config = IMAGE_MODEL_CONFIGS.get(dataset_key, {})
    num_classes = config.get("num_classes", 2)

    model = MedicalImageClassifier(
        dataset_key=dataset_key,
        num_classes=num_classes,
        pretrained=False,
    )
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info(f"[{dataset_key}] Model loaded from {checkpoint_path}")
    return model


def get_model_config(dataset_key: str) -> dict:
    """Get the full configuration for a dataset's image model."""
    if dataset_key not in IMAGE_MODEL_CONFIGS:
        raise ValueError(
            f"Unknown dataset key '{dataset_key}'. "
            f"Available: {list(IMAGE_MODEL_CONFIGS.keys())}"
        )
    return IMAGE_MODEL_CONFIGS[dataset_key].copy()


def get_all_model_keys() -> List[str]:
    """Return all registered model keys."""
    return list(IMAGE_MODEL_CONFIGS.keys())
