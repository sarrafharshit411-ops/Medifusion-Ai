"""
MediFusion AI - Image Classification Model
ResNet18-based transfer learning model for Chest X-Ray classification (NORMAL vs PNEUMONIA).

Architecture:
- ResNet18 pretrained on ImageNet
- Frozen early layers (conv1 through layer2)
- Fine-tuned layer3, layer4, and new FC head
- Returns both classification logits and 512-dim embedding for fusion
"""

import logging
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

logger = logging.getLogger(__name__)


class ChestXRayModel(nn.Module):
    """
    ResNet18-based chest X-ray classifier.
    
    Outputs:
        logits: (batch, num_classes) classification logits
        embedding: (batch, 512) feature embedding from avgpool layer
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        
        # Load pretrained ResNet18
        if pretrained:
            self.backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            logger.info("Loaded ResNet18 with ImageNet pretrained weights")
        else:
            self.backbone = models.resnet18(weights=None)
        
        # Store the embedding dimension
        self.embedding_dim = self.backbone.fc.in_features  # 512
        
        # Replace the final FC layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.embedding_dim, num_classes)
        )
        
        # Freeze early layers
        self._freeze_early_layers()
    
    def _freeze_early_layers(self):
        """Freeze conv1, bn1, layer1, layer2. Fine-tune layer3, layer4, fc."""
        frozen_modules = ["conv1", "bn1", "layer1", "layer2"]
        for name, param in self.backbone.named_parameters():
            if any(name.startswith(mod) for mod in frozen_modules):
                param.requires_grad = False
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"Model parameters: {trainable:,} trainable / {total:,} total")
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 512-dim embedding without classification head."""
        # Forward through all layers except FC
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
            embedding: (batch, 512)
        """
        embedding = self.get_embedding(x)
        logits = self.backbone.fc(embedding)
        return logits, embedding


def load_image_model(
    checkpoint_path: str,
    num_classes: int = 2,
    device: Optional[torch.device] = None,
) -> ChestXRayModel:
    """Load a trained ChestXRayModel from checkpoint."""
    if device is None:
        device = torch.device("cpu")
    
    model = ChestXRayModel(num_classes=num_classes, pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info(f"Image model loaded from {checkpoint_path}")
    return model
