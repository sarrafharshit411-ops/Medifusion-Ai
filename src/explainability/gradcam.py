"""
MediFusion AI - Grad-CAM Explainability
Gradient-weighted Class Activation Mapping for visualizing which regions
of a chest X-ray most influence the model's prediction.

Targets ResNet18 layer4[-1] (the last convolutional block).
"""

import io
import logging
import base64
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Grad-CAM implementation for ResNet18-based ChestXRayModel.
    
    Usage:
        gradcam = GradCAM(model)
        heatmap, prediction = gradcam.generate(image_tensor)
        overlay_b64 = gradcam.create_overlay(original_image, heatmap)
    """
    
    def __init__(self, model, target_layer=None):
        """
        Args:
            model: ChestXRayModel instance
            target_layer: Layer to compute Grad-CAM on. Defaults to layer4[-1].
        """
        self.model = model
        self.model.eval()
        
        # Default target: last conv block of ResNet18
        if target_layer is None:
            self.target_layer = model.backbone.layer4[-1]
        else:
            self.target_layer = target_layer
        
        # Storage for activations and gradients
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    @torch.no_grad()
    def generate(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generate Grad-CAM heatmap for an input image.
        
        Args:
            image_tensor: (1, 3, H, W) preprocessed image tensor
            target_class: Class index to explain. If None, uses predicted class.
            
        Returns:
            heatmap: (H, W) normalized heatmap in [0, 1]
            predicted_class: int
            confidence: float
        """
        # Need gradients for Grad-CAM
        with torch.enable_grad():
            image_tensor = image_tensor.requires_grad_(True)
            
            # Forward pass
            logits, _ = self.model(image_tensor)
            probs = F.softmax(logits, dim=1)
            predicted_class = logits.argmax(dim=1).item()
            confidence = probs[0, predicted_class].item()
            
            if target_class is None:
                target_class = predicted_class
            
            # Backward pass for target class
            self.model.zero_grad()
            target_score = logits[0, target_class]
            target_score.backward()
        
        # Compute Grad-CAM
        gradients = self.gradients[0]         # (C, h, w)
        activations = self.activations[0]     # (C, h, w)
        
        # Global average pooling of gradients → channel weights
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU + normalize
        cam = F.relu(cam)
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Resize to input image size
        cam = cam.unsqueeze(0).unsqueeze(0)  # (1, 1, h, w)
        cam = F.interpolate(
            cam,
            size=(image_tensor.shape[2], image_tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        heatmap = cam.squeeze().cpu().numpy()
        
        return heatmap, predicted_class, confidence
    
    @staticmethod
    def create_overlay(
        original_image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: str = "jet",
    ) -> str:
        """
        Create a Grad-CAM overlay on the original image.
        
        Args:
            original_image: PIL Image
            heatmap: (H, W) normalized heatmap
            alpha: overlay transparency
            colormap: matplotlib colormap name
            
        Returns:
            base64-encoded PNG string
        """
        # Resize heatmap to match image
        img_array = np.array(original_image.resize((224, 224)).convert("RGB"))
        
        # Apply colormap to heatmap
        cmap = cm.get_cmap(colormap)
        heatmap_colored = cmap(heatmap)[:, :, :3]  # (H, W, 3) in [0, 1]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # Blend
        overlay = (alpha * heatmap_colored + (1 - alpha) * img_array).astype(np.uint8)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_array)
        axes[0].set_title("Original X-Ray")
        axes[0].axis("off")
        
        axes[1].imshow(heatmap, cmap=colormap)
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")
        
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis("off")
        
        plt.tight_layout()
        
        # Save to base64
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        
        return b64
    
    @staticmethod
    def heatmap_to_base64(heatmap: np.ndarray, colormap: str = "jet") -> str:
        """Convert raw heatmap to a standalone base64 PNG."""
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(heatmap, cmap=colormap)
        ax.axis("off")
        plt.tight_layout()
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
