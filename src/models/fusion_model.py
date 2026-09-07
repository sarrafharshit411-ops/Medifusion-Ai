"""
MediFusion AI - Multimodal Fusion Model
Combines image, symptom, and clinical embeddings for unified disease assessment.

Two architectures:
1. ConcatFusion: Concatenation + MLP baseline
2. GatedAttentionFusion: Learnable gating + cross-modal attention

Supports missing modalities via modality masks (binary 0/1 per modality).

Unified label space: Normal (0), Pneumonia (1), Malaria (2), Typhoid (3)
- Image model: predicts {Normal, Pneumonia} → indices 0, 1
- Symptom model: predicts {Malaria, Pneumonia, Typhoid} → indices 2, 1, 3
- Fusion: predicts over all 4 classes

NOTE: The initial datasets are NOT patient-paired multimodal data. The fusion
architecture is demonstrated using synthetic pairings for educational purposes.
"""

import logging
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Unified label mapping
UNIFIED_CLASSES = ["Normal", "Pneumonia", "Malaria", "Typhoid"]
NUM_UNIFIED_CLASSES = len(UNIFIED_CLASSES)

# Mapping from individual model outputs to unified indices
IMAGE_TO_UNIFIED = {0: 0, 1: 1}       # Normal->0, Pneumonia->1
SYMPTOM_TO_UNIFIED = {0: 2, 1: 1, 2: 3}  # Malaria->2, Pneumonia->1, Typhoid->3


class ConcatFusion(nn.Module):
    """
    Concatenation + MLP Fusion (Baseline).
    
    Concatenates embeddings from all modalities and passes through MLP.
    Uses modality masks to zero out missing modalities.
    
    Input dims:
        image_embedding: 512
        symptom_embedding: 15
        clinical_embedding: 22
        Total: 549
    """
    
    def __init__(
        self,
        image_dim: int = 512,
        symptom_dim: int = 15,
        clinical_dim: int = 22,
        hidden_dim: int = 256,
        num_classes: int = NUM_UNIFIED_CLASSES,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.image_dim = image_dim
        self.symptom_dim = symptom_dim
        self.clinical_dim = clinical_dim
        total_dim = image_dim + symptom_dim + clinical_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        
        logger.info(f"ConcatFusion: {total_dim} -> {hidden_dim} -> {hidden_dim // 2} -> {num_classes}")
    
    def forward(
        self,
        image_emb: torch.Tensor,
        symptom_emb: torch.Tensor,
        clinical_emb: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            image_emb: (batch, 512)
            symptom_emb: (batch, 15)
            clinical_emb: (batch, 22)
            modality_mask: (batch, 3) binary mask [image_present, symptom_present, clinical_present]
            
        Returns:
            logits: (batch, num_classes)
        """
        if modality_mask is not None:
            # Apply modality masks
            image_emb = image_emb * modality_mask[:, 0:1]
            symptom_emb = symptom_emb * modality_mask[:, 1:2]
            clinical_emb = clinical_emb * modality_mask[:, 2:3]
        
        fused = torch.cat([image_emb, symptom_emb, clinical_emb], dim=1)
        logits = self.classifier(fused)
        return logits


class GatedAttentionFusion(nn.Module):
    """
    Gated Attention Fusion (Advanced).
    
    1. Projects each modality to a shared dimension
    2. Applies learned gating (importance weighting) per modality
    3. Uses multi-head cross-modal attention
    4. Final MLP for classification
    
    Supports missing modalities via attention masking.
    """
    
    def __init__(
        self,
        image_dim: int = 512,
        symptom_dim: int = 15,
        clinical_dim: int = 22,
        projection_dim: int = 128,
        num_heads: int = 2,
        num_classes: int = NUM_UNIFIED_CLASSES,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.projection_dim = projection_dim
        
        # Per-modality projection layers
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
        )
        self.symptom_proj = nn.Sequential(
            nn.Linear(symptom_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
        )
        self.clinical_proj = nn.Sequential(
            nn.Linear(clinical_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
        )
        
        # Gating mechanism: learns importance of each modality
        self.image_gate = nn.Sequential(nn.Linear(projection_dim, 1), nn.Sigmoid())
        self.symptom_gate = nn.Sequential(nn.Linear(projection_dim, 1), nn.Sigmoid())
        self.clinical_gate = nn.Sequential(nn.Linear(projection_dim, 1), nn.Sigmoid())
        
        # Cross-modal multi-head attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=projection_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(projection_dim)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim, projection_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim // 2, num_classes),
        )
        
        logger.info(
            f"GatedAttentionFusion: proj_dim={projection_dim}, "
            f"heads={num_heads}, classes={num_classes}"
        )
    
    def forward(
        self,
        image_emb: torch.Tensor,
        symptom_emb: torch.Tensor,
        clinical_emb: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            image_emb: (batch, 512)
            symptom_emb: (batch, 15)
            clinical_emb: (batch, 22)
            modality_mask: (batch, 3) binary mask
            
        Returns:
            logits: (batch, num_classes)
        """
        batch_size = image_emb.size(0)
        
        # Project each modality
        img_proj = self.image_proj(image_emb)      # (batch, proj_dim)
        sym_proj = self.symptom_proj(symptom_emb)   # (batch, proj_dim)
        cli_proj = self.clinical_proj(clinical_emb) # (batch, proj_dim)
        
        # Apply gating
        img_gated = img_proj * self.image_gate(img_proj)
        sym_gated = sym_proj * self.symptom_gate(sym_proj)
        cli_gated = cli_proj * self.clinical_gate(cli_proj)
        
        # Apply modality masks
        if modality_mask is not None:
            img_gated = img_gated * modality_mask[:, 0:1]
            sym_gated = sym_gated * modality_mask[:, 1:2]
            cli_gated = cli_gated * modality_mask[:, 2:3]
        
        # Stack as sequence tokens: (batch, 3, proj_dim)
        tokens = torch.stack([img_gated, sym_gated, cli_gated], dim=1)
        
        # Create attention key_padding_mask for missing modalities
        # True = masked (ignored), so invert the modality mask
        key_padding_mask = None
        if modality_mask is not None:
            key_padding_mask = (modality_mask == 0)  # (batch, 3)
        
        # Cross-modal attention
        attn_out, _ = self.cross_attention(
            tokens, tokens, tokens,
            key_padding_mask=key_padding_mask,
        )
        attn_out = self.attn_norm(attn_out + tokens)  # Residual connection
        
        # Pool across modalities (mean of non-masked tokens)
        if modality_mask is not None:
            mask_expanded = modality_mask.unsqueeze(-1)  # (batch, 3, 1)
            pooled = (attn_out * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = attn_out.mean(dim=1)  # (batch, proj_dim)
        
        logits = self.classifier(pooled)
        return logits


def map_image_probs_to_unified(probs_2class: torch.Tensor) -> torch.Tensor:
    """
    Map 2-class image predictions to 4-class unified space.
    Normal -> index 0, Pneumonia -> index 1, Malaria/Typhoid -> 0
    """
    batch_size = probs_2class.size(0)
    unified = torch.zeros(batch_size, NUM_UNIFIED_CLASSES, device=probs_2class.device)
    unified[:, 0] = probs_2class[:, 0]  # Normal
    unified[:, 1] = probs_2class[:, 1]  # Pneumonia
    return unified


def map_symptom_probs_to_unified(probs_3class: torch.Tensor) -> torch.Tensor:
    """
    Map 3-class symptom predictions to 4-class unified space.
    Malaria -> index 2, Pneumonia -> index 1, Typhoid -> index 3
    """
    batch_size = probs_3class.size(0)
    unified = torch.zeros(batch_size, NUM_UNIFIED_CLASSES, device=probs_3class.device)
    unified[:, 2] = probs_3class[:, 0]  # Malaria
    unified[:, 1] = probs_3class[:, 1]  # Pneumonia
    unified[:, 3] = probs_3class[:, 2]  # Typhoid
    return unified


def load_fusion_model(
    checkpoint_path: str,
    model_type: str = "gated",
    device: Optional[torch.device] = None,
    **kwargs,
) -> nn.Module:
    """Load a trained fusion model from checkpoint."""
    if device is None:
        device = torch.device("cpu")
    
    if model_type == "concat":
        model = ConcatFusion(**kwargs)
    elif model_type == "gated":
        model = GatedAttentionFusion(**kwargs)
    else:
        raise ValueError(f"Unknown fusion model type: {model_type}")
    
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info(f"Fusion model ({model_type}) loaded from {checkpoint_path}")
    return model
