"""
MediFusion AI - Advanced Loss Functions for Class-Imbalanced Medical Imaging
Provides:
    - FocalLoss: Focuses learning on hard negative examples; addresses severe class imbalance.
    - LabelSmoothingCrossEntropy: Prevents overconfidence on borderline/noisy medical labels.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss (Lin et al., 2017).
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma (float): Focusing parameter for hard examples (default: 2.0).
        weight (torch.Tensor, optional): Pre-computed per-class weights (alpha).
        reduction (str): 'none' | 'mean' | 'sum' (default: 'mean').
        label_smoothing (float): Optional label smoothing factor [0, 1].
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Raw model logits of shape (batch_size, num_classes).
            targets: Target ground-truth class indices of shape (batch_size).
        """
        num_classes = inputs.size(-1)
        log_p = F.log_softmax(inputs, dim=-1)

        # Label smoothing if specified
        if self.label_smoothing > 0.0:
            smoothed_targets = torch.full_like(log_p, self.label_smoothing / (num_classes - 1))
            smoothed_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            ce_loss = -(smoothed_targets * log_p).sum(dim=-1)
        else:
            ce_loss = F.nll_loss(log_p, targets, reduction="none")

        # Probability of true class
        p = torch.exp(-ce_loss)

        # Focal weighting: (1 - p_t)^gamma
        focal_weight = (1.0 - p) ** self.gamma

        # Class balancing weight (alpha)
        if self.weight is not None:
            alpha = self.weight[targets]
            focal_weight = focal_weight * alpha

        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy Loss with Label Smoothing regularization.
    """

    def __init__(self, smoothing: float = 0.1, weight: Optional[torch.Tensor] = None, reduction: str = "mean"):
        super().__init__()
        self.smoothing = smoothing
        self.register_buffer("weight", weight)
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = -(true_dist * log_probs).sum(dim=-1)

        if self.weight is not None:
            weights = self.weight[targets]
            loss = loss * weights

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
