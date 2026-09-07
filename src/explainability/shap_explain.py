"""
MediFusion AI - SHAP Explainability
SHAP (SHapley Additive exPlanations) for XGBoost models.

Provides feature importance explanations for symptom and clinical models.
"""

import io
import logging
import base64
from typing import Dict, List, Optional

import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    SHAP TreeExplainer wrapper for XGBoost models.
    
    Usage:
        explainer = SHAPExplainer(xgb_model, feature_names)
        result = explainer.explain(sample, class_names)
    """
    
    def __init__(self, model, feature_names: List[str]):
        """
        Args:
            model: Trained XGBoost model (the internal .model from SymptomModel/ClinicalModel)
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)
        logger.info(f"SHAP TreeExplainer initialized with {len(feature_names)} features")
    
    def explain(
        self,
        X: np.ndarray,
        class_names: Optional[List[str]] = None,
        top_n: int = 10,
    ) -> Dict:
        """
        Generate SHAP explanation for input sample(s).
        
        Args:
            X: (1, n_features) or (n_samples, n_features) input array
            class_names: Optional class labels for multi-class
            top_n: Number of top features to return
            
        Returns:
            dict with:
                - feature_importance: sorted list of dicts with feature and importance
                - per_class_importance: dict of class_name -> list of top features
                - plot_base64: base64-encoded feature importance plot
        """
        raw_shap = self.explainer.shap_values(X)
        
        # Normalize raw_shap to 3D array: (n_samples, n_features, n_classes) or (n_samples, n_features, 1)
        if isinstance(raw_shap, list):
            # List of (n_samples, n_features) -> stack to (n_samples, n_features, n_classes)
            shap_array = np.stack(raw_shap, axis=-1)
        elif isinstance(raw_shap, np.ndarray):
            if raw_shap.ndim == 2:
                # (n_samples, n_features) -> (n_samples, n_features, 1)
                shap_array = raw_shap[:, :, np.newaxis]
            elif raw_shap.ndim == 3:
                shap_array = raw_shap
            else:
                shap_array = raw_shap.reshape(1, len(self.feature_names), -1)
        else:
            shap_array = np.array(raw_shap)
        
        # Overall feature importance: mean over samples and classes of |SHAP|
        # shap_array is (n_samples, n_features, n_classes)
        abs_shap = np.abs(shap_array)  # (n_samples, n_features, n_classes)
        mean_importance = abs_shap.mean(axis=(0, 2))  # (n_features,) 1D array
        
        # Sort features by importance
        sorted_indices = np.argsort(mean_importance)[::-1][:top_n]
        feature_importance = [
            {
                "feature": str(self.feature_names[int(idx)]),
                "importance": float(mean_importance[int(idx)]),
            }
            for idx in sorted_indices
        ]
        
        # Per-class breakdown
        per_class_importance = {}
        if class_names and shap_array.shape[2] == len(class_names):
            for cls_idx, cls_name in enumerate(class_names):
                cls_abs = abs_shap[:, :, cls_idx].mean(axis=0)  # (n_features,)
                cls_sorted = np.argsort(cls_abs)[::-1][:top_n]
                per_class_importance[cls_name] = [
                    {
                        "feature": str(self.feature_names[int(idx)]),
                        "importance": float(cls_abs[int(idx)]),
                    }
                    for idx in cls_sorted
                ]
        
        # Generate plot
        plot_b64 = self._create_importance_plot(mean_importance, sorted_indices)
        
        return {
            "feature_importance": feature_importance,
            "per_class_importance": per_class_importance,
            "plot_base64": plot_b64,
        }
    
    def _create_importance_plot(
        self,
        mean_importance: np.ndarray,
        sorted_indices: np.ndarray,
    ) -> str:
        """Create a horizontal bar chart of feature importance."""
        top_features = [self.feature_names[int(i)] for i in sorted_indices]
        top_values = [float(mean_importance[int(i)]) for i in sorted_indices]
        
        fig, ax = plt.subplots(figsize=(8, max(4, len(top_features) * 0.45)))
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
        
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_values[::-1], color=colors[::-1], edgecolor="white", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features[::-1], fontsize=10)
        ax.set_xlabel("Mean |SHAP value|", fontsize=11)
        ax.set_title("Clinical Feature Importance (SHAP)", fontsize=13, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
