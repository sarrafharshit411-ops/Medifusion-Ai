"""
MediFusion AI - Evaluation Module
Comprehensive model evaluation with metrics and visualization.

Computes: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix
Compares: Image-only vs Symptoms-only vs Clinical-only vs Multimodal
"""

import os
import sys
import logging
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.helpers import load_config, ensure_dir, save_metrics, setup_logging

logger = logging.getLogger(__name__)


def compute_metrics(y_true, y_pred, y_proba, class_names, model_name="Model"):
    """
    Compute comprehensive classification metrics.
    
    Returns dict with all metrics.
    """
    n_classes = len(class_names)
    
    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    
    # ROC-AUC (requires probabilities)
    if y_proba is not None:
        try:
            if n_classes == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
                metrics["pr_auc"] = float(average_precision_score(y_true, y_proba[:, 1]))
            else:
                metrics["roc_auc"] = float(roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="macro"
                ))
                # PR-AUC per class for multiclass
                pr_aucs = []
                for i in range(n_classes):
                    binary_true = (y_true == i).astype(int)
                    if binary_true.sum() > 0:
                        pr_aucs.append(float(average_precision_score(binary_true, y_proba[:, i])))
                metrics["pr_auc"] = float(np.mean(pr_aucs)) if pr_aucs else 0.0
        except Exception as e:
            logger.warning(f"Could not compute AUC metrics: {e}")
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None
    
    # Per-class metrics
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    metrics["per_class"] = {
        cls: {
            "precision": report[cls]["precision"],
            "recall": report[cls]["recall"],
            "f1": report[cls]["f1-score"],
            "support": report[cls]["support"],
        }
        for cls in class_names if cls in report
    }
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names, save_path, title="Confusion Matrix"):
    """Generate and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix saved to {save_path}")


def plot_roc_curves(y_true, y_proba, class_names, save_path, title="ROC Curves"):
    """Generate and save ROC curves."""
    n_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_classes))
    
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        auc = roc_auc_score(y_true, y_proba[:, 1])
        ax.plot(fpr, tpr, color=colors[0], linewidth=2, label=f"{class_names[1]} (AUC={auc:.3f})")
    else:
        for i, (cls, color) in enumerate(zip(class_names, colors)):
            binary_true = (y_true == i).astype(int)
            if binary_true.sum() > 0:
                fpr, tpr, _ = roc_curve(binary_true, y_proba[:, i])
                auc = roc_auc_score(binary_true, y_proba[:, i])
                ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{cls} (AUC={auc:.3f})")
    
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"ROC curves saved to {save_path}")


def plot_model_comparison(all_metrics, save_path):
    """Bar chart comparing metrics across models."""
    models = list(all_metrics.keys())
    metric_names = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]
    
    x = np.arange(len(metric_labels))
    width = 0.8 / len(models)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    
    for i, (model, color) in enumerate(zip(models, colors)):
        values = [all_metrics[model].get(m, 0) or 0 for m in metric_names]
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=model, color=color, edgecolor="white")
    
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Model comparison saved to {save_path}")


def run_evaluation(config: dict = None):
    """Run full evaluation on all trained models and generate reports."""
    setup_logging()
    
    if config is None:
        config = load_config()
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    reports_dir = os.path.join(project_root, config["paths"]["reports_dir"])
    ensure_dir(reports_dir)
    
    all_metrics = {}
    
    # --- Image Model ---
    image_preds_path = os.path.join(reports_dir, "image_model_test_preds.npz")
    if os.path.exists(image_preds_path):
        logger.info("Evaluating Image Model...")
        data = np.load(image_preds_path)
        image_classes = ["NORMAL", "PNEUMONIA"]
        metrics = compute_metrics(data["labels"], data["predictions"], data["probabilities"], image_classes, "Image (ResNet18)")
        all_metrics["Image-Only"] = metrics
        
        plot_confusion_matrix(
            data["labels"], data["predictions"], image_classes,
            os.path.join(reports_dir, "image_confusion_matrix.png"),
            "Image Model - Confusion Matrix"
        )
        plot_roc_curves(
            data["labels"], data["probabilities"], image_classes,
            os.path.join(reports_dir, "image_roc_curves.png"),
            "Image Model - ROC Curves"
        )
    
    # --- Symptom Model ---
    symptom_preds_path = os.path.join(reports_dir, "symptom_model_test_preds.npz")
    if os.path.exists(symptom_preds_path):
        logger.info("Evaluating Symptom Model...")
        data = np.load(symptom_preds_path)
        symptom_classes = ["Malaria", "Pneumonia", "Typhoid"]
        metrics = compute_metrics(data["labels"], data["predictions"], data["probabilities"], symptom_classes, "Symptom (XGBoost)")
        all_metrics["Symptom-Only"] = metrics
        
        plot_confusion_matrix(
            data["labels"], data["predictions"], symptom_classes,
            os.path.join(reports_dir, "symptom_confusion_matrix.png"),
            "Symptom Model - Confusion Matrix"
        )
        plot_roc_curves(
            data["labels"], data["probabilities"], symptom_classes,
            os.path.join(reports_dir, "symptom_roc_curves.png"),
            "Symptom Model - ROC Curves"
        )
    
    # --- Clinical Model ---
    clinical_preds_path = os.path.join(reports_dir, "clinical_model_test_preds.npz")
    if os.path.exists(clinical_preds_path):
        logger.info("Evaluating Clinical Model...")
        data = np.load(clinical_preds_path)
        clinical_classes = ["Malaria", "Pneumonia", "Typhoid"]
        metrics = compute_metrics(data["labels"], data["predictions"], data["probabilities"], clinical_classes, "Clinical (XGBoost)")
        all_metrics["Clinical-Only"] = metrics
    
    # --- Comparison ---
    if all_metrics:
        plot_model_comparison(all_metrics, os.path.join(reports_dir, "model_comparison.png"))
        
        # Save combined metrics
        save_metrics(all_metrics, os.path.join(reports_dir, "all_model_metrics.json"))
        
        # Print summary table
        logger.info("\n" + "="*70)
        logger.info("MODEL COMPARISON SUMMARY")
        logger.info("="*70)
        header = f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}"
        logger.info(header)
        logger.info("-" * 70)
        for name, m in all_metrics.items():
            roc = f"{m['roc_auc']:.4f}" if m.get('roc_auc') else "N/A"
            logger.info(
                f"{name:<20} {m['accuracy']:>10.4f} {m['precision_macro']:>10.4f} "
                f"{m['recall_macro']:>10.4f} {m['f1_macro']:>10.4f} {roc:>10}"
            )
        logger.info("="*70)
    
    return all_metrics


if __name__ == "__main__":
    run_evaluation()
