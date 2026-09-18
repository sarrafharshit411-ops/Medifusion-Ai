"""
MediFusion AI - Advanced Multimodal Fusion Hyperparameter Tuning & Ensembling
Optimizes:
    1. ConcatFusion (MLP with tunable depth, width, dropout, weight decay)
    2. GatedAttentionFusion (projection dimension, attention heads, dropout, learning rate)
    3. Weighted Soft-Voting Ensemble (combining both modalities for peak accuracy)
Outputs tuned model checkpoints and comprehensive evaluation metrics.
"""

import os
import sys
import copy
import time
import logging
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.models.fusion_model import (
    ConcatFusion, GatedAttentionFusion,
    UNIFIED_CLASSES, NUM_UNIFIED_CLASSES,
)
from src.training.train_fusion import generate_fusion_data
from src.utils.helpers import load_config, set_seed, get_device, ensure_dir, save_metrics, setup_logging

logger = logging.getLogger(__name__)


def train_eval_single_fusion(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> Tuple[nn.Module, float, Dict[str, Any]]:
    """Train a fusion model instance and evaluate best validation performance."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        for img, sym, clin, mask, target in train_loader:
            img, sym, clin = img.to(device), sym.to(device), clin.to(device)
            mask, target = mask.to(device), target.to(device)

            optimizer.zero_grad()
            logits = model(img, sym, clin, mask)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for img, sym, clin, mask, target in val_loader:
                img, sym, clin = img.to(device), sym.to(device), clin.to(device)
                mask = mask.to(device)
                logits = model(img, sym, clin, mask)
                preds = logits.argmax(dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(target.numpy())

        val_acc = accuracy_score(val_targets, val_preds)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    model.eval()

    # Final detailed evaluation
    all_preds, all_probs, all_targets = [], [], []
    with torch.no_grad():
        for img, sym, clin, mask, target in val_loader:
            img, sym, clin = img.to(device), sym.to(device), clin.to(device)
            mask = mask.to(device)
            logits = model(img, sym, clin, mask)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.append(probs)
            all_targets.extend(target.numpy())

    all_probs = np.vstack(all_probs)
    report = classification_report(
        all_targets, all_preds,
        target_names=UNIFIED_CLASSES,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": float(best_val_acc),
        "precision_macro": float(precision_score(all_targets, all_preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(all_targets, all_preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(all_targets, all_preds, average="macro", zero_division=0)),
        "classification_report": report,
        "probabilities": all_probs,
        "predictions": np.array(all_preds),
        "targets": np.array(all_targets),
    }

    return model, best_val_acc, metrics


def tune_concat_fusion(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 25,
) -> Tuple[ConcatFusion, Dict[str, Any]]:
    """Grid search hyperparameter configurations for ConcatFusion."""
    logger.info("\n--- Tuning ConcatFusion Architecture & Hyperparameters ---")
    grid = [
        {"hidden_dim": 128, "dropout": 0.2, "lr": 5e-4, "weight_decay": 1e-4},
        {"hidden_dim": 256, "dropout": 0.2, "lr": 3e-4, "weight_decay": 1e-4},
        {"hidden_dim": 256, "dropout": 0.3, "lr": 5e-4, "weight_decay": 1e-4},
        {"hidden_dim": 512, "dropout": 0.3, "lr": 3e-4, "weight_decay": 1e-5},
        {"hidden_dim": 512, "dropout": 0.4, "lr": 5e-4, "weight_decay": 1e-4},
    ]

    best_acc = 0.0
    best_model = None
    best_cfg = None
    best_metrics = None

    for i, cfg in enumerate(grid, 1):
        logger.info(f"Concat Trial {i}/{len(grid)}: {cfg}")
        model = ConcatFusion(
            hidden_dim=cfg["hidden_dim"],
            dropout=cfg["dropout"],
        )
        trained_model, val_acc, metrics = train_eval_single_fusion(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            device=device,
        )
        logger.info(f"  -> Val Accuracy: {val_acc:.4f}, F1: {metrics['f1_macro']:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = trained_model
            best_cfg = cfg
            best_metrics = metrics

    logger.info(f"[BEST ConcatFusion] Acc: {best_acc:.4f} with params: {best_cfg}")
    return best_model, {"config": best_cfg, "metrics": best_metrics}


def tune_gated_attention_fusion(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 25,
) -> Tuple[GatedAttentionFusion, Dict[str, Any]]:
    """Grid search hyperparameter configurations for GatedAttentionFusion."""
    logger.info("\n--- Tuning GatedAttentionFusion Architecture & Hyperparameters ---")
    grid = [
        {"projection_dim": 64, "num_heads": 2, "dropout": 0.2, "lr": 5e-4, "weight_decay": 1e-4},
        {"projection_dim": 128, "num_heads": 2, "dropout": 0.2, "lr": 3e-4, "weight_decay": 1e-4},
        {"projection_dim": 128, "num_heads": 4, "dropout": 0.3, "lr": 5e-4, "weight_decay": 1e-4},
        {"projection_dim": 256, "num_heads": 4, "dropout": 0.3, "lr": 3e-4, "weight_decay": 1e-5},
    ]

    best_acc = 0.0
    best_model = None
    best_cfg = None
    best_metrics = None

    for i, cfg in enumerate(grid, 1):
        logger.info(f"Gated Trial {i}/{len(grid)}: {cfg}")
        model = GatedAttentionFusion(
            projection_dim=cfg["projection_dim"],
            num_heads=cfg["num_heads"],
            dropout=cfg["dropout"],
        )
        trained_model, val_acc, metrics = train_eval_single_fusion(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            device=device,
        )
        logger.info(f"  -> Val Accuracy: {val_acc:.4f}, F1: {metrics['f1_macro']:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = trained_model
            best_cfg = cfg
            best_metrics = metrics

    logger.info(f"[BEST GatedAttentionFusion] Acc: {best_acc:.4f} with params: {best_cfg}")
    return best_model, {"config": best_cfg, "metrics": best_metrics}


def evaluate_ensemble(
    concat_probs: np.ndarray,
    gated_probs: np.ndarray,
    targets: np.ndarray,
    w_concat: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate soft-voting ensemble of ConcatFusion and GatedAttentionFusion."""
    w_gated = 1.0 - w_concat
    ensemble_probs = w_concat * concat_probs + w_gated * gated_probs
    ensemble_preds = ensemble_probs.argmax(axis=1)

    acc = float(accuracy_score(targets, ensemble_preds))
    prec = float(precision_score(targets, ensemble_preds, average="macro", zero_division=0))
    rec = float(recall_score(targets, ensemble_preds, average="macro", zero_division=0))
    f1 = float(f1_score(targets, ensemble_preds, average="macro", zero_division=0))

    report = classification_report(
        targets, ensemble_preds,
        target_names=UNIFIED_CLASSES,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
        "w_concat": w_concat,
        "w_gated": w_gated,
        "classification_report": report,
        "probabilities": ensemble_probs,
        "predictions": ensemble_preds,
    }


def run_fusion_tuning(epochs_per_trial: int = 25):
    """Run full hyperparameter search and soft-voting ensembling on multimodal fusion."""
    setup_logging()
    config = load_config(os.path.join(project_root, "configs", "config.yaml"))
    set_seed(config["training"]["seed"])
    device = get_device()

    models_dir = os.path.join(project_root, config["paths"]["models_dir"])
    reports_dir = os.path.join(project_root, config["paths"]["reports_dir"])
    ensure_dir(models_dir)
    ensure_dir(reports_dir)

    logger.info("=" * 65)
    logger.info("MULTIMODAL FUSION HYPERPARAMETER OPTIMIZATION & ENSEMBLING")
    logger.info("=" * 65)

    data = generate_fusion_data(config, device)

    # Train / Val Split
    n = len(data["labels"])
    generator = torch.Generator().manual_seed(config["training"]["seed"])
    indices = torch.randperm(n, generator=generator)
    val_size = int(n * 0.15)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    batch_size = config["fusion_model"].get("batch_size", 32)
    train_dataset = TensorDataset(
        data["image_emb"][train_idx],
        data["symptom_emb"][train_idx],
        data["clinical_emb"][train_idx],
        data["masks"][train_idx],
        data["labels"][train_idx],
    )
    val_dataset = TensorDataset(
        data["image_emb"][val_idx],
        data["symptom_emb"][val_idx],
        data["clinical_emb"][val_idx],
        data["masks"][val_idx],
        data["labels"][val_idx],
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 1. Tune ConcatFusion
    best_concat, concat_res = tune_concat_fusion(train_loader, val_loader, device, epochs=epochs_per_trial)
    torch.save(best_concat.state_dict(), os.path.join(models_dir, "fusion_concat.pth"))

    # 2. Tune GatedAttentionFusion
    best_gated, gated_res = tune_gated_attention_fusion(train_loader, val_loader, device, epochs=epochs_per_trial)
    torch.save(best_gated.state_dict(), os.path.join(models_dir, "fusion_gated.pth"))

    # 3. Optimize Weighted Ensemble
    val_targets = concat_res["metrics"]["targets"]
    best_ensemble_metrics = None
    best_w = 0.5
    for w in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        ens_metrics = evaluate_ensemble(
            concat_res["metrics"]["probabilities"],
            gated_res["metrics"]["probabilities"],
            val_targets,
            w_concat=w,
        )
        if best_ensemble_metrics is None or ens_metrics["accuracy"] > best_ensemble_metrics["accuracy"]:
            best_ensemble_metrics = ens_metrics
            best_w = w

    logger.info("\n" + "=" * 65)
    logger.info("FINAL FUSION BENCHMARK")
    logger.info("=" * 65)
    logger.info(f"ConcatFusion Val Acc:         {concat_res['metrics']['accuracy']:.4f}, F1: {concat_res['metrics']['f1_macro']:.4f}")
    logger.info(f"GatedAttentionFusion Val Acc:  {gated_res['metrics']['accuracy']:.4f}, F1: {gated_res['metrics']['f1_macro']:.4f}")
    logger.info(f"Ensemble (w={best_w:.2f}) Val Acc:     {best_ensemble_metrics['accuracy']:.4f}, F1: {best_ensemble_metrics['f1_macro']:.4f}")
    logger.info("=" * 65)

    # Save summary report
    summary = {
        "concat_fusion": {
            "best_config": concat_res["config"],
            "accuracy": concat_res["metrics"]["accuracy"],
            "f1_macro": concat_res["metrics"]["f1_macro"],
            "classification_report": concat_res["metrics"]["classification_report"],
        },
        "gated_attention_fusion": {
            "best_config": gated_res["config"],
            "accuracy": gated_res["metrics"]["accuracy"],
            "f1_macro": gated_res["metrics"]["f1_macro"],
            "classification_report": gated_res["metrics"]["classification_report"],
        },
        "ensemble": {
            "best_w_concat": best_w,
            "accuracy": best_ensemble_metrics["accuracy"],
            "f1_macro": best_ensemble_metrics["f1_macro"],
            "classification_report": best_ensemble_metrics["classification_report"],
        },
    }

    report_path = os.path.join(reports_dir, "fusion_tuning_report.json")
    save_metrics(summary, report_path)
    logger.info(f"Saved fusion tuning report to {report_path}")

    return summary


if __name__ == "__main__":
    run_fusion_tuning()
