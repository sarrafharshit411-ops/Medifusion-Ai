"""
MediFusion AI - Unified Multi-Dataset Image Model Training Pipeline
Trains MedicalImageClassifier models on any registered medical image dataset.

Supports:
    - chest_xray:   Chest X-Ray Pneumonia (2-class)
    - brain_tumor:  Brain Tumor MRI (4-class)
    - skin_cancer:  Skin Cancer HAM10000 (7-class)
    - retinopathy:  Diabetic Retinopathy (5-class)
    - blood_cell:   Blood Cell Classification (4-class)

Usage:
    python -m src.training.train_multi_image --dataset brain_tumor
    python -m src.training.train_multi_image --dataset skin_cancer --epochs 15
    python -m src.training.train_multi_image --all
"""

import os
import sys
import time
import logging
import argparse
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.multi_image_model import MedicalImageClassifier, IMAGE_MODEL_CONFIGS
from src.data.dataset_downloader import resolve_dataset_path, get_dataset_info
from src.utils.helpers import load_config, set_seed, get_device, ensure_dir, save_metrics, setup_logging

logger = logging.getLogger(__name__)


# ─── Dataset Loader Dispatch ───────────────────────────────────────

def _get_data_loaders(dataset_key: str, config: dict):
    """Dispatch to the correct dataset loader based on dataset_key."""
    data_root = resolve_dataset_path(dataset_key, config)
    model_config = config.get("image_models", {}).get(dataset_key, {})
    input_size = model_config.get("input_size", 224)
    batch_size = model_config.get("batch_size", 32)
    val_split = config.get("training", {}).get("val_split", 0.15)
    seed = config.get("training", {}).get("seed", 42)

    if dataset_key == "chest_xray":
        from src.data.xray_dataset import load_xray_data
        return load_xray_data(
            data_root=data_root,
            input_size=input_size,
            val_split=val_split,
            batch_size=batch_size,
            seed=seed,
        )
    elif dataset_key == "brain_tumor":
        from src.data.brain_tumor_dataset import load_brain_tumor_data
        return load_brain_tumor_data(
            data_root=data_root,
            input_size=input_size,
            val_split=val_split,
            batch_size=batch_size,
            seed=seed,
        )
    elif dataset_key == "skin_cancer":
        from src.data.skin_cancer_dataset import load_skin_cancer_data
        return load_skin_cancer_data(
            data_root=data_root,
            input_size=input_size,
            val_split=val_split,
            batch_size=batch_size,
            seed=seed,
        )
    elif dataset_key == "retinopathy":
        from src.data.retinopathy_dataset import load_retinopathy_data
        return load_retinopathy_data(
            data_root=data_root,
            input_size=input_size,
            val_split=val_split,
            batch_size=batch_size,
            seed=seed,
        )
    elif dataset_key == "blood_cell":
        from src.data.blood_cell_dataset import load_blood_cell_data
        return load_blood_cell_data(
            data_root=data_root,
            input_size=input_size,
            val_split=val_split,
            batch_size=batch_size,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown dataset key: {dataset_key}")


# ─── Training Loop ──────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device) -> Dict:
    """Train for one epoch. Returns loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits, _ = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 30 == 0:
            logger.info(f"  Batch {batch_idx + 1}/{len(loader)}, Loss: {loss.item():.4f}")

    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
    }


def evaluate(model, loader, criterion, device) -> Dict:
    """Evaluate model on a data loader."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(logits, dim=1)
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
        "predictions": np.array(all_preds),
        "labels": np.array(all_labels),
        "probabilities": np.array(all_probs),
    }


# ─── Main Training Function ─────────────────────────────────────────

def train_single_image_model(
    dataset_key: str,
    config: dict,
    epochs: Optional[int] = None,
    learning_rate: Optional[float] = None,
) -> Dict:
    """
    Full training pipeline for a single image dataset.

    Args:
        dataset_key: Key identifying the dataset (e.g., 'brain_tumor')
        config: Full application config dict
        epochs: Override epoch count (optional)
        learning_rate: Override learning rate (optional)

    Returns:
        Dictionary with training history and final metrics.
    """
    set_seed(config["training"]["seed"])
    device = get_device()

    model_config = config.get("image_models", {}).get(dataset_key, {})
    reg_config = IMAGE_MODEL_CONFIGS.get(dataset_key, {})

    # Resolve hyperparameters
    num_epochs = epochs or model_config.get("epochs", 10)
    lr = learning_rate or model_config.get("learning_rate", 0.0001)
    weight_decay = model_config.get("weight_decay", 0.0001)
    patience = model_config.get("patience", 3)
    num_classes = reg_config.get("num_classes", 2)

    logger.info(f"")
    logger.info(f"{'─' * 60}")
    logger.info(f"Training [{dataset_key.upper()}] image model on {device}")
    logger.info(f"  Classes: {num_classes}, Epochs: {num_epochs}, LR: {lr}")
    logger.info(f"{'─' * 60}")

    # Paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(project_root, config["paths"]["models_dir"])
    reports_dir = os.path.join(project_root, config["paths"]["reports_dir"])
    ensure_dir(models_dir)
    ensure_dir(reports_dir)

    # Load data
    train_loader, val_loader, test_loader, data_info = _get_data_loaders(dataset_key, config)

    # Create model
    model = MedicalImageClassifier(
        dataset_key=dataset_key,
        num_classes=num_classes,
        pretrained=True,
    )
    model.to(device)

    # Loss with class weights for imbalance
    class_weights = data_info["class_weights"].to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer (only trainable params)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training loop with early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    model_filename = f"{dataset_key}_image_model.pth"

    logger.info(f"Starting training for {num_epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

        epoch_time = time.time() - epoch_start
        logger.info(
            f"[{dataset_key}] Epoch {epoch}/{num_epochs} ({epoch_time:.1f}s) | "
            f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}"
        )

        # Early stopping
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(models_dir, model_filename))
            logger.info(f"  → New best model saved (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    total_time = time.time() - start_time
    logger.info(f"[{dataset_key}] Training completed in {total_time:.1f}s")

    # Load best model and evaluate on test set
    model.load_state_dict(
        torch.load(os.path.join(models_dir, model_filename), map_location=device, weights_only=True)
    )
    test_metrics = evaluate(model, test_loader, criterion, device)

    logger.info(f"[{dataset_key}] Test Accuracy: {test_metrics['accuracy']:.4f}")

    # Save metrics
    results = {
        "model": f"MedicalImageClassifier ({dataset_key})",
        "backbone": reg_config.get("backbone", "resnet18"),
        "dataset": dataset_key,
        "description": reg_config.get("description", ""),
        "classes": data_info.get("class_names", []),
        "display_names": data_info.get("display_names", []),
        "num_classes": num_classes,
        "train_size": data_info["train_size"],
        "val_size": data_info["val_size"],
        "test_size": data_info["test_size"],
        "epochs_trained": len(history["train_loss"]),
        "best_val_loss": best_val_loss,
        "test_accuracy": test_metrics["accuracy"],
        "training_time_seconds": total_time,
        "history": history,
    }

    save_metrics(
        {k: v for k, v in results.items()
         if k not in ["test_predictions", "test_labels", "test_probabilities"]},
        os.path.join(reports_dir, f"{dataset_key}_image_model_metrics.json"),
    )

    # Save test predictions
    np.savez(
        os.path.join(reports_dir, f"{dataset_key}_image_model_test_preds.npz"),
        predictions=test_metrics["predictions"],
        labels=test_metrics["labels"],
        probabilities=test_metrics["probabilities"],
    )

    return results


def train_all_image_models(config: dict, datasets: Optional[list] = None) -> Dict[str, Dict]:
    """
    Train image models for all (or selected) datasets.

    Args:
        config: Full application config
        datasets: List of dataset keys to train. If None, trains all new datasets
                  (excludes chest_xray which has its own pipeline).

    Returns:
        Dictionary mapping dataset_key → training results.
    """
    if datasets is None:
        datasets = ["brain_tumor", "skin_cancer", "retinopathy", "blood_cell"]

    all_results = {}

    for i, key in enumerate(datasets, 1):
        logger.info(f"\n{'═' * 60}")
        logger.info(f"IMAGE MODEL {i}/{len(datasets)}: {key.upper()}")
        logger.info(f"{'═' * 60}")

        try:
            results = train_single_image_model(key, config)
            all_results[key] = results
            logger.info(f"[OK] {key} model trained. Test accuracy: {results['test_accuracy']:.4f}")
        except Exception as e:
            logger.error(f"[FAIL] {key} model training failed: {e}", exc_info=True)
            all_results[key] = {"error": str(e)}

    return all_results


# ─── CLI Entry Point ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train medical image classification models")
    parser.add_argument(
        "--dataset", "-d",
        choices=list(IMAGE_MODEL_CONFIGS.keys()),
        help="Dataset to train on",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Train all new image models",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    setup_logging()
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config = load_config(os.path.join(project_root, "configs", "config.yaml"))

    if args.all:
        train_all_image_models(config)
    elif args.dataset:
        train_single_image_model(args.dataset, config, epochs=args.epochs, learning_rate=args.lr)
    else:
        parser.print_help()
        print("\nExample: python -m src.training.train_multi_image --dataset brain_tumor")


if __name__ == "__main__":
    main()
