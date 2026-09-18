"""
MediFusion AI - Image Model Training Pipeline
Trains ResNet18 on the Chest X-Ray Pneumonia dataset.
"""

import os
import sys
import time
import logging
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.xray_dataset import load_xray_data
from src.data.dataset_downloader import resolve_dataset_path
from src.models.image_model import ChestXRayModel
from src.utils.helpers import load_config, set_seed, ensure_dir, save_metrics, setup_logging

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Get the best available device: CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
        
        if (batch_idx + 1) % 20 == 0:
            logger.info(f"  Batch {batch_idx + 1}/{len(loader)}, Loss: {loss.item():.4f}")
    
    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
    }


def evaluate(model, loader, criterion, device) -> Dict:
    """Evaluate model on a data loader. Returns loss, accuracy, all predictions and labels."""
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


def train_image_model(config: dict = None) -> Dict:
    """
    Full training pipeline for the chest X-ray image model.
    
    Returns:
        Dictionary with training history and final metrics.
    """
    setup_logging()
    
    if config is None:
        config = load_config()
    
    set_seed(config["training"]["seed"])
    device = get_device()
    logger.info(f"Training image model on device: {device}")
    
    # Paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(project_root, config["paths"]["models_dir"])
    reports_dir = os.path.join(project_root, config["paths"]["reports_dir"])
    ensure_dir(models_dir)
    ensure_dir(reports_dir)
    
    img_config = config["image_model"]
    
    # Load data — auto-download if xray_path is empty
    xray_path = resolve_dataset_path("chest_xray", config)
    train_loader, val_loader, test_loader, data_info = load_xray_data(
        data_root=xray_path,
        input_size=img_config["input_size"],
        val_split=config["training"]["val_split"],
        batch_size=img_config["batch_size"],
        seed=config["training"]["seed"],
    )
    
    # Create model
    model = ChestXRayModel(num_classes=img_config["num_classes"], pretrained=True)
    model.to(device)
    
    # Loss with class weights for imbalance
    class_weights = data_info["class_weights"].to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer (only trainable params)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=img_config["learning_rate"],
        weight_decay=img_config["weight_decay"],
    )
    
    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=img_config["epochs"])
    
    # Training loop with early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    logger.info(f"Starting training for {img_config['epochs']} epochs...")
    start_time = time.time()
    
    for epoch in range(1, img_config["epochs"] + 1):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        # Record history
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        
        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch}/{img_config['epochs']} ({epoch_time:.1f}s) | "
            f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}"
        )
        
        # Early stopping
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(models_dir, "image_model.pth"))
            logger.info(f"  → New best model saved (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= img_config["patience"]:
                logger.info(f"Early stopping at epoch {epoch} (patience={img_config['patience']})")
                break
    
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.1f}s")
    
    # Load best model and evaluate on test set
    model.load_state_dict(
        torch.load(os.path.join(models_dir, "image_model.pth"), map_location=device, weights_only=True)
    )
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    
    # Save metrics
    results = {
        "model": "ResNet18",
        "dataset": "Chest X-Ray Pneumonia",
        "classes": data_info["class_names"],
        "train_size": data_info["train_size"],
        "val_size": data_info["val_size"],
        "test_size": data_info["test_size"],
        "epochs_trained": len(history["train_loss"]),
        "best_val_loss": best_val_loss,
        "test_accuracy": test_metrics["accuracy"],
        "test_predictions": test_metrics["predictions"],
        "test_labels": test_metrics["labels"],
        "test_probabilities": test_metrics["probabilities"],
        "training_time_seconds": total_time,
        "history": history,
    }
    
    save_metrics(
        {k: v for k, v in results.items() if k not in ["test_predictions", "test_labels", "test_probabilities"]},
        os.path.join(reports_dir, "image_model_metrics.json")
    )
    
    # Save test predictions for evaluation module
    np.savez(
        os.path.join(reports_dir, "image_model_test_preds.npz"),
        predictions=test_metrics["predictions"],
        labels=test_metrics["labels"],
        probabilities=test_metrics["probabilities"],
    )
    
    return results


if __name__ == "__main__":
    train_image_model()
