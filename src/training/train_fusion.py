"""
MediFusion AI - Fusion Model Training Pipeline
Trains both ConcatFusion and GatedAttentionFusion models.

NOTE: Since the two datasets are NOT patient-paired, we create synthetic pairings
for demonstration:
- For "Pneumonia" class: pair X-ray embeddings with symptom rows both labeled Pneumonia
- For "Normal" class: use only image embeddings (symptom/clinical set to zero)
- For "Malaria"/"Typhoid": use only symptom/clinical embeddings (image set to zero)

This clearly demonstrates the fusion architecture while being honest about data limitations.
"""

import os
import sys
import logging
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.xray_dataset import load_xray_data
from src.data.symptom_dataset import (
    load_symptom_data, generate_synthetic_clinical_features,
    SYMPTOM_FEATURES, SYMPTOM_LABELS,
)
from src.models.image_model import ChestXRayModel
from src.models.fusion_model import (
    ConcatFusion, GatedAttentionFusion,
    UNIFIED_CLASSES, NUM_UNIFIED_CLASSES,
)
from src.utils.helpers import load_config, set_seed, get_device, ensure_dir, save_metrics, setup_logging

logger = logging.getLogger(__name__)

# Unified labels: Normal=0, Pneumonia=1, Malaria=2, Typhoid=3


def generate_fusion_data(config: dict, device: torch.device):
    """
    Generate training data for the fusion model by extracting embeddings
    from trained single-modality models and creating synthetic pairings.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(project_root, config["paths"]["models_dir"])
    
    # --- Image embeddings ---
    logger.info("Extracting image embeddings...")
    image_model = ChestXRayModel(num_classes=2, pretrained=False)
    image_model.load_state_dict(
        torch.load(os.path.join(models_dir, "image_model.pth"), map_location=device, weights_only=True)
    )
    image_model.to(device)
    image_model.eval()
    
    _, _, test_loader, data_info = load_xray_data(
        data_root=config["dataset"]["xray_path"],
        input_size=config["image_model"]["input_size"],
        val_split=config["training"]["val_split"],
        batch_size=config["image_model"]["batch_size"],
        seed=config["training"]["seed"],
    )
    
    # Extract embeddings from training data
    train_loader, _, _, _ = load_xray_data(
        data_root=config["dataset"]["xray_path"],
        input_size=config["image_model"]["input_size"],
        val_split=0.0001,  # Use almost all for embedding extraction
        batch_size=config["image_model"]["batch_size"],
        seed=config["training"]["seed"],
    )
    
    image_embeddings = []
    image_labels = []  # 0=NORMAL, 1=PNEUMONIA
    
    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            emb = image_model.get_embedding(images)
            image_embeddings.append(emb.cpu().numpy())
            image_labels.append(labels.numpy())
    
    image_embeddings = np.vstack(image_embeddings)
    image_labels = np.concatenate(image_labels)
    logger.info(f"Extracted {len(image_embeddings)} image embeddings")
    
    # --- Symptom data ---
    X_train_sym, X_test_sym, y_train_sym, y_test_sym, le, sym_info = load_symptom_data(
        csv_path=config["dataset"]["symptom_path"],
        test_size=0.2,
        seed=config["training"]["seed"],
    )
    
    # --- Clinical data ---
    clinical_train, clinical_names = generate_synthetic_clinical_features(
        X_train_sym, seed=config["training"]["seed"]
    )
    clinical_test, _ = generate_synthetic_clinical_features(
        X_test_sym, seed=config["training"]["seed"] + 1
    )
    
    # Combine symptom + clinical for clinical embedding
    clin_train_full = np.hstack([X_train_sym, clinical_train])
    clin_test_full = np.hstack([X_test_sym, clinical_test])
    
    # --- Create synthetic paired dataset ---
    # Map symptom labels to unified: Malaria=0->2, Pneumonia=1->1, Typhoid=2->3
    sym_to_unified = {0: 2, 1: 1, 2: 3}
    
    all_image_emb = []
    all_symptom_emb = []
    all_clinical_emb = []
    all_masks = []
    all_unified_labels = []
    
    rng = np.random.RandomState(config["training"]["seed"])
    
    # 1) Normal class: image-only (no symptom dataset has "Normal")
    normal_mask = image_labels == 0
    normal_embs = image_embeddings[normal_mask]
    n_normal = len(normal_embs)
    for i in range(n_normal):
        all_image_emb.append(normal_embs[i])
        all_symptom_emb.append(np.zeros(15, dtype=np.float32))
        all_clinical_emb.append(np.zeros(22, dtype=np.float32))
        all_masks.append([1, 0, 0])  # Only image available
        all_unified_labels.append(0)  # Normal
    
    # 2) Pneumonia class: pair image + symptom + clinical
    pneumonia_img_mask = image_labels == 1
    pneumonia_img_embs = image_embeddings[pneumonia_img_mask]
    pneumonia_sym_mask = y_train_sym == 1  # Pneumonia in symptom dataset
    pneumonia_sym_feats = X_train_sym[pneumonia_sym_mask]
    pneumonia_clin_feats = clin_train_full[pneumonia_sym_mask]
    
    n_pneumonia_pairs = min(len(pneumonia_img_embs), len(pneumonia_sym_feats))
    for i in range(n_pneumonia_pairs):
        sym_idx = i % len(pneumonia_sym_feats)
        all_image_emb.append(pneumonia_img_embs[i])
        all_symptom_emb.append(pneumonia_sym_feats[sym_idx])
        all_clinical_emb.append(pneumonia_clin_feats[sym_idx])
        all_masks.append([1, 1, 1])  # All modalities
        all_unified_labels.append(1)  # Pneumonia
    
    # 3) Malaria class: symptom + clinical only
    malaria_mask = y_train_sym == 0
    malaria_sym = X_train_sym[malaria_mask]
    malaria_clin = clin_train_full[malaria_mask]
    for i in range(len(malaria_sym)):
        all_image_emb.append(np.zeros(512, dtype=np.float32))
        all_symptom_emb.append(malaria_sym[i])
        all_clinical_emb.append(malaria_clin[i])
        all_masks.append([0, 1, 1])  # No image
        all_unified_labels.append(2)  # Malaria
    
    # 4) Typhoid class: symptom + clinical only
    typhoid_mask = y_train_sym == 2
    typhoid_sym = X_train_sym[typhoid_mask]
    typhoid_clin = clin_train_full[typhoid_mask]
    for i in range(len(typhoid_sym)):
        all_image_emb.append(np.zeros(512, dtype=np.float32))
        all_symptom_emb.append(typhoid_sym[i])
        all_clinical_emb.append(typhoid_clin[i])
        all_masks.append([0, 1, 1])  # No image
        all_unified_labels.append(3)  # Typhoid
    
    # Convert to tensors
    data = {
        "image_emb": torch.tensor(np.array(all_image_emb), dtype=torch.float32),
        "symptom_emb": torch.tensor(np.array(all_symptom_emb), dtype=torch.float32),
        "clinical_emb": torch.tensor(np.array(all_clinical_emb), dtype=torch.float32),
        "masks": torch.tensor(np.array(all_masks), dtype=torch.float32),
        "labels": torch.tensor(all_unified_labels, dtype=torch.long),
    }
    
    logger.info(f"Fusion dataset: {len(all_unified_labels)} samples")
    for i, cls in enumerate(UNIFIED_CLASSES):
        count = sum(1 for l in all_unified_labels if l == i)
        logger.info(f"  {cls}: {count}")
    
    return data


def train_fusion_model(config: dict = None) -> Dict:
    """Train both ConcatFusion and GatedAttentionFusion models."""
    setup_logging()
    
    if config is None:
        config = load_config()
    
    set_seed(config["training"]["seed"])
    device = get_device()
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(project_root, config["paths"]["models_dir"])
    reports_dir = os.path.join(project_root, config["paths"]["reports_dir"])
    ensure_dir(models_dir)
    ensure_dir(reports_dir)
    
    fusion_config = config["fusion_model"]
    
    # Generate fusion data
    data = generate_fusion_data(config, device)
    
    # Train/val split
    n = len(data["labels"])
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(config["training"]["seed"]))
    val_size = int(n * 0.15)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=fusion_config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=fusion_config["batch_size"], shuffle=False)
    
    results = {}
    
    for model_name, ModelClass in [("concat", ConcatFusion), ("gated", GatedAttentionFusion)]:
        logger.info(f"\n{'='*50}\nTraining {model_name} fusion model\n{'='*50}")
        
        model = ModelClass(
            image_dim=fusion_config["image_embedding_dim"],
            symptom_dim=fusion_config["symptom_embedding_dim"],
            clinical_dim=fusion_config["clinical_embedding_dim"],
            num_classes=NUM_UNIFIED_CLASSES,
            dropout=fusion_config["dropout"],
        )
        model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=fusion_config["learning_rate"])
        
        best_val_acc = 0
        
        for epoch in range(1, fusion_config["epochs"] + 1):
            # Train
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for img_e, sym_e, cli_e, mask, labels in train_loader:
                img_e, sym_e, cli_e = img_e.to(device), sym_e.to(device), cli_e.to(device)
                mask, labels = mask.to(device), labels.to(device)
                
                optimizer.zero_grad()
                logits = model(img_e, sym_e, cli_e, mask)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * labels.size(0)
                train_correct += (logits.argmax(1) == labels).sum().item()
                train_total += labels.size(0)
            
            # Validate
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for img_e, sym_e, cli_e, mask, labels in val_loader:
                    img_e, sym_e, cli_e = img_e.to(device), sym_e.to(device), cli_e.to(device)
                    mask, labels = mask.to(device), labels.to(device)
                    
                    logits = model(img_e, sym_e, cli_e, mask)
                    val_correct += (logits.argmax(1) == labels).sum().item()
                    val_total += labels.size(0)
            
            val_acc = val_correct / val_total if val_total > 0 else 0
            
            if epoch % 5 == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch}/{fusion_config['epochs']} | "
                    f"Train Loss: {train_loss/train_total:.4f}, "
                    f"Acc: {train_correct/train_total:.4f} | "
                    f"Val Acc: {val_acc:.4f}"
                )
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    model.state_dict(),
                    os.path.join(models_dir, f"fusion_{model_name}.pth")
                )
        
        logger.info(f"{model_name} fusion: Best val accuracy = {best_val_acc:.4f}")
        results[model_name] = {
            "best_val_accuracy": best_val_acc,
            "model_path": os.path.join(models_dir, f"fusion_{model_name}.pth"),
        }
    
    save_metrics(results, os.path.join(reports_dir, "fusion_model_metrics.json"))
    return results


if __name__ == "__main__":
    train_fusion_model()
