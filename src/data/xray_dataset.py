"""
MediFusion AI - X-Ray Dataset Module
PyTorch Dataset and data loading utilities for the Chest X-Ray Pneumonia dataset.

Dataset: Chest X-Ray Images (Pneumonia) by Paul Mooney
Structure: ImageFolder with train/test/val splits, each containing NORMAL/ and PNEUMONIA/ subdirectories.
"""

import os
import logging
from typing import Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from PIL import Image

logger = logging.getLogger(__name__)


def get_train_transforms(input_size: int = 224) -> transforms.Compose:
    """Training augmentation pipeline."""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_eval_transforms(input_size: int = 224) -> transforms.Compose:
    """Evaluation / inference transform pipeline."""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def load_xray_data(
    data_root: str,
    input_size: int = 224,
    val_split: float = 0.15,
    batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Load Chest X-Ray dataset and create train/val/test DataLoaders.
    
    Since the provided val set is tiny (8+8 images), we split 15% from
    the training set to use as validation and keep the original test set.
    
    Returns:
        train_loader, val_loader, test_loader, info_dict
    """
    train_dir = os.path.join(data_root, "train")
    test_dir = os.path.join(data_root, "test")
    
    # Full training set with augmentation
    full_train_dataset = datasets.ImageFolder(train_dir, transform=get_train_transforms(input_size))
    # Same training set but with eval transforms (for val split)
    full_train_eval = datasets.ImageFolder(train_dir, transform=get_eval_transforms(input_size))
    # Test set
    test_dataset = datasets.ImageFolder(test_dir, transform=get_eval_transforms(input_size))
    
    class_names = full_train_dataset.classes  # ['NORMAL', 'PNEUMONIA']
    num_samples = len(full_train_dataset)
    
    # Create train/val split
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_samples, generator=generator).tolist()
    val_size = int(num_samples * val_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    train_subset = Subset(full_train_dataset, train_indices)
    val_subset = Subset(full_train_eval, val_indices)
    
    # Compute class weights for imbalanced data
    targets = np.array(full_train_dataset.targets)
    train_targets = targets[train_indices]
    class_counts = np.bincount(train_targets)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    
    # Weighted sampler for balanced training
    sample_weights = class_weights[train_targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_indices),
        replacement=True,
        generator=generator,
    )
    
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )
    
    info = {
        "class_names": class_names,
        "num_classes": len(class_names),
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "test_size": len(test_dataset),
        "class_weights": torch.tensor(class_weights, dtype=torch.float32),
        "class_counts_train": class_counts.tolist(),
    }
    
    logger.info(f"X-Ray dataset loaded: train={info['train_size']}, val={info['val_size']}, test={info['test_size']}")
    logger.info(f"Classes: {class_names}, Train class counts: {class_counts.tolist()}")
    
    return train_loader, val_loader, test_loader, info


def preprocess_single_image(image_path: str, input_size: int = 224) -> torch.Tensor:
    """Preprocess a single image for inference. Returns tensor of shape (1, 3, H, W)."""
    transform = get_eval_transforms(input_size)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    return tensor


def preprocess_image_bytes(image_bytes: bytes, input_size: int = 224) -> torch.Tensor:
    """Preprocess image from bytes (e.g., from an API upload). Returns tensor of shape (1, 3, H, W)."""
    import io
    transform = get_eval_transforms(input_size)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    return tensor
