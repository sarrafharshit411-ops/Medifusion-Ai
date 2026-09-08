"""
MediFusion AI - Brain Tumor MRI Dataset Module
PyTorch Dataset and data loading utilities for the Brain Tumor MRI Classification dataset.

Dataset: Brain Tumor MRI Dataset by Masoud Nickparvar (Kaggle)
Structure: ImageFolder with Training/ and Testing/ splits, each containing
           glioma/, meningioma/, notumor/, pituitary/ subdirectories.
Classes: 4 (Glioma, Meningioma, No Tumor, Pituitary)
"""

import os
import logging
from typing import Tuple, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

# Human-readable class labels
BRAIN_TUMOR_CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]


def get_train_transforms(input_size: int = 224) -> transforms.Compose:
    """Training augmentation pipeline for brain MRI scans."""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),
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


def load_brain_tumor_data(
    data_root: str,
    input_size: int = 224,
    val_split: float = 0.15,
    batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Load Brain Tumor MRI dataset and create train/val/test DataLoaders.

    The dataset structure is:
      data_root/Training/{glioma, meningioma, notumor, pituitary}/
      data_root/Testing/{glioma, meningioma, notumor, pituitary}/

    We split 15% of Training as validation.

    Returns:
        train_loader, val_loader, test_loader, info_dict
    """
    train_dir = os.path.join(data_root, "Training")
    test_dir = os.path.join(data_root, "Testing")

    # Fallback: try lowercase if capitalized doesn't exist
    if not os.path.exists(train_dir):
        train_dir = os.path.join(data_root, "training")
    if not os.path.exists(test_dir):
        test_dir = os.path.join(data_root, "testing")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(
            f"Brain tumor training directory not found. Tried: "
            f"{os.path.join(data_root, 'Training')} and {train_dir}"
        )

    # Load datasets
    full_train_dataset = datasets.ImageFolder(
        train_dir, transform=get_train_transforms(input_size)
    )
    full_train_eval = datasets.ImageFolder(
        train_dir, transform=get_eval_transforms(input_size)
    )
    test_dataset = datasets.ImageFolder(
        test_dir, transform=get_eval_transforms(input_size)
    )

    class_names = full_train_dataset.classes
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
    class_counts = np.bincount(train_targets, minlength=len(class_names))
    class_weights = 1.0 / np.maximum(class_counts, 1)
    class_weights = class_weights / class_weights.sum() * len(class_names)

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
        "display_names": BRAIN_TUMOR_CLASSES,
        "num_classes": len(class_names),
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "test_size": len(test_dataset),
        "class_weights": torch.tensor(class_weights, dtype=torch.float32),
        "class_counts_train": class_counts.tolist(),
    }

    logger.info(
        f"Brain Tumor MRI dataset loaded: "
        f"train={info['train_size']}, val={info['val_size']}, test={info['test_size']}"
    )
    logger.info(f"Classes: {class_names}, Train class counts: {class_counts.tolist()}")

    return train_loader, val_loader, test_loader, info
