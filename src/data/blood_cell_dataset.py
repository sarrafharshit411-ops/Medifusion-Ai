"""
MediFusion AI - Blood Cell Classification Dataset Module
PyTorch Dataset and data loading utilities for the Blood Cell Images dataset.

Dataset: Blood Cells by Paul Timothy Mooney (Kaggle)
Structure: ImageFolder with TRAIN/TEST/PREDICT splits, each containing
           EOSINOPHIL/ LYMPHOCYTE/ MONOCYTE/ NEUTROPHIL/ subdirectories.
Classes: 4 (Eosinophil, Lymphocyte, Monocyte, Neutrophil)
"""

import os
import logging
from typing import Tuple, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

BLOOD_CELL_CLASSES = ["Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]


def get_train_transforms(input_size: int = 224) -> transforms.Compose:
    """Training augmentation pipeline for blood cell microscopy images."""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180),  # Microscopy: orientation-invariant
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
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


def _find_train_test_dirs(data_root: str):
    """Auto-detect train and test directories with flexible naming."""
    train_dir = None
    test_dir = None

    if not os.path.exists(data_root):
        return None, None

    for d in os.listdir(data_root):
        full = os.path.join(data_root, d)
        if not os.path.isdir(full):
            continue
        dl = d.lower()
        if ("train" in dl) and train_dir is None:
            train_dir = full
        elif ("test" in dl) and test_dir is None:
            test_dir = full

    return train_dir, test_dir


def load_blood_cell_data(
    data_root: str,
    input_size: int = 224,
    val_split: float = 0.15,
    batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Load Blood Cell Classification dataset and create train/val/test DataLoaders.

    The dataset has structure:
      data_root/TRAIN/{EOSINOPHIL, LYMPHOCYTE, MONOCYTE, NEUTROPHIL}/
      data_root/TEST/{...}/

    Returns:
        train_loader, val_loader, test_loader, info_dict
    """
    generator = torch.Generator().manual_seed(seed)

    train_dir, test_dir = _find_train_test_dirs(data_root)

    # Nested structure check
    if train_dir is None:
        for subdir in os.listdir(data_root):
            sub_path = os.path.join(data_root, subdir)
            if os.path.isdir(sub_path):
                t, te = _find_train_test_dirs(sub_path)
                if t is not None:
                    train_dir, test_dir = t, te
                    data_root = sub_path
                    break

    if train_dir is None:
        raise FileNotFoundError(
            f"Blood cell training directory not found in: {data_root}. "
            f"Expected TRAIN/ or train/ subdirectory."
        )

    full_train = datasets.ImageFolder(train_dir, transform=get_train_transforms(input_size))
    full_train_eval = datasets.ImageFolder(train_dir, transform=get_eval_transforms(input_size))
    class_names = full_train.classes

    num_samples = len(full_train)
    indices = torch.randperm(num_samples, generator=generator).tolist()
    val_size = int(num_samples * val_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_subset = Subset(full_train, train_indices)
    val_subset = Subset(full_train_eval, val_indices)

    if test_dir and os.path.exists(test_dir):
        test_dataset = datasets.ImageFolder(test_dir, transform=get_eval_transforms(input_size))
    else:
        test_size = int(num_samples * 0.10)
        test_indices = train_indices[:test_size]
        train_indices = train_indices[test_size:]
        train_subset = Subset(full_train, train_indices)
        test_dataset = Subset(full_train_eval, test_indices)

    # Class weights
    targets = np.array(full_train.targets)
    train_targets = targets[train_indices]
    class_counts = np.bincount(train_targets, minlength=len(class_names))
    class_weights = 1.0 / np.maximum(class_counts, 1)
    class_weights = class_weights / class_weights.sum() * len(class_names)

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
        "class_names": list(class_names),
        "display_names": BLOOD_CELL_CLASSES,
        "num_classes": len(class_names),
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "test_size": len(test_dataset),
        "class_weights": torch.tensor(class_weights, dtype=torch.float32),
        "class_counts_train": class_counts.tolist(),
    }

    logger.info(
        f"Blood Cell dataset loaded: "
        f"train={info['train_size']}, val={info['val_size']}, test={info['test_size']}"
    )
    logger.info(f"Classes: {class_names}, Train counts: {class_counts.tolist()}")

    return train_loader, val_loader, test_loader, info
