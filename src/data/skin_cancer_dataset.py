"""
MediFusion AI - Skin Cancer HAM10000 Dataset Module
PyTorch Dataset and data loading utilities for the HAM10000 skin lesion dataset.

Dataset: Skin Cancer MNIST: HAM10000 by Kaggle (kmader/skin-cancer-mnist-ham10000)
Structure: Contains HAM10000_metadata.csv + image directories (HAM10000_images_part_1/2)
           OR pre-organized ImageFolder structure.
Classes: 7
    - akiec: Actinic Keratoses
    - bcc: Basal Cell Carcinoma
    - bkl: Benign Keratosis
    - df: Dermatofibroma
    - mel: Melanoma
    - nv: Melanocytic Nevi
    - vasc: Vascular Lesions
"""

import os
import glob
import logging
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from PIL import Image

logger = logging.getLogger(__name__)

SKIN_CANCER_CLASSES = [
    "Actinic Keratoses",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic Nevi",
    "Vascular Lesions",
]
SKIN_CANCER_SHORT = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def get_train_transforms(input_size: int = 224) -> transforms.Compose:
    """Heavy augmentation pipeline for skin lesion images (addresses severe class imbalance)."""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.1)),
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


class HAM10000Dataset(Dataset):
    """
    Custom Dataset for HAM10000 that reads images from flat directories
    and uses metadata CSV for labels.
    Falls back to ImageFolder if the data is already organized by class.
    """

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def _find_images_from_metadata(data_root: str):
    """
    Try to load HAM10000 from metadata CSV + flat image directories.
    Returns (image_paths, labels, class_names) or None if structure doesn't match.
    """
    meta_path = None
    for candidate in [
        os.path.join(data_root, "HAM10000_metadata.csv"),
        os.path.join(data_root, "HAM10000_metadata"),
    ]:
        if os.path.exists(candidate):
            meta_path = candidate
            break

    if meta_path is None:
        return None

    try:
        df = pd.read_csv(meta_path)
    except Exception:
        return None

    if "dx" not in df.columns or "image_id" not in df.columns:
        return None

    # Build image path lookup
    image_dirs = []
    for d in os.listdir(data_root):
        full = os.path.join(data_root, d)
        if os.path.isdir(full) and "image" in d.lower():
            image_dirs.append(full)

    if not image_dirs:
        image_dirs = [data_root]

    image_lookup = {}
    for d in image_dirs:
        for f in os.listdir(d):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                name = os.path.splitext(f)[0]
                image_lookup[name] = os.path.join(d, f)

    # Map dx labels to integer indices
    class_names = sorted(df["dx"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    image_paths = []
    labels = []
    for _, row in df.iterrows():
        img_id = row["image_id"]
        if img_id in image_lookup:
            image_paths.append(image_lookup[img_id])
            labels.append(class_to_idx[row["dx"]])

    if len(image_paths) == 0:
        return None

    return image_paths, labels, class_names


def load_skin_cancer_data(
    data_root: str,
    input_size: int = 224,
    val_split: float = 0.15,
    test_split: float = 0.10,
    batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Load HAM10000 Skin Cancer dataset.

    Tries two strategies:
    1. Metadata CSV + flat image directories (original HAM10000 structure)
    2. ImageFolder with train/test subdirectories

    Returns:
        train_loader, val_loader, test_loader, info_dict
    """
    generator = torch.Generator().manual_seed(seed)

    # Strategy 1: Try metadata-based loading
    meta_result = _find_images_from_metadata(data_root)

    if meta_result is not None:
        image_paths, labels, class_names = meta_result
        labels = np.array(labels)
        num_samples = len(image_paths)

        # Create train/val/test split
        indices = torch.randperm(num_samples, generator=generator).tolist()
        test_size = int(num_samples * test_split)
        val_size = int(num_samples * val_split)

        test_indices = indices[:test_size]
        val_indices = indices[test_size:test_size + val_size]
        train_indices = indices[test_size + val_size:]

        train_paths = [image_paths[i] for i in train_indices]
        train_labels = labels[train_indices].tolist()
        val_paths = [image_paths[i] for i in val_indices]
        val_labels = labels[val_indices].tolist()
        test_paths = [image_paths[i] for i in test_indices]
        test_labels = labels[test_indices].tolist()

        train_dataset = HAM10000Dataset(train_paths, train_labels, get_train_transforms(input_size))
        val_dataset = HAM10000Dataset(val_paths, val_labels, get_eval_transforms(input_size))
        test_dataset = HAM10000Dataset(test_paths, test_labels, get_eval_transforms(input_size))

        train_targets = np.array(train_labels)

    else:
        # Strategy 2: ImageFolder with train/test structure
        train_dir = os.path.join(data_root, "train")
        test_dir = os.path.join(data_root, "test")

        if not os.path.exists(train_dir):
            # Try the hmnist organized structure
            for candidate in os.listdir(data_root):
                full = os.path.join(data_root, candidate)
                if os.path.isdir(full) and "train" in candidate.lower():
                    train_dir = full
                if os.path.isdir(full) and "test" in candidate.lower():
                    test_dir = full

        full_train = datasets.ImageFolder(train_dir, transform=get_train_transforms(input_size))
        full_train_eval = datasets.ImageFolder(train_dir, transform=get_eval_transforms(input_size))
        class_names = full_train.classes

        num_samples = len(full_train)
        indices = torch.randperm(num_samples, generator=generator).tolist()
        val_size = int(num_samples * val_split)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        train_dataset = Subset(full_train, train_indices)
        val_dataset = Subset(full_train_eval, val_indices)

        if os.path.exists(test_dir):
            test_dataset = datasets.ImageFolder(test_dir, transform=get_eval_transforms(input_size))
        else:
            # Use a small portion as test if no test dir
            test_size_count = int(num_samples * test_split)
            extra_test_indices = train_indices[:test_size_count]
            train_indices = train_indices[test_size_count:]
            train_dataset = Subset(full_train, train_indices)
            test_dataset = Subset(full_train_eval, extra_test_indices)

        train_targets = np.array(full_train.targets)[train_indices if isinstance(train_indices, list) else train_indices]

    # Compute class weights
    class_counts = np.bincount(train_targets.astype(int), minlength=len(class_names))
    class_weights = 1.0 / np.maximum(class_counts, 1)
    class_weights = class_weights / class_weights.sum() * len(class_names)

    # Weighted sampler
    sample_weights = class_weights[train_targets.astype(int)]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_targets),
        replacement=True,
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )

    info = {
        "class_names": list(class_names),
        "display_names": SKIN_CANCER_CLASSES,
        "num_classes": len(class_names),
        "train_size": len(train_targets),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "class_weights": torch.tensor(class_weights, dtype=torch.float32),
        "class_counts_train": class_counts.tolist(),
    }

    logger.info(
        f"Skin Cancer HAM10000 loaded: "
        f"train={info['train_size']}, val={info['val_size']}, test={info['test_size']}"
    )
    logger.info(f"Classes: {class_names}, Train counts: {class_counts.tolist()}")

    return train_loader, val_loader, test_loader, info
