"""
MediFusion AI - Centralized Kaggle Dataset Download Manager
Downloads and caches all medical image datasets via kagglehub.
Provides a single entry point to resolve dataset paths for training and inference.
"""

import os
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# ─── Dataset Registry ───────────────────────────────────────────────
# Maps dataset keys to their Kaggle slugs and expected subdirectory structures.

DATASET_REGISTRY: Dict[str, dict] = {
    "chest_xray": {
        "slug": "paultimothymooney/chest-xray-pneumonia",
        "version": 2,
        "subdir": "chest_xray",
        "description": "Chest X-Ray Pneumonia (2-class: NORMAL, PNEUMONIA)",
        "num_classes": 2,
        "class_names": ["NORMAL", "PNEUMONIA"],
    },
    "brain_tumor": {
        "slug": "masoudnickparvar/brain-tumor-mri-dataset",
        "version": None,  # latest
        "subdir": "",
        "description": "Brain Tumor MRI (4-class: Glioma, Meningioma, No Tumor, Pituitary)",
        "num_classes": 4,
        "class_names": ["glioma", "meningioma", "notumor", "pituitary"],
    },
    "skin_cancer": {
        "slug": "kmader/skin-cancer-mnist-ham10000",
        "version": None,
        "subdir": "",
        "description": "Skin Cancer HAM10000 (7-class dermatological lesion classification)",
        "num_classes": 7,
        "class_names": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
    },
    "retinopathy": {
        "slug": "amanneo/diabetic-retinopathy-resized-arranged",
        "version": None,
        "subdir": "",
        "description": "Diabetic Retinopathy (5-class severity grading)",
        "num_classes": 5,
        "class_names": ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"],
    },
    "blood_cell": {
        "slug": "paultimothymooney/blood-cells",
        "version": None,
        "subdir": "dataset2-master/dataset2-master",
        "description": "Blood Cell Classification (4-class hematological microscopy)",
        "num_classes": 4,
        "class_names": ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"],
    },
}


def download_dataset(dataset_key: str) -> str:
    """
    Download a dataset from Kaggle using kagglehub and return its local path.

    Args:
        dataset_key: One of the keys in DATASET_REGISTRY.

    Returns:
        Absolute path to the downloaded dataset root directory.
    """
    if dataset_key not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset key '{dataset_key}'. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    info = DATASET_REGISTRY[dataset_key]
    slug = info["slug"]
    version = info["version"]
    subdir = info["subdir"]

    try:
        import kagglehub

        logger.info(f"Downloading dataset '{dataset_key}' ({slug})...")

        if version is not None:
            path = kagglehub.dataset_download(slug, force_download=False)
        else:
            path = kagglehub.dataset_download(slug, force_download=False)

        # Navigate to the expected subdirectory if needed
        if subdir:
            full_path = os.path.join(path, subdir)
        else:
            full_path = path

        if not os.path.exists(full_path):
            logger.warning(
                f"Expected path '{full_path}' does not exist. "
                f"Falling back to download root: {path}"
            )
            full_path = path

        logger.info(f"Dataset '{dataset_key}' available at: {full_path}")
        return full_path

    except ImportError:
        raise ImportError(
            "kagglehub is required for dataset downloads. "
            "Install it with: pip install kagglehub"
        )
    except Exception as e:
        logger.error(f"Failed to download dataset '{dataset_key}': {e}")
        raise


def download_all_datasets() -> Dict[str, str]:
    """
    Download all registered datasets and return a mapping of key → path.
    Skips datasets that fail to download (logs error but continues).
    """
    paths = {}
    for key in DATASET_REGISTRY:
        try:
            paths[key] = download_dataset(key)
        except Exception as e:
            logger.error(f"Skipping dataset '{key}': {e}")
    return paths


def get_dataset_info(dataset_key: str) -> dict:
    """Get metadata for a registered dataset."""
    if dataset_key not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset key: {dataset_key}")
    return DATASET_REGISTRY[dataset_key].copy()


def resolve_dataset_path(dataset_key: str, config: dict) -> str:
    """
    Resolve the dataset path from config or download if not available.

    Checks config['dataset'][dataset_key + '_path'] first. If not present
    or path doesn't exist, downloads via kagglehub.
    """
    config_key = f"{dataset_key}_path"
    config_path = config.get("dataset", {}).get(config_key)

    if config_path and os.path.exists(config_path):
        logger.info(f"Using configured path for '{dataset_key}': {config_path}")
        return config_path

    logger.info(f"No valid config path for '{dataset_key}', downloading from Kaggle...")
    return download_dataset(dataset_key)
