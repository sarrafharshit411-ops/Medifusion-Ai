"""
MediFusion AI - Symptom Dataset Module
Data loading and preprocessing for the Symptom-Based Disease Prediction dataset.

Dataset: Symptom-Based Disease Prediction by miltonmacgyver
Structure: Single CSV with 15 binary symptom columns + 'label' target column.
Labels: Malaria, Pneumonia, Typhoid (each 1666 samples, perfectly balanced).
"""

import os
import logging
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# The 15 symptom features as found in the actual dataset
SYMPTOM_FEATURES = [
    "fever", "cough", "headache", "nausea", "vomiting",
    "fatigue", "sore_throat", "chills", "body_pain",
    "loss_of_appetite", "abdominal_pain", "diarrhea",
    "sweating", "rapid_breathing", "dizziness"
]

SYMPTOM_LABELS = ["Malaria", "Pneumonia", "Typhoid"]


def load_symptom_data(
    csv_path: str,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder, Dict]:
    """
    Load and split symptom dataset.
    
    Returns:
        X_train, X_test, y_train, y_test, label_encoder, info_dict
    """
    df = pd.read_csv(csv_path)
    
    logger.info(f"Symptom dataset loaded: shape={df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Label distribution:\n{df['label'].value_counts().to_string()}")
    
    # Extract features and target
    X = df[SYMPTOM_FEATURES].values.astype(np.float32)
    y_raw = df["label"].values
    
    # Encode labels
    le = LabelEncoder()
    le.fit(SYMPTOM_LABELS)  # Fixed order
    y = le.transform(y_raw)
    
    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    
    info = {
        "feature_names": SYMPTOM_FEATURES,
        "num_features": len(SYMPTOM_FEATURES),
        "label_classes": list(le.classes_),
        "num_classes": len(le.classes_),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "label_distribution": {
            cls: int((y_raw == cls).sum()) for cls in SYMPTOM_LABELS
        },
    }
    
    logger.info(f"Train/test split: {info['train_size']}/{info['test_size']}")
    
    return X_train, X_test, y_train, y_test, le, info


def preprocess_symptoms(symptoms: Dict[str, int]) -> np.ndarray:
    """
    Convert a dict of symptom_name -> 0/1 values to a feature vector.
    Missing symptoms default to 0.
    
    Args:
        symptoms: Dict like {"fever": 1, "cough": 1, "headache": 0, ...}
        
    Returns:
        numpy array of shape (1, 15) with binary features in the correct order
    """
    vector = np.zeros((1, len(SYMPTOM_FEATURES)), dtype=np.float32)
    for i, feat in enumerate(SYMPTOM_FEATURES):
        vector[0, i] = float(symptoms.get(feat, 0))
    return vector


def generate_synthetic_clinical_features(
    X_symptoms: np.ndarray,
    seed: int = 42,
) -> Tuple[np.ndarray, List[str]]:
    """
    Generate synthetic clinical features correlated with symptom patterns.
    
    NOTE: These are NOT real clinical measurements. They are synthetic features
    generated for educational/demonstration purposes to showcase the clinical
    modality in the fusion architecture. In a real system, these would come
    from actual patient clinical data.
    
    Args:
        X_symptoms: Symptom feature matrix (n_samples, 15)
        
    Returns:
        clinical_features: (n_samples, 7) synthetic clinical data
        clinical_feature_names: list of 7 feature names
    """
    rng = np.random.RandomState(seed)
    n = X_symptoms.shape[0]
    
    clinical_feature_names = [
        "age", "temperature", "heart_rate",
        "bp_systolic", "bp_diastolic",
        "respiratory_rate", "oxygen_saturation"
    ]
    
    # Symptom severity score (sum of symptoms, normalized)
    severity = X_symptoms.sum(axis=1) / X_symptoms.shape[1]
    
    # Generate clinical features with symptom-correlated noise
    age = rng.normal(45, 18, n).clip(5, 90)
    temperature = 36.5 + severity * 2.5 + rng.normal(0, 0.3, n)
    heart_rate = 72 + severity * 30 + rng.normal(0, 8, n)
    bp_systolic = 120 + severity * 15 + rng.normal(0, 10, n)
    bp_diastolic = 80 + severity * 8 + rng.normal(0, 6, n)
    respiratory_rate = 16 + severity * 10 + rng.normal(0, 2, n)
    oxygen_saturation = 98 - severity * 6 + rng.normal(0, 1, n)
    oxygen_saturation = oxygen_saturation.clip(85, 100)
    
    clinical = np.column_stack([
        age, temperature, heart_rate,
        bp_systolic, bp_diastolic,
        respiratory_rate, oxygen_saturation
    ]).astype(np.float32)
    
    return clinical, clinical_feature_names


def preprocess_clinical(clinical_data: Dict[str, float], scaler=None) -> np.ndarray:
    """
    Convert a dict of clinical feature values to a feature vector.
    
    Args:
        clinical_data: Dict like {"age": 45, "temperature": 38.2, ...}
        scaler: Optional fitted StandardScaler
        
    Returns:
        numpy array of shape (1, 7)
    """
    clinical_features = [
        "age", "temperature", "heart_rate",
        "bp_systolic", "bp_diastolic",
        "respiratory_rate", "oxygen_saturation"
    ]
    vector = np.zeros((1, len(clinical_features)), dtype=np.float32)
    for i, feat in enumerate(clinical_features):
        if feat in clinical_data:
            vector[0, i] = float(clinical_data[feat])
        else:
            # Use reasonable defaults for missing values
            defaults = {
                "age": 45, "temperature": 37.0, "heart_rate": 75,
                "bp_systolic": 120, "bp_diastolic": 80,
                "respiratory_rate": 16, "oxygen_saturation": 98
            }
            vector[0, i] = defaults.get(feat, 0)
    
    if scaler is not None:
        vector = scaler.transform(vector)
    
    return vector
