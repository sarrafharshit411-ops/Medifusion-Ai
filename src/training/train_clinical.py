"""
MediFusion AI - Clinical Model Training Pipeline
Trains XGBoost on symptom data augmented with synthetic clinical features.

NOTE: The clinical features are SYNTHETIC (generated from symptom correlations).
This model demonstrates the clinical modality architecture. In a real deployment,
actual patient clinical data would be used.
"""

import os
import sys
import logging
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.symptom_dataset import (
    load_symptom_data, generate_synthetic_clinical_features,
    SYMPTOM_FEATURES, SYMPTOM_LABELS,
)
from src.models.clinical_model import ClinicalModel
from src.utils.helpers import load_config, set_seed, ensure_dir, save_metrics, setup_logging

logger = logging.getLogger(__name__)


def train_clinical_model(config: dict = None) -> Dict:
    """
    Full training pipeline for the clinical risk model.
    
    Uses symptom data augmented with synthetic clinical features.
    """
    setup_logging()
    
    if config is None:
        config = load_config()
    
    set_seed(config["training"]["seed"])
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(project_root, config["paths"]["models_dir"])
    reports_dir = os.path.join(project_root, config["paths"]["reports_dir"])
    ensure_dir(models_dir)
    ensure_dir(reports_dir)
    
    clin_config = config["clinical_model"]
    
    # Load symptom data — resolve path relative to project root
    sym_path = config["dataset"]["symptom_path"]
    if not os.path.isabs(sym_path):
        sym_path = os.path.join(project_root, sym_path)
    X_train_sym, X_test_sym, y_train, y_test, label_encoder, data_info = load_symptom_data(
        csv_path=sym_path,
        test_size=0.2,
        seed=config["training"]["seed"],
    )
    
    # Generate synthetic clinical features
    clinical_train, clinical_feature_names = generate_synthetic_clinical_features(
        X_train_sym, seed=config["training"]["seed"]
    )
    clinical_test, _ = generate_synthetic_clinical_features(
        X_test_sym, seed=config["training"]["seed"] + 1
    )
    
    # Combine symptom + clinical features
    all_feature_names = SYMPTOM_FEATURES + clinical_feature_names
    X_train = np.hstack([X_train_sym, clinical_train])
    X_test = np.hstack([X_test_sym, clinical_test])
    
    logger.info(f"Clinical model: {X_train.shape[1]} features ({len(SYMPTOM_FEATURES)} symptom + {len(clinical_feature_names)} clinical)")
    logger.info(f"Training: {X_train.shape[0]} samples, Testing: {X_test.shape[0]} samples")
    logger.info("WARNING: Clinical features are SYNTHETIC for demonstration purposes")
    
    # Create and train model
    model = ClinicalModel(config=clin_config)
    model.train(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        feature_names=all_feature_names,
        label_encoder=label_encoder,
    )
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    report = classification_report(
        y_test, y_pred,
        target_names=SYMPTOM_LABELS,
        output_dict=True,
    )
    
    logger.info(f"Clinical model test accuracy: {accuracy:.4f}")
    logger.info(f"Classification report:\n{classification_report(y_test, y_pred, target_names=SYMPTOM_LABELS)}")
    
    # Save model
    model.save(
        model_path=os.path.join(models_dir, "clinical_model.joblib"),
        scaler_path=os.path.join(models_dir, "clinical_scaler.joblib"),
        encoder_path=os.path.join(models_dir, "clinical_label_encoder.joblib"),
    )
    
    # Save results
    results = {
        "model": "XGBoost",
        "dataset": "Symptom + Synthetic Clinical",
        "classes": SYMPTOM_LABELS,
        "num_features": len(all_feature_names),
        "feature_names": all_feature_names,
        "clinical_features_note": "SYNTHETIC - generated from symptom correlations for demonstration",
        "train_size": len(X_train),
        "test_size": len(X_test),
        "test_accuracy": accuracy,
        "classification_report": report,
    }
    
    save_metrics(results, os.path.join(reports_dir, "clinical_model_metrics.json"))
    
    np.savez(
        os.path.join(reports_dir, "clinical_model_test_preds.npz"),
        predictions=y_pred,
        labels=y_test,
        probabilities=y_proba,
    )
    
    return results


if __name__ == "__main__":
    train_clinical_model()
