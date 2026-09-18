"""
MediFusion AI - Symptom Model Training Pipeline
Trains XGBoost on the Symptom-Based Disease Prediction dataset.
"""

import os
import sys
import logging
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.symptom_dataset import load_symptom_data, SYMPTOM_FEATURES, SYMPTOM_LABELS
from src.models.symptom_model import SymptomModel
from src.utils.helpers import load_config, set_seed, ensure_dir, save_metrics, setup_logging

logger = logging.getLogger(__name__)


def train_symptom_model(config: dict = None) -> Dict:
    """
    Full training pipeline for the symptom disease prediction model.
    
    Returns:
        Dictionary with training results and test metrics.
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
    
    sym_config = config["symptom_model"]
    
    # Load data — resolve path relative to project root
    sym_path = config["dataset"]["symptom_path"]
    if not os.path.isabs(sym_path):
        sym_path = os.path.join(project_root, sym_path)
    X_train, X_test, y_train, y_test, label_encoder, data_info = load_symptom_data(
        csv_path=sym_path,
        test_size=0.2,
        seed=config["training"]["seed"],
    )
    
    logger.info(f"Training symptom model: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    
    # Create and train model
    model = SymptomModel(config=sym_config)
    model.train(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        feature_names=SYMPTOM_FEATURES,
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
    
    logger.info(f"Symptom model test accuracy: {accuracy:.4f}")
    logger.info(f"Classification report:\n{classification_report(y_test, y_pred, target_names=SYMPTOM_LABELS)}")
    
    # Save model
    model.save(
        model_path=os.path.join(models_dir, "symptom_model.joblib"),
        encoder_path=os.path.join(models_dir, "symptom_label_encoder.joblib"),
    )
    
    # Save results
    results = {
        "model": "XGBoost",
        "dataset": "Symptom-Based Disease Prediction",
        "classes": SYMPTOM_LABELS,
        "num_features": len(SYMPTOM_FEATURES),
        "feature_names": SYMPTOM_FEATURES,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "test_accuracy": accuracy,
        "classification_report": report,
        "test_predictions": y_pred,
        "test_labels": y_test,
        "test_probabilities": y_proba,
    }
    
    save_metrics(
        {k: v for k, v in results.items() if k not in ["test_predictions", "test_labels", "test_probabilities"]},
        os.path.join(reports_dir, "symptom_model_metrics.json"),
    )
    
    np.savez(
        os.path.join(reports_dir, "symptom_model_test_preds.npz"),
        predictions=y_pred,
        labels=y_test,
        probabilities=y_proba,
    )
    
    return results


if __name__ == "__main__":
    train_symptom_model()
