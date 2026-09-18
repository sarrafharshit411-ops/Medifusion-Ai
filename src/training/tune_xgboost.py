"""
MediFusion AI - Advanced Hyperparameter Tuning for XGBoost Models
Performs randomized search with stratified K-fold cross-validation on:
    1. Symptom Disease Prediction Model
    2. Clinical Risk Stratification Model
Outputs tuned models, test evaluation metrics, and comparison reports.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Tuple

import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.data.symptom_dataset import (
    load_symptom_data, generate_synthetic_clinical_features,
    SYMPTOM_FEATURES, SYMPTOM_LABELS,
)
from src.utils.helpers import load_config, set_seed, ensure_dir, save_metrics, setup_logging

logger = logging.getLogger(__name__)


PARAM_DISTRIBUTIONS = {
    "max_depth": [3, 4, 5, 6, 7, 8, 10],
    "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2],
    "n_estimators": [100, 150, 200, 250, 300, 400],
    "subsample": [0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 2, 3, 5],
    "gamma": [0.0, 0.05, 0.1, 0.2, 0.3],
    "reg_alpha": [0.0, 0.001, 0.01, 0.1, 1.0],
    "reg_lambda": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
}


def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray, class_names: list) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics on test set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    return {
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
        "classification_report": report,
        "predictions": y_pred,
        "probabilities": y_proba,
    }


def tune_symptom_model(config: dict, n_iter: int = 35) -> Dict[str, Any]:
    """Tune hyperparameters for the Symptom Disease Classification Model."""
    logger.info("=" * 60)
    logger.info(f"TUNING: Symptom XGBoost Model ({n_iter} iterations, 5-fold CV)")
    logger.info("=" * 60)

    models_dir = os.path.join(project_root, config["paths"]["models_dir"])
    reports_dir = os.path.join(project_root, config["paths"]["reports_dir"])
    ensure_dir(models_dir)
    ensure_dir(reports_dir)

    sym_path = config["dataset"]["symptom_path"]
    if not os.path.isabs(sym_path):
        sym_path = os.path.join(project_root, sym_path)

    X_train, X_test, y_train, y_test, label_encoder, _ = load_symptom_data(
        csv_path=sym_path,
        test_size=0.2,
        seed=config["training"]["seed"],
    )

    # 1. Baseline Model
    baseline = XGBClassifier(
        max_depth=config["symptom_model"]["xgboost"].get("max_depth", 6),
        n_estimators=config["symptom_model"]["xgboost"].get("n_estimators", 200),
        learning_rate=config["symptom_model"]["xgboost"].get("learning_rate", 0.1),
        eval_metric="mlogloss",
        objective="multi:softprob",
        num_class=len(SYMPTOM_LABELS),
        random_state=config["training"]["seed"],
        verbosity=0,
    )
    baseline.fit(X_train, y_train)
    baseline_metrics = evaluate_model(baseline, X_test, y_test, SYMPTOM_LABELS)
    logger.info(f"Baseline Test Accuracy: {baseline_metrics['accuracy']:.4f}, F1: {baseline_metrics['f1_macro']:.4f}")

    # 2. Hyperparameter Search
    base_estimator = XGBClassifier(
        eval_metric="mlogloss",
        objective="multi:softprob",
        num_class=len(SYMPTOM_LABELS),
        random_state=config["training"]["seed"],
        verbosity=0,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config["training"]["seed"])
    search = RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring="accuracy",
        cv=cv,
        verbose=1,
        random_state=config["training"]["seed"],
        n_jobs=-1,
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_
    logger.info(f"Best CV Score: {search.best_score_:.4f}")
    logger.info(f"Best Hyperparameters: {best_params}")

    tuned_metrics = evaluate_model(best_model, X_test, y_test, SYMPTOM_LABELS)
    logger.info(f"Tuned Test Accuracy: {tuned_metrics['accuracy']:.4f}, F1: {tuned_metrics['f1_macro']:.4f}")

    # Save tuned model and encoder
    joblib.dump(best_model, os.path.join(models_dir, "symptom_model.joblib"))
    joblib.dump(label_encoder, os.path.join(models_dir, "symptom_label_encoder.joblib"))
    logger.info(f"Saved tuned symptom model to {os.path.join(models_dir, 'symptom_model.joblib')}")

    # Save predictions
    np.savez(
        os.path.join(reports_dir, "symptom_model_test_preds.npz"),
        predictions=tuned_metrics["predictions"],
        labels=y_test,
        probabilities=tuned_metrics["probabilities"],
    )

    return {
        "model_name": "Symptom XGBoost",
        "best_cv_accuracy": float(search.best_score_),
        "best_params": best_params,
        "baseline_accuracy": baseline_metrics["accuracy"],
        "baseline_f1": baseline_metrics["f1_macro"],
        "tuned_accuracy": tuned_metrics["accuracy"],
        "tuned_f1": tuned_metrics["f1_macro"],
        "accuracy_delta": float(tuned_metrics["accuracy"] - baseline_metrics["accuracy"]),
        "classification_report": tuned_metrics["classification_report"],
    }


def tune_clinical_model(config: dict, n_iter: int = 35) -> Dict[str, Any]:
    """Tune hyperparameters for the Clinical Risk XGBoost Model."""
    logger.info("=" * 60)
    logger.info(f"TUNING: Clinical XGBoost Model ({n_iter} iterations, 5-fold CV)")
    logger.info("=" * 60)

    models_dir = os.path.join(project_root, config["paths"]["models_dir"])
    reports_dir = os.path.join(project_root, config["paths"]["reports_dir"])

    sym_path = config["dataset"]["symptom_path"]
    if not os.path.isabs(sym_path):
        sym_path = os.path.join(project_root, sym_path)

    X_train_sym, X_test_sym, y_train, y_test, label_encoder, _ = load_symptom_data(
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

    all_features = SYMPTOM_FEATURES + clinical_feature_names
    X_train_raw = np.hstack([X_train_sym, clinical_train])
    X_test_raw = np.hstack([X_test_sym, clinical_test])

    # Standard scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # 1. Baseline Model
    baseline = XGBClassifier(
        max_depth=config["clinical_model"]["xgboost"].get("max_depth", 5),
        n_estimators=config["clinical_model"]["xgboost"].get("n_estimators", 150),
        learning_rate=config["clinical_model"]["xgboost"].get("learning_rate", 0.1),
        eval_metric="mlogloss",
        objective="multi:softprob",
        num_class=len(SYMPTOM_LABELS),
        random_state=config["training"]["seed"],
        verbosity=0,
    )
    baseline.fit(X_train, y_train)
    baseline_metrics = evaluate_model(baseline, X_test, y_test, SYMPTOM_LABELS)
    logger.info(f"Baseline Test Accuracy: {baseline_metrics['accuracy']:.4f}, F1: {baseline_metrics['f1_macro']:.4f}")

    # 2. Hyperparameter Search
    base_estimator = XGBClassifier(
        eval_metric="mlogloss",
        objective="multi:softprob",
        num_class=len(SYMPTOM_LABELS),
        random_state=config["training"]["seed"],
        verbosity=0,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config["training"]["seed"])
    search = RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring="accuracy",
        cv=cv,
        verbose=1,
        random_state=config["training"]["seed"],
        n_jobs=-1,
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_
    logger.info(f"Best CV Score: {search.best_score_:.4f}")
    logger.info(f"Best Hyperparameters: {best_params}")

    tuned_metrics = evaluate_model(best_model, X_test, y_test, SYMPTOM_LABELS)
    logger.info(f"Tuned Test Accuracy: {tuned_metrics['accuracy']:.4f}, F1: {tuned_metrics['f1_macro']:.4f}")

    # Save tuned model, scaler, and label encoder
    joblib.dump(best_model, os.path.join(models_dir, "clinical_model.joblib"))
    joblib.dump(scaler, os.path.join(models_dir, "clinical_scaler.joblib"))
    joblib.dump(label_encoder, os.path.join(models_dir, "clinical_label_encoder.joblib"))
    logger.info(f"Saved tuned clinical model to {os.path.join(models_dir, 'clinical_model.joblib')}")

    # Save predictions
    np.savez(
        os.path.join(reports_dir, "clinical_model_test_preds.npz"),
        predictions=tuned_metrics["predictions"],
        labels=y_test,
        probabilities=tuned_metrics["probabilities"],
    )

    return {
        "model_name": "Clinical XGBoost",
        "best_cv_accuracy": float(search.best_score_),
        "best_params": best_params,
        "baseline_accuracy": baseline_metrics["accuracy"],
        "baseline_f1": baseline_metrics["f1_macro"],
        "tuned_accuracy": tuned_metrics["accuracy"],
        "tuned_f1": tuned_metrics["f1_macro"],
        "accuracy_delta": float(tuned_metrics["accuracy"] - baseline_metrics["accuracy"]),
        "classification_report": tuned_metrics["classification_report"],
    }


def run_xgboost_tuning():
    """Run hyperparameter tuning for both XGBoost models and save summary report."""
    setup_logging()
    config = load_config(os.path.join(project_root, "configs", "config.yaml"))
    set_seed(config["training"]["seed"])

    reports_dir = os.path.join(project_root, config["paths"]["reports_dir"])
    ensure_dir(reports_dir)

    sym_results = tune_symptom_model(config, n_iter=30)
    clin_results = tune_clinical_model(config, n_iter=30)

    summary = {
        "symptom_model": sym_results,
        "clinical_model": clin_results,
    }

    report_path = os.path.join(reports_dir, "xgboost_tuning_report.json")
    save_metrics(summary, report_path)
    logger.info(f"XGBoost tuning summary saved to {report_path}")

    print("\n" + "=" * 65)
    print("XGBOOST HYPERPARAMETER TUNING RESULTS")
    print("=" * 65)
    print(f"{'Model':<20} {'Baseline Acc':>15} {'Tuned Acc':>12} {'Gain':>10}")
    print("-" * 65)
    print(f"{sym_results['model_name']:<20} {sym_results['baseline_accuracy']:>15.4f} {sym_results['tuned_accuracy']:>12.4f} {sym_results['accuracy_delta']:>+10.4f}")
    print(f"{clin_results['model_name']:<20} {clin_results['baseline_accuracy']:>15.4f} {clin_results['tuned_accuracy']:>12.4f} {clin_results['accuracy_delta']:>+10.4f}")
    print("=" * 65)

    return summary


if __name__ == "__main__":
    run_xgboost_tuning()
