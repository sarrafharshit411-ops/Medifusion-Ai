"""
MediFusion AI - Clinical Risk Model
XGBoost-based risk assessment from clinical data.

NOTE: No real clinical dataset is available. This model is trained on the symptom
dataset augmented with synthetic clinical features for educational/demonstration
purposes. In a real deployment, this would use actual patient clinical records.
"""

import logging
from typing import Dict, Optional, List

import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)


class ClinicalModel:
    """
    XGBoost clinical risk assessment model.
    
    Takes combined symptom + clinical features (22 features total)
    and predicts disease risk probabilities.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        xgb_params = config.get("xgboost", {}) if config else {}
        self.model = XGBClassifier(
            max_depth=xgb_params.get("max_depth", 5),
            n_estimators=xgb_params.get("n_estimators", 150),
            learning_rate=xgb_params.get("learning_rate", 0.1),
            eval_metric="mlogloss",
            objective="multi:softprob",
            num_class=3,
            use_label_encoder=False,
            random_state=42,
            verbosity=0,
        )
        self.scaler = StandardScaler()
        self.label_encoder: Optional[LabelEncoder] = None
        self.feature_names: Optional[List[str]] = None
        self.is_fitted = False
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        feature_names: List[str] = None,
        label_encoder: LabelEncoder = None,
    ):
        """Train the clinical XGBoost classifier with feature scaling."""
        self.feature_names = feature_names
        self.label_encoder = label_encoder
        
        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        eval_set = [(X_train_scaled, y_train)]
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            eval_set.append((X_val_scaled, y_val))
        
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=eval_set,
            verbose=False,
        )
        self.is_fitted = True
        logger.info("Clinical model training complete")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities. Returns (n_samples, 3)."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_embedding(self, X: np.ndarray) -> np.ndarray:
        """
        Get clinical feature embedding for fusion.
        Returns scaled features (22-dim).
        """
        return self.scaler.transform(X).astype(np.float32)
    
    def save(self, model_path: str, scaler_path: str, encoder_path: str):
        """Save model, scaler, and label encoder."""
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        if self.label_encoder is not None:
            joblib.dump(self.label_encoder, encoder_path)
        logger.info(f"Clinical model saved to {model_path}")
    
    def load(self, model_path: str, scaler_path: str, encoder_path: str):
        """Load model, scaler, and label encoder."""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(encoder_path)
        self.is_fitted = True
        logger.info(f"Clinical model loaded from {model_path}")


def load_clinical_model(
    model_path: str, scaler_path: str, encoder_path: str
) -> ClinicalModel:
    """Convenience function to load a trained ClinicalModel."""
    cm = ClinicalModel()
    cm.load(model_path, scaler_path, encoder_path)
    return cm
