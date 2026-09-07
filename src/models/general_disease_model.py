"""
MediFusion AI — General Disease Ensemble Model Definition & Inference Class
Soft-voting ensemble of tuned Random Forest and multi-class XGBoost
trained on the 41-disease, 132-symptom clinical dataset.
"""

import numpy as np
import pandas as pd

class GeneralDiseaseEnsemble:
    """Calibrated soft-voting ensemble of Random Forest and XGBoost."""
    def __init__(self, rf_model, xgb_model, label_encoder, feature_names):
        self.rf = rf_model
        self.xgb = xgb_model
        self.le = label_encoder
        self.feature_names = feature_names

    def _to_df(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.feature_names]
        X_arr = np.asarray(X)
        if len(X_arr.shape) == 1:
            X_arr = X_arr.reshape(1, -1)
        return pd.DataFrame(X_arr, columns=self.feature_names)

    def predict_proba(self, X):
        """Compute weighted ensemble probabilities."""
        df = self._to_df(X)
        rf_proba = self.rf.predict_proba(df)
        xgb_proba = self.xgb.predict_proba(df)
        # 50% RF + 50% XGBoost soft-voting
        return 0.5 * rf_proba + 0.5 * xgb_proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.le.inverse_transform(np.argmax(proba, axis=1))

    def predict_top_k(self, feature_vector, k=5):
        """Returns top-k predicted diseases with calibrated confidence probabilities."""
        proba = self.predict_proba(feature_vector)[0]
        top_indices = np.argsort(proba)[::-1][:k]
        
        results = []
        for idx in top_indices:
            disease_name = self.le.inverse_transform([idx])[0]
            conf = float(proba[idx])
            results.append({
                "disease": disease_name,
                "confidence": round(conf * 100, 2),
                "probability": round(conf, 4)
            })
        return results
