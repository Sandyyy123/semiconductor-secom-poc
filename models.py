"""
models.py - Anomaly detection (Isolation Forest) + failure prediction (XGBoost).
Produces per-sample SHAP values for cross-tool root-cause attribution.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def train_isolation_forest(X: np.ndarray, contamination: float = 0.065, random_state: int = 42):
    """Unsupervised anomaly detection - no labels required."""
    clf = IsolationForest(contamination=contamination, n_estimators=200,
                          random_state=random_state, n_jobs=-1)
    clf.fit(X)
    scores = clf.decision_function(X)        # higher = more normal
    anomaly_flags = clf.predict(X) == -1     # True = anomaly
    print(f"  Isolation Forest flagged {anomaly_flags.sum()} anomalies ({anomaly_flags.mean():.1%} of samples)")
    return clf, scores, anomaly_flags


def train_xgboost(X: np.ndarray, y: np.ndarray, scale_pos_weight: float = None):
    """Supervised binary classifier for pass/fail prediction."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    if scale_pos_weight is None:
        neg, pos = np.bincount(y_tr.astype(int))
        scale_pos_weight = neg / max(pos, 1)

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, y_prob)
    print(f"  XGBoost AUC-ROC: {auc:.4f}")
    print(classification_report(y_te, y_pred, target_names=["Pass", "Fail"]))

    return model, X_te, y_te, y_prob


def compute_shap(model, X: np.ndarray, feature_names: list = None):
    """Return SHAP values for the positive (Fail) class."""
    if not SHAP_AVAILABLE:
        print("SHAP not installed - skipping attribution")
        return None
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]   # index 1 = Fail class
    print(f"  SHAP values computed for {X.shape[0]} samples x {X.shape[1]} features")
    return shap_values
