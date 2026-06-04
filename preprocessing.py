"""
preprocessing.py - SECOM dataset cleaning and feature engineering.
Simulates a 3-tool wafer fab workflow by partitioning features into
Deposition / Etch / Inspection blocks.
"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE


TOOL_BLOCKS = {
    "Deposition": (0, 196),
    "Etch":        (196, 393),
    "Inspection":  (393, 590),
}


def load_secom(data_path: str = "secom.data", labels_path: str = "secom_labels.labels"):
    """Load raw SECOM files. Falls back to synthetic data when files are absent."""
    try:
        X = pd.read_csv(data_path, sep=" ", header=None)
        y_raw = pd.read_csv(labels_path, sep=" ", header=None)
        y = (y_raw.iloc[:, 0] == 1).astype(int)
        print(f"Loaded SECOM: {X.shape[0]} samples, {X.shape[1]} features")
    except FileNotFoundError:
        print("SECOM files not found - generating synthetic data for demo")
        rng = np.random.default_rng(42)
        n, f = 1567, 590
        X = pd.DataFrame(rng.standard_normal((n, f)) * 10 + 5)
        X.iloc[rng.choice(n, size=int(n * 0.38), replace=False), :] = np.nan
        y = pd.Series((rng.random(n) < 0.065).astype(int))
    return X, y


def label_tool_block(feature_idx: int) -> str:
    for tool, (lo, hi) in TOOL_BLOCKS.items():
        if lo <= feature_idx < hi:
            return tool
    return "Unknown"


def clean(X: pd.DataFrame, y: pd.Series, variance_threshold: float = 0.01,
          corr_threshold: float = 0.95, n_neighbors: int = 5):
    """Full preprocessing pipeline: impute -> filter -> balance."""

    # 1. Remove near-zero variance features
    selector = VarianceThreshold(threshold=variance_threshold)
    X_var = selector.fit_transform(X)
    kept_cols = np.where(selector.get_support())[0]
    print(f"  After variance filter: {X_var.shape[1]} / {X.shape[1]} features")

    # 2. KNN imputation
    imputer = KNNImputer(n_neighbors=n_neighbors)
    X_imp = imputer.fit_transform(X_var)
    X_df = pd.DataFrame(X_imp, columns=[f"f{c}" for c in kept_cols])

    # 3. Remove highly correlated features
    corr_matrix = X_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_cols = [c for c in upper.columns if any(upper[c] > corr_threshold)]
    X_df = X_df.drop(columns=drop_cols)
    print(f"  After correlation filter: {X_df.shape[1]} features retained")

    # 4. SMOTE oversampling to handle class imbalance
    sm = SMOTE(random_state=42)
    X_bal, y_bal = sm.fit_resample(X_df, y)
    print(f"  After SMOTE: {X_bal.shape[0]} samples (was {len(y)})")

    # 5. Annotate each feature with its tool block
    feature_meta = []
    for col in X_df.columns:
        idx = int(col[1:])
        feature_meta.append({"feature": col, "original_idx": idx, "tool_block": label_tool_block(idx)})

    return X_bal, y_bal, pd.DataFrame(feature_meta)
