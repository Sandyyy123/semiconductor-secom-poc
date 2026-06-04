"""
main.py - Entry point. Runs the full SECOM PoC pipeline end-to-end.
Usage: python main.py
"""

import numpy as np
import pandas as pd

from preprocessing import load_secom, clean
from models import train_isolation_forest, train_xgboost, compute_shap
from rca_engine import attribute_shap_to_tools, run_rca


def main():
    print("=== Semiconductor Manufacturing AI PoC ===\n")

    # 1. Load + preprocess
    print("[1/5] Loading and preprocessing SECOM data...")
    X_raw, y_raw = load_secom()
    X_bal, y_bal, feature_meta = clean(X_raw, y_raw)
    X_np = X_bal.values if hasattr(X_bal, "values") else np.array(X_bal)
    y_np = y_bal.values if hasattr(y_bal, "values") else np.array(y_bal)

    # 2. Anomaly detection (unsupervised)
    print("\n[2/5] Training Isolation Forest anomaly detector...")
    iso_model, anomaly_scores, anomaly_flags = train_isolation_forest(X_np)

    # 3. Failure prediction (supervised)
    print("\n[3/5] Training XGBoost failure predictor...")
    xgb_model, X_test, y_test, y_prob = train_xgboost(X_np, y_np)

    # 4. SHAP attribution
    print("\n[4/5] Computing SHAP feature attributions...")
    shap_vals = compute_shap(xgb_model, X_test)

    if shap_vals is not None:
        # 5. Cross-tool RCA
        print("\n[5/5] Running cross-tool root-cause analysis...")
        y_pred = (y_prob >= 0.5).astype(int)
        tool_summary = attribute_shap_to_tools(shap_vals, feature_meta.head(X_test.shape[1]))
        print("\n--- Tool Attribution Summary ---")
        print(tool_summary.to_string(index=False))

        rca_df = run_rca(shap_vals, y_pred, y_test, feature_meta.head(X_test.shape[1]), X_test)
        rca_df.to_csv("rca_results.csv", index=False)
        print(f"\nRCA results saved to rca_results.csv ({len(rca_df)} failures traced)")
    else:
        print("\n[5/5] SHAP not available - skipping RCA (pip install shap)")

    print("\n=== Pipeline complete. Run: streamlit run dashboard.py ===")


if __name__ == "__main__":
    main()
