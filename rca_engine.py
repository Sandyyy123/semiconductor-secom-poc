"""
rca_engine.py - Cross-tool root-cause analysis using SHAP attribution.
Given a failing wafer, traces the defect back to its originating tool block.
"""

import numpy as np
import pandas as pd


TOOL_BLOCKS = {
    "Deposition": (0, 196),
    "Etch":        (196, 393),
    "Inspection":  (393, 590),
}


def attribute_shap_to_tools(shap_values: np.ndarray, feature_meta: pd.DataFrame) -> pd.DataFrame:
    """Sum absolute SHAP values by tool block for every sample."""
    records = []
    for i, row in feature_meta.iterrows():
        col = row["feature"]
        tool = row["tool_block"]
        col_idx = i  # positional index after preprocessing aligns with shap_values columns
        if col_idx < shap_values.shape[1]:
            records.append({"feature": col, "tool_block": tool,
                             "mean_abs_shap": float(np.abs(shap_values[:, col_idx]).mean())})
    df = pd.DataFrame(records)
    tool_summary = df.groupby("tool_block")["mean_abs_shap"].sum().reset_index()
    tool_summary["pct_attribution"] = (tool_summary["mean_abs_shap"] /
                                        tool_summary["mean_abs_shap"].sum() * 100).round(1)
    return tool_summary.sort_values("pct_attribution", ascending=False).reset_index(drop=True)


def trace_single_failure(sample_idx: int, shap_values: np.ndarray,
                         feature_meta: pd.DataFrame, X: np.ndarray,
                         feature_names: list = None) -> dict:
    """
    Full RCA trace for one failing wafer.
    Returns root-cause tool, top contributing features, and SHAP waterfall data.
    """
    sample_shap = shap_values[sample_idx]   # shape: (n_features,)

    # Map each feature to its tool block
    block_scores = {}
    top_features = []
    for i, (shap_val, meta_row) in enumerate(zip(sample_shap, feature_meta.itertuples())):
        tool = meta_row.tool_block
        block_scores[tool] = block_scores.get(tool, 0.0) + abs(shap_val)
        top_features.append({
            "feature": meta_row.feature,
            "tool_block": tool,
            "shap_value": float(shap_val),
            "abs_shap": float(abs(shap_val)),
        })

    top_features_df = pd.DataFrame(top_features).sort_values("abs_shap", ascending=False)
    root_tool = max(block_scores, key=block_scores.get)
    total = sum(block_scores.values())
    attribution_pct = {t: round(s / total * 100, 1) for t, s in block_scores.items()}

    return {
        "sample_idx": sample_idx,
        "root_cause_tool": root_tool,
        "attribution_pct": attribution_pct,
        "top_features": top_features_df.head(10).to_dict("records"),
    }


def run_rca(shap_values: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray,
            feature_meta: pd.DataFrame, X: np.ndarray) -> pd.DataFrame:
    """Run RCA on all true positive (correctly predicted) failures."""
    fail_indices = np.where((y_pred == 1) & (y_true == 1))[0]
    print(f"  Running RCA on {len(fail_indices)} confirmed failure samples")

    results = []
    for idx in fail_indices[:50]:   # cap at 50 for demo speed
        trace = trace_single_failure(idx, shap_values, feature_meta, X)
        results.append({
            "sample": idx,
            "root_tool": trace["root_cause_tool"],
            **{f"pct_{k}": v for k, v in trace["attribution_pct"].items()},
        })
    df = pd.DataFrame(results)
    if len(df):
        print(df["root_tool"].value_counts().to_string())
    return df
