# Semiconductor Manufacturing AI PoC - SECOM Dataset

**Anomaly Detection | Failure Prediction | Cross-Tool Root-Cause Analysis**

Built for semiconductor fabs that need to trace downstream defects back to upstream process steps.

---

## Architecture

```
secom.data (590 features, 1,567 wafers)
      |
      v
preprocessing.py     # KNN impute, variance filter, SMOTE balance
      |
      +------> Isolation Forest  (unsupervised anomaly detection)
      |
      +------> XGBoost Classifier (supervised pass/fail prediction)
                    |
                    v
              SHAP Attribution
                    |
                    v
              rca_engine.py     # attribute defects to Deposition / Etch / Inspection
                    |
                    v
              dashboard.py      # Streamlit interactive dashboard
```

## Multi-Tool Workflow Simulation

Features are partitioned to simulate a 3-step wafer fab:

| Tool Block    | Feature Range | Simulated Process           |
|---------------|---------------|-----------------------------|
| Deposition    | f0 - f195     | Film thickness, uniformity  |
| Etch          | f196 - f392   | Etch rate, chamber pressure |
| Inspection    | f393 - f589   | Defect density, CD, overlay |

## Cross-Tool Root-Cause Analysis

When a wafer fails at Inspection, `rca_engine.py` traces the failure upstream:

1. SHAP values are computed per feature per failing wafer
2. Absolute SHAP contributions are summed within each tool block
3. The tool block with the highest cumulative SHAP attribution is flagged as root cause
4. Top contributing features within that block are surfaced for process engineers

## Setup

```bash
pip install -r requirements.txt
python main.py                  # full pipeline (synthetic data if SECOM not present)
streamlit run dashboard.py      # interactive dashboard
```

Download SECOM data from: https://archive.ics.uci.edu/dataset/179/secom
Place `secom.data` and `secom_labels.labels` in the project root.

## Files

| File | Description |
|------|-------------|
| `preprocessing.py` | Data cleaning, imputation, feature selection, SMOTE |
| `models.py` | Isolation Forest + XGBoost + SHAP computation |
| `rca_engine.py` | Cross-tool defect attribution engine |
| `main.py` | End-to-end pipeline runner |
| `dashboard.py` | Streamlit interactive dashboard |
| `requirements.txt` | Python dependencies |

---

*Dr. Sandeep Grover - PhD Data Science (CSIR-IGIB Delhi)*
