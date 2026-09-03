"""
01_train_RF_models_random80_20.py

Final reproducibility script for the Mendeley Data repository.

Purpose
-------
This script trains Random Forest surrogate models for predicting:
1) Average Nusselt number (Nuavg)
2) Pressure drop (DelP_Pa)

using the cleaned long-form CFD-derived dataset and the fixed random
80/20 train-test split included in the repository.

Expected repository structure
-----------------------------
Run this script from the top-level dataset folder:

CFD_ML_Wavy_Channel_Surrogate_Data/
│
├── 02_processed_data/
│   └── ML_dataset_longform.csv
│
├── 03_splits/
│   └── random80_20_split.csv
│
├── 04_models/
├── 05_predictions/
├── 07_tables/
└── 06_scripts/
    └── 01_train_RF_models_random80_20.py

Required input columns
----------------------
The processed dataset must include:
case_uid, raw_row_id, geometry_id,
Re, Pr, Da, epsi, Hp_mm, a_mm, Lw_mm,
Nuavg, DelP_Pa

The split file must include:
case_uid, split

where split is either "train" or "test".

Outputs generated
-----------------
04_models/random80_20/RF_Nuavg_model.pkl
04_models/random80_20/RF_DelP_model.pkl
05_predictions/random80_20/test_predictions.csv
05_predictions/random80_20/test_predictions.xlsx
07_tables/random80_20/test_metrics.csv
07_tables/random80_20/test_metrics.xlsx
07_tables/random80_20/training_summary.md
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ============================================================
# 1. USER SETTINGS
# ============================================================

BASE_DIR = Path(".")

DATA_FILE = BASE_DIR / "02_processed_data" / "ML_dataset_longform.csv"
SPLIT_FILE = BASE_DIR / "03_splits" / "random80_20_split.csv"

MODEL_DIR = BASE_DIR / "04_models" / "random80_20"
PRED_DIR = BASE_DIR / "05_predictions" / "random80_20"
TABLE_DIR = BASE_DIR / "07_tables" / "random80_20"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

RF_NU_MODEL_FILE = MODEL_DIR / "RF_Nuavg_model.pkl"
RF_DP_MODEL_FILE = MODEL_DIR / "RF_DelP_model.pkl"

PRED_CSV_FILE = PRED_DIR / "test_predictions.csv"
PRED_XLSX_FILE = PRED_DIR / "test_predictions.xlsx"

METRICS_CSV_FILE = TABLE_DIR / "test_metrics.csv"
METRICS_XLSX_FILE = TABLE_DIR / "test_metrics.xlsx"
SUMMARY_MD_FILE = TABLE_DIR / "training_summary.md"

RANDOM_STATE = 42

FEATURES = ["Re", "Pr", "Da", "epsi", "Hp_mm", "a_mm", "Lw_mm"]
TARGET_NU = "Nuavg"
TARGET_DP = "DelP_Pa"


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def require_file(path: Path) -> None:
    """Raise a clear error if an input file is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Please run this script from the top-level repository folder."
        )


def metrics_row(y_true, y_pred, model_name: str, output_name: str) -> dict:
    """Return standard regression metrics."""
    return {
        "model": model_name,
        "output": output_name,
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


# ============================================================
# 3. LOAD DATA AND SPLIT FILE
# ============================================================

require_file(DATA_FILE)
require_file(SPLIT_FILE)

df = pd.read_csv(DATA_FILE)
split_df = pd.read_csv(SPLIT_FILE)

# Merge fixed split labels into dataset
df = df.merge(split_df, on="case_uid", how="inner")

required_cols = ["case_uid", "raw_row_id", "geometry_id"] + FEATURES + [TARGET_NU, TARGET_DP, "split"]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# Enforce numeric values for model inputs and targets
for col in FEATURES + [TARGET_NU, TARGET_DP]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=FEATURES + [TARGET_NU, TARGET_DP]).reset_index(drop=True)

train_df = df[df["split"] == "train"].copy()
test_df = df[df["split"] == "test"].copy()

if train_df.empty or test_df.empty:
    raise ValueError(
        "Train or test subset is empty. Check that the split column contains "
        '"train" and "test" labels.'
    )

X_train = train_df[FEATURES].copy()
X_test = test_df[FEATURES].copy()

y_nu_train = train_df[TARGET_NU].copy()
y_nu_test = test_df[TARGET_NU].copy()

y_dp_train = train_df[TARGET_DP].copy()
y_dp_test = test_df[TARGET_DP].copy()

print("Loaded dataset:", df.shape)
print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)


# ============================================================
# 4. TRAIN RANDOM FOREST MODELS
# ============================================================

rf_nu = RandomForestRegressor(
    n_estimators=800,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

rf_dp = RandomForestRegressor(
    n_estimators=800,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

rf_nu.fit(X_train, y_nu_train)
rf_dp.fit(X_train, y_dp_train)


# ============================================================
# 5. PREDICT AND EVALUATE
# ============================================================

nu_pred = rf_nu.predict(X_test)
dp_pred = rf_dp.predict(X_test)

metrics = [
    metrics_row(y_nu_test, nu_pred, "RandomForest", "Nuavg"),
    metrics_row(y_dp_test, dp_pred, "RandomForest", "DelP_Pa"),
]

metrics_df = pd.DataFrame(metrics)


# ============================================================
# 6. SAVE MODELS, PREDICTIONS, AND METRICS
# ============================================================

joblib.dump(rf_nu, RF_NU_MODEL_FILE)
joblib.dump(rf_dp, RF_DP_MODEL_FILE)

pred_df = test_df[
    ["case_uid", "raw_row_id", "geometry_id"] + FEATURES + [TARGET_NU, TARGET_DP]
].copy()

pred_df["Nuavg_pred_RF"] = nu_pred
pred_df["DelP_pred_RF"] = dp_pred

pred_df["Nuavg_abs_error"] = (pred_df[TARGET_NU] - pred_df["Nuavg_pred_RF"]).abs()
pred_df["DelP_abs_error"] = (pred_df[TARGET_DP] - pred_df["DelP_pred_RF"]).abs()

pred_df["Nuavg_residual"] = pred_df["Nuavg_pred_RF"] - pred_df[TARGET_NU]
pred_df["DelP_residual"] = pred_df["DelP_pred_RF"] - pred_df[TARGET_DP]

pred_df["Nuavg_percent_error"] = 100.0 * pred_df["Nuavg_residual"] / pred_df[TARGET_NU]
pred_df["DelP_percent_error"] = 100.0 * pred_df["DelP_residual"] / pred_df[TARGET_DP]

pred_df.to_csv(PRED_CSV_FILE, index=False)
pred_df.to_excel(PRED_XLSX_FILE, index=False)

metrics_df.to_csv(METRICS_CSV_FILE, index=False)
metrics_df.to_excel(METRICS_XLSX_FILE, index=False)


# ============================================================
# 7. SAVE TRAINING SUMMARY
# ============================================================

summary_lines = [
    "# Training Summary",
    "",
    "## Baseline protocol",
    "- Protocol: fixed random 80/20 train-test split",
    f"- Source dataset: `{DATA_FILE}`",
    f"- Split file: `{SPLIT_FILE}`",
    "",
    "## Dataset size",
    f"- Total rows after cleaning and split merge: **{len(df)}**",
    f"- Train rows: **{len(train_df)}**",
    f"- Test rows: **{len(test_df)}**",
    f"- Number of input features: **{len(FEATURES)}**",
    "",
    "## Input features",
]

summary_lines.extend([f"- `{feature}`" for feature in FEATURES])

summary_lines.extend(
    [
        "",
        "## Target outputs",
        f"- `{TARGET_NU}`",
        f"- `{TARGET_DP}`",
        "",
        "## Random Forest hyperparameters",
        "- n_estimators = 800",
        "- min_samples_leaf = 2",
        f"- random_state = {RANDOM_STATE}",
        "- n_jobs = -1",
        "",
        "## Test metrics",
        metrics_df.to_string(index=False),
        "",
        "## Files generated",
        f"- `{RF_NU_MODEL_FILE}`",
        f"- `{RF_DP_MODEL_FILE}`",
        f"- `{PRED_CSV_FILE}`",
        f"- `{PRED_XLSX_FILE}`",
        f"- `{METRICS_CSV_FILE}`",
        f"- `{METRICS_XLSX_FILE}`",
    ]
)

SUMMARY_MD_FILE.write_text("\n".join(summary_lines), encoding="utf-8")


# ============================================================
# 8. CONSOLE SUMMARY
# ============================================================

print("\nSaved:")
print(" -", RF_NU_MODEL_FILE)
print(" -", RF_DP_MODEL_FILE)
print(" -", PRED_CSV_FILE)
print(" -", PRED_XLSX_FILE)
print(" -", METRICS_CSV_FILE)
print(" -", METRICS_XLSX_FILE)
print(" -", SUMMARY_MD_FILE)

print("\n=== TEST METRICS ===")
print(metrics_df.to_string(index=False))
