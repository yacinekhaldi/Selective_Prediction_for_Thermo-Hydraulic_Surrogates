from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.compose import TransformedTargetRegressor

try:
    import xgboost as xgb
except Exception:  # pragma: no cover - optional dependency
    xgb = None

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data" / "raw" / "mendeley_wavy_channel" / "ML-CFD-Wavy-Channel-Surrogate"
DATA_FILE = DATASET_ROOT / "02_processed_data" / "ML_dataset_longform.csv"
SPLIT_FILE = DATASET_ROOT / "03_splits" / "random80_20_split.csv"
OUT = ROOT / "results"

FEATURES = ["Re", "Pr", "Da", "epsi", "Hp_mm", "a_mm", "Lw_mm"]
TARGETS = ["Nuavg", "DelP_Pa"]
TARGET_TOLERANCE_PCT = {"Nuavg": 5.0, "DelP_Pa": 10.0}
RANDOM_STATE = 42
COVERAGE_GRID = sorted(set([0.05] + [round(x, 2) for x in np.linspace(0.1, 1.0, 19)] + [0.7, 0.8, 0.9, 0.95]))
MATCHED_COVERAGES = [0.95, 0.90, 0.80, 0.70]


def ensure_dirs() -> dict[str, Path]:
    dirs = {
        "data_processed": ROOT / "data" / "processed",
        "models": OUT / "models",
        "predictions": OUT / "predictions",
        "tables": OUT / "tables",
        "figures": OUT / "figures",
        "plot_data": OUT / "plot_data",
        "reports": OUT / "reports",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_").lower()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)


def save_table(df: pd.DataFrame, path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_base.with_suffix(".csv"), index=False)
    try:
        df.to_excel(path_base.with_suffix(".xlsx"), index=False)
    except Exception as exc:
        print(f"[warn] could not write Excel for {path_base.name}: {exc}")
    try:
        write_latex_table(df, path_base.with_suffix(".tex"))
    except Exception as exc:
        print(f"[warn] could not write LaTeX table for {path_base.name}: {exc}")


def latex_escape(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""
        return f"{value:.4g}"
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def write_latex_table(df: pd.DataFrame, path: Path, max_rows: int = 2000) -> None:
    table = df.copy()
    truncated = len(table) > max_rows
    if truncated:
        table = table.head(max_rows).copy()
    colspec = "l" * len(table.columns)
    lines = [
        "% Auto-generated table snippet. Full data are available in the matching CSV/XLSX file.",
    ]
    if truncated:
        lines.append(f"% Truncated to the first {max_rows} of {len(df)} rows for LaTeX usability.")
    lines.extend(
        [
            r"\begin{tabular}{" + colspec + "}",
            r"\hline",
            " & ".join(latex_escape(col) for col in table.columns) + r" \\",
            r"\hline",
        ]
    )
    for _, row in table.iterrows():
        lines.append(" & ".join(latex_escape(row[col]) for col in table.columns) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    abs_err = np.abs(err)
    denom = np.maximum(np.abs(y_true), 1e-12)
    ape = 100.0 * abs_err / denom
    return {
        "n": int(len(y_true)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE_pct": float(np.mean(ape)),
        "Median_APE_pct": float(np.median(ape)),
        "P95_APE_pct": float(np.percentile(ape, 95)),
        "Max_APE_pct": float(np.max(ape)),
        "Bias": float(np.mean(err)),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
    }


def failure_scores(y_true: np.ndarray, y_pred: np.ndarray, tolerance_pct: float) -> np.ndarray:
    denom = np.maximum(np.abs(y_true), 1e-12)
    ape = 100.0 * np.abs(y_pred - y_true) / denom
    return ape > tolerance_pct


def transform_distance_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in features:
        vals = pd.to_numeric(df[col], errors="coerce").astype(float)
        if col in {"Re", "Pr", "Da"}:
            out[f"log10_{col}"] = np.log10(np.maximum(vals, 1e-30))
        else:
            out[col] = vals
    return out


class OODDistance:
    def __init__(self, feature_cols: list[str], n_neighbors: int = 5):
        self.feature_cols = feature_cols
        self.n_neighbors = n_neighbors
        self.scaler = StandardScaler()
        self.knn: NearestNeighbors | None = None
        self.cov: EmpiricalCovariance | None = None

    def fit(self, train_df: pd.DataFrame) -> "OODDistance":
        X = transform_distance_features(train_df, self.feature_cols)
        Xs = self.scaler.fit_transform(X)
        k = min(self.n_neighbors, max(1, len(Xs)))
        self.knn = NearestNeighbors(n_neighbors=k)
        self.knn.fit(Xs)
        self.cov = EmpiricalCovariance().fit(Xs)
        return self

    def score(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.knn is None or self.cov is None:
            raise RuntimeError("OODDistance is not fitted.")
        X = transform_distance_features(df, self.feature_cols)
        Xs = self.scaler.transform(X)
        distances, _ = self.knn.kneighbors(Xs)
        knn_distance = distances.mean(axis=1)
        try:
            maha = self.cov.mahalanobis(Xs)
        except Exception:
            maha = np.zeros(len(df), dtype=float)
        return knn_distance, maha


@dataclass
class ScoreNormalizer:
    low: float
    high: float

    @classmethod
    def fit(cls, values: np.ndarray) -> "ScoreNormalizer":
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return cls(0.0, 1.0)
        low = float(np.percentile(values, 5))
        high = float(np.percentile(values, 95))
        if not np.isfinite(high - low) or abs(high - low) < 1e-12:
            high = low + 1.0
        return cls(low, high)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.clip((np.asarray(values, dtype=float) - self.low) / (self.high - self.low), 0.0, 1.0)


class XGBEnsemble:
    def __init__(self, seeds: list[int], use_gpu: bool, target: str):
        if xgb is None:
            raise RuntimeError("xgboost is not installed.")
        self.seeds = seeds
        self.use_gpu = use_gpu
        self.target = target
        self.models: list[xgb.XGBRegressor] = []
        self.log_target = target == "DelP_Pa"

    def _make_model(self, seed: int) -> "xgb.XGBRegressor":
        params = {
            "objective": "reg:squarederror",
            "n_estimators": 800,
            "max_depth": 6,
            "learning_rate": 0.03,
            "subsample": 0.95,
            "colsample_bytree": 0.95,
            "reg_lambda": 1.0,
            "min_child_weight": 1.0,
            "random_state": seed,
            "tree_method": "hist",
            "device": "cuda" if self.use_gpu else "cpu",
            "verbosity": 0,
        }
        return xgb.XGBRegressor(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "XGBEnsemble":
        yy = np.asarray(y, dtype=float)
        if self.log_target:
            yy = np.log1p(np.maximum(yy, 0.0))
        self.models = []
        for seed in self.seeds:
            model = self._make_model(seed)
            model.fit(X, yy)
            self.models.append(model)
        return self

    def predict_members(self, X: pd.DataFrame) -> np.ndarray:
        preds = []
        for model in self.models:
            pred = np.asarray(model.predict(X), dtype=float)
            if self.log_target:
                pred = np.expm1(pred)
            preds.append(pred)
        return np.column_stack(preds)

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        member_preds = self.predict_members(X)
        return member_preds.mean(axis=1), member_preds.std(axis=1)

    def feature_importance(self) -> np.ndarray:
        importances = []
        for model in self.models:
            if hasattr(model, "feature_importances_"):
                importances.append(np.asarray(model.feature_importances_, dtype=float))
        if not importances:
            return np.zeros(len(FEATURES), dtype=float)
        return np.vstack(importances).mean(axis=0)


def probe_xgboost_gpu() -> tuple[bool, str]:
    if xgb is None:
        return False, "xgboost is not installed"
    try:
        X = np.random.default_rng(0).normal(size=(128, 4))
        y = X[:, 0] - 0.3 * X[:, 1]
        model = xgb.XGBRegressor(
            n_estimators=4,
            max_depth=2,
            objective="reg:squarederror",
            tree_method="hist",
            device="cuda",
            verbosity=0,
            random_state=0,
        )
        model.fit(X, y)
        return True, "xgboost CUDA probe trained successfully"
    except Exception as exc:
        return False, f"xgboost CUDA probe failed: {type(exc).__name__}: {exc}"


def environment_report(use_gpu: bool, gpu_note: str, dirs: dict[str, Path]) -> dict[str, object]:
    def run_cmd(args: list[str]) -> str:
        try:
            return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=20)
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"

    packages: dict[str, str] = {}
    for module_name in ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "xgboost", "torch"]:
        try:
            module = __import__(module_name)
            packages[module_name] = str(getattr(module, "__version__", "available"))
        except Exception as exc:
            packages[module_name] = f"not available: {type(exc).__name__}"

    report = {
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "xgboost_gpu_requested": True,
        "xgboost_gpu_enabled": use_gpu,
        "xgboost_gpu_note": gpu_note,
        "torch_cuda_available": bool(torch is not None and torch.cuda.is_available()),
        "torch_cuda_version": None if torch is None else getattr(torch.version, "cuda", None),
        "nvidia_smi": run_cmd(["nvidia-smi"]),
        "data_file": str(DATA_FILE),
        "data_file_sha256": sha256_file(DATA_FILE) if DATA_FILE.exists() else None,
    }
    save_json(report, dirs["reports"] / "environment_report.json")
    md = [
        "# Environment Report",
        "",
        f"- Python executable: `{report['python_executable']}`",
        f"- Platform: `{report['platform']}`",
        f"- XGBoost GPU enabled: **{report['xgboost_gpu_enabled']}**",
        f"- XGBoost GPU note: {report['xgboost_gpu_note']}",
        f"- Torch CUDA available: **{report['torch_cuda_available']}**",
        "",
        "## Package versions",
    ]
    md.extend([f"- {name}: `{version}`" for name, version in packages.items()])
    (dirs["reports"] / "environment_report.md").write_text("\n".join(md), encoding="utf-8")
    return report


def load_primary_dataset(dirs: dict[str, Path]) -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Expected data file not found: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    required = ["case_uid", "raw_row_id", "geometry_id"] + FEATURES + TARGETS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required dataset columns: {missing}")
    for col in FEATURES + TARGETS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURES + TARGETS).reset_index(drop=True)
    processed_copy = dirs["data_processed"] / "thermal_wavy_channel_longform.csv"
    try:
        if not processed_copy.exists() or sha256_file(DATA_FILE) != sha256_file(processed_copy):
            shutil.copy2(DATA_FILE, processed_copy)
    except PermissionError as exc:
        print(f"[warn] could not refresh processed dataset copy: {exc}")
    return df


def save_dataset_profile(df: pd.DataFrame, dirs: dict[str, Path]) -> None:
    desc = df[FEATURES + TARGETS].describe().T.reset_index().rename(columns={"index": "variable"})
    save_table(desc, dirs["tables"] / "dataset_numeric_summary")

    levels = []
    for col in ["geometry_id", "Re", "Pr", "Da", "epsi", "Hp_mm", "a_mm", "Lw_mm"]:
        values = sorted(df[col].dropna().unique().tolist())
        levels.append(
            {
                "variable": col,
                "n_unique": len(values),
                "levels": ", ".join([f"{v:g}" if isinstance(v, float) else str(v) for v in values]),
            }
        )
    save_table(pd.DataFrame(levels), dirs["tables"] / "dataset_factor_levels")

    corr = df[FEATURES + TARGETS].corr(method="spearman").reset_index().rename(columns={"index": "variable"})
    save_table(corr, dirs["tables"] / "spearman_correlation_matrix")

    profile = {
        "n_rows": int(len(df)),
        "n_geometries": int(df["geometry_id"].nunique()),
        "geometry_ids": sorted(df["geometry_id"].unique().tolist()),
        "feature_columns": FEATURES,
        "target_columns": TARGETS,
        "parameter_levels": {row["variable"]: row["levels"] for row in levels},
    }
    save_json(profile, dirs["reports"] / "dataset_profile.json")
    lines = [
        "# Dataset Profile",
        "",
        f"- Rows: **{profile['n_rows']}**",
        f"- Geometry IDs: **{profile['n_geometries']}** ({', '.join(profile['geometry_ids'])})",
        f"- Features: {', '.join(FEATURES)}",
        f"- Targets: {', '.join(TARGETS)}",
        "",
        "## Parameter levels",
    ]
    lines.extend([f"- {key}: {value}" for key, value in profile["parameter_levels"].items()])
    (dirs["reports"] / "dataset_profile.md").write_text("\n".join(lines), encoding="utf-8")


@dataclass
class SplitSpec:
    name: str
    family: str
    train_cases: np.ndarray
    test_cases: np.ndarray
    description: str


def build_splits(df: pd.DataFrame, dirs: dict[str, Path]) -> list[SplitSpec]:
    splits: list[SplitSpec] = []
    split_records = []
    if SPLIT_FILE.exists():
        split_df = pd.read_csv(SPLIT_FILE)
        merged = df[["case_uid"]].merge(split_df, on="case_uid", how="left")
        train = merged.loc[merged["split"] == "train", "case_uid"].to_numpy()
        test = merged.loc[merged["split"] == "test", "case_uid"].to_numpy()
        splits.append(
            SplitSpec(
                "random80_20",
                "in_distribution",
                train,
                test,
                "Fixed random 80/20 split supplied with the dataset.",
            )
        )
    else:
        train_cases, test_cases = train_test_split(df["case_uid"], test_size=0.2, random_state=RANDOM_STATE)
        splits.append(
            SplitSpec("random80_20_generated", "in_distribution", train_cases.to_numpy(), test_cases.to_numpy(), "Generated random 80/20 split.")
        )

    for re_value in sorted(df["Re"].unique()):
        train = df.loc[df["Re"] != re_value, "case_uid"].to_numpy()
        test = df.loc[df["Re"] == re_value, "case_uid"].to_numpy()
        splits.append(
            SplitSpec(
                f"holdout_Re{int(re_value)}",
                "withheld_reynolds",
                train,
                test,
                f"All Re={re_value:g} cases are withheld from training.",
            )
        )

    for geom in sorted(df["geometry_id"].unique()):
        train = df.loc[df["geometry_id"] != geom, "case_uid"].to_numpy()
        test = df.loc[df["geometry_id"] == geom, "case_uid"].to_numpy()
        splits.append(
            SplitSpec(
                f"holdout_{geom}",
                "withheld_geometry",
                train,
                test,
                f"All {geom} geometry cases are withheld from training.",
            )
        )

    max_re = df["Re"].max()
    max_geom = sorted(df["geometry_id"].unique())[-1]
    train_mask = (df["Re"] != max_re) & (df["geometry_id"] != max_geom)
    test_mask = (df["Re"] == max_re) & (df["geometry_id"] == max_geom)
    splits.append(
        SplitSpec(
            f"combined_{max_geom}_Re{int(max_re)}",
            "combined_shift",
            df.loc[train_mask, "case_uid"].to_numpy(),
            df.loc[test_mask, "case_uid"].to_numpy(),
            f"Training excludes both {max_geom} and Re={max_re:g}; testing uses their intersection.",
        )
    )

    for spec in splits:
        for case_uid in spec.train_cases:
            split_records.append({"split_name": spec.name, "family": spec.family, "case_uid": case_uid, "split": "train"})
        for case_uid in spec.test_cases:
            split_records.append({"split_name": spec.name, "family": spec.family, "case_uid": case_uid, "split": "test"})
    save_table(pd.DataFrame(split_records), dirs["tables"] / "split_assignments")

    summary = pd.DataFrame(
        [
            {
                "split_name": spec.name,
                "family": spec.family,
                "n_train": len(spec.train_cases),
                "n_test": len(spec.test_cases),
                "description": spec.description,
            }
            for spec in splits
        ]
    )
    save_table(summary, dirs["tables"] / "split_summary")
    return splits


def train_validation_split(train_df: pd.DataFrame, split_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    strat = train_df["Re"].astype(str) + "_" + train_df["geometry_id"].astype(str)
    counts = strat.value_counts()
    stratify = strat if counts.min() >= 2 else None
    stable_offset = int(hashlib.sha256(split_name.encode("utf-8")).hexdigest()[:8], 16) % 1000
    fit_idx, val_idx = train_test_split(
        np.arange(len(train_df)),
        test_size=0.2,
        random_state=RANDOM_STATE + stable_offset,
        stratify=stratify,
    )
    return train_df.iloc[fit_idx].copy(), train_df.iloc[val_idx].copy()


def physics_violation_score(
    predictor: XGBEnsemble,
    df: pd.DataFrame,
    target: str,
    y_train: pd.Series,
    re_levels: np.ndarray,
) -> np.ndarray:
    X = df[FEATURES].copy()
    y_pred, _ = predictor.predict(X)
    y_train_arr = np.asarray(y_train, dtype=float)
    span = max(float(np.max(y_train_arr) - np.min(y_train_arr)), 1e-9)
    lower = float(np.min(y_train_arr)) - 0.05 * span
    upper = float(np.max(y_train_arr)) + 0.05 * span
    range_violation = np.maximum(0.0, lower - y_pred) / span + np.maximum(0.0, y_pred - upper) / span
    positivity_violation = np.maximum(0.0, -y_pred) / span

    re_levels = np.asarray(sorted(np.unique(re_levels)), dtype=float)
    if len(re_levels) < 2:
        return range_violation + positivity_violation

    repeated = pd.DataFrame(np.repeat(X.to_numpy(), len(re_levels), axis=0), columns=FEATURES)
    repeated["Re"] = np.tile(re_levels, len(X))
    grid_pred, _ = predictor.predict(repeated[FEATURES])
    grid_pred = grid_pred.reshape(len(X), len(re_levels))
    diffs = np.diff(grid_pred, axis=1)
    monotonic_violation = np.maximum(0.0, -diffs).sum(axis=1) / span
    return range_violation + positivity_violation + monotonic_violation


def raw_reliability_scores(
    predictor: XGBEnsemble,
    ood: OODDistance,
    df_eval: pd.DataFrame,
    target: str,
    y_train: pd.Series,
    re_levels: np.ndarray,
) -> pd.DataFrame:
    X_eval = df_eval[FEATURES].copy()
    y_pred, y_std = predictor.predict(X_eval)
    knn, maha = ood.score(df_eval)
    phys = physics_violation_score(predictor, df_eval, target, y_train, re_levels)
    return pd.DataFrame(
        {
            "prediction": y_pred,
            "uncertainty_raw": y_std / np.maximum(np.abs(y_pred), 1e-9),
            "knn_distance_raw": knn,
            "mahalanobis_raw": maha,
            "physics_violation_raw": phys,
        },
        index=df_eval.index,
    )


def normalize_signal_frames(val_raw: pd.DataFrame, test_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    normalizers = {
        "U": ScoreNormalizer.fit(val_raw["uncertainty_raw"].to_numpy()),
        "D_knn": ScoreNormalizer.fit(val_raw["knn_distance_raw"].to_numpy()),
        "D_maha": ScoreNormalizer.fit(val_raw["mahalanobis_raw"].to_numpy()),
        "V": ScoreNormalizer.fit(val_raw["physics_violation_raw"].to_numpy()),
    }
    def apply(raw: pd.DataFrame) -> pd.DataFrame:
        out = raw.copy()
        out["U_norm"] = normalizers["U"].transform(raw["uncertainty_raw"].to_numpy())
        out["D_knn_norm"] = normalizers["D_knn"].transform(raw["knn_distance_raw"].to_numpy())
        out["D_maha_norm"] = normalizers["D_maha"].transform(raw["mahalanobis_raw"].to_numpy())
        out["D_norm"] = np.maximum(out["D_knn_norm"], out["D_maha_norm"])
        out["V_norm"] = normalizers["V"].transform(raw["physics_violation_raw"].to_numpy())
        out["U_D_equal"] = 0.5 * (out["U_norm"] + out["D_norm"])
        out["U_V_equal"] = 0.5 * (out["U_norm"] + out["V_norm"])
        out["D_V_equal"] = 0.5 * (out["D_norm"] + out["V_norm"])
        out["U_D_V_equal"] = (out["U_norm"] + out["D_norm"] + out["V_norm"]) / 3.0
        out["U_D_V_max"] = np.maximum.reduce([out["U_norm"].to_numpy(), out["D_norm"].to_numpy(), out["V_norm"].to_numpy()])
        out["U_2D_V_weighted"] = (out["U_norm"] + 2.0 * out["D_norm"] + out["V_norm"]) / 4.0
        return out
    meta = {key: {"low": norm.low, "high": norm.high} for key, norm in normalizers.items()}
    return apply(val_raw), apply(test_raw), meta


def fit_learned_gate(val_scores: pd.DataFrame, val_failure: np.ndarray) -> Callable[[pd.DataFrame], np.ndarray] | None:
    if len(np.unique(val_failure)) < 2:
        return None
    X = val_scores[["U_norm", "D_norm", "V_norm"]].to_numpy()
    try:
        clf = LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE, max_iter=1000)
        clf.fit(X, val_failure.astype(int))
        return lambda df: clf.predict_proba(df[["U_norm", "D_norm", "V_norm"]].to_numpy())[:, 1]
    except Exception as exc:
        print(f"[warn] learned gate failed: {exc}")
        return None


def method_scores(
    score_df: pd.DataFrame,
    learned_gate: Callable[[pd.DataFrame], np.ndarray] | None,
    oracle_ape: np.ndarray | None = None,
    random_seed: int = RANDOM_STATE,
) -> dict[str, np.ndarray]:
    scores = {
        "U_only": score_df["U_norm"].to_numpy(),
        "D_only": score_df["D_norm"].to_numpy(),
        "V_only": score_df["V_norm"].to_numpy(),
        "U_plus_D": score_df["U_D_equal"].to_numpy(),
        "U_plus_V": score_df["U_V_equal"].to_numpy(),
        "D_plus_V": score_df["D_V_equal"].to_numpy(),
        "U_plus_D_plus_V": score_df["U_D_V_equal"].to_numpy(),
        "U_2D_plus_V": score_df["U_2D_V_weighted"].to_numpy(),
        "conservative_U_D_V_max": score_df["U_D_V_max"].to_numpy(),
    }
    if learned_gate is not None:
        scores["learned_logistic_gate"] = learned_gate(score_df)
    if oracle_ape is not None:
        scores["oracle_error_rank"] = oracle_ape
    rng = np.random.default_rng(random_seed)
    scores["random_rejection"] = rng.random(len(score_df))
    return scores


def risk_coverage_rows(
    score_name: str,
    scores: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split_name: str,
    target: str,
) -> list[dict[str, float | str | int]]:
    order = np.argsort(scores)
    rows = []
    for coverage in COVERAGE_GRID:
        n_accept = max(1, int(math.ceil(coverage * len(order))))
        idx = order[:n_accept]
        metrics = regression_metrics(y_true[idx], y_pred[idx])
        rows.append(
            {
                "split_name": split_name,
                "target": target,
                "score_method": score_name,
                "coverage": float(n_accept / len(order)),
                **metrics,
            }
        )
    return rows


def summarize_score_method(
    score_name: str,
    scores: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tolerance_pct: float,
    split_name: str,
    family: str,
    target: str,
) -> dict[str, float | str | int]:
    curve = pd.DataFrame(risk_coverage_rows(score_name, scores, y_true, y_pred, split_name, target))
    auc = float(np.trapezoid(curve["MAPE_pct"].to_numpy(), curve["coverage"].to_numpy()) / (curve["coverage"].max() - curve["coverage"].min()))
    failure = failure_scores(y_true, y_pred, tolerance_pct)
    if len(np.unique(failure)) > 1:
        auroc = float(roc_auc_score(failure.astype(int), scores))
        auprc = float(average_precision_score(failure.astype(int), scores))
    else:
        auroc = float("nan")
        auprc = float("nan")
    full = regression_metrics(y_true, y_pred)
    return {
        "split_name": split_name,
        "family": family,
        "target": target,
        "score_method": score_name,
        "tolerance_pct": tolerance_pct,
        "failure_rate": float(np.mean(failure)),
        "failure_AUROC": auroc,
        "failure_AUPRC": auprc,
        "AURC_MAPE_pct": auc,
        "full_MAE": full["MAE"],
        "full_RMSE": full["RMSE"],
        "full_MAPE_pct": full["MAPE_pct"],
        "full_R2": full["R2"],
    }


def matched_coverage_rows(
    score_name: str,
    scores: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split_name: str,
    family: str,
    target: str,
) -> list[dict[str, float | str | int]]:
    order = np.argsort(scores)
    rows = []
    for coverage in MATCHED_COVERAGES:
        n_accept = max(1, int(math.ceil(coverage * len(order))))
        idx = order[:n_accept]
        metrics = regression_metrics(y_true[idx], y_pred[idx])
        rows.append(
            {
                "split_name": split_name,
                "family": family,
                "target": target,
                "score_method": score_name,
                "coverage_target": coverage,
                "actual_coverage": float(n_accept / len(order)),
                **metrics,
            }
        )
    return rows


def calibration_rows(
    split_name: str,
    target: str,
    scores: np.ndarray,
    failure: np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, float | str | int]]:
    bins = np.linspace(0, 1, n_bins + 1)
    clipped = np.clip(scores, 0, 1)
    out = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (clipped >= lo) & (clipped <= hi)
        else:
            mask = (clipped >= lo) & (clipped < hi)
        if not np.any(mask):
            continue
        out.append(
            {
                "split_name": split_name,
                "target": target,
                "bin_low": float(lo),
                "bin_high": float(hi),
                "count": int(mask.sum()),
                "mean_score": float(np.mean(clipped[mask])),
                "empirical_failure_rate": float(np.mean(failure[mask])),
            }
        )
    return out


def gate_decisions(scores: np.ndarray, tau_predict: float, tau_warn: float) -> np.ndarray:
    return np.where(scores < tau_predict, "predict", np.where(scores < tau_warn, "warn", "abstain"))


def run_selective_experiment(
    df: pd.DataFrame,
    spec: SplitSpec,
    target: str,
    use_gpu: bool,
    dirs: dict[str, Path],
    ensemble_size: int,
) -> dict[str, list[dict[str, object]]]:
    train_all = df[df["case_uid"].isin(spec.train_cases)].copy()
    test_df = df[df["case_uid"].isin(spec.test_cases)].copy()
    if train_all.empty or test_df.empty:
        raise ValueError(f"Split {spec.name} has empty train or test set.")

    fit_df, val_df = train_validation_split(train_all, spec.name)
    re_levels = df["Re"].dropna().unique()
    seeds = [RANDOM_STATE + i * 17 for i in range(ensemble_size)]
    predictor = XGBEnsemble(seeds=seeds, use_gpu=use_gpu, target=target)
    predictor.fit(fit_df[FEATURES], fit_df[target])

    ood = OODDistance(FEATURES).fit(fit_df)
    val_raw = raw_reliability_scores(predictor, ood, val_df, target, fit_df[target], re_levels)
    test_raw = raw_reliability_scores(predictor, ood, test_df, target, fit_df[target], re_levels)
    val_scores, test_scores, norm_meta = normalize_signal_frames(val_raw, test_raw)

    y_val = val_df[target].to_numpy(dtype=float)
    y_val_pred = val_scores["prediction"].to_numpy(dtype=float)
    tolerance = TARGET_TOLERANCE_PCT[target]
    val_failure = failure_scores(y_val, y_val_pred, tolerance)
    learned_gate = fit_learned_gate(val_scores, val_failure)

    y_test = test_df[target].to_numpy(dtype=float)
    y_test_pred = test_scores["prediction"].to_numpy(dtype=float)
    test_ape = 100.0 * np.abs(y_test_pred - y_test) / np.maximum(np.abs(y_test), 1e-12)
    test_failure = test_ape > tolerance

    val_methods = method_scores(val_scores, learned_gate, oracle_ape=None, random_seed=RANDOM_STATE)
    test_methods = method_scores(test_scores, learned_gate, oracle_ape=test_ape, random_seed=RANDOM_STATE)

    primary = "conservative_U_D_V_max"
    tau_predict = float(np.quantile(val_methods[primary], 0.80))
    tau_warn = float(np.quantile(val_methods[primary], 0.90))
    decisions = gate_decisions(test_methods[primary], tau_predict, tau_warn)

    pred_out = test_df[["case_uid", "raw_row_id", "geometry_id"] + FEATURES + [target]].copy()
    pred_out["split_name"] = spec.name
    pred_out["split_family"] = spec.family
    pred_out["target"] = target
    pred_out["prediction"] = y_test_pred
    pred_out["residual"] = y_test_pred - y_test
    pred_out["abs_error"] = np.abs(y_test_pred - y_test)
    pred_out["ape_pct"] = test_ape
    pred_out["failure_tolerance_pct"] = tolerance
    pred_out["is_failure"] = test_failure
    for col in test_scores.columns:
        pred_out[col] = test_scores[col].to_numpy()
    pred_out["reliability_score"] = test_methods[primary]
    pred_out["gate_decision"] = decisions
    pred_out["tau_predict"] = tau_predict
    pred_out["tau_warn"] = tau_warn
    save_table(pred_out, dirs["predictions"] / f"{slug(spec.name)}_{slug(target)}_predictions")

    val_out = val_df[["case_uid", "raw_row_id", "geometry_id"] + FEATURES + [target]].copy()
    val_out["split_name"] = spec.name
    val_out["target"] = target
    val_out["prediction"] = y_val_pred
    val_out["ape_pct"] = 100.0 * np.abs(y_val_pred - y_val) / np.maximum(np.abs(y_val), 1e-12)
    val_out["is_failure"] = val_failure
    for col in val_scores.columns:
        val_out[col] = val_scores[col].to_numpy()
    save_table(val_out, dirs["plot_data"] / f"{slug(spec.name)}_{slug(target)}_validation_scores")

    model_path = dirs["models"] / f"{slug(spec.name)}_{slug(target)}_xgb_gpu_ensemble.joblib"
    joblib.dump(
        {
            "split": spec,
            "target": target,
            "feature_columns": FEATURES,
            "predictor": predictor,
            "ood": ood,
            "normalizers": norm_meta,
            "tau_predict": tau_predict,
            "tau_warn": tau_warn,
            "primary_score": primary,
            "xgboost_gpu_enabled": use_gpu,
        },
        model_path,
    )

    method_summary_rows = []
    curve_rows = []
    matched_rows = []
    for name, score in test_methods.items():
        method_summary_rows.append(summarize_score_method(name, score, y_test, y_test_pred, tolerance, spec.name, spec.family, target))
        curve_rows.extend(risk_coverage_rows(name, score, y_test, y_test_pred, spec.name, target))
        matched_rows.extend(matched_coverage_rows(name, score, y_test, y_test_pred, spec.name, spec.family, target))

    gate_counts = pd.Series(decisions).value_counts().to_dict()
    accepted_predict = decisions == "predict"
    warn_only = decisions == "warn"
    abstain_only = decisions == "abstain"
    accepted_predict_warn = accepted_predict | warn_only
    physics_violation = test_scores["physics_violation_raw"].to_numpy() > 1e-12
    gate_summary = {
        "split_name": spec.name,
        "family": spec.family,
        "target": target,
        "primary_score": primary,
        "tau_predict": tau_predict,
        "tau_warn": tau_warn,
        "n_test": int(len(test_df)),
        "n_predict": int(gate_counts.get("predict", 0)),
        "n_warn": int(gate_counts.get("warn", 0)),
        "n_abstain": int(gate_counts.get("abstain", 0)),
        "coverage_predict_only": float(np.mean(accepted_predict)),
        "coverage_warn_only": float(np.mean(warn_only)),
        "coverage_abstain_only": float(np.mean(abstain_only)),
        "coverage_predict_or_warn": float(np.mean(accepted_predict_warn)),
        "failure_rate_all": float(np.mean(test_failure)),
        "failure_rate_predict_only": float(np.mean(test_failure[accepted_predict])) if np.any(accepted_predict) else float("nan"),
        "failure_rate_warn_only": float(np.mean(test_failure[warn_only])) if np.any(warn_only) else float("nan"),
        "failure_rate_abstain_only": float(np.mean(test_failure[abstain_only])) if np.any(abstain_only) else float("nan"),
        "failure_rate_predict_or_warn": float(np.mean(test_failure[accepted_predict_warn])) if np.any(accepted_predict_warn) else float("nan"),
        "MAPE_all_pct": regression_metrics(y_test, y_test_pred)["MAPE_pct"],
        "MAPE_predict_only_pct": regression_metrics(y_test[accepted_predict], y_test_pred[accepted_predict])["MAPE_pct"] if np.any(accepted_predict) else float("nan"),
        "MAPE_warn_only_pct": regression_metrics(y_test[warn_only], y_test_pred[warn_only])["MAPE_pct"] if np.any(warn_only) else float("nan"),
        "MAPE_abstain_only_pct": regression_metrics(y_test[abstain_only], y_test_pred[abstain_only])["MAPE_pct"] if np.any(abstain_only) else float("nan"),
        "MAPE_predict_or_warn_pct": regression_metrics(y_test[accepted_predict_warn], y_test_pred[accepted_predict_warn])["MAPE_pct"] if np.any(accepted_predict_warn) else float("nan"),
        "physics_violation_rate_all": float(np.mean(physics_violation)),
        "physics_violation_rate_predict_only": float(np.mean(physics_violation[accepted_predict])) if np.any(accepted_predict) else float("nan"),
        "physics_violation_rate_warn_only": float(np.mean(physics_violation[warn_only])) if np.any(warn_only) else float("nan"),
        "physics_violation_rate_abstain_only": float(np.mean(physics_violation[abstain_only])) if np.any(abstain_only) else float("nan"),
        "physics_violation_rate_predict_or_warn": float(np.mean(physics_violation[accepted_predict_warn])) if np.any(accepted_predict_warn) else float("nan"),
    }

    feature_importance = pd.DataFrame(
        {
            "split_name": spec.name,
            "target": target,
            "feature": FEATURES,
            "importance": predictor.feature_importance(),
        }
    ).sort_values("importance", ascending=False)
    save_table(feature_importance, dirs["tables"] / f"{slug(spec.name)}_{slug(target)}_feature_importance")

    calibration = calibration_rows(spec.name, target, test_methods[primary], test_failure)

    return {
        "score_summary": method_summary_rows,
        "curves": curve_rows,
        "matched": matched_rows,
        "gate": [gate_summary],
        "calibration": calibration,
        "feature_importance": feature_importance.to_dict(orient="records"),
    }


def maybe_log_transform_estimator(estimator, target: str):
    if target != "DelP_Pa":
        return estimator
    return TransformedTargetRegressor(regressor=estimator, func=np.log1p, inverse_func=np.expm1)


def benchmark_id_models(df: pd.DataFrame, splits: list[SplitSpec], use_gpu: bool, dirs: dict[str, Path]) -> pd.DataFrame:
    id_split = next(spec for spec in splits if spec.name.startswith("random80_20"))
    train_df = df[df["case_uid"].isin(id_split.train_cases)].copy()
    test_df = df[df["case_uid"].isin(id_split.test_cases)].copy()
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=450, learning_rate=0.045, random_state=RANDOM_STATE),
        "SVR_RBF": make_pipeline(StandardScaler(), SVR(C=80.0, epsilon=0.02, gamma="scale")),
    }
    if xgb is not None:
        models["XGBoost_GPU" if use_gpu else "XGBoost_CPU"] = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=800,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.95,
            colsample_bytree=0.95,
            random_state=RANDOM_STATE,
            tree_method="hist",
            device="cuda" if use_gpu else "cpu",
            verbosity=0,
        )

    rows = []
    prediction_frames = []
    for target in TARGETS:
        X_train = train_df[FEATURES]
        X_test = test_df[FEATURES]
        y_train = train_df[target]
        y_test = test_df[target].to_numpy(dtype=float)
        for name, estimator in models.items():
            print(f"[benchmark] {name} target={target}")
            model = maybe_log_transform_estimator(estimator, target)
            started = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            elapsed = time.perf_counter() - started
            pred = np.asarray(model.predict(X_test), dtype=float)
            metrics = regression_metrics(y_test, pred)
            rows.append(
                {
                    "model": name,
                    "target": target,
                    "split_name": id_split.name,
                    "train_seconds": elapsed,
                    **metrics,
                }
            )
            frame = test_df[["case_uid", "raw_row_id", "geometry_id"] + FEATURES + [target]].copy()
            frame["model"] = name
            frame["target"] = target
            frame["prediction"] = pred
            frame["ape_pct"] = 100.0 * np.abs(pred - y_test) / np.maximum(np.abs(y_test), 1e-12)
            prediction_frames.append(frame)
            joblib.dump(model, dirs["models"] / f"benchmark_{slug(name)}_{slug(target)}.joblib")
    bench_df = pd.DataFrame(rows).sort_values(["target", "MAPE_pct"])
    save_table(bench_df, dirs["tables"] / "benchmark_random80_20_models")
    save_table(pd.concat(prediction_frames, ignore_index=True), dirs["predictions"] / "benchmark_random80_20_predictions")
    return bench_df


def plot_framework(dirs: dict[str, Path]) -> None:
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.axis("off")
    boxes = [
        ("Input case\n(Re, Pr, Da,\ngeometry)", 0.04, 0.45),
        ("Predictive\nsurrogate", 0.25, 0.45),
        ("Uncertainty\nU(x)", 0.47, 0.68),
        ("OOD distance\nD(x)", 0.47, 0.45),
        ("Physics check\nV(x)", 0.47, 0.22),
        ("Reliability gate\nR(x)", 0.70, 0.45),
        ("Predict\nWarn\nAbstain", 0.89, 0.45),
    ]
    for text, x, y in boxes:
        ax.text(x, y, text, ha="center", va="center", fontsize=11, bbox=dict(boxstyle="round,pad=0.35", fc="#f4f6f8", ec="#263238", lw=1.0))
    arrows = [
        ((0.12, 0.45), (0.19, 0.45)),
        ((0.33, 0.45), (0.41, 0.68)),
        ((0.33, 0.45), (0.41, 0.45)),
        ((0.33, 0.45), (0.41, 0.22)),
        ((0.55, 0.68), (0.64, 0.49)),
        ((0.55, 0.45), (0.64, 0.45)),
        ((0.55, 0.22), (0.64, 0.41)),
        ((0.77, 0.45), (0.84, 0.45)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.3, color="#263238"))
    fig.tight_layout()
    fig.savefig(dirs["figures"] / "framework_diagram.png", dpi=300, bbox_inches="tight")
    fig.savefig(dirs["figures"] / "framework_diagram.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_dataset(df: pd.DataFrame, dirs: dict[str, Path]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()
    axes[0].hist(df["Nuavg"], bins=35, color="#3b6ea8", edgecolor="white")
    axes[0].set_title("Average Nusselt number")
    axes[1].hist(df["DelP_Pa"], bins=35, color="#b55a30", edgecolor="white")
    axes[1].set_title("Pressure drop")
    axes[2].scatter(df["Re"], df["Nuavg"], s=10, alpha=0.5, c=df["a_mm"], cmap="viridis")
    axes[2].set_xlabel("Re")
    axes[2].set_ylabel("Nuavg")
    axes[3].scatter(df["Re"], df["DelP_Pa"], s=10, alpha=0.5, c=df["Hp_mm"], cmap="plasma")
    axes[3].set_xlabel("Re")
    axes[3].set_ylabel("DelP_Pa")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(dirs["figures"] / "dataset_overview.png", dpi=300, bbox_inches="tight")
    fig.savefig(dirs["figures"] / "dataset_overview.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_results(
    df: pd.DataFrame,
    bench: pd.DataFrame,
    curves: pd.DataFrame,
    predictions: pd.DataFrame,
    gate: pd.DataFrame,
    feature_importance: pd.DataFrame,
    calibration: pd.DataFrame,
    dirs: dict[str, Path],
) -> None:
    plot_framework(dirs)
    plot_dataset(df, dirs)

    primary_curves = curves[
        (curves["score_method"].isin(["U_only", "D_only", "V_only", "U_plus_D_plus_V", "conservative_U_D_V_max", "learned_logistic_gate", "random_rejection", "oracle_error_rank"]))
        & (curves["split_name"].isin(["random80_20", "holdout_Re500", "combined_G06_Re500"]))
    ].copy()
    for target in TARGETS:
        subset = primary_curves[primary_curves["target"] == target]
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        for method, part in subset.groupby("score_method"):
            part = part.sort_values("coverage")
            ax.plot(part["coverage"], part["MAPE_pct"], marker="o", ms=3, label=method)
        ax.set_xlabel("Coverage")
        ax.set_ylabel("Selective risk (MAPE, %)")
        ax.set_title(f"Risk-coverage curves: {target}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(dirs["figures"] / f"risk_coverage_{slug(target)}.png", dpi=300, bbox_inches="tight")
        fig.savefig(dirs["figures"] / f"risk_coverage_{slug(target)}.pdf", bbox_inches="tight")
        plt.close(fig)

    random_preds = predictions[(predictions["split_name"] == "random80_20") & (predictions["target"].isin(TARGETS))]
    for target in TARGETS:
        part = random_preds[random_preds["target"] == target]
        if part.empty:
            continue
        fig, ax = plt.subplots(figsize=(5.5, 5))
        actual = part[target].to_numpy(dtype=float)
        pred = part["prediction"].to_numpy(dtype=float)
        ax.scatter(actual, pred, c=part["reliability_score"], cmap="magma_r", s=16, alpha=0.75)
        lims = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
        ax.plot(lims, lims, color="black", lw=1)
        ax.set_xlabel("CFD-derived target")
        ax.set_ylabel("Prediction")
        ax.set_title(f"Parity plot: {target}")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(dirs["figures"] / f"parity_random80_20_{slug(target)}.png", dpi=300, bbox_inches="tight")
        fig.savefig(dirs["figures"] / f"parity_random80_20_{slug(target)}.pdf", bbox_inches="tight")
        plt.close(fig)

    for target in TARGETS:
        part = predictions[predictions["target"] == target]
        if part.empty:
            continue
        sample = part.sample(n=min(len(part), 6000), random_state=RANDOM_STATE)
        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.scatter(sample["D_norm"], sample["ape_pct"], c=sample["U_norm"], cmap="viridis", s=13, alpha=0.65)
        ax.set_xlabel("Normalized OOD distance")
        ax.set_ylabel("Absolute percentage error (%)")
        ax.set_title(f"Error versus OOD distance: {target}")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(dirs["figures"] / f"error_vs_ood_{slug(target)}.png", dpi=300, bbox_inches="tight")
        fig.savefig(dirs["figures"] / f"error_vs_ood_{slug(target)}.pdf", bbox_inches="tight")
        plt.close(fig)

    for target in TARGETS:
        part = predictions[(predictions["target"] == target) & (predictions["split_name"].isin(["random80_20", "holdout_Re500", "combined_G06_Re500"]))]
        if part.empty:
            continue
        fig, ax = plt.subplots(figsize=(6.5, 5))
        sc = ax.scatter(part["Re"], part["Da"], c=part["reliability_score"], cmap="magma_r", s=24, alpha=0.85)
        ax.set_yscale("log")
        ax.set_xlabel("Re")
        ax.set_ylabel("Da")
        ax.set_title(f"Reliability map in operating space: {target}")
        fig.colorbar(sc, ax=ax, label="Reliability score")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(dirs["figures"] / f"reliability_map_re_da_{slug(target)}.png", dpi=300, bbox_inches="tight")
        fig.savefig(dirs["figures"] / f"reliability_map_re_da_{slug(target)}.pdf", bbox_inches="tight")
        plt.close(fig)

    gate_plot = gate.melt(
        id_vars=["split_name", "family", "target"],
        value_vars=["n_predict", "n_warn", "n_abstain"],
        var_name="decision",
        value_name="count",
    )
    for target in TARGETS:
        part = gate_plot[gate_plot["target"] == target]
        if part.empty:
            continue
        pivot = part.pivot_table(index="split_name", columns="decision", values="count", aggfunc="sum").fillna(0)
        pivot = pivot.reindex(sorted(pivot.index))
        fig, ax = plt.subplots(figsize=(10, 5))
        pivot.plot(kind="bar", stacked=True, ax=ax, color=["#9e2f2f", "#3b6ea8", "#d9a441"])
        ax.set_ylabel("Number of test cases")
        ax.set_title(f"Reliability-gate decisions: {target}")
        ax.legend(title="")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(dirs["figures"] / f"gate_decisions_{slug(target)}.png", dpi=300, bbox_inches="tight")
        fig.savefig(dirs["figures"] / f"gate_decisions_{slug(target)}.pdf", bbox_inches="tight")
        plt.close(fig)

    if not bench.empty:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        bench_plot = bench.copy()
        bench_plot["label"] = bench_plot["model"] + " / " + bench_plot["target"]
        bench_plot.sort_values("MAPE_pct").plot.barh(x="label", y="MAPE_pct", ax=ax, color="#4d7c5f", legend=False)
        ax.set_xlabel("MAPE (%)")
        ax.set_title("Random split benchmark accuracy")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(dirs["figures"] / "benchmark_accuracy.png", dpi=300, bbox_inches="tight")
        fig.savefig(dirs["figures"] / "benchmark_accuracy.pdf", bbox_inches="tight")
        plt.close(fig)

    if not feature_importance.empty:
        fi = (
            feature_importance.groupby(["target", "feature"], as_index=False)["importance"]
            .mean()
            .sort_values(["target", "importance"], ascending=[True, False])
        )
        for target in TARGETS:
            part = fi[fi["target"] == target]
            if part.empty:
                continue
            fig, ax = plt.subplots(figsize=(6.2, 4.6))
            ax.barh(part["feature"], part["importance"], color="#3b6ea8")
            ax.invert_yaxis()
            ax.set_xlabel("Mean XGBoost gain importance")
            ax.set_title(f"Feature importance: {target}")
            ax.grid(axis="x", alpha=0.25)
            fig.tight_layout()
            fig.savefig(dirs["figures"] / f"feature_importance_{slug(target)}.png", dpi=300, bbox_inches="tight")
            fig.savefig(dirs["figures"] / f"feature_importance_{slug(target)}.pdf", bbox_inches="tight")
            plt.close(fig)

    if not calibration.empty:
        for target in TARGETS:
            part = calibration[calibration["target"] == target]
            if part.empty:
                continue
            fig, ax = plt.subplots(figsize=(5.8, 4.8))
            for split, group in part.groupby("split_name"):
                if split not in ["random80_20", "holdout_Re500", "combined_G06_Re500"]:
                    continue
                ax.plot(group["mean_score"], group["empirical_failure_rate"], marker="o", ms=4, label=split)
            ax.plot([0, 1], [0, 1], color="black", lw=1, alpha=0.6)
            ax.set_xlabel("Mean reliability score")
            ax.set_ylabel("Empirical failure rate")
            ax.set_title(f"Reliability calibration: {target}")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(dirs["figures"] / f"calibration_{slug(target)}.png", dpi=300, bbox_inches="tight")
            fig.savefig(dirs["figures"] / f"calibration_{slug(target)}.pdf", bbox_inches="tight")
            plt.close(fig)


def write_summary_report(
    df: pd.DataFrame,
    env: dict[str, object],
    bench: pd.DataFrame,
    score_summary: pd.DataFrame,
    matched: pd.DataFrame,
    gate: pd.DataFrame,
    dirs: dict[str, Path],
) -> None:
    primary = score_summary[score_summary["score_method"] == "conservative_U_D_V_max"].copy()
    best_id = bench.sort_values(["target", "MAPE_pct"]).groupby("target").head(1)

    def fmt_number(value: object, digits: int = 3) -> str:
        return "--" if pd.isna(value) else f"{float(value):.{digits}f}"

    lines = [
        "# Experiment Summary",
        "",
        "## Dataset",
        f"- Rows: {len(df)}",
        f"- Geometry IDs: {', '.join(sorted(df['geometry_id'].unique()))}",
        f"- Features: {', '.join(FEATURES)}",
        f"- Targets: {', '.join(TARGETS)}",
        "",
        "## Compute",
        f"- XGBoost GPU enabled: {env['xgboost_gpu_enabled']}",
        f"- XGBoost note: {env['xgboost_gpu_note']}",
        f"- Torch CUDA available: {env['torch_cuda_available']}",
        "",
        "## Best in-distribution benchmark",
    ]
    for _, row in best_id.iterrows():
        lines.append(
            f"- {row['target']}: {row['model']} with MAPE={row['MAPE_pct']:.3f}%, RMSE={row['RMSE']:.4g}, R2={row['R2']:.4f}"
        )
    lines.extend(["", "## Primary reliability gate highlights"])
    for target in TARGETS:
        for split in ["random80_20", "holdout_Re500", "combined_G06_Re500"]:
            part = gate[(gate["target"] == target) & (gate["split_name"] == split)]
            if part.empty:
                continue
            row = part.iloc[0]
            lines.append(
                f"- {target}, {split}: all-case MAPE={row['MAPE_all_pct']:.3f}%, "
                f"predict={int(row['n_predict'])} cases (MAPE={fmt_number(row.get('MAPE_predict_only_pct'))}%), "
                f"warn={int(row['n_warn'])} cases (MAPE={fmt_number(row.get('MAPE_warn_only_pct'))}%), "
                f"abstain={int(row['n_abstain'])} cases (MAPE={fmt_number(row.get('MAPE_abstain_only_pct'))}%), "
                f"predict+warn coverage={row['coverage_predict_or_warn']:.3f}"
            )
    lines.extend(
        [
            "",
            "## Exported artifacts",
            f"- Tables: `{dirs['tables']}`",
            f"- Predictions: `{dirs['predictions']}`",
            f"- Figures: `{dirs['figures']}`",
            f"- Models: `{dirs['models']}`",
            f"- Reports: `{dirs['reports']}`",
        ]
    )
    (dirs["reports"] / "experiment_summary.md").write_text("\n".join(lines), encoding="utf-8")

    article = {
        "dataset": {
            "n_rows": int(len(df)),
            "n_geometries": int(df["geometry_id"].nunique()),
            "geometry_ids": sorted(df["geometry_id"].unique().tolist()),
            "features": FEATURES,
            "targets": TARGETS,
        },
        "environment": {
            "xgboost_gpu_enabled": env["xgboost_gpu_enabled"],
            "xgboost_gpu_note": env["xgboost_gpu_note"],
            "torch_cuda_available": env["torch_cuda_available"],
        },
        "best_id_benchmark": best_id.to_dict(orient="records"),
        "primary_score_summary": primary.to_dict(orient="records"),
        "matched_coverage_primary": matched[matched["score_method"] == "conservative_U_D_V_max"].to_dict(orient="records"),
        "gate_summary": gate.to_dict(orient="records"),
    }
    save_json(article, dirs["reports"] / "article_results_summary.json")


def export_artifact_manifest(dirs: dict[str, Path]) -> None:
    rows = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("artifact_manifest."):
            continue
        rel = path.relative_to(ROOT).as_posix()
        rows.append(
            {
                "relative_path": rel,
                "extension": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(dirs["reports"] / "artifact_manifest.csv", index=False)
    save_json(manifest.to_dict(orient="records"), dirs["reports"] / "artifact_manifest.json")


def summarize_gate_decisions_from_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split_name, target), part in predictions.groupby(["split_name", "target"], sort=True):
        decisions = part["gate_decision"].astype(str).to_numpy()
        predict_only = decisions == "predict"
        warn_only = decisions == "warn"
        abstain_only = decisions == "abstain"
        retained = predict_only | warn_only
        failure = part["is_failure"].astype(bool).to_numpy()
        ape = part["ape_pct"].to_numpy(dtype=float)
        physics_raw = (
            part["physics_violation_raw"].to_numpy(dtype=float)
            if "physics_violation_raw" in part.columns
            else np.full(len(part), np.nan)
        )
        physics_violation = physics_raw > 1e-12

        def mean_for(values: np.ndarray, mask: np.ndarray) -> float:
            return float(np.mean(values[mask])) if np.any(mask) else float("nan")

        def physics_rate(mask: np.ndarray) -> float:
            if not np.any(mask) or np.isnan(physics_raw[mask]).all():
                return float("nan")
            return float(np.mean(physics_violation[mask]))

        rows.append(
            {
                "split_name": split_name,
                "family": part["split_family"].iloc[0],
                "target": target,
                "primary_score": "conservative_U_D_V_max",
                "tau_predict": float(part["tau_predict"].iloc[0]) if "tau_predict" in part else float("nan"),
                "tau_warn": float(part["tau_warn"].iloc[0]) if "tau_warn" in part else float("nan"),
                "n_test": int(len(part)),
                "n_predict": int(np.sum(predict_only)),
                "n_warn": int(np.sum(warn_only)),
                "n_abstain": int(np.sum(abstain_only)),
                "coverage_predict_only": float(np.mean(predict_only)),
                "coverage_warn_only": float(np.mean(warn_only)),
                "coverage_abstain_only": float(np.mean(abstain_only)),
                "coverage_predict_or_warn": float(np.mean(retained)),
                "failure_rate_all": float(np.mean(failure)),
                "failure_rate_predict_only": mean_for(failure, predict_only),
                "failure_rate_warn_only": mean_for(failure, warn_only),
                "failure_rate_abstain_only": mean_for(failure, abstain_only),
                "failure_rate_predict_or_warn": mean_for(failure, retained),
                "MAPE_all_pct": float(np.mean(ape)),
                "MAPE_predict_only_pct": mean_for(ape, predict_only),
                "MAPE_warn_only_pct": mean_for(ape, warn_only),
                "MAPE_abstain_only_pct": mean_for(ape, abstain_only),
                "MAPE_predict_or_warn_pct": mean_for(ape, retained),
                "physics_violation_rate_all": physics_rate(np.ones(len(part), dtype=bool)),
                "physics_violation_rate_predict_only": physics_rate(predict_only),
                "physics_violation_rate_warn_only": physics_rate(warn_only),
                "physics_violation_rate_abstain_only": physics_rate(abstain_only),
                "physics_violation_rate_predict_or_warn": physics_rate(retained),
            }
        )
    return pd.DataFrame(rows)


def export_article_tables(
    bench: pd.DataFrame,
    score_summary: pd.DataFrame,
    matched: pd.DataFrame,
    gate: pd.DataFrame,
    predictions: pd.DataFrame,
    dirs: dict[str, Path],
) -> None:
    benchmark_cols = ["model", "target", "RMSE", "MAPE_pct", "R2", "train_seconds"]
    save_table(
        bench[benchmark_cols].sort_values(["target", "MAPE_pct"]),
        dirs["tables"] / "article_table_1_random_split_benchmark",
    )

    selected_splits = ["random80_20", "holdout_Re500", "combined_G06_Re500"]
    gate_cols = [
        "split_name",
        "target",
        "n_test",
        "n_predict",
        "n_warn",
        "n_abstain",
        "coverage_predict_only",
        "coverage_warn_only",
        "coverage_abstain_only",
        "MAPE_all_pct",
        "MAPE_predict_only_pct",
        "MAPE_warn_only_pct",
        "MAPE_abstain_only_pct",
        "failure_rate_all",
        "failure_rate_predict_only",
        "failure_rate_warn_only",
        "failure_rate_abstain_only",
        "coverage_predict_or_warn",
        "MAPE_predict_or_warn_pct",
        "failure_rate_predict_or_warn",
    ]
    save_table(
        gate[gate["split_name"].isin(selected_splits)][[col for col in gate_cols if col in gate.columns]].sort_values(["split_name", "target"]),
        dirs["tables"] / "article_table_2_selected_gate_behavior",
    )

    if not predictions.empty:
        aggregate_rows = []
        for (family, target), part in predictions.groupby(["split_family", "target"]):
            predict = part["gate_decision"].eq("predict")
            warn = part["gate_decision"].eq("warn")
            abstain = part["gate_decision"].eq("abstain")
            retained = predict | warn
            aggregate_rows.append(
                {
                    "split_family": family,
                    "target": target,
                    "n_cases": int(len(part)),
                    "n_predict": int(predict.sum()),
                    "n_warn": int(warn.sum()),
                    "n_abstain": int((part["gate_decision"] == "abstain").sum()),
                    "coverage_predict_pct": 100.0 * float(predict.mean()),
                    "coverage_warn_pct": 100.0 * float(warn.mean()),
                    "coverage_abstain_pct": 100.0 * float(abstain.mean()),
                    "MAPE_all_pct": float(part["ape_pct"].mean()),
                    "MAPE_predict_only_pct": float(part.loc[predict, "ape_pct"].mean()) if predict.any() else float("nan"),
                    "MAPE_warn_only_pct": float(part.loc[warn, "ape_pct"].mean()) if warn.any() else float("nan"),
                    "MAPE_abstain_only_pct": float(part.loc[abstain, "ape_pct"].mean()) if abstain.any() else float("nan"),
                    "failure_rate_all_pct": 100.0 * float(part["is_failure"].mean()),
                    "failure_rate_predict_only_pct": 100.0 * float(part.loc[predict, "is_failure"].mean()) if predict.any() else float("nan"),
                    "failure_rate_warn_only_pct": 100.0 * float(part.loc[warn, "is_failure"].mean()) if warn.any() else float("nan"),
                    "failure_rate_abstain_only_pct": 100.0 * float(part.loc[abstain, "is_failure"].mean()) if abstain.any() else float("nan"),
                    "coverage_predict_or_warn_pct": 100.0 * float(retained.mean()),
                    "MAPE_predict_or_warn_pct": float(part.loc[retained, "ape_pct"].mean()) if retained.any() else float("nan"),
                }
            )
        save_table(
            pd.DataFrame(aggregate_rows).sort_values(["split_family", "target"]),
            dirs["tables"] / "article_table_3_family_gate_aggregate",
        )

    ablation_methods = [
        "U_only",
        "D_only",
        "V_only",
        "U_plus_D_plus_V",
        "conservative_U_D_V_max",
        "random_rejection",
        "oracle_error_rank",
    ]
    ablation_cols = ["split_name", "target", "score_method", "AURC_MAPE_pct", "failure_AUROC", "failure_AUPRC"]
    save_table(
        score_summary[
            score_summary["split_name"].isin(selected_splits) & score_summary["score_method"].isin(ablation_methods)
        ][ablation_cols].sort_values(["split_name", "target", "score_method"]),
        dirs["tables"] / "article_table_4_selected_ablation",
    )

    matched_cols = ["split_name", "target", "coverage_target", "actual_coverage", "MAPE_pct", "RMSE", "R2"]
    save_table(
        matched[
            matched["split_name"].isin(selected_splits) & (matched["score_method"] == "conservative_U_D_V_max")
        ][matched_cols].sort_values(["split_name", "target", "coverage_target"], ascending=[True, True, False]),
        dirs["tables"] / "article_table_5_matched_coverage_primary_gate",
    )


def load_selective_prediction_files(dirs: dict[str, Path]) -> pd.DataFrame:
    prediction_files = sorted(dirs["predictions"].glob("*_predictions.csv"))
    excluded = {"benchmark_random80_20_predictions.csv", "all_selective_predictions.csv"}
    selective_prediction_files = [p for p in prediction_files if p.name not in excluded]
    if not selective_prediction_files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(path) for path in selective_prediction_files], ignore_index=True)


def rebuild_artifacts_from_existing(dirs: dict[str, Path]) -> None:
    env_path = dirs["reports"] / "environment_report.json"
    if not env_path.exists():
        raise FileNotFoundError(f"Missing environment report: {env_path}")
    with env_path.open("r", encoding="utf-8") as handle:
        env = json.load(handle)
    df = load_primary_dataset(dirs)
    bench = pd.read_csv(dirs["tables"] / "benchmark_random80_20_models.csv")
    curves = pd.read_csv(dirs["plot_data"] / "risk_coverage_curves.csv")
    gate = pd.read_csv(dirs["tables"] / "gate_decision_summary.csv")
    feature_importance = pd.read_csv(dirs["tables"] / "xgb_ensemble_feature_importance_all_splits.csv")
    calibration = pd.read_csv(dirs["plot_data"] / "reliability_calibration.csv")
    matched = pd.read_csv(dirs["tables"] / "matched_coverage_metrics.csv")
    score_summary = pd.read_csv(dirs["tables"] / "selective_score_summary.csv")
    predictions = load_selective_prediction_files(dirs)
    if not predictions.empty:
        save_table(predictions, dirs["predictions"] / "all_selective_predictions")
        gate = summarize_gate_decisions_from_predictions(predictions)
        save_table(gate, dirs["tables"] / "gate_decision_summary")
    export_article_tables(bench, score_summary, matched, gate, predictions, dirs)
    plot_results(df, bench, curves, predictions, gate, feature_importance, calibration, dirs)
    write_summary_report(df, env, bench, score_summary, matched, gate, dirs)
    export_artifact_manifest(dirs)
    print(f"[done] rebuilt aggregate predictions, plots, and reports from {OUT}")


def run(args: argparse.Namespace) -> None:
    dirs = ensure_dirs()
    np.random.seed(RANDOM_STATE)

    if args.rebuild_artifacts:
        rebuild_artifacts_from_existing(dirs)
        return

    use_gpu, gpu_note = probe_xgboost_gpu()
    if args.cpu:
        use_gpu = False
        gpu_note = "CPU mode requested from command line."
    env = environment_report(use_gpu, gpu_note, dirs)

    df = load_primary_dataset(dirs)
    save_dataset_profile(df, dirs)
    splits = build_splits(df, dirs)
    plot_dataset(df, dirs)

    bench = benchmark_id_models(df, splits, use_gpu, dirs)

    all_score_summary: list[dict[str, object]] = []
    all_curves: list[dict[str, object]] = []
    all_matched: list[dict[str, object]] = []
    all_gate: list[dict[str, object]] = []
    all_calibration: list[dict[str, object]] = []
    all_importance: list[dict[str, object]] = []

    for spec in splits:
        for target in TARGETS:
            print(f"[selective] split={spec.name} target={target} family={spec.family}")
            result = run_selective_experiment(df, spec, target, use_gpu, dirs, args.ensemble_size)
            all_score_summary.extend(result["score_summary"])
            all_curves.extend(result["curves"])
            all_matched.extend(result["matched"])
            all_gate.extend(result["gate"])
            all_calibration.extend(result["calibration"])
            all_importance.extend(result["feature_importance"])

    score_summary = pd.DataFrame(all_score_summary)
    curves = pd.DataFrame(all_curves)
    matched = pd.DataFrame(all_matched)
    gate = pd.DataFrame(all_gate)
    calibration = pd.DataFrame(all_calibration)
    feature_importance = pd.DataFrame(all_importance)

    save_table(score_summary, dirs["tables"] / "selective_score_summary")
    save_table(curves, dirs["plot_data"] / "risk_coverage_curves")
    save_table(matched, dirs["tables"] / "matched_coverage_metrics")
    save_table(gate, dirs["tables"] / "gate_decision_summary")
    save_table(calibration, dirs["plot_data"] / "reliability_calibration")
    save_table(feature_importance, dirs["tables"] / "xgb_ensemble_feature_importance_all_splits")

    predictions = load_selective_prediction_files(dirs)
    if not predictions.empty:
        save_table(predictions, dirs["predictions"] / "all_selective_predictions")
    export_article_tables(bench, score_summary, matched, gate, predictions, dirs)

    plot_results(df, bench, curves, predictions, gate, feature_importance, calibration, dirs)
    write_summary_report(df, env, bench, score_summary, matched, gate, dirs)
    export_artifact_manifest(dirs)
    print(f"[done] results written to {OUT}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run selective prediction experiments for the wavy-channel dataset.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode even if XGBoost CUDA is available.")
    parser.add_argument("--ensemble-size", type=int, default=5, help="Number of XGBoost members used for epistemic uncertainty.")
    parser.add_argument("--rebuild-artifacts", action="store_true", help="Rebuild aggregate predictions, plots, and reports from existing tables.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
