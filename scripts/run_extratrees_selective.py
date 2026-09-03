from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from run_experiments import (
    FEATURES,
    OUT,
    RANDOM_STATE,
    TARGETS,
    TARGET_TOLERANCE_PCT,
    OODDistance,
    SplitSpec,
    build_splits,
    calibration_rows,
    ensure_dirs,
    failure_scores,
    gate_decisions,
    load_primary_dataset,
    matched_coverage_rows,
    method_scores,
    normalize_signal_frames,
    raw_reliability_scores,
    regression_metrics,
    risk_coverage_rows,
    save_table,
    slug,
    summarize_score_method,
    train_validation_split,
)


ET_OUT = OUT / "extratrees_selective"


class ExtraTreesEnsemble:
    def __init__(self, seeds: list[int], target: str):
        self.seeds = seeds
        self.target = target
        self.models: list[ExtraTreesRegressor] = []
        self.log_target = target == "DelP_Pa"

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "ExtraTreesEnsemble":
        yy = np.asarray(y, dtype=float)
        if self.log_target:
            yy = np.log1p(np.maximum(yy, 0.0))
        self.models = []
        for seed in self.seeds:
            model = ExtraTreesRegressor(
                n_estimators=500,
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=-1,
            )
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
        importances = [np.asarray(model.feature_importances_, dtype=float) for model in self.models]
        return np.vstack(importances).mean(axis=0)


def ensure_et_dirs() -> dict[str, Path]:
    dirs = ensure_dirs()
    et_dirs = {
        "models": ET_OUT / "models",
        "predictions": ET_OUT / "predictions",
        "tables": ET_OUT / "tables",
        "plot_data": ET_OUT / "plot_data",
        "reports": ET_OUT / "reports",
        "data_processed": dirs["data_processed"],
    }
    for path in et_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return et_dirs


def run_extratrees_experiment(
    df: pd.DataFrame,
    spec: SplitSpec,
    target: str,
    dirs: dict[str, Path],
    ensemble_size: int = 5,
) -> dict[str, list[dict[str, object]]]:
    train_all = df[df["case_uid"].isin(spec.train_cases)].copy()
    test_df = df[df["case_uid"].isin(spec.test_cases)].copy()
    if train_all.empty or test_df.empty:
        raise ValueError(f"Split {spec.name} has empty train or test set.")

    fit_df, val_df = train_validation_split(train_all, spec.name)
    re_levels = df["Re"].dropna().unique()
    seeds = [RANDOM_STATE + i * 17 for i in range(ensemble_size)]
    predictor = ExtraTreesEnsemble(seeds=seeds, target=target)

    started = time.perf_counter()
    predictor.fit(fit_df[FEATURES], fit_df[target])
    train_seconds = time.perf_counter() - started

    ood = OODDistance(FEATURES).fit(fit_df)
    val_raw = raw_reliability_scores(predictor, ood, val_df, target, fit_df[target], re_levels)
    test_raw = raw_reliability_scores(predictor, ood, test_df, target, fit_df[target], re_levels)
    val_scores, test_scores, norm_meta = normalize_signal_frames(val_raw, test_raw)

    y_val = val_df[target].to_numpy(dtype=float)
    y_val_pred = val_scores["prediction"].to_numpy(dtype=float)
    tolerance = TARGET_TOLERANCE_PCT[target]
    val_failure = failure_scores(y_val, y_val_pred, tolerance)
    learned_gate = None

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
    pred_out["model_family"] = "ExtraTrees"
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
    save_table(pred_out, dirs["predictions"] / f"extratrees_{slug(spec.name)}_{slug(target)}_predictions")

    val_out = val_df[["case_uid", "raw_row_id", "geometry_id"] + FEATURES + [target]].copy()
    val_out["model_family"] = "ExtraTrees"
    val_out["split_name"] = spec.name
    val_out["target"] = target
    val_out["prediction"] = y_val_pred
    val_out["ape_pct"] = 100.0 * np.abs(y_val_pred - y_val) / np.maximum(np.abs(y_val), 1e-12)
    val_out["is_failure"] = val_failure
    for col in val_scores.columns:
        val_out[col] = val_scores[col].to_numpy()
    save_table(val_out, dirs["plot_data"] / f"extratrees_{slug(spec.name)}_{slug(target)}_validation_scores")

    model_path = dirs["models"] / f"extratrees_{slug(spec.name)}_{slug(target)}_ensemble.joblib"
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
        },
        model_path,
    )

    method_summary_rows = []
    curve_rows = []
    matched_rows = []
    for name, score in test_methods.items():
        row = summarize_score_method(name, score, y_test, y_test_pred, tolerance, spec.name, spec.family, target)
        row["model_family"] = "ExtraTrees"
        method_summary_rows.append(row)
        curve_rows.extend(
            {
                **curve_row,
                "model_family": "ExtraTrees",
            }
            for curve_row in risk_coverage_rows(name, score, y_test, y_test_pred, spec.name, target)
        )
        matched_rows.extend(
            {
                **matched_row,
                "model_family": "ExtraTrees",
            }
            for matched_row in matched_coverage_rows(name, score, y_test, y_test_pred, spec.name, spec.family, target)
        )

    gate_counts = pd.Series(decisions).value_counts().to_dict()
    predict_only = decisions == "predict"
    warn_only = decisions == "warn"
    abstain_only = decisions == "abstain"
    retained = np.isin(decisions, ["predict", "warn"])
    physics_violation = test_scores["physics_violation_raw"].to_numpy() > 1e-12
    gate_summary = {
        "model_family": "ExtraTrees",
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
        "coverage_predict_only": float(np.mean(predict_only)),
        "coverage_warn_only": float(np.mean(warn_only)),
        "coverage_abstain_only": float(np.mean(abstain_only)),
        "coverage_predict_or_warn": float(np.mean(retained)),
        "failure_rate_all": float(np.mean(test_failure)),
        "failure_rate_predict_only": float(np.mean(test_failure[predict_only])) if np.any(predict_only) else float("nan"),
        "failure_rate_warn_only": float(np.mean(test_failure[warn_only])) if np.any(warn_only) else float("nan"),
        "failure_rate_abstain_only": float(np.mean(test_failure[abstain_only])) if np.any(abstain_only) else float("nan"),
        "failure_rate_predict_or_warn": float(np.mean(test_failure[retained])) if np.any(retained) else float("nan"),
        "MAPE_all_pct": regression_metrics(y_test, y_test_pred)["MAPE_pct"],
        "MAPE_predict_only_pct": regression_metrics(y_test[predict_only], y_test_pred[predict_only])["MAPE_pct"] if np.any(predict_only) else float("nan"),
        "MAPE_warn_only_pct": regression_metrics(y_test[warn_only], y_test_pred[warn_only])["MAPE_pct"] if np.any(warn_only) else float("nan"),
        "MAPE_abstain_only_pct": regression_metrics(y_test[abstain_only], y_test_pred[abstain_only])["MAPE_pct"] if np.any(abstain_only) else float("nan"),
        "MAPE_predict_or_warn_pct": regression_metrics(y_test[retained], y_test_pred[retained])["MAPE_pct"] if np.any(retained) else float("nan"),
        "train_seconds": float(train_seconds),
        "physics_violation_rate_all": float(np.mean(physics_violation)),
        "physics_violation_rate_predict_only": float(np.mean(physics_violation[predict_only])) if np.any(predict_only) else float("nan"),
        "physics_violation_rate_warn_only": float(np.mean(physics_violation[warn_only])) if np.any(warn_only) else float("nan"),
        "physics_violation_rate_abstain_only": float(np.mean(physics_violation[abstain_only])) if np.any(abstain_only) else float("nan"),
        "physics_violation_rate_predict_or_warn": float(np.mean(physics_violation[retained])) if np.any(retained) else float("nan"),
    }

    feature_importance = pd.DataFrame(
        {
            "model_family": "ExtraTrees",
            "split_name": spec.name,
            "target": target,
            "feature": FEATURES,
            "importance": predictor.feature_importance(),
        }
    ).sort_values("importance", ascending=False)
    save_table(feature_importance, dirs["tables"] / f"extratrees_{slug(spec.name)}_{slug(target)}_feature_importance")

    calibration = [
        {**row, "model_family": "ExtraTrees"}
        for row in calibration_rows(spec.name, target, test_methods[primary], test_failure)
    ]

    return {
        "score_summary": method_summary_rows,
        "curves": curve_rows,
        "matched": matched_rows,
        "gate": [gate_summary],
        "calibration": calibration,
        "feature_importance": feature_importance.to_dict(orient="records"),
    }


def build_family_gate_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_family, family, target), part in predictions.groupby(["model_family", "split_family", "target"], sort=True):
        for decision in ["predict", "warn", "abstain"]:
            subset = part[part["gate_decision"] == decision]
            rows.append(
                {
                    "model_family": model_family,
                    "split_family": family,
                    "target": target,
                    "decision": decision,
                    "n_cases": int(len(subset)),
                    "coverage_pct": 100.0 * len(subset) / len(part),
                    "MAPE_pct": float(subset["ape_pct"].mean()) if len(subset) else np.nan,
                    "failure_rate_pct": 100.0 * float(subset["is_failure"].mean()) if len(subset) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_model_comparison(xgb_predictions: pd.DataFrame, et_predictions: pd.DataFrame) -> pd.DataFrame:
    xgb = xgb_predictions.copy()
    xgb["model_family"] = "XGBoost"
    combined = pd.concat([xgb, et_predictions], ignore_index=True)
    rows = []
    for (model_family, family, target), part in combined.groupby(["model_family", "split_family", "target"], sort=True):
        predict_only = part["gate_decision"].eq("predict")
        warn_only = part["gate_decision"].eq("warn")
        abstain_only = part["gate_decision"].eq("abstain")
        rows.append(
            {
                "model_family": model_family,
                "split_family": family,
                "target": target,
                "n_cases": int(len(part)),
                "predict_cov_pct": 100.0 * float(predict_only.mean()),
                "warn_cov_pct": 100.0 * float(warn_only.mean()),
                "abstain_cov_pct": 100.0 * float(abstain_only.mean()),
                "MAPE_all_pct": float(part["ape_pct"].mean()),
                "MAPE_predict_pct": float(part.loc[predict_only, "ape_pct"].mean()) if predict_only.any() else np.nan,
                "MAPE_warn_pct": float(part.loc[warn_only, "ape_pct"].mean()) if warn_only.any() else np.nan,
                "MAPE_abstain_pct": float(part.loc[abstain_only, "ape_pct"].mean()) if abstain_only.any() else np.nan,
                "failure_predict_pct": 100.0 * float(part.loc[predict_only, "is_failure"].mean()) if predict_only.any() else np.nan,
                "failure_warn_pct": 100.0 * float(part.loc[warn_only, "is_failure"].mean()) if warn_only.any() else np.nan,
                "failure_abstain_pct": 100.0 * float(part.loc[abstain_only, "is_failure"].mean()) if abstain_only.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    dirs = ensure_et_dirs()
    df = load_primary_dataset(dirs)
    splits = build_splits(df, dirs)

    all_score_summary: list[dict[str, object]] = []
    all_curves: list[dict[str, object]] = []
    all_matched: list[dict[str, object]] = []
    all_gate: list[dict[str, object]] = []
    all_calibration: list[dict[str, object]] = []
    all_importance: list[dict[str, object]] = []
    all_predictions: list[pd.DataFrame] = []

    for spec in splits:
        for target in TARGETS:
            print(f"[extratrees-selective] split={spec.name} target={target}")
            result = run_extratrees_experiment(df, spec, target, dirs)
            all_score_summary.extend(result["score_summary"])
            all_curves.extend(result["curves"])
            all_matched.extend(result["matched"])
            all_gate.extend(result["gate"])
            all_calibration.extend(result["calibration"])
            all_importance.extend(result["feature_importance"])
            prediction_path = dirs["predictions"] / f"extratrees_{slug(spec.name)}_{slug(target)}_predictions.csv"
            all_predictions.append(pd.read_csv(prediction_path))

    score_summary = pd.DataFrame(all_score_summary)
    curves = pd.DataFrame(all_curves)
    matched = pd.DataFrame(all_matched)
    gate = pd.DataFrame(all_gate)
    calibration = pd.DataFrame(all_calibration)
    feature_importance = pd.DataFrame(all_importance)
    predictions = pd.concat(all_predictions, ignore_index=True)

    save_table(score_summary, dirs["tables"] / "extratrees_selective_score_summary")
    save_table(curves, dirs["plot_data"] / "extratrees_risk_coverage_curves")
    save_table(matched, dirs["tables"] / "extratrees_matched_coverage_metrics")
    save_table(gate, dirs["tables"] / "extratrees_gate_decision_summary")
    save_table(calibration, dirs["plot_data"] / "extratrees_reliability_calibration")
    save_table(feature_importance, dirs["tables"] / "extratrees_feature_importance_all_splits")
    save_table(predictions, dirs["predictions"] / "extratrees_all_selective_predictions")
    save_table(build_family_gate_table(predictions), dirs["tables"] / "extratrees_gate_by_decision_family")

    xgb_predictions = pd.read_csv(OUT / "predictions" / "all_selective_predictions.csv")
    comparison = build_model_comparison(xgb_predictions, predictions)
    save_table(comparison, dirs["tables"] / "xgb_vs_extratrees_gate_by_decision_family")
    save_table(comparison, OUT / "tables" / "article_table_9_xgb_vs_extratrees_gate_by_decision_family")

    report = {
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_splits": len(splits),
        "n_targets": len(TARGETS),
        "n_predictions": int(len(predictions)),
    }
    (dirs["reports"] / "extratrees_selective_report.json").write_text(
        pd.Series(report).to_json(indent=2),
        encoding="utf-8",
    )
    print(f"[done] ExtraTrees selective results written to {ET_OUT}")


if __name__ == "__main__":
    main()
