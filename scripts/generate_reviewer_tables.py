from pathlib import Path
import shutil

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional for table export
    plt = None


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
SUBMISSION_TABLES = RESULTS / "submission_tables"
PREDICTIONS = RESULTS / "predictions" / "all_selective_predictions.csv"
EXTRATREES_PREDICTIONS = (
    RESULTS / "extratrees_selective" / "predictions" / "extratrees_all_selective_predictions.csv"
)
PLOT_DATA = RESULTS / "plot_data"
FIGURES = RESULTS / "figures"
SUBMISSION_FIGURES = RESULTS / "submission_figures"
XGB_SCORE_SUMMARY = TABLES / "selective_score_summary.csv"
EXTRATREES_SCORE_SUMMARY = RESULTS / "extratrees_selective" / "tables" / "extratrees_selective_score_summary.csv"
XGB_RISK_CURVES = PLOT_DATA / "risk_coverage_curves.csv"
EXTRATREES_RISK_CURVES = RESULTS / "extratrees_selective" / "plot_data" / "extratrees_risk_coverage_curves.csv"
COVERAGE_GRID = sorted(set([0.05] + [round(x, 2) for x in np.linspace(0.1, 1.0, 19)] + [0.7, 0.8, 0.9, 0.95]))
BOOTSTRAP_REPS = 500
RANDOM_SEED = 42


SPLIT_LABELS = {
    "random80_20": "Random 80/20",
    "holdout_Re25": "Holdout Re=25",
    "holdout_Re100": "Holdout Re=100",
    "holdout_Re250": "Holdout Re=250",
    "holdout_Re500": "Holdout Re=500",
    "holdout_G01": "Holdout G01",
    "holdout_G02": "Holdout G02",
    "holdout_G03": "Holdout G03",
    "holdout_G04": "Holdout G04",
    "holdout_G05": "Holdout G05",
    "holdout_G06": "Holdout G06",
    "combined_G06_Re500": "Combined G06/Re500",
}

FAMILY_LABELS = {
    "in_distribution": "Random 80/20",
    "withheld_reynolds": "Withheld Reynolds",
    "withheld_geometry": "Withheld geometry",
    "combined_shift": "Combined G06/Re500",
}

STRUCTURED_SPLIT_ORDER = [
    "holdout_Re25",
    "holdout_Re100",
    "holdout_Re250",
    "holdout_Re500",
    "holdout_G01",
    "holdout_G02",
    "holdout_G03",
    "holdout_G04",
    "holdout_G05",
    "holdout_G06",
]

SELECTED_SPLITS = ["random80_20", "holdout_Re500", "combined_G06_Re500"]


def mape(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else np.nan


def failure_rate(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else np.nan


def pct(mask: pd.Series | np.ndarray) -> float:
    return 100.0 * float(np.mean(mask)) if len(mask) else np.nan


def mean_or_nan(values: pd.Series | np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)) if len(values) and np.isfinite(values).any() else np.nan


def ci_text(values: np.ndarray, precision: int = 1) -> str:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return "--"
    mean = float(np.mean(values))
    lo, hi = np.percentile(values, [2.5, 97.5])
    return f"{mean:.{precision}f} [{lo:.{precision}f}, {hi:.{precision}f}]"


def value_text(value: float, precision: int = 1) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.{precision}f}"


def pair_text(left: float, right: float, precision: int = 1) -> str:
    return f"{value_text(left, precision)}/{value_text(right, precision)}"


def split_order(name: str) -> int:
    order = {split: i for i, split in enumerate(["random80_20"] + STRUCTURED_SPLIT_ORDER + ["combined_G06_Re500"])}
    return order.get(name, 10_000)


def ensure_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def load_model_predictions() -> pd.DataFrame:
    xgb = pd.read_csv(PREDICTIONS)
    xgb["model_family"] = "XGBoost"
    frames = [xgb]
    if EXTRATREES_PREDICTIONS.exists():
        et = pd.read_csv(EXTRATREES_PREDICTIONS)
        et["model_family"] = et.get("model_family", "ExtraTrees")
        frames.append(et)
    out = pd.concat(frames, ignore_index=True)
    out["is_failure"] = ensure_bool(out["is_failure"])
    out["is_retained"] = out["gate_decision"].isin(["predict", "warn"])
    signed = []
    for _, row in out.iterrows():
        target = row["target"]
        y_true = row[target]
        signed.append(100.0 * (row["prediction"] - y_true) / max(abs(y_true), 1e-12))
    out["signed_pct_error"] = signed
    return out


def summarize_group(group: pd.DataFrame) -> dict[str, float | int]:
    retained = group["is_retained"]
    abstain = group["gate_decision"].eq("abstain")
    failed = group["is_failure"]
    false_accept = retained & failed
    false_abstain = abstain & ~failed
    return {
        "n_cases": int(len(group)),
        "n_retained": int(retained.sum()),
        "n_abstain": int(abstain.sum()),
        "n_failed": int(failed.sum()),
        "retained_cov_pct": pct(retained),
        "abstain_cov_pct": pct(abstain),
        "MAPE_all_pct": mape(group["ape_pct"]),
        "MAPE_retained_pct": mape(group.loc[retained, "ape_pct"]),
        "failure_all_pct": pct(failed),
        "failure_retained_pct": pct(group.loc[retained, "is_failure"]) if retained.any() else np.nan,
        "false_accept_n": int(false_accept.sum()),
        "false_accept_pct_all": pct(false_accept),
        "false_abstain_n": int(false_abstain.sum()),
        "false_abstain_pct_all": pct(false_abstain),
        "bias_pct": mean_or_nan(group["signed_pct_error"]),
    }


def compact_pair_table(full: pd.DataFrame, row_keys: list[str], value_cols: list[tuple[str, str, int]]) -> pd.DataFrame:
    rows = []
    for keys, group in full.groupby(row_keys, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(row_keys, keys))
        lookup = {record["target"]: record for _, record in group.iterrows()}
        nu = lookup.get("Nuavg", {})
        dp = lookup.get("DelP_Pa", {})
        for output_col, source_col, precision in value_cols:
            row[output_col] = pair_text(nu.get(source_col, np.nan), dp.get(source_col, np.nan), precision)
        rows.append(row)
    return pd.DataFrame(rows)


def load_score_summaries() -> pd.DataFrame:
    xgb = pd.read_csv(XGB_SCORE_SUMMARY)
    xgb["model_family"] = "XGBoost"
    frames = [xgb]
    if EXTRATREES_SCORE_SUMMARY.exists():
        et = pd.read_csv(EXTRATREES_SCORE_SUMMARY)
        et["model_family"] = et.get("model_family", "ExtraTrees")
        frames.append(et)
    return pd.concat(frames, ignore_index=True)


def load_risk_curves() -> pd.DataFrame:
    xgb = pd.read_csv(XGB_RISK_CURVES)
    xgb["model_family"] = "XGBoost"
    frames = [xgb]
    if EXTRATREES_RISK_CURVES.exists():
        et = pd.read_csv(EXTRATREES_RISK_CURVES)
        et["model_family"] = et.get("model_family", "ExtraTrees")
        frames.append(et)
    return pd.concat(frames, ignore_index=True)


def aurc_from_arrays(scores: np.ndarray, ape: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    ape = np.asarray(ape, dtype=float)
    if len(scores) == 0:
        return np.nan
    order = np.argsort(scores)
    coverages = []
    risks = []
    for coverage in COVERAGE_GRID:
        n_accept = max(1, int(np.ceil(coverage * len(order))))
        idx = order[:n_accept]
        coverages.append(n_accept / len(order))
        risks.append(float(np.mean(ape[idx])))
    denom = max(coverages[-1] - coverages[0], 1e-12)
    return float(np.trapezoid(risks, coverages) / denom)


def build_per_fold_structured_tables(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    structured = pred[pred["split_name"].isin(STRUCTURED_SPLIT_ORDER)].copy()
    rows = []
    for (model, split_name, split_family, target), group in structured.groupby(
        ["model_family", "split_name", "split_family", "target"], sort=False
    ):
        row = {
            "model_family": model,
            "split_name": split_name,
            "split": SPLIT_LABELS.get(split_name, split_name),
            "split_family": FAMILY_LABELS.get(split_family, split_family),
            "target": target,
            "split_order": split_order(split_name),
        }
        row.update(summarize_group(group))
        rows.append(row)
    full = pd.DataFrame(rows).sort_values(["model_family", "split_order", "target"])

    compact = compact_pair_table(
        full,
        ["model_family", "split", "split_family", "split_order"],
        [
            ("retained_cov_nu_delp_pct", "retained_cov_pct", 1),
            ("retained_mape_nu_delp_pct", "MAPE_retained_pct", 3),
            ("retained_fail_nu_delp_pct", "failure_retained_pct", 1),
            ("abstain_cov_nu_delp_pct", "abstain_cov_pct", 1),
            ("bias_nu_delp_pct", "bias_pct", 1),
        ],
    ).sort_values(["model_family", "split_order"])
    return full.drop(columns=["split_order"]), compact.drop(columns=["split_order"])


def build_threshold_clipping_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    eps = 1e-12
    for (model, split_name, split_family, target), group in pred.groupby(
        ["model_family", "split_name", "split_family", "target"], sort=False
    ):
        tau_predict = float(group["tau_predict"].iloc[0])
        tau_warn = float(group["tau_warn"].iloc[0])
        score_eq_one = group["reliability_score"] >= 1.0 - eps
        abstain = group["gate_decision"].eq("abstain")
        rows.append(
            {
                "model_family": model,
                "split_name": split_name,
                "split": SPLIT_LABELS.get(split_name, split_name),
                "split_family": FAMILY_LABELS.get(split_family, split_family),
                "target": target,
                "tau_predict": tau_predict,
                "tau_warn": tau_warn,
                "score_eq_1_pct": pct(score_eq_one),
                "abstain_eq_1_pct": pct(score_eq_one & abstain),
                "retained_cov_pct": pct(group["is_retained"]),
                "abstain_cov_pct": pct(abstain),
                "split_order": split_order(split_name),
            }
        )
    return pd.DataFrame(rows).sort_values(["model_family", "split_order", "target"]).drop(columns=["split_order"])


def build_threshold_clipping_compact(pred: pd.DataFrame, model_family: str = "XGBoost") -> pd.DataFrame:
    full = build_threshold_clipping_table(pred)
    selected = full[full["model_family"].eq(model_family)].copy()
    selected["split_order"] = selected["split_name"].map(split_order)
    compact = compact_pair_table(
        selected,
        ["split", "split_family", "split_order"],
        [
            ("tau_predict_nu_delp", "tau_predict", 3),
            ("tau_warn_nu_delp", "tau_warn", 3),
            ("score_eq_1_nu_delp_pct", "score_eq_1_pct", 1),
            ("retained_cov_nu_delp_pct", "retained_cov_pct", 1),
        ],
    ).sort_values("split_order")
    compact.insert(0, "model_family", model_family)
    return compact.drop(columns=["split_order"])


def build_component_driver_table(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    eps = 1e-12
    for (model, split_name, split_family, target), group in pred.groupby(
        ["model_family", "split_name", "split_family", "target"], sort=False
    ):
        abstained = group[group["gate_decision"].eq("abstain")]
        tau_warn = float(group["tau_warn"].iloc[0])
        if len(abstained):
            u_alarm = abstained["U_norm"] >= tau_warn - eps
            d_alarm = abstained["D_norm"] >= tau_warn - eps
            v_alarm = abstained["V_norm"] >= tau_warn - eps
            alarm_count = u_alarm.astype(int) + d_alarm.astype(int) + v_alarm.astype(int)
            max_score = abstained[["U_norm", "D_norm", "V_norm"]].max(axis=1)
            u_max = abstained["U_norm"] >= max_score - eps
            d_max = abstained["D_norm"] >= max_score - eps
            v_max = abstained["V_norm"] >= max_score - eps
        else:
            u_alarm = d_alarm = v_alarm = alarm_count = u_max = d_max = v_max = pd.Series(dtype=float)
        rows.append(
            {
                "model_family": model,
                "split_name": split_name,
                "split": SPLIT_LABELS.get(split_name, split_name),
                "split_family": FAMILY_LABELS.get(split_family, split_family),
                "target": target,
                "n_abstain": int(len(abstained)),
                "U_alarm_pct": pct(u_alarm),
                "D_alarm_pct": pct(d_alarm),
                "V_alarm_pct": pct(v_alarm),
                "multi_alarm_pct": pct(alarm_count > 1) if len(abstained) else np.nan,
                "U_max_driver_pct": pct(u_max),
                "D_max_driver_pct": pct(d_max),
                "V_max_driver_pct": pct(v_max),
                "split_order": split_order(split_name),
            }
        )
    full = pd.DataFrame(rows).sort_values(["model_family", "split_order", "target"]).drop(columns=["split_order"])

    family_rows = []
    for (model, split_family, target), group in pred.groupby(["model_family", "split_family", "target"], sort=False):
        abstained = group[group["gate_decision"].eq("abstain")]
        if len(abstained):
            tau = abstained["tau_warn"].to_numpy(dtype=float)
            u_alarm = abstained["U_norm"].to_numpy(dtype=float) >= tau - eps
            d_alarm = abstained["D_norm"].to_numpy(dtype=float) >= tau - eps
            v_alarm = abstained["V_norm"].to_numpy(dtype=float) >= tau - eps
            alarm_count = u_alarm.astype(int) + d_alarm.astype(int) + v_alarm.astype(int)
        else:
            u_alarm = d_alarm = v_alarm = alarm_count = np.array([], dtype=float)
        family_rows.append(
            {
                "model_family": model,
                "split_family": FAMILY_LABELS.get(split_family, split_family),
                "target": target,
                "n_abstain": int(len(abstained)),
                "U_alarm_pct": pct(u_alarm),
                "D_alarm_pct": pct(d_alarm),
                "V_alarm_pct": pct(v_alarm),
                "multi_alarm_pct": pct(alarm_count > 1) if len(abstained) else np.nan,
            }
        )
    compact = pd.DataFrame(family_rows).sort_values(["model_family", "split_family", "target"])
    paired = compact_pair_table(
        compact,
        ["model_family", "split_family"],
        [
            ("n_abstain_nu_delp", "n_abstain", 0),
            ("U_alarm_nu_delp_pct", "U_alarm_pct", 1),
            ("D_alarm_nu_delp_pct", "D_alarm_pct", 1),
            ("V_alarm_nu_delp_pct", "V_alarm_pct", 1),
            ("multi_alarm_nu_delp_pct", "multi_alarm_pct", 1),
        ],
    )
    return full, compact, paired


def build_aurc_comparison_tables(score_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = score_summary[score_summary["score_method"].eq("conservative_U_D_V_max")].copy()
    primary["split"] = primary["split_name"].map(SPLIT_LABELS).fillna(primary["split_name"])
    primary["split_order"] = primary["split_name"].map(split_order)
    full_cols = [
        "model_family",
        "split_name",
        "split",
        "family",
        "target",
        "full_MAPE_pct",
        "AURC_MAPE_pct",
        "failure_AUROC",
        "failure_AUPRC",
    ]
    full = primary[full_cols + ["split_order"]].sort_values(["model_family", "split_order", "target"]).drop(columns=["split_order"])

    selected = primary[primary["split_name"].isin(SELECTED_SPLITS)].copy()
    selected = selected.sort_values(["split_order", "target", "model_family"])
    selected = selected[
        [
            "model_family",
            "split",
            "target",
            "full_MAPE_pct",
            "AURC_MAPE_pct",
            "failure_AUROC",
            "failure_AUPRC",
        ]
    ]
    return full, selected


def build_false_accept_abstain_tables(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (model, split_name, split_family, target), group in pred.groupby(
        ["model_family", "split_name", "split_family", "target"], sort=False
    ):
        row = {
            "model_family": model,
            "split_name": split_name,
            "split": SPLIT_LABELS.get(split_name, split_name),
            "split_family": FAMILY_LABELS.get(split_family, split_family),
            "target": target,
            "split_order": split_order(split_name),
        }
        row.update(summarize_group(group))
        rows.append(row)
    full = pd.DataFrame(rows).sort_values(["model_family", "split_order", "target"])
    compact = compact_pair_table(
        full,
        ["model_family", "split", "split_family", "split_order"],
        [
            ("false_accept_nu_delp_pct_all", "false_accept_pct_all", 1),
            ("false_abstain_nu_delp_pct_all", "false_abstain_pct_all", 1),
            ("retained_fail_nu_delp_pct", "failure_retained_pct", 1),
        ],
    ).sort_values(["model_family", "split_order"])
    return full.drop(columns=["split_order"]), compact.drop(columns=["split_order"])


def build_false_accept_abstain_family_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, split_family, target), group in pred.groupby(["model_family", "split_family", "target"], sort=False):
        row = {
            "model_family": model,
            "split_family": FAMILY_LABELS.get(split_family, split_family),
            "target": target,
        }
        row.update(summarize_group(group))
        rows.append(row)
    full = pd.DataFrame(rows).sort_values(["model_family", "split_family", "target"])
    return compact_pair_table(
        full,
        ["model_family", "split_family"],
        [
            ("false_accept_nu_delp_pct_all", "false_accept_pct_all", 1),
            ("false_abstain_nu_delp_pct_all", "false_abstain_pct_all", 1),
            ("retained_fail_nu_delp_pct", "failure_retained_pct", 1),
        ],
    ).sort_values(["model_family", "split_family"])


def build_bootstrap_ci_table(pred: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for (model, split_family, target), group in pred.groupby(["model_family", "split_family", "target"], sort=False):
        ape = group["ape_pct"].to_numpy(dtype=float)
        retained = group["is_retained"].to_numpy(dtype=bool)
        failed = group["is_failure"].to_numpy(dtype=bool)
        score = group["reliability_score"].to_numpy(dtype=float)
        mape_samples = []
        coverage_samples = []
        retained_failure_samples = []
        aurc_samples = []
        n = len(group)
        for _ in range(BOOTSTRAP_REPS):
            idx = rng.integers(0, n, n)
            mape_samples.append(float(np.mean(ape[idx])))
            coverage_samples.append(100.0 * float(np.mean(retained[idx])))
            retained_idx = idx[retained[idx]]
            if len(retained_idx):
                retained_failure_samples.append(100.0 * float(np.mean(failed[retained_idx])))
            else:
                retained_failure_samples.append(np.nan)
            aurc_samples.append(aurc_from_arrays(score[idx], ape[idx]))
        rows.append(
            {
                "model_family": model,
                "split_family": FAMILY_LABELS.get(split_family, split_family),
                "target": target,
                "n_cases": int(n),
                "MAPE_all_pct_CI": ci_text(np.asarray(mape_samples), 2),
                "retained_cov_pct_CI": ci_text(np.asarray(coverage_samples), 1),
                "retained_failure_pct_CI": ci_text(np.asarray(retained_failure_samples), 1),
                "AURC_MAPE_pct_CI": ci_text(np.asarray(aurc_samples), 2),
            }
        )
    return pd.DataFrame(rows).sort_values(["model_family", "split_family", "target"])


def build_signed_error_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    selected_families = ["withheld_reynolds", "withheld_geometry", "combined_shift"]
    for (model, split_family, target), group in pred[pred["split_family"].isin(selected_families)].groupby(
        ["model_family", "split_family", "target"], sort=False
    ):
        retained = group["is_retained"]
        rows.append(
            {
                "model_family": model,
                "split_family": FAMILY_LABELS.get(split_family, split_family),
                "target": target,
                "bias_all_pct": mean_or_nan(group["signed_pct_error"]),
                "bias_retained_pct": mean_or_nan(group.loc[retained, "signed_pct_error"]),
                "bias_abstain_pct": mean_or_nan(group.loc[~retained, "signed_pct_error"]),
                "median_signed_error_pct": float(np.median(group["signed_pct_error"])),
            }
        )
    return pd.DataFrame(rows).sort_values(["model_family", "split_family", "target"])


def validation_scores_path(model: str, split_name: str, target: str) -> Path:
    stem = f"{split_name.lower()}_{target.lower()}_validation_scores.csv"
    if model == "ExtraTrees":
        return RESULTS / "extratrees_selective" / "plot_data" / f"extratrees_{stem}"
    return PLOT_DATA / stem


def save_figure_to_outputs(fig: "plt.Figure", stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    SUBMISSION_FIGURES.mkdir(parents=True, exist_ok=True)
    png_path = FIGURES / f"{stem}.png"
    pdf_path = FIGURES / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    shutil.copyfile(png_path, SUBMISSION_FIGURES / png_path.name)
    shutil.copyfile(pdf_path, SUBMISSION_FIGURES / pdf_path.name)


def plot_score_distributions(pred: pd.DataFrame) -> None:
    if plt is None:
        print("Skipped score distribution plots: matplotlib is unavailable")
        return
    for model in ["XGBoost", "ExtraTrees"]:
        model_pred = pred[pred["model_family"].eq(model)]
        if model_pred.empty:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
        for row_idx, target in enumerate(["Nuavg", "DelP_Pa"]):
            for col_idx, split_name in enumerate(SELECTED_SPLITS):
                ax = axes[row_idx, col_idx]
                test = model_pred[(model_pred["split_name"].eq(split_name)) & (model_pred["target"].eq(target))]
                val_path = validation_scores_path(model, split_name, target)
                if test.empty or not val_path.exists():
                    ax.axis("off")
                    continue
                val = pd.read_csv(val_path)
                ax.hist(val["U_D_V_max"], bins=np.linspace(0, 1, 21), alpha=0.55, density=True, label="Validation")
                ax.hist(test["reliability_score"], bins=np.linspace(0, 1, 21), alpha=0.55, density=True, label="Test")
                tau_predict = float(test["tau_predict"].iloc[0])
                tau_warn = float(test["tau_warn"].iloc[0])
                ax.axvline(tau_predict, color="black", linewidth=1.0, linestyle="--")
                ax.axvline(tau_warn, color="black", linewidth=1.0, linestyle=":")
                ax.set_title(f"{SPLIT_LABELS.get(split_name, split_name)}: {target}")
                if row_idx == 1:
                    ax.set_xlabel("Reliability score")
                if col_idx == 0:
                    ax.set_ylabel("Density")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
        fig.suptitle(f"{model} validation and test reliability-score distributions", y=1.03)
        fig.tight_layout()
        save_figure_to_outputs(fig, f"score_distribution_{model.lower()}")
        plt.close(fig)


def plot_model_risk_coverage_comparison(curves: pd.DataFrame) -> None:
    if plt is None:
        print("Skipped model risk-coverage comparison plot: matplotlib is unavailable")
        return
    selected = curves[
        curves["score_method"].eq("conservative_U_D_V_max")
        & curves["split_name"].isin(SELECTED_SPLITS)
        & curves["model_family"].isin(["XGBoost", "ExtraTrees"])
    ].copy()
    if selected.empty:
        return
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    colors = {"XGBoost": "#1f77b4", "ExtraTrees": "#d62728"}
    for row_idx, target in enumerate(["Nuavg", "DelP_Pa"]):
        for col_idx, split_name in enumerate(SELECTED_SPLITS):
            ax = axes[row_idx, col_idx]
            for model in ["XGBoost", "ExtraTrees"]:
                part = selected[
                    selected["target"].eq(target)
                    & selected["split_name"].eq(split_name)
                    & selected["model_family"].eq(model)
                ].sort_values("coverage")
                if part.empty:
                    continue
                ax.plot(part["coverage"], part["MAPE_pct"], marker="o", markersize=2.5, linewidth=1.3, label=model, color=colors[model])
            ax.set_title(f"{SPLIT_LABELS.get(split_name, split_name)}: {target}")
            if row_idx == 1:
                ax.set_xlabel("Coverage")
            if col_idx == 0:
                ax.set_ylabel("Retained MAPE (%)")
            ax.grid(alpha=0.25, linewidth=0.6)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Risk-coverage comparison for the conservative max gate", y=1.03)
    fig.tight_layout()
    save_figure_to_outputs(fig, "risk_coverage_model_comparison")
    plt.close(fig)


def write_table_files(df: pd.DataFrame, table_dir: Path, stem: str) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(table_dir / f"{stem}.csv", index=False)
    try:
        df.to_excel(table_dir / f"{stem}.xlsx", index=False)
    except Exception as exc:  # pragma: no cover - depends on optional writer engines
        print(f"Skipped XLSX export for {stem}: {exc}")

    latex_path = table_dir / f"{stem}.tex"
    with latex_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\\begin{tabular}{" + "l" * len(df.columns) + "}\n")
        handle.write("\\hline\n")
        handle.write(" & ".join(map(str, df.columns)) + " \\\\\n")
        handle.write("\\hline\n")
        for _, row in df.iterrows():
            cells = []
            for value in row:
                if pd.isna(value):
                    cells.append("--")
                elif isinstance(value, (float, np.floating)):
                    cells.append(f"{value:.3f}")
                else:
                    cells.append(str(value))
            handle.write(" & ".join(cells) + " \\\\\n")
        handle.write("\\hline\n")
        handle.write("\\end{tabular}\n")


def export_table(df: pd.DataFrame, stem: str) -> None:
    write_table_files(df, TABLES, stem)
    write_table_files(df, SUBMISSION_TABLES, stem)


def build_family_gate_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, target), group in pred.groupby(["split_family", "target"], sort=True):
        predict = group["gate_decision"].eq("predict")
        warn = group["gate_decision"].eq("warn")
        abstain = group["gate_decision"].eq("abstain")
        retained = predict | warn
        rows.append(
            {
                "split_family": family,
                "target": target,
                "n_cases": len(group),
                "n_predict": int(predict.sum()),
                "n_warn": int(warn.sum()),
                "n_abstain": int(abstain.sum()),
                "coverage_predict_pct": 100.0 * float(predict.mean()),
                "coverage_warn_pct": 100.0 * float(warn.mean()),
                "coverage_abstain_pct": 100.0 * float(abstain.mean()),
                "MAPE_all_pct": mape(group["ape_pct"]),
                "MAPE_predict_only_pct": mape(group.loc[predict, "ape_pct"]),
                "MAPE_warn_only_pct": mape(group.loc[warn, "ape_pct"]),
                "MAPE_abstain_only_pct": mape(group.loc[abstain, "ape_pct"]),
                "failure_rate_all_pct": 100.0 * failure_rate(group["is_failure"]),
                "failure_rate_predict_only_pct": 100.0
                * failure_rate(group.loc[predict, "is_failure"]),
                "failure_rate_warn_only_pct": 100.0 * failure_rate(group.loc[warn, "is_failure"]),
                "failure_rate_abstain_only_pct": 100.0
                * failure_rate(group.loc[abstain, "is_failure"]),
                "coverage_predict_or_warn_pct": 100.0 * float(retained.mean()),
                "MAPE_predict_or_warn_pct": mape(group.loc[retained, "ape_pct"]),
                "failure_rate_predict_or_warn_pct": 100.0
                * failure_rate(group.loc[retained, "is_failure"]),
            }
        )
    return pd.DataFrame(rows)


def build_threshold_sensitivity(pred: pd.DataFrame) -> pd.DataFrame:
    threshold_pairs = [(0.70, 0.85), (0.75, 0.90), (0.80, 0.90), (0.85, 0.95)]
    scored = []

    for (split_name, target), test in pred.groupby(["split_name", "target"], sort=True):
        validation_path = (
            PLOT_DATA / f"{split_name.lower()}_{target.lower()}_validation_scores.csv"
        )
        validation = pd.read_csv(validation_path)
        validation_score = validation["U_D_V_max"]
        score = test["reliability_score"]

        for q_predict, q_warn in threshold_pairs:
            tau_predict = float(validation_score.quantile(q_predict))
            tau_warn = float(validation_score.quantile(q_warn))
            gate = np.where(score < tau_predict, "predict", np.where(score < tau_warn, "warn", "abstain"))
            predict = gate == "predict"
            warn = gate == "warn"
            retained = predict | warn

            scored.append(
                {
                    "split_name": split_name,
                    "split_family": test["split_family"].iloc[0],
                    "target": target,
                    "q_predict": q_predict,
                    "q_warn": q_warn,
                    "n_cases": len(test),
                    "n_predict": int(predict.sum()),
                    "n_warn": int(warn.sum()),
                    "n_abstain": int((gate == "abstain").sum()),
                    "coverage_predict_or_warn_pct": 100.0 * float(retained.mean()),
                    "MAPE_all_pct": mape(test["ape_pct"]),
                    "MAPE_predict_or_warn_pct": mape(test.loc[retained, "ape_pct"]),
                    "failure_rate_all_pct": 100.0 * failure_rate(test["is_failure"]),
                    "failure_rate_predict_or_warn_pct": 100.0
                    * failure_rate(test.loc[retained, "is_failure"]),
                }
            )

    detailed = pd.DataFrame(scored)
    family_rows = []
    for keys, group in detailed.groupby(["split_family", "target", "q_predict", "q_warn"], sort=True):
        family, target, q_predict, q_warn = keys
        n_cases = group["n_cases"].sum()
        n_predict = group["n_predict"].sum()
        n_warn = group["n_warn"].sum()
        n_abstain = group["n_abstain"].sum()

        source = pred[(pred["split_family"] == family) & (pred["target"] == target)]
        retained_parts = []
        for split_name in group["split_name"]:
            g = pred[(pred["split_name"] == split_name) & (pred["target"] == target)]
            validation = pd.read_csv(
                PLOT_DATA / f"{split_name.lower()}_{target.lower()}_validation_scores.csv"
            )
            tau_predict = float(validation["U_D_V_max"].quantile(q_predict))
            tau_warn = float(validation["U_D_V_max"].quantile(q_warn))
            gate = np.where(
                g["reliability_score"] < tau_predict,
                "predict",
                np.where(g["reliability_score"] < tau_warn, "warn", "abstain"),
            )
            retained_parts.append(g[pd.Series(gate, index=g.index).isin(["predict", "warn"])])
        retained_source = pd.concat(retained_parts, ignore_index=True) if retained_parts else source.iloc[0:0]

        family_rows.append(
            {
                "split_family": family,
                "target": target,
                "q_predict": q_predict,
                "q_warn": q_warn,
                "n_cases": int(n_cases),
                "n_predict": int(n_predict),
                "n_warn": int(n_warn),
                "n_abstain": int(n_abstain),
                "coverage_predict_or_warn_pct": 100.0 * (n_predict + n_warn) / n_cases,
                "MAPE_all_pct": mape(source["ape_pct"]),
                "MAPE_predict_or_warn_pct": mape(retained_source["ape_pct"]),
                "failure_rate_all_pct": 100.0 * failure_rate(source["is_failure"]),
                "failure_rate_predict_or_warn_pct": 100.0
                * failure_rate(retained_source["is_failure"]),
            }
        )

    family = pd.DataFrame(family_rows)
    return detailed, family


def build_compact_sensitivity_table(family: pd.DataFrame) -> pd.DataFrame:
    scenario_labels = {
        "in_distribution": "Random 80/20",
        "withheld_reynolds": "Withheld Reynolds",
        "combined_shift": "Combined G06/Re500",
    }
    rows = []
    selected = family[family["split_family"].isin(scenario_labels)]
    for (q_predict, q_warn, family_name), group in selected.groupby(
        ["q_predict", "q_warn", "split_family"], sort=True
    ):
        by_target = {row["target"]: row for _, row in group.iterrows()}
        nu = by_target.get("Nuavg")
        dp = by_target.get("DelP_Pa")
        if nu is None or dp is None:
            continue
        rows.append(
            {
                "q_predict/q_warn": f"{q_predict:.2f}/{q_warn:.2f}",
                "split_family": scenario_labels[family_name],
                "predict_cov_nu_delp_pct": f"{100.0 * nu['n_predict'] / nu['n_cases']:.1f}/{100.0 * dp['n_predict'] / dp['n_cases']:.1f}",
                "retained_cov_nu_delp_pct": f"{nu['coverage_predict_or_warn_pct']:.1f}/{dp['coverage_predict_or_warn_pct']:.1f}",
                "retained_mape_nu_delp_pct": (
                    "--/--"
                    if pd.isna(nu["MAPE_predict_or_warn_pct"]) and pd.isna(dp["MAPE_predict_or_warn_pct"])
                    else f"{nu['MAPE_predict_or_warn_pct']:.3f}/{dp['MAPE_predict_or_warn_pct']:.3f}"
                ),
            }
        )
    return pd.DataFrame(rows)


def format_pair(left: float, right: float, precision: int = 1) -> str:
    if pd.isna(left) and pd.isna(right):
        return "--/--"
    left_text = "--" if pd.isna(left) else f"{left:.{precision}f}"
    right_text = "--" if pd.isna(right) else f"{right:.{precision}f}"
    return f"{left_text}/{right_text}"


def build_model_decision_tables(xgb_pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not EXTRATREES_PREDICTIONS.exists():
        return pd.DataFrame(), pd.DataFrame()

    xgb = xgb_pred.copy()
    xgb["model_family"] = "XGBoost"
    et = pd.read_csv(EXTRATREES_PREDICTIONS)
    combined = pd.concat([xgb, et], ignore_index=True)

    full_rows = []
    for (model, family, target), part in combined.groupby(["model_family", "split_family", "target"], sort=True):
        for decision in ["predict", "warn", "abstain"]:
            subset = part[part["gate_decision"] == decision]
            full_rows.append(
                {
                    "model_family": model,
                    "split_family": family,
                    "target": target,
                    "decision": decision,
                    "n_cases": int(len(subset)),
                    "coverage_pct": 100.0 * len(subset) / len(part),
                    "MAPE_pct": mape(subset["ape_pct"]),
                    "failure_rate_pct": 100.0 * failure_rate(subset["is_failure"]),
                }
            )
    full = pd.DataFrame(full_rows)

    split_labels = {
        "in_distribution": "Random 80/20",
        "withheld_geometry": "Withheld geometry",
        "withheld_reynolds": "Withheld Reynolds",
        "combined_shift": "Combined G06/Re500",
    }
    compact_rows = []
    selected = full[full["split_family"].isin(split_labels)]
    for (model, family), group in selected.groupby(["model_family", "split_family"], sort=True):
        lookup = {
            (row["target"], row["decision"]): row
            for _, row in group.iterrows()
        }
        def pair(metric: str, decision: str, precision: int) -> str:
            nu = lookup.get(("Nuavg", decision), {})
            dp = lookup.get(("DelP_Pa", decision), {})
            return format_pair(nu.get(metric, np.nan), dp.get(metric, np.nan), precision)

        compact_rows.append(
            {
                "model_family": model,
                "split_family": split_labels[family],
                "predict_cov_nu_delp_pct": pair("coverage_pct", "predict", 1),
                "warn_cov_nu_delp_pct": pair("coverage_pct", "warn", 1),
                "abstain_cov_nu_delp_pct": pair("coverage_pct", "abstain", 1),
                "predict_mape_nu_delp_pct": pair("MAPE_pct", "predict", 3),
                "warn_mape_nu_delp_pct": pair("MAPE_pct", "warn", 3),
                "abstain_mape_nu_delp_pct": pair("MAPE_pct", "abstain", 3),
                "predict_fail_nu_delp_pct": pair("failure_rate_pct", "predict", 1),
                "warn_fail_nu_delp_pct": pair("failure_rate_pct", "warn", 1),
                "abstain_fail_nu_delp_pct": pair("failure_rate_pct", "abstain", 1),
            }
        )
    compact = pd.DataFrame(compact_rows)
    return full, compact


def main() -> None:
    pred = pd.read_csv(PREDICTIONS)
    combined_pred = load_model_predictions()
    score_summary = load_score_summaries()
    risk_curves = load_risk_curves()

    family_gate = build_family_gate_table(pred)
    detailed_sensitivity, family_sensitivity = build_threshold_sensitivity(pred)
    compact_sensitivity = build_compact_sensitivity_table(family_sensitivity)
    model_decision_full, model_decision_compact = build_model_decision_tables(pred)
    per_fold_full, per_fold_compact = build_per_fold_structured_tables(combined_pred)
    threshold_clipping = build_threshold_clipping_table(combined_pred)
    threshold_clipping_compact = build_threshold_clipping_compact(combined_pred)
    component_full, component_compact, component_paired = build_component_driver_table(combined_pred)
    aurc_full, aurc_selected = build_aurc_comparison_tables(score_summary)
    false_tradeoff_full, false_tradeoff_compact = build_false_accept_abstain_tables(combined_pred)
    false_tradeoff_family = build_false_accept_abstain_family_table(combined_pred)
    bootstrap_ci = build_bootstrap_ci_table(combined_pred)
    signed_error = build_signed_error_table(combined_pred)

    export_table(family_gate, "article_table_6_family_gate_full")
    export_table(detailed_sensitivity, "article_table_7_threshold_sensitivity_by_split")
    export_table(family_sensitivity, "article_table_7_threshold_sensitivity_family")
    export_table(compact_sensitivity, "article_table_8_threshold_sensitivity_compact")
    if not model_decision_full.empty:
        export_table(model_decision_full, "article_table_9_xgb_vs_extratrees_gate_by_decision_family")
        export_table(model_decision_compact, "article_table_10_xgb_vs_extratrees_decision_compact")
    export_table(per_fold_full, "article_table_11_per_fold_structured_gate_full")
    export_table(per_fold_compact, "article_table_12_per_fold_structured_gate_compact")
    export_table(threshold_clipping, "article_table_13_threshold_clipping")
    export_table(component_full, "article_table_14_component_gate_drivers_full")
    export_table(component_compact, "article_table_15_component_gate_drivers_family")
    export_table(aurc_full, "article_table_16_xgb_extratrees_aurc_full")
    export_table(aurc_selected, "article_table_17_xgb_extratrees_aurc_selected")
    export_table(false_tradeoff_full, "article_table_18_false_accept_abstain_full")
    export_table(false_tradeoff_compact, "article_table_19_false_accept_abstain_compact")
    export_table(bootstrap_ci, "article_table_20_bootstrap_uncertainty")
    export_table(signed_error, "article_table_21_signed_error_by_shift")
    export_table(threshold_clipping_compact, "article_table_22_threshold_clipping_xgboost_compact")
    export_table(component_paired, "article_table_23_component_gate_drivers_paired")
    export_table(false_tradeoff_family, "article_table_24_false_accept_abstain_family")
    plot_score_distributions(combined_pred)
    plot_model_risk_coverage_comparison(risk_curves)

    print("reviewer tables exported")


if __name__ == "__main__":
    main()
