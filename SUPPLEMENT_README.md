# Scientific supplement: thermo-hydraulic selective prediction

The archive contains complete numerical evidence for two separately fitted/calibrated systems: the primary wavy-channel case study and cross-problem U-bend replication. Each system supplies 468 model/fold/target/seed experiments. The total is 936 experiments and 1,940,760 selective-prediction rows. Serialized seed-42 point models are included in the prepared repository; the scientific archive supplies their exact settings and baseline predictions without duplicating the large model binaries.

## Paths and reproduction

`results/revision_round1/` contains wavy-channel metrics, predictions, partitions, conformal endpoints, bootstrap intervals, sensitivity/tolerance sweeps and mixed tests. `results/ubend_round1/` provides the same measurement layout plus 21 CSV/LaTeX tables, 10 vector/PNG figures, exact plot data, 60 point benchmarks and 36 fit-budget/tuned-SVR controls. Numerical validation reports accompany the results. Both systems include gp_pressure_audit.csv (all 78 GP pressure configurations in total) and cqr_calibration_audit.csv (all 156 CQR configurations in total), with per-run replay records under predictive_audit/. No reported model results are replaced by these verification records.

Follow `README.md` and `docs/reproducibility_guide.md`; use Python 3.14 and `requirements.txt`. Completed checkpoints are resumable. Source datasets, attributions, fixed source commit, checksums and clean-data schemas are preserved under `data/`. The updated U-bend release declares CC-BY-NC-4.0. `SHA256SUMS.json` identifies exact members of this archive.

## Files and columns

- `metrics.csv`: seed, fold, target, model, role, method, cutoff, coverage, all/retained MAPE, retained failure, false accept/abstain, prevalence, AUROC/AUPRC, Spearman, AURC and metric status.
- `runs/*/predictions.csv`: case IDs, input coordinates, reference/prediction, APE, raw/scaled scores and primary binary decision. `partitions.csv` assigns every used case to its allowed role.
- `runs/*/complete.json`: settings, counts, seeds, fit timing and selected kernel/degree/penalty. U-bend `calibration_scores.csv` permits independent cutoff replay.
- `matched.csv`, `curves.csv`: expected random tie-order risk versus coverage, separate from deterministic cutoff decisions.
- `sensitivity.csv`, `binary_coverage_sweep.csv`, `tolerance_sweep.csv`: actual cutoffs/state risks, binary quantile sensitivity and error-tolerance dependence.
- `cluster_intervals.csv`, `coarse_cluster_intervals.csv`: conditional descriptive bands with sampling units and insufficient-cluster flags.
- `mixed_shift.csv`: common-model support mixtures, failure metrics and distinct `ood_` detection columns.
- `conformal_group_coverage.csv`, `runs/*_CQR/interval_endpoints.csv`: row/group interval coverage and endpoints; these are not selective-risk guarantees.
- `fit_budget_svr_controls.csv`, `point_baselines.csv`, `svr_tuning_candidates.csv`: fit counts, selected settings, convergence and point accuracy. Benchmark budgets differ from selective common-fit budgets.
- U-bend `error_distribution.csv`: median/tail APE, bounded symmetric APE, raw ranges and numerical-cap activation counts. `physics_activation.csv` and `geometry_diagnostics.csv` report descriptive geometry/range diagnostics.

MAPE/APE are percentages; fractions in raw metrics become percentages only in labeled tables. Empty accepted sets have undefined retained risk. Single-class detection metrics and constant-score discriminatory AURC remain undefined. Tolerances are illustrative. U-bend failed-simulation sentinel codes are never regression references; solver-iteration outputs are never input features. Full column names are preserved in CSV headers. Each system's limits are stated in its protocol and data/results guide.

The manuscript gives a compact U-bend summary. `supplementary_ubend.pdf` contains the complete protocol, tables and figures. All per-case records remain in this archive; the main article does not need to reproduce those long tables.
