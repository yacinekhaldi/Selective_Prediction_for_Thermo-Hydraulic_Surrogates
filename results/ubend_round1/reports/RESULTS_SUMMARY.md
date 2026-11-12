# U-bend independent-system evaluation

All 468 planned experiments are included: six models, thirteen designs, two physical-unit targets, three complete partition/training seeds. Models are fitted and calibrated separately for this system.

## Design-random primary results

| Model | Target | All MAPE % | Retained MAPE % | Coverage % | Retained failure % |
|---|---|---:|---:|---:|---:|
| CQR | Cooling objective | 3.494 | 3.284 | 89.0 | 20.0 |
| ET-bag | Cooling objective | 4.974 | 4.805 | 89.4 | 37.6 |
| ET-seed | Cooling objective | 4.865 | 4.713 | 89.1 | 36.9 |
| GP-ARD | Cooling objective | 2.325 | 2.204 | 89.9 | 7.5 |
| Log-poly | Cooling objective | 2.554 | 2.480 | 90.4 | 10.6 |
| XGB | Cooling objective | 3.489 | 3.360 | 89.5 | 21.4 |
| CQR | Pressure drop | 10.825 | 10.108 | 90.1 | 39.9 |
| ET-bag | Pressure drop | 14.464 | 13.976 | 89.7 | 55.6 |
| ET-seed | Pressure drop | 13.769 | 13.327 | 89.6 | 53.2 |
| GP-ARD | Pressure drop | 7.583 | 7.292 | 89.8 | 25.9 |
| Log-poly | Pressure drop | 10.481 | 10.173 | 90.4 | 40.5 |
| XGB | Pressure drop | 10.333 | 9.750 | 89.2 | 38.8 |

## Comparative findings and limits

Across 132 nonempty shifted model/target/fold mean results, calibrated U+D max reduces retained MAPE relative to all-case MAPE in 131; 0 shifted combinations have zero mean coverage. This descriptive count compares different accepted populations and is not a guarantee or significance test.
At approximately 90% expected coverage, max fusion is better than both U-only and D-only in 28 of 156 averaged comparisons, worse than the stronger component in 101, and tied in the remainder. Expected random tie breaking need not be attainable by a deterministic cutoff. Complete per-run results remain available.
GP-ARD cooling objective on coordinate shifts: coverage ranges from 27.9% to 95.0%; retained MAPE among nonempty sets ranges from 2.236% to 3.281%. Read these jointly with all-case error and retained failure, not as a statement of universal shifted reliability.
GP-ARD pressure drop on coordinate shifts: coverage ranges from 32.5% to 93.7%; retained MAPE among nonempty sets ranges from 6.945% to 11.403%. Read these jointly with all-case error and retained failure, not as a statement of universal shifted reliability.

The second system provides evidence beyond the wavy-channel grid, but it does not add methodological novelty to max fusion. The complete comparisons include both selective-error improvements and remaining failures at measured coverage.

## Protocol and interpretation

- Primary failure tolerances are illustrative: cooling objective 5%, pressure 10%; full sweeps are exported.
- Low/high support tests and middle-slab interpolation tests are on fixed coordinates x0, x6 and x12. They are not Reynolds shifts or transfer to independently defined physical geometry families. Fold boundaries use covariates only, not response errors.
- The supplied split has 8,051 outer-training / 895 test cases. Design-random and structured evaluations use independent fit/tune/scale/calibration roles. All models share assignments.
- Geometry coordinates are signed and unlogged; fit-only standardization is used. Pressure is modeled using log1p except the log-polynomial log(y) comparator. GP pressure points are inverse-log medians; ensembles use physical-unit member means.
- CQR intervals have no shifted selective-risk guarantee. One row per design makes row/group coverage identical; Sobol sampling does not establish exchangeable random groups.
- Positivity and expanded-fit-range checks are descriptive diagnostics. No unsupported monotonic response law is imposed on geometry coordinates.
- Bootstrap intervals condition on the fitted model and observed successful designs. Six covariate-only KMeans strata and coordinate bins are coarse descriptive blocks, not physical simulation batches or confidence intervals for unseen families.
- Failed mesh/solver runs remain excluded from regression; sentinel values are not reference pressures. This study does not evaluate failure detection on those runs.
- Calibration-fixed selective results and expected matched-coverage rankings answer different questions. Empty accepted sets have undefined risk; single-class detection metrics and constant-score discriminatory AURC are undefined.
- Full-outer-training point baselines have a different budget from selective models. Same-fit single/ensemble and outer-training controls expose that distinction.

## Export layout

- `runs/`: complete metadata, per-case predictions, assignments, calibration scores, per-run metrics, curves, matched coverage, sensitivity; CQR interval endpoints.
- `tables/`: measured CSV tables and article-ready LaTeX fragments.
- `figures/`: vector PDF and high-resolution PNG plots.
- `plot_data/`: exact records/aggregates used in every figure.
- `predictions/point_baselines.csv`: point-benchmark predictions; selective predictions remain in run folders.
- `models/`: saved seed-42/supplied-split point baseline estimators and settings; selective checkpoint estimators can be reproduced from code/settings and are not serialized.
- `reports/`: this assessment, protocol and figure/table PDFs.

Reproduction from the project root: `python scripts/test_ubend.py`; `python scripts/run_ubend_study.py`; `python scripts/ubend_controls.py`; `python scripts/ubend_diagnostics.py`; `python scripts/validate_ubend_outputs.py`; `python scripts/ubend_exports.py`.

## Data provenance

Source: JDecke/ubend-cfd, updated tabular release, pinned revision 838b96a5c7e72eaf1177f8b8b49c854b80f20592.
Original dataset article: Decke, Wuensch and Sick, Data in Brief 50 (2023), 109477; DOI 10.1016/j.dib.2023.109477.
The updated release declares CC-BY-NC-4.0. Source documentation and hashes are preserved under data/raw/ubend_cfd. Results concern 8,946 successful simulations, conditional on exclusion of failed mesh/solver runs.
Models are fitted and calibrated separately on U-bend. This is replication of the evaluation across systems, not transfer of the fitted wavy-channel predictor.
Reproduce with requirements-ubend.txt. PDF exports additionally require pdflatex, booktabs, longtable, graphicx, hyperref and lmodern.
