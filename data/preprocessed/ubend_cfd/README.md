# Preprocessed U-bend CFD data

This follows the original project's long-form CSV preparation: numeric inputs and physical-unit targets, stable case/row identifiers, and a separate fixed split file. Raw source files remain in `../../raw/ubend_cfd/`.

- `ubend_longform.csv`: **8,946 successful simulations**, 28 input columns, pressure loss `DelP_Pa` and `CoolingObjective_K2m2`.
- `random90_10_split.csv`: original supplied assignments, **8,051 train / 895 test**, joined by `case_uid`. No split was regenerated.
- `failed_simulations.csv`: **1,054 excluded unsuccessful runs**, original target sentinel values and failure reasons; preserved for a separate failure-detection study.
- `solver_diagnostics.csv`: last-iteration target mean/std values, excluded from model inputs. Missing auxiliary diagnostics are preserved rather than imputed.
- `numeric_summary.csv`: physical-unit feature/target distributions.
- `preprocessing.json`: feature/target lists, row counts, validation checks, source revision and exact hashes.

The 28 released `input_*` columns are the only model features. IDs, targets, source split, and solver statistics are not features. `geometry_id` identifies an individual design; it is not analogous to the six wavy-channel geometry families. Every retained input vector is unique. No Reynolds/Prandtl/Darcy columns are invented for this fixed-operating-condition dataset.

The cooling objective is the integrated squared temperature difference over the heated surface, not Nusselt number. The scalar source targets `target_0` and `target_1` are retained as the response values; solver-iteration means are auxiliary diagnostics. Both response columns remain in physical units. Signed geometry inputs are not logarithmically transformed. Scaling and any model-specific target transformation must be fitted/applied inside the training workflow, using fit data only. No scaler is fitted on the complete dataset.

All successful cases are retained; no imputation or error-based outlier filtering is performed. Pressure sentinel codes 10000/11000 are kept only in the failed-run file and cannot enter the regression labels. The full aggregate raw file overlaps the provided splits and must not be appended as additional observations.

Source revision: `838b96a5c7e72eaf1177f8b8b49c854b80f20592`. Source and citation/license details are preserved with the raw data. Reproduce from the project root with `python scripts/preprocess_ubend.py` (requires pandas, NumPy and pyarrow). Existing wavy-channel experiment runners still use their seven-feature schema; this preparation does not run or silently change those experiments.
