# Experiment Summary

## Dataset
- Rows: 4608
- Geometry IDs: G01, G02, G03, G04, G05, G06
- Features: Re, Pr, Da, epsi, Hp_mm, a_mm, Lw_mm
- Targets: Nuavg, DelP_Pa

## Compute
- XGBoost GPU enabled: True
- XGBoost note: xgboost CUDA probe trained successfully
- Torch CUDA available: False

## Best in-distribution benchmark
- DelP_Pa: ExtraTrees with MAPE=0.039%, RMSE=1.34, R2=1.0000
- Nuavg: ExtraTrees with MAPE=0.200%, RMSE=0.1914, R2=0.9996

## Primary reliability gate highlights
- Nuavg, random80_20: all-case MAPE=0.515%, predict=716 cases (MAPE=0.398%), warn=73 cases (MAPE=0.724%), abstain=133 cases (MAPE=1.033%), predict+warn coverage=0.856
- Nuavg, holdout_Re500: all-case MAPE=22.256%, predict=0 cases (MAPE=--%), warn=186 cases (MAPE=18.485%), abstain=966 cases (MAPE=22.982%), predict+warn coverage=0.161
- Nuavg, combined_G06_Re500: all-case MAPE=29.005%, predict=0 cases (MAPE=--%), warn=0 cases (MAPE=--%), abstain=192 cases (MAPE=29.005%), predict+warn coverage=0.000
- DelP_Pa, random80_20: all-case MAPE=0.431%, predict=718 cases (MAPE=0.328%), warn=78 cases (MAPE=0.437%), abstain=126 cases (MAPE=1.019%), predict+warn coverage=0.863
- DelP_Pa, holdout_Re500: all-case MAPE=58.371%, predict=0 cases (MAPE=--%), warn=187 cases (MAPE=58.266%), abstain=965 cases (MAPE=58.391%), predict+warn coverage=0.162
- DelP_Pa, combined_G06_Re500: all-case MAPE=68.184%, predict=0 cases (MAPE=--%), warn=0 cases (MAPE=--%), abstain=192 cases (MAPE=68.184%), predict+warn coverage=0.000

## Exported artifacts
- Tables: `results/tables`
- Predictions: `results/predictions`
- Figures: `results/figures`
- Models: `results/models`
- Reports: `results/reports`
