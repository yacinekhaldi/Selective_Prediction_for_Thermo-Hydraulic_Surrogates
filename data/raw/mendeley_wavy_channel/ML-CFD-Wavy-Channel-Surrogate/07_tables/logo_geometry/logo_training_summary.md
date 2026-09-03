# LOGO Training Summary

## Source files
- Dataset: `02_processed_data\ML_dataset_longform.csv`
- LOGO fold file: `03_splits\logo_geometry_folds.csv`

## Model setup
- Model: RandomForestRegressor
- n_estimators = 800
- min_samples_leaf = 2
- random_state = 42
- n_jobs = -1

## Fold names
- `holdout_G01`
- `holdout_G02`
- `holdout_G03`
- `holdout_G04`
- `holdout_G05`
- `holdout_G06`

## Fold-wise metrics
protocol   fold_name  output       R2       MAE      RMSE  n_test
    LOGO holdout_G01   Nuavg 0.990710  0.606878  0.881417     768
    LOGO holdout_G01 DelP_Pa 0.999622  7.778364 12.353902     768
    LOGO holdout_G02   Nuavg 0.994441  0.231210  0.637815     768
    LOGO holdout_G02 DelP_Pa 0.999994  0.811320  1.559319     768
    LOGO holdout_G03   Nuavg 0.993429  0.296307  0.708563     768
    LOGO holdout_G03 DelP_Pa 0.999997  0.641302  1.110001     768
    LOGO holdout_G04   Nuavg 0.977773  0.640747  1.275562     768
    LOGO holdout_G04 DelP_Pa 0.999926  3.237918  5.699851     768
    LOGO holdout_G05   Nuavg 0.981978  0.580236  1.097326     768
    LOGO holdout_G05 DelP_Pa 0.999924  3.295831  5.762076     768
    LOGO holdout_G06   Nuavg 0.935189  1.352805  2.735325     768
    LOGO holdout_G06 DelP_Pa 0.985742 51.807239 84.744178     768

## Summary metrics
protocol  output  R2_mean   R2_std  MAE_mean   MAE_std  RMSE_mean  RMSE_std  total_test_cases  n_folds
    LOGO DelP_Pa 0.997534 0.005779 11.261996 20.029547  18.538221 32.685024              4608        6
    LOGO   Nuavg 0.978920 0.022421  0.618031  0.398773   1.222668  0.778389              4608        6

## Files generated
- `logo_pooled_predictions.csv`
- `logo_pooled_predictions.xlsx`
- `logo_fold_metrics.csv`
- `logo_fold_metrics.xlsx`
- `logo_summary_metrics.csv`
- `logo_summary_metrics.xlsx`
- `logo_training_summary.md`