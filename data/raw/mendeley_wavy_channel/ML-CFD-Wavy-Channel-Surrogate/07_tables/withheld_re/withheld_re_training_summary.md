# Withheld-Re Training Summary

## Source files
- Dataset: `02_processed_data\ML_dataset_longform.csv`
- Withheld-Re fold file: `03_splits\withheld_re_folds.csv`

## Model setup
- Model: RandomForestRegressor
- n_estimators = 800
- min_samples_leaf = 2
- random_state = 42
- n_jobs = -1

## Fold names
- `holdout_Re100`
- `holdout_Re25`
- `holdout_Re250`
- `holdout_Re500`

## Fold-wise metrics
   protocol     fold_name  output         R2        MAE       RMSE  n_test
WITHHELD_RE holdout_Re100   Nuavg  -0.037487   4.318321   4.876199    1152
WITHHELD_RE holdout_Re100 DelP_Pa   0.168308  98.084235 174.481574    1152
WITHHELD_RE  holdout_Re25   Nuavg  -2.574230   4.318295   4.876172    1152
WITHHELD_RE  holdout_Re25 DelP_Pa -12.767805  98.085976 174.483455    1152
WITHHELD_RE holdout_Re250   Nuavg   0.413539   4.902803   5.593073    1152
WITHHELD_RE holdout_Re250 DelP_Pa   0.424530 220.631420 376.606299    1152
WITHHELD_RE holdout_Re500   Nuavg   0.561568   6.019749   7.090710    1152
WITHHELD_RE holdout_Re500 DelP_Pa   0.546946 437.886140 710.602822    1152

## Summary metrics
   protocol  output   R2_mean   R2_std   MAE_mean    MAE_std  RMSE_mean   RMSE_std  total_test_cases  n_folds
WITHHELD_RE DelP_Pa -2.907005 6.575759 213.671943 160.250922 359.043537 253.000608              4608        4
WITHHELD_RE   Nuavg -0.409153 1.465698   4.889792   0.802114   5.609038   1.043991              4608        4

## Files generated
- `withheld_re_pooled_predictions.csv`
- `withheld_re_pooled_predictions.xlsx`
- `withheld_re_fold_metrics.csv`
- `withheld_re_fold_metrics.xlsx`
- `withheld_re_summary_metrics.csv`
- `withheld_re_summary_metrics.xlsx`
- `withheld_re_training_summary.md`