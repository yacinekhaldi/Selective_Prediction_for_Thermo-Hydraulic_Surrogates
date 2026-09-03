# Split Summary

## Source dataset
- File: `02_processed_data\ML_dataset_longform.csv`
- Total cases: **4608**

## Random 80/20 split
- Train cases: **3686**
- Test cases: **922**

## Leave-one-geometry-out folds
  logo_fold  test  train
holdout_G01   768   3840
holdout_G02   768   3840
holdout_G03   768   3840
holdout_G04   768   3840
holdout_G05   768   3840
holdout_G06   768   3840

## Withheld-Re folds
    fold_name  test  train
holdout_Re100  1152   3456
 holdout_Re25  1152   3456
holdout_Re250  1152   3456
holdout_Re500  1152   3456

## Files generated
- `random80_20_split.csv`
- `logo_geometry_folds.csv`
- `withheld_re_folds.csv`
- `split_summary.md`