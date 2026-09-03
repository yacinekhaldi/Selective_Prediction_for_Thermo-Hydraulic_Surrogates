# Prediction Export Summary

## Source files
- Dataset: `02_processed_data\ML_dataset_longform.csv`
- Split: `03_splits\random80_20_split.csv`
- Nu model: `04_models\random80_20\RF_Nuavg_model.pkl`
- DelP model: `04_models\random80_20\RF_DelP_model.pkl`

## Row counts
- All rows: **4608**
- Train rows: **3686**
- Test rows: **922**

## Exported columns
- `case_uid`
- `raw_row_id`
- `geometry_id`
- `split`
- `Re`
- `Pr`
- `Da`
- `epsi`
- `Hp_mm`
- `a_mm`
- `Lw_mm`
- `Nuavg`
- `Nuavg_pred_RF`
- `Nuavg_residual`
- `Nuavg_abs_error`
- `Nuavg_pct_error`
- `DelP_Pa`
- `DelP_pred_RF`
- `DelP_residual`
- `DelP_abs_error`
- `DelP_pct_error`

## Files generated
- `train_predictions.csv`
- `train_predictions.xlsx`
- `test_predictions_full.csv`
- `test_predictions_full.xlsx`
- `all_predictions_with_split.csv`
- `all_predictions_with_split.xlsx`