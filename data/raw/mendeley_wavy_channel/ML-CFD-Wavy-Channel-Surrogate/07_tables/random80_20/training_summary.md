# Training Summary

## Baseline protocol
- Protocol: fixed random 80/20 split
- Source dataset: `02_processed_data\ML_dataset_longform.csv`
- Split file: `03_splits\random80_20_split.csv`

## Shapes
- Train rows: **3686**
- Test rows: **922**
- Number of input features: **7**

## Features
- `Re`
- `Pr`
- `Da`
- `epsi`
- `Hp_mm`
- `a_mm`
- `Lw_mm`

## Targets
- `Nuavg`
- `DelP_Pa`

## Random Forest hyperparameters
- n_estimators = 800
- min_samples_leaf = 2
- random_state = 42
- n_jobs = -1

## Test metrics
       model  output       R2      MAE     RMSE
RandomForest   Nuavg 0.999143 0.083819 0.276695
RandomForest DelP_Pa 0.999991 0.639853 1.997940

## Files generated
- `04_models\random80_20\RF_Nuavg_model.pkl`
- `04_models\random80_20\RF_DelP_model.pkl`
- `05_predictions\random80_20\test_predictions.csv`
- `05_predictions\random80_20\test_predictions.xlsx`
- `07_tables\random80_20\test_metrics.csv`
- `07_tables\random80_20\test_metrics.xlsx`