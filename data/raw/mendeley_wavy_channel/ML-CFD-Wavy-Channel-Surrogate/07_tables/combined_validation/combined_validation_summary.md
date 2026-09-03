# Combined Validation Summary

## Metrics table
              protocol  output   R2_mean    R2_std   MAE_mean     MAE_std  RMSE_mean    RMSE_std total_test_cases  n_folds
    Random 80/20 split   Nuavg  0.999143             0.083819               0.276695                                     1
    Random 80/20 split DelP_Pa  0.999991             0.639853               1.997940                                     1
Leave-one-geometry-out DelP_Pa  0.997534  0.005779  11.261996   20.029547  18.538221   32.685024             4608        6
Leave-one-geometry-out   Nuavg  0.978920  0.022421   0.618031    0.398773   1.222668    0.778389             4608        6
    Withheld Re regime DelP_Pa -2.907005  6.575759 213.671943  160.250922 359.043537  253.000608             4608        4
    Withheld Re regime   Nuavg -0.409153  1.465698   4.889792    0.802114   5.609038    1.043991             4608        4

## Files generated
- `combined_validation_summary.csv`
- `combined_validation_summary.xlsx`
- `combined_validation_summary.md`