# Trend Consistency Export Summary

## Fixed common parameters
- `Pr` = 20
- `epsi` = 0.8
- `Hp_mm` = 0.2
- `a_mm` = 0.2
- `Lw_mm` = 5.0

- Re sweep fixed Da = 0.0001
- Da sweep fixed Re = 250

## Re sweep data
 Re  Pr     Da  epsi  Hp_mm  a_mm  Lw_mm  Nuavg  Nuavg_pred_RF    DelP_Pa  DelP_pred_RF
 25  20 0.0001   0.8    0.2   0.2    5.0 10.723      10.722276   7.965503      7.964000
100  20 0.0001   0.8    0.2   0.2    5.0 15.847      15.862461  34.963696     34.950084
250  20 0.0001   0.8    0.2   0.2    5.0 20.841      20.828147 104.912881    104.812854
500  20 0.0001   0.8    0.2   0.2    5.0 25.872      25.901108 267.339190    267.135778

## Da sweep data
 Re  Pr       Da  epsi  Hp_mm  a_mm  Lw_mm  Nuavg  Nuavg_pred_RF    DelP_Pa  DelP_pred_RF
250  20 0.000001   0.8    0.2   0.2    5.0 20.659      20.692125 866.752230    867.369127
250  20 0.000010   0.8    0.2   0.2    5.0 20.785      20.784777 189.900716    189.821804
250  20 0.000100   0.8    0.2   0.2    5.0 20.841      20.828147 104.912881    104.812854
250  20 0.001000   0.8    0.2   0.2    5.0 20.740      20.752378  80.716712     80.706969

## Files generated
- `trend_vs_Re.csv`
- `trend_vs_Re.xlsx`
- `trend_vs_Da.csv`
- `trend_vs_Da.xlsx`
- `trend_consistency_summary.md`