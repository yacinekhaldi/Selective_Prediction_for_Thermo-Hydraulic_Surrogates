# Raw Data Dictionary

## Raw files
- Nuavg.xlsx
- DelP.xlsx

## Rows used
- Rows 1-768 only

## Common input columns in both files
| Excel Column | Variable | Pattern |
|---|---|---|
| 1 | Re | 192-row blocks: 25, 100, 250, 500 |
| 2 | Pr | 48-row blocks: 3, 6, 20, 50 |
| 3 | Da | 12-row blocks: 1e-3, 1e-4, 1e-5, 1e-6 |
| 4 | epsi | 3-row blocks: 0.85, 0.80, 0.75, 0.70 |
| 5 | Hp_mm | row-wise cycle: 0.1, 0.2, 0.3, repeat |

## Parameter nesting order
Slowest to fastest:
Re -> Pr -> Da -> epsi -> Hp_mm

## Nuavg.xlsx output columns
| Excel Column | Output | a_mm | Lw_mm |
|---|---|---:|---:|
| 6 | Nuavg | 0.0 | 0.0 |
| 7 | Nuavg | 0.1 | 4.0 |
| 8 | Nuavg | 0.1 | 5.0 |
| 9 | Nuavg | 0.2 | 4.0 |
| 10 | Nuavg | 0.2 | 5.0 |
| 11 | Nuavg | 0.3 | 4.0 |

## DelP.xlsx output columns
| Excel Column | Output | a_mm | Lw_mm |
|---|---|---:|---:|
| 8 | DelP_Pa | 0.0 | 0.0 |
| 11 | DelP_Pa | 0.1 | 4.0 |
| 14 | DelP_Pa | 0.1 | 5.0 |
| 17 | DelP_Pa | 0.2 | 4.0 |
| 20 | DelP_Pa | 0.2 | 5.0 |
| 23 | DelP_Pa | 0.3 | 4.0 |

## Notes
- Columns 1-5 are identical in both raw files.
- Raw rows represent the full factorial operating design space.
- Geometry/output columns are mapped separately for Nuavg and DelP.
- Lw_mm = 0.0 is used for the straight-channel case where wavelength is not applicable.
