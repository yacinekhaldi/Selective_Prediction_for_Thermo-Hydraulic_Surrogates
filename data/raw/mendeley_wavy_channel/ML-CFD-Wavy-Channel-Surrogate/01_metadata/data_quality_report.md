# Data Quality Report

## Master dataset
- Source file: `02_processed_data\ML_dataset_longform.csv`
- Shape: **4608 rows × 12 columns**
- Total missing values: **0**
- Duplicate `case_uid` count: **0**
- Total unique operating combinations: **768**

## Expected design checks
- Expected total rows: **4608**
- Expected total columns: **12**
- Expected rows per geometry: **768**
- Expected count per Re level: **1152**
- Expected count per Pr level: **1152**
- Expected count per Da level: **1152**
- Expected count per epsi level: **1152**
- Expected count per Hp_mm level: **1536**
- Expected unique operating combinations total: **768**
- Expected unique operating combinations per geometry: **768**

## Frequency summary

### geometry_id
| geometry_id   |   count |
|:--------------|--------:|
| G01           |     768 |
| G02           |     768 |
| G03           |     768 |
| G04           |     768 |
| G05           |     768 |
| G06           |     768 |

### Re
|   Re |   count |
|-----:|--------:|
|   25 |    1152 |
|  100 |    1152 |
|  250 |    1152 |
|  500 |    1152 |

### Pr
|   Pr |   count |
|-----:|--------:|
|    3 |    1152 |
|    6 |    1152 |
|   20 |    1152 |
|   50 |    1152 |

### Da
|     Da |   count |
|-------:|--------:|
| 1e-06  |    1152 |
| 1e-05  |    1152 |
| 0.0001 |    1152 |
| 0.001  |    1152 |

### epsi
|   epsi |   count |
|-------:|--------:|
|   0.7  |    1152 |
|   0.75 |    1152 |
|   0.8  |    1152 |
|   0.85 |    1152 |

### Hp_mm
|   Hp_mm |   count |
|--------:|--------:|
|     0.1 |    1536 |
|     0.2 |    1536 |
|     0.3 |    1536 |

### a_mm
|   a_mm |   count |
|-------:|--------:|
|    0   |     768 |
|    0.1 |    1536 |
|    0.2 |    1536 |
|    0.3 |     768 |

### Lw_mm
|   Lw_mm |   count |
|--------:|--------:|
|       0 |     768 |
|       4 |    2304 |
|       5 |    1536 |

## Per-geometry unique operating combinations
| geometry_id   |   unique_operating_combinations |
|:--------------|--------------------------------:|
| G01           |                             768 |
| G02           |                             768 |
| G03           |                             768 |
| G04           |                             768 |
| G05           |                             768 |
| G06           |                             768 |

## Files generated
- `data_quality_report.md`
- `dataset_frequency_tables.xlsx`