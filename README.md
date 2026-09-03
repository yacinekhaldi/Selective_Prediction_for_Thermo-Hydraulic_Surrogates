# Selective Prediction for Thermo-Hydraulic Surrogates

This repository contains the code, datasets, trained models, predictions,
figures, tables, and reports for the selective-prediction study of a
partially porous wavy-channel heat-sink surrogate.

The repository is associated with:

`https://github.com/yacinekhaldi/Selective_Prediction_for_Thermo-Hydraulic_Surrogates`

## Repository map

- `data/`: processed data and the complete archived dataset release.
- `scripts/`: experiment, ExtraTrees, and reviewer-table generation scripts.
- `src/`: Python package namespace reserved for reusable project code.
- `results/models/`: benchmark and XGBoost model artifacts.
- `results/extratrees_selective/`: ExtraTrees selective-prediction artifacts.
- `results/predictions/`: case-level predictions and gate decisions.
- `results/tables/`: CSV, XLSX, and LaTeX result tables.
- `results/figures/`: PNG and PDF publication figures.
- `results/plot_data/`: calibration and risk-coverage data.
- `results/reports/`: experiment summaries, profiles, manifests, and environment reports.
- `docs/`: reproducibility, data/results, and GitHub upload guides.

## Requirements

Use Python 3.10 or newer. Install the scientific Python dependencies with:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The XGBoost workflow can use a CUDA-enabled installation when available. The
CPU path is supported for machines without a compatible GPU.

## Reproduce the experiments

Run these commands from the repository root:

```powershell
python scripts/run_experiments.py --cpu
python scripts/run_extratrees_selective.py
python scripts/generate_reviewer_tables.py
```

The first command regenerates the benchmark and XGBoost results. The second
regenerates the ExtraTrees selective results. The third creates the detailed
reviewer diagnostics and writes submission-ready copies under
`results/submission_tables/` and `results/submission_figures/`.

To use the XGBoost GPU path, omit `--cpu` and install a compatible XGBoost
build and CUDA runtime.

## Dataset

The primary cleaned dataset is:

`data/raw/mendeley_wavy_channel/ML-CFD-Wavy-Channel-Surrogate/02_processed_data/ML_dataset_longform.csv`

The fixed split and structured split definitions are under the corresponding
`03_splits/` directory. The original archived dataset is retained under
`data/raw/CFD-informed machine learning surrogate dataset fo/` for provenance.
See `data/raw/mendeley_wavy_channel/ML-CFD-Wavy-Channel-Surrogate/README.txt`
for the source dataset description and citation information.

## Existing results

The committed `results/` tree is the generated result snapshot used for the
manuscript. It includes raw case-level outputs, fitted models, figures, tables,
calibration data, risk-coverage data, bootstrap summaries, and environment
metadata. Re-running the scripts may update these files as software or hardware
settings change.

## Guides

- `docs/reproducibility_guide.md`: complete run order and expected outputs.
- `docs/data_and_results_guide.md`: dataset provenance and result-file map.
- `docs/github_upload_guide.md`: Git and Git LFS upload instructions.

## Citation

Use `CITATION.cff` for the project citation metadata. Cite the associated
manuscript and the Mendeley Data record when reusing the dataset.

## Large files

The archived dataset and trained model artifacts are large. `.gitattributes`
configures Git LFS for ZIP, PKL, JOBLIB, and XLSX files. Install Git LFS before
the first commit; the upload guide explains the required commands.
