# Reproducibility Guide

## 1. Prepare the environment

From the repository root, create an isolated Python environment and install
the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The generated environment report records the package versions used for the
committed result snapshot. The XGBoost experiment supports a CPU mode and a
CUDA-enabled mode.

## 2. Confirm the input files

The main scripts resolve paths relative to the repository root. Confirm that
these files exist before running:

- `data/raw/mendeley_wavy_channel/ML-CFD-Wavy-Channel-Surrogate/02_processed_data/ML_dataset_longform.csv`
- `data/raw/mendeley_wavy_channel/ML-CFD-Wavy-Channel-Surrogate/03_splits/random80_20_split.csv`
- `data/raw/mendeley_wavy_channel/ML-CFD-Wavy-Channel-Surrogate/03_splits/withheld_re_folds.csv`
- `data/raw/mendeley_wavy_channel/ML-CFD-Wavy-Channel-Surrogate/03_splits/logo_geometry_folds.csv`

The scripts also maintain a normalized copy at
`data/processed/thermal_wavy_channel_longform.csv`.

## 3. Run the workflows

Run the benchmark and primary selective workflow:

```powershell
python scripts/run_experiments.py --cpu
```

Run the companion selective ExtraTrees workflow:

```powershell
python scripts/run_extratrees_selective.py
```

Generate the reviewer diagnostics and publication exports:

```powershell
python scripts/generate_reviewer_tables.py
```

For GPU execution, remove `--cpu` from the first command and verify the
XGBoost/CUDA combination in the environment report.

## 4. Expected outputs

The workflows write to `results/`:

- `models/` and `extratrees_selective/models/`: fitted model ensembles.
- `predictions/` and `extratrees_selective/predictions/`: case-level outputs.
- `tables/` and `extratrees_selective/tables/`: metrics and diagnostics.
- `figures/` and `extratrees_selective/figures/`: publication graphics.
- `plot_data/` and `extratrees_selective/plot_data/`: score and curve data.
- `reports/` and `extratrees_selective/reports/`: summaries and manifests.
- `submission_tables/` and `submission_figures/`: reviewer-table exports.

The `artifact_manifest.csv` and `artifact_manifest.json` files provide output
paths, sizes, and SHA-256 checksums for the tracked result snapshot.

## 5. Reproducibility notes

The workflow uses fixed seeds and fixed split definitions. Exact floating-point
results can still vary with Python, scikit-learn, XGBoost, BLAS, CPU, and GPU
versions. Treat the committed result files as the reference snapshot for the
manuscript.
