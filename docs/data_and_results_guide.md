# Data and Results Guide

## Dataset provenance

The project uses the CFD-informed partially porous wavy-channel dataset
distributed through Mendeley Data. The extracted source release is retained in
`data/raw/mendeley_wavy_channel/ML-CFD-Wavy-Channel-Surrogate/`, together with
its metadata, split definitions, source models, predictions, tables, and plot
data. The cleaned long-form file used by the main workflow is in
`02_processed_data/ML_dataset_longform.csv`.

The dataset contains 4,608 CFD-derived cases, 18 geometry configurations, and
256 operating-condition combinations. Inputs include Reynolds number, Prandtl
number, Darcy number, porosity, porous-slab thickness, wave amplitude, and
wavelength. Targets are average Nusselt number and pressure drop.

## Result directories

### Models

`results/models/` contains benchmark and XGBoost artifacts. The
`results/extratrees_selective/models/` directory contains the ExtraTrees
selective ensembles. These files are binary artifacts and are configured for
Git LFS.

### Predictions

Prediction files contain input identifiers, reference values, ensemble outputs,
errors, reliability components, thresholds, and gate decisions. CSV is the
portable format; XLSX and LaTeX files are presentation-oriented exports.

### Tables

The tables directory includes benchmark accuracy, calibration, split summaries,
per-fold structured results, threshold clipping, component gate drivers,
risk-coverage/AURC, false accepts, bootstrap intervals, and signed-error
diagnostics.

### Figures and plot data

PNG and PDF figures are stored together. The matching plot-data files contain
the numerical values used to draw calibration, reliability-score, and
risk-coverage figures.

### Reports

Reports summarize the dataset, environment, generated artifacts, and manuscript
results. They are useful for checking that an experiment run produced the
expected files before preparing a release.

## Interpretation cautions

The dataset is structured rather than an independent random sample. Case counts
should therefore be interpreted together with the split definition and the
bootstrap or per-fold summaries. Warning decisions are cautionary outputs and
do not validate a withheld operating regime. Abstention identifies cases that
should receive additional CFD or experimental validation before design use.
