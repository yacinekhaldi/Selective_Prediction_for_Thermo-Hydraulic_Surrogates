# Selective Prediction for Thermo-Hydraulic Surrogates

This project evaluates binary acceptance and abstention for scalar thermo-hydraulic surrogates. The primary case study uses a partially porous wavy-channel dataset; cross-problem replication uses parameterized U-bend conjugate heat transfer. Models are fitted and calibrated separately for each system.

The evaluation compares uncertainty, distance, fusion, leverage, support-gap, learned-gate and conformal-interval scores on common partitions. Fit, tune, score-scale and threshold-calibration roles are separate. Three complete seeds repeat each six-model, thirteen-design, two-target study. Empirical score calibration does not establish selective-risk control under distribution shift.

## Environment

Use Python 3.14 and the pinned dependencies:

```powershell
python -m pip install -r requirements.txt
```

Vector-figure generation uses Matplotlib. PDF compendiums additionally require `pdflatex` and standard LaTeX packages. Exact execution versions and source hashes are included with each result set.

## Run the studies

```powershell
python scripts/wavy_channel_study.py --stage all
python scripts/download_ubend.py
python scripts/preprocess_ubend.py
python scripts/test_ubend.py
python scripts/run_ubend_study.py
python scripts/ubend_controls.py
python scripts/ubend_diagnostics.py
python scripts/ubend_transform_audit.py
python scripts/audit_predictive_transforms.py
python scripts/ubend_exports.py
python scripts/validate_ubend_outputs.py --with-exports
```

Completed model checkpoints are skipped automatically. The downloader uses the pinned source commit recorded in its manifest. The supplied U-bend tables are already included locally; downloading is unnecessary for reproducing the delivered results. The scientific archive omits large serialized point models; when using that archive, validate numerical outputs with `python scripts/validate_ubend_outputs.py` without `--with-exports`. The prepared repository includes the binaries and supports the full artifact check. To run individual wavy-channel stages, use `--stage tests`, `experiments`, `controls`, `intervals`, `diagnostics`, `exports`, or `validate`.

## Data and output layout

- `data/raw/mendeley_wavy_channel/`: released wavy-channel data and attribution.
- `data/raw/ubend_cfd/`: pinned U-bend Parquet tables, source documentation and checksums.
- `data/preprocessed/ubend_cfd/`: cleaned physical-unit CSV, supplied split, separate failed cases and preprocessing metadata.
- `results/revision_round1/`: wavy-channel measurements, complete runs and diagnostics.
- `results/ubend_round1/`: U-bend measurements, complete runs, controls, tables, figures, plot data, predictions, saved point models and reports.
- `scripts/`: reproducible scientific workflows.
- `docs/`: data definitions and reproduction instructions.

Each study contains 468 model experiments. Together they provide 936 experiments and 1,940,760 stored selective-prediction rows. The U-bend study additionally supplies 60 point benchmarks and 36 fit-budget/tuned-SVR control records. Detailed counts are verified in the validation reports.

## Interpretation and attribution

The U-bend cooling objective is an integrated squared temperature difference, not Nusselt number. Failed simulation codes are not regression targets. Coordinate-support holdouts are distinguished from Reynolds shifts and physical geometry-family transfer. Tolerances are illustrative and their sweeps are supplied. Undefined detection/ranking metrics and empty accepted sets are explicitly flagged. Source licenses and citations are preserved with the raw data; the updated U-bend release declares CC-BY-NC-4.0.

See [the data/results guide](docs/data_and_results_guide.md), [reproduction guide](docs/reproducibility_guide.md), and [scientific supplement guide](SUPPLEMENT_README.md). Saved `.joblib` artifacts use the repository's Git LFS configuration. Cite the manuscript using `CITATION.cff` and cite each dataset's source record when using its data.
