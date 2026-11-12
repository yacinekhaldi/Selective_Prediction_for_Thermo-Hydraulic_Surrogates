# Reproducible scientific workflows

Run commands from the repository root with Python 3.14 and `requirements.txt`. `README.md` lists the complete run order. `wavy_channel_study.py` provides stages for the primary system; the U-bend scripts provide the independent system. Scientific scripts resolve data/output paths relative to their checkout, without depending on the author's manuscript directory.

The wavy-channel model schema has seven inputs and two targets. U-bend has 28 signed geometry inputs, pressure loss and a cooling objective. The model schemas must remain separate. No standardizer is fitted on the complete dataset, and solver-iteration outputs never serve as input features. Model/hyperparameter selection uses tune labels only; acceptance cutoffs use independent calibration scores.

Each completed run contains assignments, predictions, score metrics, calibration/cutoff diagnostics, matched coverage and completion metadata. U-bend runs additionally preserve calibration scores; CQR endpoints are supplied. Different seeds repeat the complete fitting/partition workflow; they are not independent physical systems. The code does not count ensemble members as independent repeats.

Validate the delivered outputs with `python scripts/wavy_channel_study.py --stage validate` and `python scripts/validate_ubend_outputs.py --with-exports`. Numerical transform/replay checks are in the corresponding diagnostic files. The U-bend validator checks saved point-model predictions against their serialized artifacts. Full scientific outputs are delivered; no numerical result needs to be inferred from a manuscript image.

Wavy-channel plots/tables are regenerated with `python scripts/wavy_channel_exports.py`. U-bend exports use `python scripts/ubend_exports.py`; its CSV/LaTeX tables, vector PDF/PNG figures, plot-data CSVs and PDF reports remain together under the result folder. Partial checkpoints must not be represented as a complete study. If scientific settings change, use a separate output location or explicitly regenerate affected checkpoints rather than mixing settings.
