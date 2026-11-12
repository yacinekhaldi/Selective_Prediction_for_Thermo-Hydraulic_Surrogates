# U-bend CFD — local raw dataset

Source: https://huggingface.co/datasets/JDecke/ubend-cfd
Pinned revision: `838b96a5c7e72eaf1177f8b8b49c854b80f20592`
Original article: https://doi.org/10.1016/j.dib.2023.109477
License declared by this updated dataset release: **CC-BY-NC-4.0**. See the preserved upstream `README.md` for attribution and version details.

The original downloaded bytes are preserved under `params/`: `data.parquet`, `train.parquet`, `test.parquet`, and `failed.parquet`. The full CFD fields/animations are not needed for tabular surrogate evaluation and are not downloaded.

There are 28 geometry input parameters (`input_0` through `input_27`) and two main outputs: pressure loss (`target_0`) and a cooling-performance objective (`target_1`, integrated squared temperature deviation, not Nusselt number). Solver-iteration target means and standard deviations are auxiliary outputs, not geometry input features.

Failed runs use pressure-target sentinel values 10000 (mesh failure) and 11000 (solver divergence). Preserve these failures separately; do not treat sentinels as physical regression targets. Check sample IDs and overlap before using `data.parquet` alongside the supplied splits: the aggregate and split files are not independent datasets.

`download_manifest.json` records exact source URLs, revision, file sizes and SHA-256 checksums. `validation.json`, when present, records actual local schema and row counts. No model experiments have been run on this dataset yet.

To download this same pinned revision again from the project root: `python scripts/download_ubend.py`.

## Verified local counts

Aggregate: 10,000 cases. Supplied train: 8,051; test: 895; failed: 1,054. The 8,946 successful cases and failed cases partition the aggregate without overlapping IDs. Split values exactly match the aggregate by sample ID. All 28 inputs and both main targets are finite.
