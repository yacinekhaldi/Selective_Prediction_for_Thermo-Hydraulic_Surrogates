README

Dataset title:
CFD-informed Random Forest surrogate dataset for thermo-hydraulic prediction in partially porous wavy channels

Associated manuscript:
CFD-Informed Machine Learning Surrogate Modeling for Thermo-Hydraulic Prediction in Partially Porous Wavy Channels for Heat Sink Applications

Description:
This repository supports a CFD-informed machine-learning surrogate modeling study for predicting thermo-hydraulic performance in partially porous sinusoidal wavy channels. The cleaned long-form CFD-derived dataset was used to train and validate Random Forest regression models for predicting the average Nusselt number and pressure drop.

The repository contains the cleaned dataset, fixed train-test split information, trained-model outputs, prediction files, validation tables, plot data, and one final reproducibility script. Earlier exploratory notebooks, draft plotting files, old data folders, raw headerless spreadsheets, and intermediate working files were excluded to improve clarity and avoid duplication.

Dataset summary:
- Total CFD-derived samples: 4,608
- Geometry configurations: 18
- Operating-condition combinations: 256
- Geometry variables: wave amplitude, porous slab thickness, and wavelength
- Operating variables: Reynolds number, Prandtl number, Darcy number, and porosity
- Target outputs: average Nusselt number and pressure drop

Main input features:
- Re: Reynolds number
- Pr: Prandtl number
- Da: Darcy number
- epsi: porosity of porous slab
- Hp_mm: porous slab thickness in mm
- a_mm: wave amplitude in mm
- Lw_mm: wavelength in mm

Target variables:
- Nuavg: average Nusselt number
- DelP_Pa: pressure drop in Pa

Repository folder structure:
01_metadata/
    Contains dataset descriptions, feature definitions, parameter ranges, or related metadata.

02_processed_data/
    Contains the cleaned long-form CFD-derived dataset used for machine-learning training and validation.
    Expected main file:
    - ML_dataset_longform.csv

03_splits/
    Contains the fixed random 80/20 train-test split and any blocked-validation split definitions, if included.
    Expected main file:
    - random80_20_split.csv

04_models/
    Contains trained Random Forest model files generated from the reproducibility script, if included.

05_predictions/
    Contains prediction files used to generate parity plots, residual plots, and validation results.

06_scripts/
    Contains the final reproducibility script:
    - 01_train_RF_models_random80_20.py

07_tables/
    Contains model-performance metrics, validation-summary tables, and other tabulated results.

08_plot_data/
    Contains data used to generate manuscript and supplementary figures.

Reproducibility script:
06_scripts/01_train_RF_models_random80_20.py

This script trains Random Forest surrogate models using the cleaned long-form dataset and the fixed random 80/20 split. It uses the following input features:
Re, Pr, Da, epsi, Hp_mm, a_mm, and Lw_mm.

The target outputs are:
Nuavg and DelP_Pa.

The script generates:
- 04_models/random80_20/RF_Nuavg_model.pkl
- 04_models/random80_20/RF_DelP_model.pkl
- 05_predictions/random80_20/test_predictions.csv
- 05_predictions/random80_20/test_predictions.xlsx
- 07_tables/random80_20/test_metrics.csv
- 07_tables/random80_20/test_metrics.xlsx
- 07_tables/random80_20/training_summary.md

Software requirements:
The script was prepared using Python and common scientific-computing libraries. To reproduce the workflow, the following packages are required:

- Python 3.x
- numpy
- pandas
- scikit-learn
- matplotlib
- joblib
- openpyxl

Example installation command:
pip install numpy pandas scikit-learn matplotlib joblib openpyxl

How to reproduce the random 80/20 Random Forest training:
1. Download the complete dataset folder from Mendeley Data.
2. Keep the folder structure unchanged.
3. Open a terminal or command prompt in the top-level repository folder.
4. Run:
   python 06_scripts/01_train_RF_models_random80_20.py
5. The trained models, prediction files, test metrics, and training summary will be saved automatically in the corresponding output folders.

Notes on raw CFD files:
The original CFD output files were generated as separate headerless spreadsheet files for average Nusselt number and pressure drop. For clarity and reproducibility, this repository provides the cleaned long-form dataset, where each row corresponds to one CFD-derived geometry-operating-condition case and includes both target outputs. The processed dataset contains the numerical information required for the machine-learning workflow.

Validation:
The associated manuscript reports random train-test validation, leave-one-geometry-out validation, withheld-Reynolds-number validation, and physics-consistency checks. Corresponding prediction files, metric tables, and plot data are included where applicable. The reproducibility script included here focuses on the fixed random 80/20 Random Forest training protocol.

Citation:
If using this dataset, please cite the associated manuscript and this Mendeley Data repository.

Contact:
For questions about the dataset or additional raw CFD simulation files, please contact the corresponding author.
