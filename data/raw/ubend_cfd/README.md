---
license: cc-by-nc-4.0
task_categories:
  - tabular-regression
tags:
  - cfd
  - openfoam
  - surrogate-modeling
  - scientific-computing
  - scientific-machine-learning
  - physics-informed-neural-networks
  - neural-operator
  - heat-transfer
  - engineering
  - simulation
  - fluid-dynamics
  - rans
language:
  - en
size_categories:
  - 1K<n<10K
configs:
  - config_name: params
    data_files:
      - split: train
        path: "params/train.parquet"
      - split: test
        path: "params/test.parquet"
      - split: failed
        path: "params/failed.parquet"
    default: true
  - config_name: fields
    data_files:
      - split: train
        path: "fields/train/*.safetensors"
      - split: test
        path: "fields/test/*.safetensors"
---

# U-bend Conjugate Heat Transfer Dataset

Parameterized U-bend CFD dataset for surrogate modeling and scientific machine learning.

![Sample overview](overview.png)

*Five randomly selected samples showing the geometry and the solution fields (velocity magnitude, pressure, temperature, turbulent kinetic energy, turbulent viscosity).*

## Dataset Description

This dataset contains **~9,000 conjugate heat transfer simulations** of a parameterized U-bend cooling channel, generated with OpenFOAM (`chtMultiRegionSimpleFoam`). Each sample represents a unique channel geometry defined by 28 design parameters, with corresponding flow and temperature solution fields.

The dataset is designed for scientific machine learning tasks including:
- **Parameter-to-objective regression**: predict pressure loss and cooling effectiveness from geometry parameters
- **Field prediction / Neural operators**: predict full solution fields (velocity, pressure, temperature, turbulence quantities) from mesh coordinates 
- **Physics-informed neural networks (PINNs)**: mesh coordinates and boundary conditions available for physics-constrained training
- **Surrogate modeling**: fast approximation of expensive CFD simulations for design optimization

## Configurations

### `params` — Tabular parameter data
Lightweight CSV/Parquet with 28 input parameters and 2 scalar targets per sample.

```python
from datasets import load_dataset
ds = load_dataset("JDecke/ubend-cfd", "params")
```

| Column | Description |
|---|---|
| `input_0` ... `input_27` | 28 Sobol-sampled geometry parameters |
| `target_0` | Pressure loss (area-averaged inlet pressure) |
| `target_1` | Cooling effectiveness (integrated squared temperature deviation) |
| `target_0_mean`, `target_0_std` | Mean and std over last 100 solver iterations |
| `target_1_mean`, `target_1_std` | Mean and std over last 100 solver iterations |

### `fields` — Solution fields on the mesh
Per-sample safetensors files containing mesh coordinates and solution fields, cropped to the bend region (240 x 60 cells).

```python
from safetensors.numpy import load_file
data = load_file("fields/sample_0.safetensors")
```

| Key | Shape | Description |
|---|---|---|
| `sample_id` | (1,) | Sample index |
| `params` | (28,) | Geometry parameters |
| `targets` | (6,) | Target values (last, mean, std for both objectives) |
| `coords` | (3, 240, 60) | Cell center coordinates (x, y, z) |
| `U` | (3, 240, 60) | Velocity field (Ux, Uy, Uz) |
| `p` | (240, 60) | Pressure field |
| `T` | (240, 60) | Temperature field (fluid) |
| `k` | (240, 60) | Turbulent kinetic energy |
| `nut` | (240, 60) | Turbulent viscosity |
| `p_rgh` | (240, 60) | Pressure minus hydrostatic (equal to `p` for constant density) |
| `solid_coords` | (3, 240, 15) | Solid region cell centers |
| `solid_T` | (240, 15) | Solid temperature field |

## Simulation Setup

- **Solver**: `chtMultiRegionSimpleFoam` (run in the [`jdecke/foamy`](https://hub.docker.com/r/jdecke/foamy) container)
- **Turbulence**: k-omega SST with wall functions
- **Reynolds number**: 40,000
- **Fluid**: Air, constant density (rho = 1.2 kg/m³)1
- **Boundary conditions**:
  - *Inlet (fluid)*: fully-developed turbulent velocity profile, T = 300 K
  - *Outlet (fluid)*: `inletOutlet` for U, p = 0, zeroGradient for T
  - *Fluid/solid interface*: thermally coupled (`turbulentTemperatureCoupledBaffleMixed`), noSlip
  - *Heated wall*: constant heat flux q = 100,000 W/m² on the **outer solid wall in the U-bend (curved) section only**
  - *All other solid outer walls*: adiabatic (zeroGradient)
  - *Front/back*: empty (2D simulation)
- **Mesh**: Structured 2D, 57,000 cells (45,600 fluid + 11,400 solid), wall grading ratio 10. The fluid region contains the flow channel (60 cells across), the solid region models the heated channel wall (15 cells across). Both regions are thermally coupled at the interface.
- **Convergence**: Residual-based termination (p_rgh < 1e-6), max 15,000 iterations

## Parameterization

The geometry is controlled by 28 parameters sampled via a scrambled Sobol sequence (seed=42) to achieve space-filling parameter space. The 28 parameters comprise:
- 6 anchor point displacements (outer wall shape)
- 6 midpoint displacements (inner wall shape)
- 16 cubic Bezier control weights (wall curvature)

<img src="animation_rotgruen.gif" alt="Animation" width="400">

## Dataset Statistics

- **Total samples**: 10,000
- **Successful simulations**: ~8,950 (89.5%)
- **Mesh generation failures**: ~990 (9.9%) — extreme parameter combinations producing invalid geometries
- **Solver divergence**: ~65 (0.7%) — physically unsteady flow in strongly constricted geometries
- **Pressure loss range**: 24 – 800 (successful samples)
- **Temperature range**: 3.7 – 10.0 (successful samples)

### Failed samples and error codes

The `params` configuration provides three splits: `train` (≈8,050 samples) and `test` (≈895 samples) — a 90/10 split with seed=42 over successful simulations — plus `failed` (≈1,055 samples) containing all unsuccessful runs. Failed samples are indicated by sentinel values in the target columns:

| `target_0` value | Meaning |
|---|---|
| `10000` | Mesh generation failed (`checkMesh` reported errors) — geometry too extreme for valid mesh |
| `11000` | Simulation diverged (final residual >= 1e-4) — flow is physically unsteady or geometry too constricted |
| other | Actual pressure loss value (successful simulation) |

The `fields` configuration contains **only successful samples** (~8,950) since no solution fields are available for failed cases. It uses the same train/test split as `params`.

Note that the failed samples are not randomly distributed in parameter space — they cluster in regions with extreme parameter combinations (sharp bends, strong constrictions). This means the parameter space has **gaps** where no valid simulation results exist. Users training surrogate models should be aware of this and filter out samples with `target_0 >= 10000`.

## Mesh Quality

The computational mesh was selected based on a mesh convergence study across multiple wall grading configurations. The final mesh provides a good balance between numerical accuracy and simulation stability. Convergence quality is confirmed by recording the mean and standard deviation of each target over the last 100 solver iterations (median relative std < 0.05%).

## Quick Start

### Parameter regression
```python
from datasets import load_dataset

ds = load_dataset("JDecke/ubend-cfd", "params")
print(ds["train"][0])     # ~8,050 training samples
print(ds["test"][0])      # ~895 test samples (90/10 split, seed=42)
# ds["failed"]            # samples with error codes (mesh/divergence)
```

### Field prediction with PyTorch
```python
from torch.utils.data import Dataset, DataLoader
from safetensors.numpy import load_file
from pathlib import Path
import torch

class UBendFieldDataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted(Path(data_dir).glob("*.safetensors"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = load_file(str(self.files[idx]))
        return {
            "params": torch.from_numpy(data["params"]),              # (28,)
            # Fluid region
            "coords": torch.from_numpy(data["coords"]),             # (3, 240, 60)
            "U": torch.from_numpy(data["U"]),                        # (3, 240, 60)
            "p": torch.from_numpy(data["p"]),                        # (240, 60)
            "T": torch.from_numpy(data["T"]),                        # (240, 60)
            "k": torch.from_numpy(data["k"]),                        # (240, 60)
            "nut": torch.from_numpy(data["nut"]),                    # (240, 60)
            # Solid region
            "solid_coords": torch.from_numpy(data["solid_coords"]),  # (3, 240, 15)
            "solid_T": torch.from_numpy(data["solid_T"]),            # (240, 15)
        }

train_ds = UBendFieldDataset("fields/train")
test_ds = UBendFieldDataset("fields/test")
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=4)
```

### Visualization

A ready-to-use visualization script ([`visualize_sample.py`](visualize_sample.py)) is included. It downloads N random samples from the hub and renders the geometry and all solution fields as PNGs:

```bash
python visualize_sample.py --n 5
```

This produces individual plots per sample plus an `overview.png` (as shown above).

## Version History

This is an updated version of the original U-bend dataset with improved mesh quality (mesh convergence study), corrected Sobol sampling, and additional output fields (mean/std of targets, solution fields in safetensors format). The underlying geometry parameterization and simulation setup remain the same.

## Citation

If you use this dataset, please cite the original publication:

```bibtex
@article{decke2023dataset,
  title={Dataset of a parameterized U-bend flow for Deep Learning Applications},
  author={Decke, Jens and W{\"u}nsch, Olaf and Sick, Bernhard},
  booktitle={Data in Brief},
  volume={50},
  year={2023},
  publisher={Elsevier}
}
```

## Authors

Jens Decke, Olaf Wunsch, Bernhard Sick

## Acknowledgments

This research has been funded by the Federal Ministry for Economic Affairs and Climate Action (BMWK) within the project "KI-basierte Topologieoptimierung elektrischer Maschinen (KITE)" (19I21034C).
