# PhysicsNeMo and Omniverse Data Center Thermal Visualization Twin Pilot

This repository is a pilot workflow for converting Cadence Reality Design Pro CFD
thermal results into AI-ready tensors, training a field surrogate, exporting
predicted thermal fields to VTK/VTU, and preparing the path toward a
PhysicsNeMo-powered Omniverse digital twin.

## Current Status

The default AI backend is now `physicsnemo.models.unet.UNet`. The older compact
PyTorch 3D UNet is retained only for loading legacy checkpoints.

The completed first-layer workflow is:

1. Inspect Cadence VTK/VTU/VTM exports.
2. Convert the selected fluid region to regular-grid tensors.
3. Train a masked temperature-field surrogate.
4. Predict a new heat-load scenario.
5. Export prediction and validation fields to VTK/VTU for ParaView.

The current two-case dataset is not enough for engineering-grade prediction
accuracy. Treat the model as a workflow validation artifact.

## Repository Layout

```text
configs/
  field_model.json
physicsnemo_dc_twin/
  data.py
  models.py
  train.py
  infer.py
  validation.py
  heat_loads.py
  vtk_export.py
scripts/
  01_inspect_vtk.py
  01b_search_rack_power_metadata.py
  02_convert_vtk_to_tensor.py
  03_train_field_model.py
  04_predict_export_vtk.py
  05_validate_known_case.py
```

## Python Environment

Install the pilot dependencies:

```bash
pip install -r requirements.txt
```

When migrating the backend, install PhysicsNeMo following NVIDIA's official
instructions. The intended package is `nvidia-physicsnemo`, but use NVIDIA's
current compatibility matrix for CUDA, PyTorch, and container choices.

This workspace has been tested with a local Python 3.12 virtual environment:

```powershell
.\.venv\Scripts\python.exe scripts\03_train_field_model.py --epochs 1
```

PhysicsNeMo core is installed with:

```bash
pip install nvidia-physicsnemo
pip install warp-lang==1.12.1
```

On native Windows, the full `nvidia-physicsnemo[cu13,nn-extras]` install can
fail because NVIDIA DALI publishes Linux wheels for the CUDA 13 package path.
For GPU training and the full CUDA extras stack, use WSL2 or an NVIDIA container.

## Data Preparation

Cadence case folders are expected under:

```text
data/raw_cadence_vtk/case_001/
data/raw_cadence_vtk/case_002/
```

Each case needs a `rack_loads.csv` file:

```csv
rack_id,x_min,x_max,y_min,y_max,z_min,z_max,heat_kw
TEST_LOAD_ZONE,4.5,7.6,0.0,4.8,8.9,18.1,50
```

Convert CFD outputs to tensors:

```bash
python scripts/02_convert_vtk_to_tensor.py
```

The converter writes files such as:

```text
data/tensors/case_001.npz
data/tensors/case_002.npz
```

## Training

Train with the default config:

```bash
python scripts/03_train_field_model.py
```

Fast smoke test:

```bash
python scripts/03_train_field_model.py --epochs 1
```

The best PhysicsNeMo checkpoint is written to:

```text
models/best_physicsnemo_unet.pt
```

## Prediction Export

Predict a new heat-load scenario from `new_rack_loads.csv`:

```bash
python scripts/04_predict_export_vtk.py
```

Outputs:

```text
outputs/predicted_vtk/pred_case_new.npz
outputs/predicted_vtk/pred_case_new.vtk
outputs/predicted_vtk/pred_case_new.vtu
```

Open the VTU in ParaView and color by `T_pred_C`.

## Validation Export

Export true, predicted, and error fields for a known CFD tensor case:

```bash
python scripts/05_validate_known_case.py --case data/tensors/case_002.npz
```

Outputs include:

```text
T_true_C
T_pred_C
T_error_C
T_abs_error_C
rack_heat_kw
valid_mask
```

This step is the minimum required bridge from a visual demo to a measurable
surrogate workflow.

## PhysicsNeMo Migration Plan

The package is now structured so the PhysicsNeMo migration can happen in layers:

1. Keep `physicsnemo_dc_twin.data.FieldDataset` as the tensor datapipe contract.
2. Use `physicsnemo_dc_twin.models.build_physicsnemo_unet` as the native
   PhysicsNeMo model backend.
3. Move `configs/field_model.json` to Hydra/OmegaConf once the PhysicsNeMo
   backend is active.
4. Extend outputs from temperature-only to temperature, pressure, and velocity.
5. Keep `physicsnemo_dc_twin.vtk_export` as the visualization export boundary.
6. Add rack-level thermal state extraction for Omniverse/OpenUSD mapping.

Near-term quality gates:

1. At least 10 to 20 CFD cases for meaningful validation.
2. Dedicated train/validation/test split by scenario, not by random grid points.
3. MAE, max error, and hotspot-location error reporting.
4. Rack-by-rack heat-load metadata instead of one pilot test zone.
5. Reproducible config, checkpoint metadata, and inference export naming.
