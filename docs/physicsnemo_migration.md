# PhysicsNeMo Migration Notes

The current project is now organized around a stable data contract:

```text
input:
  heat:  [X, Y, Z]
  valid: [X, Y, Z]
target:
  T:     [X, Y, Z]
optional future targets:
  P:     [X, Y, Z]
  U:     [X, Y, Z, 3]
```

The first migration should not change Cadence preprocessing or VTK export.
Only the model and training backend should change first.

## Stage 1: PhysicsNeMo-Ready Torch Baseline

Completed in this repository:

- Shared dataset module.
- Shared model module.
- Shared checkpoint utilities.
- Shared prediction and VTK export module.
- Known-case validation export.
- Config-driven training entry point.

## Stage 2: Native PhysicsNeMo Backend

Completed in this repository:

- Installed `nvidia-physicsnemo==2.0.0` in the local `.venv`.
- Pinned `warp-lang==1.12.1` because newer Warp builds broke
  `physicsnemo.models.unet` import on this Windows environment.
- Replaced the default model factory with `physicsnemo.models.unet.UNet`.
- Saved PhysicsNeMo model configuration into checkpoint metadata.
- Generated `models/best_physicsnemo_unet.pt` with a 1 epoch smoke run.

Recommended next implementation:

1. Move `configs/field_model.json` to Hydra/OmegaConf.
2. Keep the tensor shapes and checkpoint metadata compatible with current
   inference exports.
3. Run full GPU training in WSL2 or an NVIDIA container for CUDA extras support.

## Stage 3: Multi-Field Prediction

Update the target tensor from temperature-only to:

```text
T_pred_C
P_pred_Pa
U_pred_mps
```

Use masked losses for all fields and report engineering units during validation.

## Stage 4: Omniverse/OpenUSD Interface

After reliable validation metrics exist, add a rack-state extractor:

```text
rack_id
inlet_T_avg_C
inlet_T_max_C
inlet_T_p95_C
risk_level
over_temperature
```

This table can drive rack colors, labels, and scenario panels in Omniverse.
