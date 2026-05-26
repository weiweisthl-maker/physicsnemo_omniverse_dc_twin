from pathlib import Path

import numpy as np
import torch

from .checkpoints import load_checkpoint
from .metrics import compute_validation_metrics, save_metrics
from .models import build_model
from .vtk_export import export_structured_prediction


def export_validation_case(model_path: Path, tensor_case: Path, out_dir: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load(tensor_case)
    x, y, z = data["x"], data["y"], data["z"]
    valid = data["valid"].astype(np.float32)
    heat = data["heat"].astype(np.float32)
    true_c = data["T"].astype(np.float32)

    ckpt = load_checkpoint(model_path, device)
    stats = ckpt["stats"]
    heat_n = (heat - stats["heat_mean"]) / stats["heat_std"]
    input_np = np.stack([heat_n, valid], axis=0)[None, ...].astype(np.float32)

    model = build_model(
        name=ckpt.get("model_name", "small_3d_unet"),
        base_channels=int(ckpt.get("base_channels", 16)),
        model_config=ckpt.get("model_config", {}),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        pred_n = model(torch.from_numpy(input_np).to(device)).cpu().numpy()[0, 0]

    pred_c = pred_n * stats["T_std"] + stats["T_mean"]
    pred_v = np.where(valid > 0.5, pred_c, np.nan).astype(np.float32)
    true_v = np.where(valid > 0.5, true_c, np.nan).astype(np.float32)
    error_v = (pred_v - true_v).astype(np.float32)
    abs_error = np.abs(error_v)

    case_name = tensor_case.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / f"{case_name}_validation.npz"
    out_metrics = out_dir / f"{case_name}_validation_metrics.json"
    out_vtk = out_dir / f"{case_name}_validation.vtk"
    out_vtu = out_dir / f"{case_name}_validation.vtu"

    np.savez_compressed(
        out_npz,
        T_true_C=true_v,
        T_pred_C=pred_v,
        T_error_C=error_v,
        T_abs_error_C=abs_error,
        heat=heat,
        valid=valid,
        x=x,
        y=y,
        z=z,
    )
    export_structured_prediction(
        x,
        y,
        z,
        {
            "T_true_C": true_v,
            "T_pred_C": pred_v,
            "T_error_C": error_v,
            "T_abs_error_C": abs_error.astype(np.float32),
            "rack_heat_kw": heat,
            "valid_mask": valid,
        },
        out_vtk,
        out_vtu,
    )
    metrics = compute_validation_metrics(true_v, pred_v, valid, x, y, z)
    save_metrics(metrics, out_metrics)

    print("\nValidation export summary:")
    print(f"  Case: {tensor_case}")
    print(f"  MAE valid: {metrics['mae_C']:.3f} C")
    print(f"  RMSE valid: {metrics['rmse_C']:.3f} C")
    print(f"  P95 abs error: {metrics['p95_abs_error_C']:.3f} C")
    print(f"  P99 abs error: {metrics['p99_abs_error_C']:.3f} C")
    print(f"  Max abs error valid: {metrics['max_abs_error_C']:.3f} C")
    print(f"  Hotspot location error: {metrics['hotspot']['location_error_m']:.3f} m")
    print(f"  Saved NPZ: {out_npz}")
    print(f"  Saved metrics: {out_metrics}")
    print(f"  Saved VTK: {out_vtk}")
    print(f"  Saved VTU: {out_vtu}")
