from pathlib import Path

import numpy as np
import torch

from .checkpoints import load_checkpoint
from .heat_loads import build_heat_load_field
from .models import build_model
from .vtk_export import export_structured_prediction


def predict_temperature(
    model_path: Path,
    reference_tensor: Path,
    rack_loads_csv: Path,
    out_npz: Path,
    out_vtk: Path,
    out_vtu: Path,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Starting field prediction and VTK export...")
    print(f"Device: {device}")

    ref = np.load(reference_tensor)
    x, y, z = ref["x"], ref["y"], ref["z"]
    valid = ref["valid"].astype(np.float32)
    heat = build_heat_load_field(rack_loads_csv, x, y, z)

    ckpt = load_checkpoint(model_path, device)
    stats = ckpt["stats"]
    model_name = ckpt.get("model_name", "small_3d_unet")
    base_channels = int(ckpt.get("base_channels", 16))
    model_config = ckpt.get("model_config", {})

    heat_n = (heat - stats["heat_mean"]) / stats["heat_std"]
    input_np = np.stack([heat_n, valid], axis=0)[None, ...].astype(np.float32)
    input_tensor = torch.from_numpy(input_np).to(device)

    model = build_model(
        name=model_name,
        base_channels=base_channels,
        model_config=model_config,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        pred_n = model(input_tensor).cpu().numpy()[0, 0]

    pred_c = pred_n * stats["T_std"] + stats["T_mean"]
    pred_visual = np.where(valid > 0.5, pred_c, np.nan).astype(np.float32)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        T_pred=pred_visual,
        heat=heat.astype(np.float32),
        valid=valid.astype(np.float32),
        x=x,
        y=y,
        z=z,
        source_model=str(model_path),
        source_rack_loads=str(rack_loads_csv),
    )

    export_structured_prediction(
        x,
        y,
        z,
        {
            "T_pred_C": pred_visual,
            "rack_heat_kw": heat.astype(np.float32),
            "valid_mask": valid.astype(np.float32),
        },
        out_vtk,
        out_vtu,
    )

    print("\nPrediction summary:")
    print(f"  T_pred valid min: {np.nanmin(pred_visual):.3f} C")
    print(f"  T_pred valid max: {np.nanmax(pred_visual):.3f} C")
    print(f"  Saved NPZ: {out_npz}")
    print(f"  Saved VTK: {out_vtk}")
    print(f"  Saved VTU: {out_vtu}")
    return pred_visual
