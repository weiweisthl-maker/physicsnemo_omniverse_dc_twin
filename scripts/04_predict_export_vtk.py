from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
import torch
import torch.nn as nn
from pandas.errors import EmptyDataError


# ============================================================
# Project paths
# ============================================================

MODEL_PATH = Path("models/best_field_model.pt")
REFERENCE_TENSOR = Path("data/tensors/case_001.npz")
NEW_RACK_LOADS = Path("new_rack_loads.csv")

OUT_DIR = Path("outputs/predicted_vtk")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_NPZ = OUT_DIR / "pred_case_new.npz"
OUT_VTK = OUT_DIR / "pred_case_new.vtk"
OUT_VTU = OUT_DIR / "pred_case_new.vtu"


# ============================================================
# Device
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Model definition
# Must match scripts/03_train_field_model.py
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Small3DUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base=16):
        super().__init__()

        self.enc1 = ConvBlock(in_channels, base)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = ConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.bottleneck = ConvBlock(base * 2, base * 4)

        self.up2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.dec2 = ConvBlock(base * 4 + base * 2, base * 2)

        self.up1 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.dec1 = ConvBlock(base * 2 + base, base)

        self.out = nn.Conv3d(base, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        b = self.bottleneck(p2)

        u2 = self.up2(b)

        if u2.shape[-3:] != e2.shape[-3:]:
            u2 = torch.nn.functional.interpolate(
                u2,
                size=e2.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )

        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)

        if u1.shape[-3:] != e1.shape[-3:]:
            u1 = torch.nn.functional.interpolate(
                u1,
                size=e1.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )

        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        return self.out(d1)


# ============================================================
# Rack heat-load field
# ============================================================

def build_heat_load_field(rack_csv: Path, x, y, z):
    """
    Required new_rack_loads.csv format:

    rack_id,x_min,x_max,y_min,y_max,z_min,z_max,heat_kw
    TEST_LOAD_ZONE,4.5,7.6,0.0,4.8,8.9,18.1,65
    """

    if not rack_csv.exists():
        raise FileNotFoundError(
            f"Cannot find {rack_csv}\n\n"
            "Please create new_rack_loads.csv in the project root, for example:\n\n"
            "rack_id,x_min,x_max,y_min,y_max,z_min,z_max,heat_kw\n"
            "TEST_LOAD_ZONE,4.5,7.6,0.0,4.8,8.9,18.1,65\n"
        )

    try:
        racks = pd.read_csv(rack_csv)
    except EmptyDataError as exc:
        raise RuntimeError(
            f"{rack_csv} is empty.\n\n"
            "Please fill it with:\n\n"
            "rack_id,x_min,x_max,y_min,y_max,z_min,z_max,heat_kw\n"
            "TEST_LOAD_ZONE,4.5,7.6,0.0,4.8,8.9,18.1,65\n"
        ) from exc

    required_cols = [
        "rack_id",
        "x_min", "x_max",
        "y_min", "y_max",
        "z_min", "z_max",
        "heat_kw",
    ]

    missing = [c for c in required_cols if c not in racks.columns]

    if missing:
        raise RuntimeError(
            f"{rack_csv} missing columns: {missing}\n"
            f"Required columns: {required_cols}"
        )

    if racks.empty:
        raise RuntimeError(
            f"{rack_csv} has header but no rows.\n"
            "Please add at least one row."
        )

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    heat = np.zeros_like(xx, dtype=np.float32)

    print("\nBuilding new heat-load field from:", rack_csv)

    for _, r in racks.iterrows():
        rack_id = str(r["rack_id"])

        x_min, x_max = float(r["x_min"]), float(r["x_max"])
        y_min, y_max = float(r["y_min"]), float(r["y_max"])
        z_min, z_max = float(r["z_min"]), float(r["z_max"])
        heat_kw = float(r["heat_kw"])

        mask = (
            (xx >= x_min) & (xx <= x_max) &
            (yy >= y_min) & (yy <= y_max) &
            (zz >= z_min) & (zz <= z_max)
        )

        selected_count = int(mask.sum())

        print(
            f"  {rack_id}: "
            f"x=[{x_min}, {x_max}], "
            f"y=[{y_min}, {y_max}], "
            f"z=[{z_min}, {z_max}], "
            f"heat={heat_kw} kW, "
            f"grid_points={selected_count}"
        )

        if selected_count == 0:
            print(
                f"  WARNING: '{rack_id}' selected 0 grid points. "
                "Check coordinates against the CFD bounds."
            )

        heat[mask] = heat_kw

    return heat.astype(np.float32)


# ============================================================
# Export VTK / VTU
# ============================================================

def export_prediction_to_vtk(x, y, z, T_pred, heat, valid):
    """
    Export predicted field to:
      - StructuredGrid legacy .vtk
      - UnstructuredGrid .vtu

    Fields:
      T_pred_C
      rack_heat_kw
      valid_mask
    """

    print("\nExporting prediction to VTK/VTU...")

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    grid = pv.StructuredGrid(xx, yy, zz)

    # PyVista StructuredGrid normally expects VTK ordering.
    # For arrays generated by meshgrid(indexing='ij'), order='F' is commonly correct.
    grid.point_data["T_pred_C"] = T_pred.ravel(order="F")
    grid.point_data["rack_heat_kw"] = heat.ravel(order="F")
    grid.point_data["valid_mask"] = valid.ravel(order="F")

    # Save legacy VTK structured grid
    grid.save(OUT_VTK)

    # Save VTU unstructured grid
    ugrid = grid.cast_to_unstructured_grid()
    ugrid.save(OUT_VTU)

    print(f"Saved legacy VTK: {OUT_VTK}")
    print(f"Saved VTU:        {OUT_VTU}")


# ============================================================
# Main
# ============================================================

def main():
    print("Starting field prediction and VTK export...")
    print(f"Device: {DEVICE}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}\n"
            "Please run scripts/03_train_field_model.py first."
        )

    if not REFERENCE_TENSOR.exists():
        raise FileNotFoundError(
            f"Reference tensor not found: {REFERENCE_TENSOR}\n"
            "Please run scripts/02_convert_vtk_to_tensor.py first."
        )

    print(f"Loading reference tensor: {REFERENCE_TENSOR}")

    ref = np.load(REFERENCE_TENSOR)

    x = ref["x"]
    y = ref["y"]
    z = ref["z"]
    valid = ref["valid"].astype(np.float32)

    print("Reference grid:")
    print(f"  x: {x.shape}, min={x.min():.3f}, max={x.max():.3f}")
    print(f"  y: {y.shape}, min={y.min():.3f}, max={y.max():.3f}")
    print(f"  z: {z.shape}, min={z.min():.3f}, max={z.max():.3f}")
    print(f"  valid shape: {valid.shape}, valid ratio={valid.mean():.3f}")

    heat = build_heat_load_field(NEW_RACK_LOADS, x, y, z)

    print("\nLoading model checkpoint:", MODEL_PATH)

    # For recent PyTorch versions, weights_only=False is safe here because
    # this checkpoint was generated locally by scripts/03_train_field_model.py.
    try:
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    except TypeError:
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

    stats = ckpt["stats"]

    print("\nLoaded stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    heat_n = (heat - stats["heat_mean"]) / stats["heat_std"]

    input_np = np.stack([heat_n, valid], axis=0)[None, ...].astype(np.float32)
    input_tensor = torch.from_numpy(input_np).to(DEVICE)

    print("\nInput tensor shape:", tuple(input_tensor.shape))

    model = Small3DUNet(in_channels=2, out_channels=1, base=16).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print("\nRunning inference...")

    with torch.no_grad():
        pred_n = model(input_tensor).cpu().numpy()[0, 0]

    T_pred = pred_n * stats["T_std"] + stats["T_mean"]

    # Invalid region is set to NaN for visualization.
    # ParaView can still color valid regions correctly.
    T_pred_visual = np.where(valid > 0.5, T_pred, np.nan).astype(np.float32)

    print("\nPrediction summary:")
    print(f"  T_pred shape: {T_pred.shape}")
    print(f"  T_pred valid min: {np.nanmin(T_pred_visual):.3f} C")
    print(f"  T_pred valid max: {np.nanmax(T_pred_visual):.3f} C")
    print(f"  Heat min/max: {heat.min():.3f} / {heat.max():.3f}")

    np.savez_compressed(
        OUT_NPZ,
        T_pred=T_pred_visual,
        heat=heat.astype(np.float32),
        valid=valid.astype(np.float32),
        x=x,
        y=y,
        z=z,
        source_model=str(MODEL_PATH),
        source_rack_loads=str(NEW_RACK_LOADS),
    )

    print(f"\nSaved prediction NPZ: {OUT_NPZ}")

    export_prediction_to_vtk(
        x=x,
        y=y,
        z=z,
        T_pred=T_pred_visual,
        heat=heat,
        valid=valid,
    )

    print("\nPrediction export finished.")
    print("\nNext step:")
    print(f"Open {OUT_VTU} in ParaView and color by 'T_pred_C'.")


if __name__ == "__main__":
    main()