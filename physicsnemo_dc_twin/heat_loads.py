from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from .rack_mapping import DEFAULT_RACK_LAYOUT_PATH, standardize_rack_load_file


REQUIRED_RACK_COLUMNS = [
    "rack_id",
    "x_min",
    "x_max",
    "y_min",
    "y_max",
    "z_min",
    "z_max",
    "heat_kw",
]


def read_rack_loads(rack_csv: Path):
    if not rack_csv.exists():
        raise FileNotFoundError(f"Rack heat-load CSV not found: {rack_csv}")
    try:
        racks = pd.read_csv(rack_csv)
    except EmptyDataError as exc:
        raise RuntimeError(f"{rack_csv} is empty.") from exc

    missing = [c for c in REQUIRED_RACK_COLUMNS if c not in racks.columns]
    if missing:
        racks = standardize_rack_load_file(rack_csv, DEFAULT_RACK_LAYOUT_PATH)
    if racks.empty:
        raise RuntimeError(f"{rack_csv} has header but no rows.")

    missing = [c for c in REQUIRED_RACK_COLUMNS if c not in racks.columns]
    if missing:
        raise RuntimeError(
            f"{rack_csv} missing columns after standardization: {missing}. "
            f"Required columns: {REQUIRED_RACK_COLUMNS}"
        )

    return racks[REQUIRED_RACK_COLUMNS].copy()


def build_heat_load_field(rack_csv: Path, x, y, z, verbose: bool = True):
    racks = read_rack_loads(rack_csv)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    heat = np.zeros_like(xx, dtype=np.float32)

    if verbose:
        print("\nBuilding heat-load field from:", rack_csv)

    for _, row in racks.iterrows():
        rack_id = str(row["rack_id"])
        x_min, x_max = float(row["x_min"]), float(row["x_max"])
        y_min, y_max = float(row["y_min"]), float(row["y_max"])
        z_min, z_max = float(row["z_min"]), float(row["z_max"])
        heat_kw = float(row["heat_kw"])
        mask = (
            (xx >= x_min)
            & (xx <= x_max)
            & (yy >= y_min)
            & (yy <= y_max)
            & (zz >= z_min)
            & (zz <= z_max)
        )
        selected_count = int(mask.sum())
        if verbose:
            print(
                f"  {rack_id}: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}], "
                f"z=[{z_min}, {z_max}], heat={heat_kw} kW, "
                f"grid_points={selected_count}"
            )
            if selected_count == 0:
                print(f"  WARNING: '{rack_id}' selected 0 grid points.")
        heat[mask] = heat_kw

    return heat.astype(np.float32)

