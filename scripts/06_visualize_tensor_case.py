import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physicsnemo_dc_twin.rack_mapping import DEFAULT_RACK_LAYOUT_PATH, read_rack_layout
from physicsnemo_dc_twin.vtk_export import export_structured_prediction


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export tensor fields and rack boxes for ParaView QC."
    )
    parser.add_argument("--case", default="data/tensors/case_001.npz")
    parser.add_argument("--rack-layout", default=str(DEFAULT_RACK_LAYOUT_PATH))
    parser.add_argument("--out-dir", default="outputs/tensor_qc")
    return parser.parse_args()


def build_rack_zone_field(layout: pd.DataFrame, x, y, z):
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    zone = np.zeros_like(xx, dtype=np.float32)

    for idx, row in enumerate(layout.itertuples(index=False), start=1):
        mask = (
            (xx >= row.x_min)
            & (xx <= row.x_max)
            & (yy >= row.y_min)
            & (yy <= row.y_max)
            & (zz >= row.z_min)
            & (zz <= row.z_max)
        )
        zone[mask] = float(idx)

    return zone


def export_rack_boxes(layout: pd.DataFrame, heat, x, y, z, out_path: Path):
    boxes = []
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    for idx, row in enumerate(layout.itertuples(index=False), start=1):
        bounds = (row.x_min, row.x_max, row.y_min, row.y_max, row.z_min, row.z_max)
        box = pv.Box(bounds=bounds)
        mask = (
            (xx >= row.x_min)
            & (xx <= row.x_max)
            & (yy >= row.y_min)
            & (yy <= row.y_max)
            & (zz >= row.z_min)
            & (zz <= row.z_max)
        )
        heat_values = heat[mask]
        heat_kw = float(np.max(heat_values)) if heat_values.size else 0.0
        box.cell_data["rack_index"] = np.full(box.n_cells, idx, dtype=np.float32)
        box.cell_data["rack_heat_kw"] = np.full(box.n_cells, heat_kw, dtype=np.float32)
        boxes.append(box)

    combined = pv.MultiBlock(boxes).combine()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(out_path)


def main():
    args = parse_args()
    tensor_case = Path(args.case)
    layout_path = Path(args.rack_layout)
    out_dir = Path(args.out_dir)
    case_name = tensor_case.stem

    data = np.load(tensor_case)
    x, y, z = data["x"], data["y"], data["z"]
    temp = data["T"].astype(np.float32)
    heat = data["heat"].astype(np.float32)
    valid = data["valid"].astype(np.float32)
    layout = read_rack_layout(layout_path)

    zone = build_rack_zone_field(layout, x, y, z)

    out_vtk = out_dir / f"{case_name}_tensor_qc.vtk"
    out_vtu = out_dir / f"{case_name}_tensor_qc.vtu"
    out_boxes = out_dir / f"{case_name}_rack_boxes.vtu"

    export_structured_prediction(
        x,
        y,
        z,
        {
            "T_C": np.where(valid > 0.5, temp, np.nan).astype(np.float32),
            "rack_heat_kw": heat,
            "valid_mask": valid,
            "rack_zone_id": zone,
        },
        out_vtk,
        out_vtu,
    )
    export_rack_boxes(layout, heat, x, y, z, out_boxes)

    print("\nTensor QC export summary:")
    print(f"  Case: {tensor_case}")
    print(f"  Rack layout: {layout_path}")
    print(f"  Saved VTK: {out_vtk}")
    print(f"  Saved VTU: {out_vtu}")
    print(f"  Saved rack boxes: {out_boxes}")
    print("\nOpen both VTU files in ParaView to inspect T_C, rack_heat_kw, and rack_zone_id.")


if __name__ == "__main__":
    main()
