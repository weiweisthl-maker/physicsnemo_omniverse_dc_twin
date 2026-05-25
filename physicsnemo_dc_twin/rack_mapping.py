import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyvista as pv


RACK_LOAD_EXPORT_CANDIDATES = [
    "rack_IT load.csv",
    "rack_it_load.csv",
    "rack_load.csv",
]


@dataclass(frozen=True)
class Bounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    @classmethod
    def from_pyvista(cls, bounds):
        return cls(
            x_min=float(bounds[0]),
            x_max=float(bounds[1]),
            y_min=float(bounds[2]),
            y_max=float(bounds[3]),
            z_min=float(bounds[4]),
            z_max=float(bounds[5]),
        )


def natural_sort_key(text: str):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text.lower())]


def find_rack_load_export(case_dir: Path) -> Path | None:
    for name in RACK_LOAD_EXPORT_CANDIDATES:
        path = case_dir / name
        if path.exists():
            return path

    csv_files = sorted(case_dir.glob("*.csv"), key=lambda p: natural_sort_key(p.name))
    for path in csv_files:
        name = path.name.lower()
        if "rack" in name and ("load" in name or "it" in name):
            return path

    return None


def read_rack_load_export(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, usecols=[0, 1])
    raw = raw.dropna(how="all")

    columns = {str(c).strip().upper(): c for c in raw.columns}
    rack_col = columns.get("NUMBER", raw.columns[0])
    load_col = columns.get("LOAD", raw.columns[1])

    loads = raw[[rack_col, load_col]].copy()
    loads.columns = ["rack_id", "heat_kw"]
    loads["rack_id"] = loads["rack_id"].astype(str).str.strip()
    loads["heat_kw"] = pd.to_numeric(loads["heat_kw"], errors="coerce")
    loads = loads.dropna(subset=["rack_id", "heat_kw"])
    loads = loads[loads["rack_id"].str.len() > 0]
    loads = loads.sort_values("rack_id", key=lambda s: s.map(natural_sort_key)).reset_index(drop=True)

    if loads.empty:
        raise RuntimeError(f"No rack load rows found in {path}")

    return loads


def _parse_vtm_datasets(case_dir: Path):
    vtm_path = case_dir / "result.vtm"
    if not vtm_path.exists():
        return []

    tree = ET.parse(vtm_path)
    root = tree.getroot()
    records = []

    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag != "DataSet":
            continue

        name = elem.attrib.get("name", "")
        file_attr = elem.attrib.get("file", "")
        if not file_attr:
            continue

        path = case_dir / file_attr
        if path.exists():
            records.append({"name": name, "path": path})

    return records


def _read_bounds(path: Path) -> Bounds:
    mesh = pv.read(path)
    if mesh.n_points == 0 or mesh.n_cells == 0:
        raise RuntimeError(f"Empty geometry block: {path}")
    return Bounds.from_pyvista(mesh.bounds)


def _axis_thickness(bounds: Bounds, axis: str) -> float:
    if axis == "x":
        return bounds.x_max - bounds.x_min
    if axis == "y":
        return bounds.y_max - bounds.y_min
    if axis == "z":
        return bounds.z_max - bounds.z_min
    raise ValueError(f"Unsupported axis: {axis}")


def infer_rack_envelope_from_case(case_dir: Path) -> Bounds:
    """
    Infer the rack bank interior from non-flow VTM blocks.

    Current Cadence exports contain Flow_Region_0 plus four thin partition wall
    volumes. Those wall blocks do not carry rack IDs, but their inner faces bound
    the equipment/rack bank. We use those inner faces as the automatic envelope.
    """

    records = _parse_vtm_datasets(case_dir)
    wall_records = [r for r in records if "flow" not in r["name"].lower()]

    if len(wall_records) < 4:
        raise RuntimeError(
            f"Expected at least 4 non-flow wall blocks in {case_dir / 'result.vtm'}, "
            f"found {len(wall_records)}."
        )

    wall_bounds = [{"name": r["name"], "path": r["path"], "bounds": _read_bounds(r["path"])} for r in wall_records]

    thin_x = sorted(wall_bounds, key=lambda r: _axis_thickness(r["bounds"], "x"))[:2]
    thin_z = sorted(wall_bounds, key=lambda r: _axis_thickness(r["bounds"], "z"))[:2]

    left_wall, right_wall = sorted(thin_x, key=lambda r: r["bounds"].x_min)
    start_wall, end_wall = sorted(thin_z, key=lambda r: r["bounds"].z_min)

    x_min = left_wall["bounds"].x_max
    x_max = right_wall["bounds"].x_min
    z_min = start_wall["bounds"].z_max
    z_max = end_wall["bounds"].z_min
    y_min = min(r["bounds"].y_min for r in wall_bounds)
    y_max = max(r["bounds"].y_max for r in wall_bounds)

    if not (x_min < x_max and y_min < y_max and z_min < z_max):
        all_bounds = [r["bounds"] for r in wall_bounds]
        x_min = min(b.x_min for b in all_bounds)
        x_max = max(b.x_max for b in all_bounds)
        y_min = min(b.y_min for b in all_bounds)
        y_max = max(b.y_max for b in all_bounds)
        z_min = min(b.z_min for b in all_bounds)
        z_max = max(b.z_max for b in all_bounds)

    return Bounds(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, z_min=z_min, z_max=z_max)


def build_rack_layout_from_export(
    case_dir: Path,
    load_csv: Path | None = None,
    split_axis: str = "z",
) -> pd.DataFrame:
    if load_csv is None:
        load_csv = find_rack_load_export(case_dir)

    if load_csv is None:
        raise FileNotFoundError(
            f"No rack load export found in {case_dir}. "
            f"Tried: {RACK_LOAD_EXPORT_CANDIDATES}"
        )

    loads = read_rack_load_export(load_csv)
    envelope = infer_rack_envelope_from_case(case_dir)
    rack_count = len(loads)

    if split_axis != "z":
        raise ValueError("Only split_axis='z' is currently supported.")

    z_edges = [
        envelope.z_min + (envelope.z_max - envelope.z_min) * i / rack_count
        for i in range(rack_count + 1)
    ]

    rows = []
    for i, row in loads.iterrows():
        rows.append(
            {
                "rack_id": row["rack_id"],
                "x_min": envelope.x_min,
                "x_max": envelope.x_max,
                "y_min": envelope.y_min,
                "y_max": envelope.y_max,
                "z_min": z_edges[i],
                "z_max": z_edges[i + 1],
                "heat_kw": float(row["heat_kw"]),
            }
        )

    return pd.DataFrame(rows)


def ensure_standard_rack_loads(case_dir: Path, output_name: str = "rack_loads.csv") -> Path:
    standard_path = case_dir / output_name
    if standard_path.exists():
        return standard_path

    layout = build_rack_layout_from_export(case_dir)
    layout.to_csv(standard_path, index=False)
    return standard_path
