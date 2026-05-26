import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
from pandas.errors import EmptyDataError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from physicsnemo_dc_twin.rack_mapping import ensure_project_rack_layout, ensure_standard_rack_loads


# ============================================================
# Project paths
# ============================================================

RAW_DIR = Path("data/raw_cadence_vtk")
OUT_DIR = Path("data/tensors")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Sampling resolution
# Pilot stage: keep it low first
# ============================================================

NX, NY, NZ = 96, 48, 32


# ============================================================
# Cadence field name variants
# VTU/VTM:
#   Temperature (C), Pressure (Pa), Velocity (m/s)
# Legacy VTK:
#   Temperature(C), Pressure(Pa), Velocity(m/s)
# ============================================================

TEMPERATURE_CANDIDATES = ["Temperature(C)", "Temperature (C)"]
PRESSURE_CANDIDATES = ["Pressure(Pa)", "Pressure (Pa)"]
VELOCITY_CANDIDATES = ["Velocity(m/s)", "Velocity (m/s)"]


# ============================================================
# Utilities
# ============================================================

def natural_sort_key(path: Path):
    text = path.name.lower()
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", text)
    ]


def find_field_name(dataset, candidates):
    point_fields = set(dataset.point_data.keys())
    cell_fields = set(dataset.cell_data.keys())

    for name in candidates:
        if name in point_fields or name in cell_fields:
            return name

    return None


def get_temperature_name(dataset):
    return find_field_name(dataset, TEMPERATURE_CANDIDATES)


def get_pressure_name(dataset):
    return find_field_name(dataset, PRESSURE_CANDIDATES)


def get_velocity_name(dataset):
    return find_field_name(dataset, VELOCITY_CANDIDATES)


def has_temperature_field(mesh) -> bool:
    return get_temperature_name(mesh) is not None


def to_unstructured_grid(dataset):
    if isinstance(dataset, pv.UnstructuredGrid):
        return dataset

    try:
        return dataset.cast_to_unstructured_grid()
    except Exception as exc:
        raise RuntimeError(
            f"Cannot cast dataset of type {type(dataset)} to UnstructuredGrid."
        ) from exc


def read_single_mesh(path: Path):
    print(f"\nReading mesh file: {path}")

    dataset = pv.read(path)

    print("Raw type:", type(dataset))
    print("Points:", getattr(dataset, "n_points", "N/A"))
    print("Cells:", getattr(dataset, "n_cells", "N/A"))
    print("Point fields:", list(dataset.point_data.keys()))
    print("Cell fields:", list(dataset.cell_data.keys()))

    return to_unstructured_grid(dataset)


# ============================================================
# Source selection
# ============================================================

def parse_flow_region_from_vtm(case_dir: Path):
    """
    Read result.vtm as XML and find the VTU file whose DataSet name
    looks like Flow_Region_0.

    Example in result.vtm:
      <DataSet index="0" name="Flow_Region_0" file="VTM/region_0.vtu">
    """

    vtm_path = case_dir / "result.vtm"

    if not vtm_path.exists():
        return None

    print(f"\nChecking VTM index file for flow region: {vtm_path}")

    try:
        tree = ET.parse(vtm_path)
        root = tree.getroot()
    except Exception as exc:
        print(f"Could not parse result.vtm as XML: {exc}")
        return None

    candidate_records = []

    for elem in root.iter():
        tag = elem.tag.split("}")[-1]

        if tag != "DataSet":
            continue

        name = elem.attrib.get("name", "")
        file_attr = elem.attrib.get("file", "")

        if not file_attr:
            continue

        record = {
            "name": name,
            "file": file_attr,
        }

        candidate_records.append(record)

    if candidate_records:
        print("VTM DataSet records:")
        for r in candidate_records:
            print(f"  name={r['name']}, file={r['file']}")

    # Prefer explicit flow region.
    for r in candidate_records:
        name_lower = r["name"].lower()

        if "flow" in name_lower:
            flow_file = case_dir / r["file"]
            if flow_file.exists():
                print(f"Selected flow region from VTM: {r['name']} -> {flow_file}")
                return flow_file
            else:
                print(f"Flow region file listed in VTM but not found: {flow_file}")

    print("No explicit Flow_Region file found in result.vtm.")
    return None


def find_vtu_files(case_dir: Path):
    """
    Find VTU files using Cadence export structure:

    case_xxx/
    ├── VTM/
    │   ├── region_0.vtu
    │   ├── region_1.vtu
    │   └── ...
    ├── result.vtk
    └── result.vtm
    """

    vtu_files = []

    vtm_folder = case_dir / "VTM"

    if vtm_folder.exists() and vtm_folder.is_dir():
        vtu_files.extend(sorted(vtm_folder.glob("*.vtu"), key=natural_sort_key))

    # Also support VTU files directly under case folder.
    vtu_files.extend(sorted(case_dir.glob("*.vtu"), key=natural_sort_key))

    # Remove duplicates while preserving order.
    seen = set()
    unique_files = []

    for f in vtu_files:
        resolved = f.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_files.append(f)

    return unique_files


def choose_largest_valid_vtu(case_dir: Path):
    """
    Fallback when result.vtm does not explicitly provide Flow_Region_0.

    Choose the largest VTU file that contains temperature.
    In your current Cadence export, region_0 is the largest and is the fluid region.
    """

    vtu_files = find_vtu_files(case_dir)

    if not vtu_files:
        return None

    print("\nVTU candidates found:")
    for f in vtu_files:
        print(f"  - {f}")

    valid_candidates = []

    for path in vtu_files:
        try:
            mesh = read_single_mesh(path)
        except Exception as exc:
            print(f"Failed to read VTU file {path}. Error: {exc}")
            continue

        if mesh.n_points == 0 or mesh.n_cells == 0:
            print(f"Skipped empty VTU: {path}")
            continue

        if not has_temperature_field(mesh):
            print(
                f"Skipped VTU without temperature field: {path}. "
                f"Accepted names: {TEMPERATURE_CANDIDATES}"
            )
            continue

        valid_candidates.append(
            {
                "path": path,
                "mesh": mesh,
                "n_cells": mesh.n_cells,
                "n_points": mesh.n_points,
                "bounds": mesh.bounds,
            }
        )

    if not valid_candidates:
        print("No valid VTU file with temperature field found.")
        return None

    # Select the largest cell count.
    valid_candidates.sort(key=lambda r: r["n_cells"], reverse=True)

    selected = valid_candidates[0]

    print("\nSelected largest valid VTU as fluid region fallback:")
    print(f"  file: {selected['path']}")
    print(f"  points: {selected['n_points']}")
    print(f"  cells: {selected['n_cells']}")
    print(f"  bounds: {selected['bounds']}")

    return selected["path"]


def read_cadence_case_mesh(case_dir: Path):
    """
    Robust source selection for training tensor:

    1. Prefer Flow_Region_0 from result.vtm.
    2. If result.vtm does not provide it, use the largest valid VTU.
    3. If no valid VTU exists, fallback to result.vtk.
    4. If result.vtk does not exist, fallback to result.vtm block reading.
    """

    print("\nSource selection:")

    # 1. Prefer explicit Flow_Region_0 from result.vtm
    flow_vtu = parse_flow_region_from_vtm(case_dir)

    if flow_vtu is not None:
        mesh = read_single_mesh(flow_vtu)

        if has_temperature_field(mesh):
            print("\nSelected source: Flow_Region from result.vtm")
            return mesh

        print(
            "Flow region file was found but does not contain temperature. "
            "Trying fallback source selection."
        )

    # 2. Fallback: largest valid VTU
    largest_vtu = choose_largest_valid_vtu(case_dir)

    if largest_vtu is not None:
        mesh = read_single_mesh(largest_vtu)

        if has_temperature_field(mesh):
            print("\nSelected source: largest valid VTU")
            return mesh

    # 3. Fallback: result.vtk
    result_vtk = case_dir / "result.vtk"

    if result_vtk.exists():
        print(f"\nFallback source: {result_vtk}")
        mesh = read_single_mesh(result_vtk)
        print("Selected source: result.vtk")
        return mesh

    # 4. Fallback: result.vtm direct read
    result_vtm = case_dir / "result.vtm"

    if result_vtm.exists():
        print(f"\nFallback source: {result_vtm}")
        dataset = pv.read(result_vtm)

        if isinstance(dataset, pv.MultiBlock):
            print(f"result.vtm returned MultiBlock with {len(dataset)} blocks.")

            # Prefer first block with temperature and largest cell count.
            candidates = []

            for i, block in enumerate(dataset):
                if block is None:
                    continue

                try:
                    mesh = to_unstructured_grid(block)
                except Exception:
                    continue

                if mesh.n_cells == 0 or mesh.n_points == 0:
                    continue

                if not has_temperature_field(mesh):
                    continue

                candidates.append((i, mesh.n_cells, mesh))

            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                block_id, _, mesh = candidates[0]
                print(f"Selected source: result.vtm block {block_id}")
                return mesh

        else:
            mesh = to_unstructured_grid(dataset)
            if has_temperature_field(mesh):
                print("Selected source: result.vtm direct dataset")
                return mesh

    raise FileNotFoundError(
        f"No valid CFD source found in {case_dir}.\n"
        f"Expected one of:\n"
        f"  - result.vtm with Flow_Region_0 pointing to VTM/region_0.vtu\n"
        f"  - case_xxx/VTM/*.vtu containing one of {TEMPERATURE_CANDIDATES}\n"
        f"  - result.vtk\n"
        f"  - result.vtm"
    )


# ============================================================
# Field conversion and sampling
# ============================================================

def ensure_point_data(mesh):
    """
    Convert Cadence cell_data to point_data.
    """

    print("\nBefore cell_data_to_point_data:")
    print("Point fields:", list(mesh.point_data.keys()))
    print("Cell fields:", list(mesh.cell_data.keys()))

    temp_name = get_temperature_name(mesh)
    p_name = get_pressure_name(mesh)
    u_name = get_velocity_name(mesh)

    needs_conversion = False

    for name in [temp_name, p_name, u_name]:
        if name is not None and name in mesh.cell_data:
            needs_conversion = True

    if needs_conversion:
        mesh = mesh.cell_data_to_point_data()

    print("\nAfter cell_data_to_point_data:")
    print("Point fields:", list(mesh.point_data.keys()))
    print("Cell fields:", list(mesh.cell_data.keys()))

    return mesh


def check_required_fields(mesh):
    temp_name = get_temperature_name(mesh)
    p_name = get_pressure_name(mesh)
    u_name = get_velocity_name(mesh)

    if temp_name is None or temp_name not in mesh.point_data:
        raise RuntimeError(
            "Required temperature field not found in point_data.\n"
            f"Accepted names: {TEMPERATURE_CANDIDATES}\n"
            f"Available point fields: {list(mesh.point_data.keys())}\n"
            f"Available cell fields: {list(mesh.cell_data.keys())}"
        )

    print("\nRequired field check:")
    print(f"Temperature field found: {temp_name}")

    if p_name is not None and p_name in mesh.point_data:
        print(f"Pressure field found: {p_name}")
    else:
        print(f"Pressure field not found. Accepted names: {PRESSURE_CANDIDATES}")

    if u_name is not None and u_name in mesh.point_data:
        print(f"Velocity field found: {u_name}")
    else:
        print(f"Velocity field not found. Accepted names: {VELOCITY_CANDIDATES}")


def build_sample_points(bounds, nx, ny, nz):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    z = np.linspace(zmin, zmax, nz)

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

    cloud = pv.PolyData(points)

    return cloud, x, y, z


def sample_field_to_grid(mesh, nx, ny, nz):
    sample_cloud, x, y, z = build_sample_points(mesh.bounds, nx, ny, nz)

    print("\nSampling mesh to regular grid...")
    sampled = sample_cloud.sample(mesh)

    sampled_fields = list(sampled.point_data.keys())
    print("Sampled point fields:", sampled_fields)

    temp_name = get_temperature_name(sampled)
    p_name = get_pressure_name(sampled)
    u_name = get_velocity_name(sampled)

    if temp_name is None or temp_name not in sampled.point_data:
        raise RuntimeError(
            "Temperature field not found after sampling.\n"
            f"Accepted names: {TEMPERATURE_CANDIDATES}\n"
            f"Available sampled fields: {sampled_fields}"
        )

    T = sampled.point_data[temp_name].reshape(nx, ny, nz).astype(np.float32)

    if np.nanmean(T) > 100:
        print("Temperature mean > 100. Assuming Kelvin. Converting to Celsius.")
        T = T - 273.15

    # Prefer vtkValidPointMask if available.
    if "vtkValidPointMask" in sampled.point_data:
        valid = sampled.point_data["vtkValidPointMask"].reshape(nx, ny, nz).astype(np.float32)
    else:
        valid = np.isfinite(T).astype(np.float32)

    T = np.nan_to_num(T, nan=0.0)

    P = None
    if p_name is not None and p_name in sampled.point_data:
        P = sampled.point_data[p_name].reshape(nx, ny, nz).astype(np.float32)
        P = np.nan_to_num(P, nan=0.0)

    U = None
    if u_name is not None and u_name in sampled.point_data:
        U_raw = sampled.point_data[u_name].astype(np.float32)

        if U_raw.ndim == 2 and U_raw.shape[1] == 3:
            U = U_raw.reshape(nx, ny, nz, 3).astype(np.float32)
            U = np.nan_to_num(U, nan=0.0)
        else:
            print(f"Velocity shape unexpected: {U_raw.shape}. Skipping U.")

    return T, P, U, valid, x, y, z


# ============================================================
# Rack heat load input
# ============================================================

def build_heat_load_field(rack_csv: Path, x, y, z):
    """
    Required rack_loads.csv format:

    rack_id,x_min,x_max,y_min,y_max,z_min,z_max,heat_kw
    TEST_LOAD_ZONE,4.5,7.6,0.0,4.8,8.9,18.1,50
    """

    try:
        racks = pd.read_csv(rack_csv)
    except EmptyDataError as exc:
        raise RuntimeError(
            f"{rack_csv} is empty.\n\n"
            "Please fill it with at least the following header and one row:\n\n"
            "rack_id,x_min,x_max,y_min,y_max,z_min,z_max,heat_kw\n"
            "TEST_LOAD_ZONE,4.5,7.6,0.0,4.8,8.9,18.1,50\n"
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
            f"Required columns: {required_cols}\n\n"
            "Example:\n"
            "rack_id,x_min,x_max,y_min,y_max,z_min,z_max,heat_kw\n"
            "TEST_LOAD_ZONE,4.5,7.6,0.0,4.8,8.9,18.1,50\n"
        )

    if racks.empty:
        raise RuntimeError(
            f"{rack_csv} has header but no rows.\n\n"
            "Please add at least one row, for example:\n"
            "TEST_LOAD_ZONE,4.5,7.6,0.0,4.8,8.9,18.1,50\n"
        )

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    heat = np.zeros_like(xx, dtype=np.float32)

    print("\nBuilding heat-load field from:", rack_csv)

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
                f"  WARNING: rack/load zone '{rack_id}' did not select any grid point. "
                "Check coordinates against CFD bounds."
            )

        heat[mask] = heat_kw

    return heat.astype(np.float32)


# ============================================================
# Main conversion
# ============================================================

def convert_one_case(case_dir: Path):
    case_id = case_dir.name
    rack_csv = case_dir / "rack_loads.csv"

    print("\n" + "=" * 100)
    print(f"Converting case: {case_id}")
    print("=" * 100)

    if not rack_csv.exists():
        print(f"\nMissing standard rack load file: {rack_csv}")
        print("Trying to generate it from Cadence rack load export and VTM geometry...")
        rack_csv = ensure_standard_rack_loads(case_dir)
        print(f"Generated standard rack load file: {rack_csv}")

    mesh = read_cadence_case_mesh(case_dir)

    print("\nSelected mesh summary:")
    print("Mesh type:", type(mesh))
    print("Points:", mesh.n_points)
    print("Cells:", mesh.n_cells)
    print("Bounds:", mesh.bounds)

    mesh = ensure_point_data(mesh)
    check_required_fields(mesh)

    T, P, U, valid, x, y, z = sample_field_to_grid(mesh, NX, NY, NZ)

    heat = build_heat_load_field(rack_csv, x, y, z)

    out_path = OUT_DIR / f"{case_id}.npz"

    save_dict = {
        "T": T,
        "heat": heat,
        "valid": valid,
        "x": x,
        "y": y,
        "z": z,
        "source_type": "Flow_Region_0_or_largest_valid_fluid_vtu",
    }

    if P is not None:
        save_dict["P"] = P

    if U is not None:
        save_dict["U"] = U

    np.savez_compressed(out_path, **save_dict)

    print("\nSaved tensor file:", out_path)
    print("T shape:", T.shape)
    print(f"T min/max: {T.min():.3f} / {T.max():.3f}")
    print(f"Heat min/max: {heat.min():.3f} / {heat.max():.3f}")
    print(f"Valid ratio: {valid.mean():.3f}")

    if P is not None:
        print("P shape:", P.shape)
        print(f"P min/max: {P.min():.3f} / {P.max():.3f}")

    if U is not None:
        print("U shape:", U.shape)
        print(f"U min/max: {U.min():.3f} / {U.max():.3f}")


def main():
    print("Starting Cadence CFD result to tensor conversion...")
    print(f"Raw data folder: {RAW_DIR}")
    print(f"Output tensor folder: {OUT_DIR}")
    print(f"Sampling grid: NX={NX}, NY={NY}, NZ={NZ}")

    case_dirs = sorted([p for p in RAW_DIR.iterdir() if p.is_dir()])

    if not case_dirs:
        raise RuntimeError(f"No case folders found in {RAW_DIR}")

    print(f"\nFound {len(case_dirs)} case folders:")
    for p in case_dirs:
        print(f"  - {p.name}")

    layout_path = ensure_project_rack_layout(case_dirs[0])
    print(f"\nRack geometry layout: {layout_path}")

    for case_dir in case_dirs:
        convert_one_case(case_dir)

    print("\nAll cases converted successfully.")


if __name__ == "__main__":
    main()
