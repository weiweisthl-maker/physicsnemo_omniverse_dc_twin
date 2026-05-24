import re
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pyvista as pv
import numpy as np


# ============================================================
# Configure case folder
# ============================================================

CASE_DIR = Path("data/raw_cadence_vtk/case_001")


# ============================================================
# Keywords to search
# ============================================================

KEYWORDS = [
    "rack",
    "cabinet",
    "server",
    "equipment",
    "device",
    "unit",
    "name",
    "id",
    "label",
    "power",
    "load",
    "heat",
    "kw",
    "watt",
    "source",
    "generation",
    "flux",
    "boundary",
    "bc",
]


# ============================================================
# File collection
# ============================================================

def natural_sort_key(path: Path):
    text = path.name.lower()
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", text)
    ]


def collect_candidate_files(case_dir: Path):
    files = []

    vtm_folder = case_dir / "VTM"

    if vtm_folder.exists() and vtm_folder.is_dir():
        files.extend(sorted(vtm_folder.glob("*.vtu"), key=natural_sort_key))
        files.extend(sorted(vtm_folder.glob("*.vtp"), key=natural_sort_key))
        files.extend(sorted(vtm_folder.glob("*.vts"), key=natural_sort_key))
        files.extend(sorted(vtm_folder.glob("*.vti"), key=natural_sort_key))

    files.extend(sorted(case_dir.glob("*.vtu"), key=natural_sort_key))
    files.extend(sorted(case_dir.glob("*.vtk"), key=natural_sort_key))
    files.extend(sorted(case_dir.glob("*.vtm"), key=natural_sort_key))
    files.extend(sorted(case_dir.glob("*.vtp"), key=natural_sort_key))

    # Remove duplicates while preserving order
    unique = []
    seen = set()

    for f in files:
        resolved = f.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(f)

    return unique


# ============================================================
# Keyword utilities
# ============================================================

def matches_keywords(text: str):
    text_lower = text.lower()
    return [kw for kw in KEYWORDS if kw in text_lower]


def print_array_brief(name, arr, indent="  "):
    shape = getattr(arr, "shape", None)
    dtype = getattr(arr, "dtype", None)

    msg = f"{indent}- {name}: shape={shape}, dtype={dtype}"

    try:
        if np.issubdtype(arr.dtype, np.number):
            msg += f", min={np.nanmin(arr):.5g}, max={np.nanmax(arr):.5g}"
            if arr.size <= 20:
                msg += f", values={arr}"
        else:
            unique_vals = np.unique(arr)
            msg += f", unique_count={len(unique_vals)}"
            msg += f", sample={unique_vals[:10]}"
    except Exception as exc:
        msg += f", value inspection failed: {exc}"

    print(msg)


def inspect_data_container(container, container_name):
    """
    Inspect point_data, cell_data, field_data.
    """
    print(f"\n{container_name}:")

    if len(container.keys()) == 0:
        print("  None")
        return []

    hits = []

    for name in container.keys():
        arr = container[name]
        kws = matches_keywords(name)

        print_array_brief(name, arr)

        if kws:
            hits.append(
                {
                    "container": container_name,
                    "field_name": name,
                    "matched_keywords": kws,
                    "shape": str(getattr(arr, "shape", None)),
                    "dtype": str(getattr(arr, "dtype", None)),
                }
            )

    return hits


# ============================================================
# PyVista inspection
# ============================================================

def inspect_dataset(dataset, label):
    print("\n" + "=" * 100)
    print(f"Dataset: {label}")
    print("=" * 100)

    print("Type:", type(dataset))

    if hasattr(dataset, "n_points"):
        print("Points:", dataset.n_points)

    if hasattr(dataset, "n_cells"):
        print("Cells:", dataset.n_cells)

    if hasattr(dataset, "bounds"):
        print("Bounds:", dataset.bounds)

    hits = []

    if hasattr(dataset, "point_data"):
        hits.extend(inspect_data_container(dataset.point_data, "point_data"))

    if hasattr(dataset, "cell_data"):
        hits.extend(inspect_data_container(dataset.cell_data, "cell_data"))

    if hasattr(dataset, "field_data"):
        hits.extend(inspect_data_container(dataset.field_data, "field_data"))

    return hits


def inspect_file_with_pyvista(path: Path):
    print("\n" + "#" * 100)
    print(f"Reading with PyVista: {path}")
    print("#" * 100)

    try:
        dataset = pv.read(path)
    except Exception as exc:
        print(f"PyVista read failed: {exc}")
        return []

    all_hits = []

    if isinstance(dataset, pv.MultiBlock):
        print("MultiBlock dataset")
        print("Number of blocks:", len(dataset))

        if hasattr(dataset, "field_data"):
            all_hits.extend(inspect_data_container(dataset.field_data, "multiblock_field_data"))

        for i, block in enumerate(dataset):
            if block is None:
                print(f"\nBlock {i}: Empty")
                continue

            hits = inspect_dataset(block, f"{path.name} / block_{i}")
            for h in hits:
                h["block"] = i
            all_hits.extend(hits)

    else:
        all_hits.extend(inspect_dataset(dataset, path.name))

    return all_hits


# ============================================================
# Raw XML inspection
# This is important because some metadata may not be exposed as PyVista arrays.
# ============================================================

def inspect_xml_names(path: Path):
    """
    Search XML-style VTK files for DataArray names and metadata names.
    Works for .vtu, .vtm, .vtp, etc.
    """
    if path.suffix.lower() not in [".vtu", ".vtm", ".vtp", ".vts", ".vti"]:
        return []

    print("\n" + "-" * 100)
    print(f"Raw XML name scan: {path}")
    print("-" * 100)

    hits = []

    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except Exception as exc:
        print(f"XML parse failed: {exc}")
        return hits

    for elem in root.iter():
        tag = elem.tag.split("}")[-1]

        attrs_to_check = {}

        for key, value in elem.attrib.items():
            attrs_to_check[key] = value

        # Common useful VTK XML elements:
        # DataArray Name="..."
        # Piece Source="..."
        # DataSet name="..."
        # Block name="..."
        for attr_name, attr_value in attrs_to_check.items():
            kws = matches_keywords(str(attr_value))

            if kws:
                hit = {
                    "xml_tag": tag,
                    "attribute": attr_name,
                    "value": attr_value,
                    "matched_keywords": kws,
                }
                hits.append(hit)
                print(f"Hit: tag={tag}, {attr_name}={attr_value}, keywords={kws}")

    if not hits:
        print("No keyword hits in XML attributes.")

    return hits


# ============================================================
# Raw text scan
# Useful for legacy .vtk and for quickly finding strings in XML files.
# ============================================================

def inspect_raw_text(path: Path, max_hits=50):
    print("\n" + "-" * 100)
    print(f"Raw text keyword scan: {path}")
    print("-" * 100)

    hits = []

    try:
        # Read as text with errors ignored.
        text = path.read_text(errors="ignore")
    except Exception as exc:
        print(f"Raw text read failed: {exc}")
        return hits

    lines = text.splitlines()

    for i, line in enumerate(lines, start=1):
        kws = matches_keywords(line)

        if kws:
            short_line = line.strip()
            if len(short_line) > 240:
                short_line = short_line[:240] + "..."

            hits.append(
                {
                    "line": i,
                    "text": short_line,
                    "matched_keywords": kws,
                }
            )

            print(f"Line {i}: {short_line} | keywords={kws}")

            if len(hits) >= max_hits:
                print(f"Reached max_hits={max_hits}. Stopping raw text scan.")
                break

    if not hits:
        print("No keyword hits in raw text.")

    return hits


# ============================================================
# Main
# ============================================================

def main():
    print("=== Cadence Rack / Power Metadata Search ===")
    print(f"Case folder: {CASE_DIR}")
    print(f"Exists: {CASE_DIR.exists()}")

    if not CASE_DIR.exists():
        raise FileNotFoundError(f"Case folder not found: {CASE_DIR}")

    files = collect_candidate_files(CASE_DIR)

    print("\nCandidate files:")
    if not files:
        print("  None")
        return

    for f in files:
        print(f"  - {f}")

    report = {
        "case_dir": str(CASE_DIR),
        "keywords": KEYWORDS,
        "files": [],
    }

    for path in files:
        file_record = {
            "file": str(path),
            "pyvista_hits": [],
            "xml_hits": [],
            "raw_text_hits": [],
        }

        pyvista_hits = inspect_file_with_pyvista(path)
        xml_hits = inspect_xml_names(path)
        raw_hits = inspect_raw_text(path)

        file_record["pyvista_hits"] = pyvista_hits
        file_record["xml_hits"] = xml_hits
        file_record["raw_text_hits"] = raw_hits

        report["files"].append(file_record)

    out_dir = Path("outputs/metadata_scan")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{CASE_DIR.name}_rack_power_metadata_scan.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 100)
    print("Scan summary")
    print("=" * 100)

    any_hit = False

    for file_record in report["files"]:
        total_hits = (
            len(file_record["pyvista_hits"])
            + len(file_record["xml_hits"])
            + len(file_record["raw_text_hits"])
        )

        if total_hits > 0:
            any_hit = True
            print(f"{file_record['file']}: {total_hits} keyword hits")

    if not any_hit:
        print("No rack/power/load/heat metadata-like keyword hits found.")

    print(f"\nFull JSON report saved to: {out_path}")

    print("\nInterpretation:")
    print("- If you see fields like rack_id, equipment_name, power, heat_load, source, etc.,")
    print("  then we can modify 02_convert_vtk_to_tensor.py to read them directly.")
    print("- If only Temperature / Pressure / Velocity appear, then the exported result")
    print("  contains CFD output fields but not the original rack power input metadata.")
    print("- If result.vtm contains block names such as rack/fan/wall, we can use those")
    print("  later for Omniverse USD object mapping.")


if __name__ == "__main__":
    main()