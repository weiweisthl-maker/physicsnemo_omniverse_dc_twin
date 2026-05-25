from pathlib import Path

import numpy as np
import pyvista as pv


def export_structured_prediction(
    x,
    y,
    z,
    fields: dict[str, np.ndarray],
    out_vtk: Path,
    out_vtu: Path,
):
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    grid = pv.StructuredGrid(xx, yy, zz)

    for name, values in fields.items():
        grid.point_data[name] = values.ravel(order="F")

    out_vtk.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_vtk)
    grid.cast_to_unstructured_grid().save(out_vtu)

