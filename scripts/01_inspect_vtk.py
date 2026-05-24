import pyvista as pv
from pathlib import Path

vtk_path = Path("data/raw_cadence_vtk/case_001/result.vtk")

mesh = pv.read(vtk_path)

print("Mesh type:", type(mesh))
print("Number of points:", mesh.n_points)
print("Number of cells:", mesh.n_cells)
print("Bounds:", mesh.bounds)

print("\nPoint data arrays:")
for name in mesh.point_data.keys():
    arr = mesh.point_data[name]
    print(f"  {name}: shape={arr.shape}, min={arr.min():.4g}, max={arr.max():.4g}")

print("\nCell data arrays:")
for name in mesh.cell_data.keys():
    arr = mesh.cell_data[name]
    print(f"  {name}: shape={arr.shape}, min={arr.min():.4g}, max={arr.max():.4g}")