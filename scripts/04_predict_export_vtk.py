import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physicsnemo_dc_twin.infer import predict_temperature


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict a thermal field and export NPZ/VTK/VTU visualization files."
    )
    parser.add_argument("--model", default="models/best_physicsnemo_unet.pt")
    parser.add_argument("--reference", default="data/tensors/case_001.npz")
    parser.add_argument("--rack-loads", default="new_rack_loads.csv")
    parser.add_argument("--out-dir", default="outputs/predicted_vtk")
    parser.add_argument("--name", default="pred_case_new")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    predict_temperature(
        model_path=Path(args.model),
        reference_tensor=Path(args.reference),
        rack_loads_csv=Path(args.rack_loads),
        out_npz=out_dir / f"{args.name}.npz",
        out_vtk=out_dir / f"{args.name}.vtk",
        out_vtu=out_dir / f"{args.name}.vtu",
    )
    print("\nNext step:")
    print(f"Open {out_dir / (args.name + '.vtu')} in ParaView and color by 'T_pred_C'.")


if __name__ == "__main__":
    main()
