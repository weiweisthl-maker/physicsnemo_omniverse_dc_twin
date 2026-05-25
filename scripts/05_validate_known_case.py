import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physicsnemo_dc_twin.validation import export_validation_case


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export true/predicted/error fields for a known CFD tensor case."
    )
    parser.add_argument("--model", default="models/best_physicsnemo_unet.pt")
    parser.add_argument("--case", default="data/tensors/case_002.npz")
    parser.add_argument("--out-dir", default="outputs/validation_vtk")
    return parser.parse_args()


def main():
    args = parse_args()
    export_validation_case(
        model_path=Path(args.model),
        tensor_case=Path(args.case),
        out_dir=Path(args.out_dir),
    )
    print("\nNext step:")
    print("Open the validation VTU in ParaView and compare T_true_C, T_pred_C, and T_error_C.")


if __name__ == "__main__":
    main()
