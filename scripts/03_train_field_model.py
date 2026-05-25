import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physicsnemo_dc_twin.physicsnemo_bridge import physicsnemo_status
from physicsnemo_dc_twin.train import train_field_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the pilot field surrogate on Cadence-derived tensors."
    )
    parser.add_argument(
        "--config",
        default="configs/field_model.json",
        help="JSON training config path.",
    )
    parser.add_argument("--epochs", type=int, help="Override epoch count.")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if args.epochs is not None:
        config["epochs"] = args.epochs

    status = physicsnemo_status()
    print("PhysicsNeMo status:", status["message"])
    train_field_model(config)


if __name__ == "__main__":
    main()
