from pathlib import Path

import torch


def load_checkpoint(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def save_checkpoint(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)

