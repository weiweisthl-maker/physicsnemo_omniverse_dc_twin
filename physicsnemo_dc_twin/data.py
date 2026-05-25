import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class FieldDataset(Dataset):
    def __init__(self, files, stats):
        self.files = [Path(f) for f in files]
        self.stats = stats

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        heat = data["heat"].astype(np.float32)
        temp = data["T"].astype(np.float32)
        valid = data["valid"].astype(np.float32)

        heat_n = (heat - self.stats["heat_mean"]) / self.stats["heat_std"]
        temp_n = (temp - self.stats["T_mean"]) / self.stats["T_std"]

        x = np.stack([heat_n, valid], axis=0).astype(np.float32)
        y = temp_n[None, ...].astype(np.float32)
        mask = valid[None, ...].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask)


def list_tensor_cases(data_dir: Path):
    return sorted(Path(data_dir).glob("case_*.npz"))


def split_train_val(files):
    if len(files) < 2:
        raise RuntimeError(
            "At least 2 tensor cases are required for train/validation split."
        )
    return files[:-1], files[-1:]


def compute_stats(files, stats_path: Path | None = None):
    heat_values = []
    temp_values = []

    for file_path in files:
        data = np.load(file_path)
        heat = data["heat"].astype(np.float32)
        temp = data["T"].astype(np.float32)
        valid = data["valid"].astype(np.float32) > 0.5
        heat_values.append(heat.reshape(-1))
        temp_values.append(temp[valid].reshape(-1))

    heat_all = np.concatenate(heat_values)
    temp_all = np.concatenate(temp_values)
    stats = {
        "heat_mean": float(np.mean(heat_all)),
        "heat_std": float(np.std(heat_all) + 1e-6),
        "T_mean": float(np.mean(temp_all)),
        "T_std": float(np.std(temp_all) + 1e-6),
    }

    if stats_path is not None:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

    return stats

