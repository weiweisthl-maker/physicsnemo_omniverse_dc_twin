import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Project paths
# ============================================================

DATA_DIR = Path("data/tensors")
MODEL_DIR = Path("models")
META_DIR = Path("data/metadata")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "best_field_model.pt"
STATS_PATH = META_DIR / "dataset_stats.json"


# ============================================================
# Training settings
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 1
EPOCHS = 300
LR = 1e-3

# Current tensor shape from 02:
# T shape: (96, 48, 32)
# Input channels:
#   channel 0 = heat field
#   channel 1 = valid mask


# ============================================================
# Dataset
# ============================================================

class FieldDataset(Dataset):
    def __init__(self, files, stats):
        self.files = [Path(f) for f in files]
        self.stats = stats

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])

        heat = data["heat"].astype(np.float32)
        T = data["T"].astype(np.float32)
        valid = data["valid"].astype(np.float32)

        # Normalize input heat
        heat_n = (heat - self.stats["heat_mean"]) / self.stats["heat_std"]

        # Normalize target T
        # Only valid region is physically meaningful, but tensor keeps full grid.
        T_n = (T - self.stats["T_mean"]) / self.stats["T_std"]

        # Input: [C, X, Y, Z]
        x = np.stack([heat_n, valid], axis=0).astype(np.float32)

        # Output: [1, X, Y, Z]
        y = T_n[None, ...].astype(np.float32)

        # Mask: [1, X, Y, Z]
        mask = valid[None, ...].astype(np.float32)

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(mask),
        )


# ============================================================
# Small 3D UNet
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Small3DUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base=16):
        super().__init__()

        self.enc1 = ConvBlock(in_channels, base)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = ConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.bottleneck = ConvBlock(base * 2, base * 4)

        self.up2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.dec2 = ConvBlock(base * 4 + base * 2, base * 2)

        self.up1 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.dec1 = ConvBlock(base * 2 + base, base)

        self.out = nn.Conv3d(base, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        b = self.bottleneck(p2)

        u2 = self.up2(b)
        if u2.shape[-3:] != e2.shape[-3:]:
            u2 = torch.nn.functional.interpolate(
                u2,
                size=e2.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )

        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        if u1.shape[-3:] != e1.shape[-3:]:
            u1 = torch.nn.functional.interpolate(
                u1,
                size=e1.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )

        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        return self.out(d1)


# ============================================================
# Stats and loss
# ============================================================

def compute_stats(files):
    """
    Compute dataset statistics using only valid regions.
    This is important because invalid sampled regions are filled with T=0.
    """

    heat_values = []
    temp_values = []

    for f in files:
        data = np.load(f)

        heat = data["heat"].astype(np.float32)
        T = data["T"].astype(np.float32)
        valid = data["valid"].astype(np.float32)

        valid_mask = valid > 0.5

        # For heat, including zeros is acceptable because zero means no heat source.
        heat_values.append(heat.reshape(-1))

        # For temperature, only valid fluid region should be used.
        temp_values.append(T[valid_mask].reshape(-1))

    heat_all = np.concatenate(heat_values)
    temp_all = np.concatenate(temp_values)

    stats = {
        "heat_mean": float(np.mean(heat_all)),
        "heat_std": float(np.std(heat_all) + 1e-6),
        "T_mean": float(np.mean(temp_all)),
        "T_std": float(np.std(temp_all) + 1e-6),
    }

    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\nDataset stats:")
    print(json.dumps(stats, indent=2))

    return stats


def masked_mse(pred, target, mask):
    """
    MSE only on valid sampled CFD region.
    """
    loss = ((pred - target) ** 2) * mask
    return loss.sum() / (mask.sum() + 1e-6)


def denormalized_mae(pred_n, target_n, mask, stats):
    """
    Report MAE in Celsius for easier engineering interpretation.
    """
    pred = pred_n * stats["T_std"] + stats["T_mean"]
    target = target_n * stats["T_std"] + stats["T_mean"]

    err = torch.abs(pred - target) * mask
    return err.sum() / (mask.sum() + 1e-6)


# ============================================================
# Training
# ============================================================

def main():
    print("Starting field model training...")
    print(f"Device: {DEVICE}")
    print(f"Data folder: {DATA_DIR}")

    files = sorted(DATA_DIR.glob("case_*.npz"))

    if len(files) < 2:
        raise RuntimeError(
            "At least 2 tensor cases are required.\n"
            "Expected files such as:\n"
            "  data/tensors/case_001.npz\n"
            "  data/tensors/case_002.npz"
        )

    print("\nTensor files found:")
    for f in files:
        print(f"  - {f}")

    # With 2 cases:
    #   train = case_001
    #   val   = case_002
    #
    # With more cases:
    #   last one is validation, rest are training.
    train_files = files[:-1]
    val_files = files[-1:]

    print("\nTraining files:")
    for f in train_files:
        print(f"  - {f}")

    print("\nValidation files:")
    for f in val_files:
        print(f"  - {f}")

    stats = compute_stats(train_files)

    train_ds = FieldDataset(train_files, stats)
    val_ds = FieldDataset(val_files, stats)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model = Small3DUNet(in_channels=2, out_channels=1, base=16).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, EPOCHS + 1):
        # ----------------------------
        # Train
        # ----------------------------
        model.train()
        train_loss_sum = 0.0
        train_mae_sum = 0.0

        for x, y, mask in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            mask = mask.to(DEVICE)

            pred = model(x)

            loss = masked_mse(pred, y, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                mae_c = denormalized_mae(pred, y, mask, stats)

            train_loss_sum += float(loss.item())
            train_mae_sum += float(mae_c.item())

        train_loss = train_loss_sum / len(train_loader)
        train_mae = train_mae_sum / len(train_loader)

        # ----------------------------
        # Validate
        # ----------------------------
        model.eval()
        val_loss_sum = 0.0
        val_mae_sum = 0.0

        with torch.no_grad():
            for x, y, mask in val_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                mask = mask.to(DEVICE)

                pred = model(x)

                loss = masked_mse(pred, y, mask)
                mae_c = denormalized_mae(pred, y, mask, stats)

                val_loss_sum += float(loss.item())
                val_mae_sum += float(mae_c.item())

        val_loss = val_loss_sum / len(val_loader)
        val_mae = val_mae_sum / len(val_loader)

        # ----------------------------
        # Save best model
        # ----------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "stats": stats,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_mae_C": val_mae,
                    "input_channels": ["heat_normalized", "valid_mask"],
                    "output_channels": ["T_normalized"],
                    "grid_shape": [96, 48, 32],
                },
                MODEL_PATH,
            )

        if epoch == 1 or epoch % 25 == 0 or epoch == EPOCHS:
            print(
                f"Epoch {epoch:04d} | "
                f"train_loss={train_loss:.6f} | "
                f"train_MAE={train_mae:.3f} C | "
                f"val_loss={val_loss:.6f} | "
                f"val_MAE={val_mae:.3f} C | "
                f"best_epoch={best_epoch}"
            )

    print("\nTraining finished.")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Saved best model to: {MODEL_PATH}")
    print(f"Saved dataset stats to: {STATS_PATH}")

    print("\nNote:")
    print("You currently have only 2 CFD cases, so this training is only for pipeline validation.")
    print("The model should not yet be interpreted as a reliable predictive surrogate.")


if __name__ == "__main__":
    main()