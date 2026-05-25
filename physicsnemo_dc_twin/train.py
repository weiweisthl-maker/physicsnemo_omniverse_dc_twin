import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .checkpoints import save_checkpoint
from .data import FieldDataset, compute_stats, list_tensor_cases, split_train_val
from .models import build_model


def masked_mse(pred, target, mask):
    loss = ((pred - target) ** 2) * mask
    return loss.sum() / (mask.sum() + 1e-6)


def denormalized_mae(pred_n, target_n, mask, stats):
    pred = pred_n * stats["T_std"] + stats["T_mean"]
    target = target_n * stats["T_std"] + stats["T_mean"]
    err = torch.abs(pred - target) * mask
    return err.sum() / (mask.sum() + 1e-6)


def train_field_model(config):
    data_dir = Path(config["data_dir"])
    model_path = Path(config["model_path"])
    stats_path = Path(config["stats_path"])
    epochs = int(config.get("epochs", 300))
    batch_size = int(config.get("batch_size", 1))
    lr = float(config.get("learning_rate", 1e-3))
    base_channels = int(config.get("base_channels", 16))
    model_name = config.get("model_name", "physicsnemo_unet")
    model_config = config.get("model", {})
    device = "cuda" if torch.cuda.is_available() else "cpu"

    files = list_tensor_cases(data_dir)
    train_files, val_files = split_train_val(files)
    stats = compute_stats(train_files, stats_path)

    print("Starting field model training...")
    print(f"Device: {device}")
    print(f"Training files: {[str(f) for f in train_files]}")
    print(f"Validation files: {[str(f) for f in val_files]}")
    print("Dataset stats:")
    print(json.dumps(stats, indent=2))

    if epochs <= 0:
        print("\nSmoke check finished.")
        print("Epoch count is 0, so no optimizer step or checkpoint write was performed.")
        print(f"Saved dataset stats to: {stats_path}")
        return

    train_loader = DataLoader(
        FieldDataset(train_files, stats),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        FieldDataset(val_files, stats),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model = build_model(
        name=model_name,
        base_channels=base_channels,
        model_config=model_config,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_mae_sum = 0.0

        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            pred = model(x)
            loss = masked_mse(pred, y, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                train_mae_sum += float(denormalized_mae(pred, y, mask, stats).item())
            train_loss_sum += float(loss.item())

        model.eval()
        val_loss_sum = 0.0
        val_mae_sum = 0.0
        with torch.no_grad():
            for x, y, mask in val_loader:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                pred = model(x)
                val_loss_sum += float(masked_mse(pred, y, mask).item())
                val_mae_sum += float(denormalized_mae(pred, y, mask, stats).item())

        train_loss = train_loss_sum / len(train_loader)
        train_mae = train_mae_sum / len(train_loader)
        val_loss = val_loss_sum / len(val_loader)
        val_mae = val_mae_sum / len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_checkpoint(
                model_path,
                {
                    "model_state": model.state_dict(),
                    "stats": stats,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_mae_C": val_mae,
                    "model_name": model_name,
                    "base_channels": base_channels,
                    "model_config": model_config,
                    "input_channels": ["heat_normalized", "valid_mask"],
                    "output_channels": ["T_normalized"],
                    "grid_shape": list(next(iter(train_loader))[0].shape[-3:]),
                    "backend": "physicsnemo" if model_name == "physicsnemo_unet" else "torch_pilot",
                },
            )

        if epoch == 1 or epoch % int(config.get("log_every", 25)) == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:04d} | train_loss={train_loss:.6f} | "
                f"train_MAE={train_mae:.3f} C | val_loss={val_loss:.6f} | "
                f"val_MAE={val_mae:.3f} C | best_epoch={best_epoch}"
            )

    print("\nTraining finished.")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Saved best model to: {model_path}")
    print(f"Saved dataset stats to: {stats_path}")
