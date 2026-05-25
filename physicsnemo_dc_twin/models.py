import os
from pathlib import Path

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
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
    """Pilot 3D UNet kept as the baseline before the PhysicsNeMo model swap."""

    def __init__(self, in_channels: int = 2, out_channels: int = 1, base: int = 16):
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
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        u2 = self.up2(b)
        if u2.shape[-3:] != e2.shape[-3:]:
            u2 = torch.nn.functional.interpolate(
                u2, size=e2.shape[-3:], mode="trilinear", align_corners=False
            )
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        if u1.shape[-3:] != e1.shape[-3:]:
            u1 = torch.nn.functional.interpolate(
                u1, size=e1.shape[-3:], mode="trilinear", align_corners=False
            )
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return self.out(d1)


def build_physicsnemo_unet(config: dict | None = None) -> nn.Module:
    """Build the native PhysicsNeMo 3D UNet backend."""

    config = dict(config or {})
    os.environ.setdefault("WARP_CACHE_PATH", str(Path(".warp-cache").resolve()))

    try:
        from physicsnemo.models.unet import UNet
    except Exception as exc:
        raise RuntimeError(
            "Could not import PhysicsNeMo UNet. Confirm that nvidia-physicsnemo "
            "is installed and that warp-lang is compatible with this platform."
        ) from exc

    model_depth = int(config.get("model_depth", 3))
    feature_map_channels = config.get(
        "feature_map_channels", [16, 16, 32, 32, 64, 64]
    )

    return UNet(
        in_channels=2,
        out_channels=1,
        model_depth=model_depth,
        feature_map_channels=feature_map_channels,
        num_conv_blocks=int(config.get("num_conv_blocks", 2)),
        pooling_type=config.get("pooling_type", "MaxPool3d"),
        normalization=config.get("normalization", "groupnorm"),
        gradient_checkpointing=bool(config.get("gradient_checkpointing", False)),
    )


def build_model(
    name: str = "physicsnemo_unet",
    base_channels: int = 16,
    model_config: dict | None = None,
) -> nn.Module:
    if name == "physicsnemo_unet":
        return build_physicsnemo_unet(model_config)
    if name == "small_3d_unet":
        return Small3DUNet(in_channels=2, out_channels=1, base=base_channels)
    raise ValueError(
        "Unsupported model "
        f"'{name}'. Supported models: physicsnemo_unet, small_3d_unet"
    )
