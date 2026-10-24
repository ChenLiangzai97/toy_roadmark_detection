"""Tiny CNN backbone (design doc 8.2).

Input [B,3,192,320] -> stride-4 feature [B,64,48,80].
Non-overlapping AvgPool keeps feature-pixel centers exact:
stride-4 feature column j maps to source continuous x = 4*(j+0.5).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _gn(channels: int) -> nn.GroupNorm:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    raise ValueError(f"cannot normalize {channels} channels")


class TinyCNN(nn.Module):
    def __init__(self, channels: list[int] | None = None,
                 in_ch: int = 3) -> None:
        super().__init__()
        c = channels or [16, 32, 64, 64]
        # stage1 / stage2: conv -> GN -> SiLU -> AvgPool2x2
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_ch, c[0], 3, stride=1, padding=1), _gn(c[0]),
            nn.SiLU(), nn.AvgPool2d(2))
        self.stage2 = nn.Sequential(
            nn.Conv2d(c[0], c[1], 3, stride=1, padding=1), _gn(c[1]),
            nn.SiLU(), nn.AvgPool2d(2))
        # stage3: two stride-1 conv blocks (no pooling)
        self.stage3 = nn.Sequential(
            nn.Conv2d(c[1], c[2], 3, stride=1, padding=1), _gn(c[2]),
            nn.SiLU(),
            nn.Conv2d(c[2], c[3], 3, stride=1, padding=1), _gn(c[3]),
            nn.SiLU())
        self.out_channels = c[3]
        self.stride = 4

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images [B,3,H,W] normalized -> [B,C,H/4,W/4]."""
        return self.stage3(self.stage2(self.stage1(images)))
