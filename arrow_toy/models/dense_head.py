"""Dense heatmap head (design doc ablation E).

Uses the SAME BEV feature tokens as the sparse head; predicts per-class
heatmaps + offset regression on the BEV grid. Purpose: check whether the
task bottleneck is the sparse query head (matching churn) or the BEV
features (design doc 12/E, 15.2).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 6


class DenseHeatmapHead(nn.Module):
    def __init__(self, in_channels: int, bev_hw: tuple[int, int],
                 bev_range: tuple[float, float, float, float]) -> None:
        super().__init__()
        self.bev_hw = bev_hw
        self.bev_range = bev_range
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(min(8, in_channels), in_channels), nn.SiLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(min(8, in_channels), in_channels), nn.SiLU(),
            nn.Conv2d(in_channels, NUM_CLASSES, 1))
        self.off = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1), nn.SiLU(),
            nn.Conv2d(in_channels, 2, 1), nn.Tanh())

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        B, HW, C = tokens.shape
        H, W = self.bev_hw
        x = tokens.transpose(1, 2).reshape(B, C, H, W)
        heat = self.net(x)                       # [B,6,H,W] logits
        off = self.off(x) * 0.5                  # [B,2,H,W] cell offsets
        return {"heatmap": heat, "offset": off}


def heatmap_targets(batch: dict, bev_hw: tuple[int, int],
                    bev_range: tuple[float, float, float, float],
                    radius_cells: int = 2, device="cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """Gaussian-ish box targets: class maps [B,6,H,W] and offsets [B,2,H,W]."""
    B = batch["cls"].shape[0]
    H, W = bev_hw
    x_min, x_max, y_min, y_max = bev_range
    cls_map = torch.zeros(B, NUM_CLASSES, H, W, device=device)
    off_map = torch.zeros(B, 2, H, W, device=device)
    for b in range(B):
        for i in range(batch["cls"].shape[1]):
            if float(batch["mask"][b, i]) <= 0:
                continue
            c = int(batch["cls"][b, i])
            xn, yn = batch["center"][b, i].tolist()
            x = x_min + xn * (x_max - x_min)
            y = y_min + yn * (y_max - y_min)
            r_f = (x_max - x) / (x_max - x_min) * H - 0.5
            c_f = (y_max - y) / (y_max - y_min) * W - 0.5
            r0, c0 = int(round(r_f)), int(round(c_f))
            for dr in range(-radius_cells, radius_cells + 1):
                for dc in range(-radius_cells, radius_cells + 1):
                    rr, cc = r0 + dr, c0 + dc
                    if 0 <= rr < H and 0 <= cc < W:
                        w = math.exp(-(dr * dr + dc * dc)
                                     / (2.0 * (radius_cells / 1.5) ** 2))
                        cls_map[b, c, rr, cc] = torch.maximum(
                            cls_map[b, c, rr, cc],
                            torch.tensor(w, device=device))
            if 0 <= r0 < H and 0 <= c0 < W:
                off_map[b, 0, r0, c0] = r_f - r0
                off_map[b, 1, r0, c0] = c_f - c0
    return cls_map, off_map


def focal_heatmap_loss(pred_logits: torch.Tensor,
                       target: torch.Tensor, alpha: float = 2.0,
                       beta: float = 4.0) -> torch.Tensor:
    """CenterNet-style focal loss on per-class heatmaps."""
    p = pred_logits.sigmoid()
    pos = target >= 0.99
    neg = ~pos
    p_t = torch.where(pos, p, 1 - p)
    w = torch.where(pos, 1 - target, 1 + beta * (target - p).pow(alpha).clamp(max=1))
    w_neg = torch.where(neg, 1 + beta * ((1 - target) - (1 - p)).clamp(min=0).pow(alpha), torch.zeros_like(w))
    w = torch.where(neg, w_neg, torch.ones_like(w))
    loss = -(p_t.clamp_min(1e-6).log()) * w
    # normalize by positives (at least 1)
    return loss.sum() / pos.sum().clamp_min(1.0)


def offset_loss(pred_off: torch.Tensor, target_off: torch.Tensor,
                cls_map: torch.Tensor) -> torch.Tensor:
    mask = cls_map >= 0.99
    if not mask.any():
        return pred_off.sum() * 0.0
    return F.l1_loss(pred_off[mask], target_off[mask])
