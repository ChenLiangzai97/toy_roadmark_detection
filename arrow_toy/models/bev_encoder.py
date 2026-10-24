"""Ground-plane BEV encoder (design doc 8.3/8.4).

Projects BEV cell centers into the image with K / T_C_E, grid_samples the
backbone feature map, applies a depthwise-separable neck and returns
flattened tokens [B, H_bev*W_bev, C] plus the geometric validity mask.
The projection grid is cached per unique calibration key.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BevGridCache:
    """Cache sampling grids keyed by the full projection contract."""

    def __init__(self) -> None:
        self._grids: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def key(K: torch.Tensor, T_C_E: torch.Tensor, feat_hw: tuple[int, int],
            img_hw: tuple[int, int], bev_range, bev_hw) -> tuple:
        return (tuple(K.flatten().tolist()), tuple(T_C_E.flatten().tolist()),
                feat_hw, img_hw, tuple(bev_range), bev_hw,
                "grid_sample_bilinear_zeros_align_false")

    def get(self, key, device: torch.device):
        hit = self._grids.get(key)
        if hit is not None:
            return hit[0].to(device), hit[1].to(device)
        return None

    def put(self, key, grid: torch.Tensor, valid: torch.Tensor) -> None:
        self._grids[key] = (grid.cpu(), valid.cpu())


_GRID_CACHE = BevGridCache()


def build_sampling_grid(K: torch.Tensor, T_C_E: torch.Tensor,
                        bev_range: tuple[float, float, float, float],
                        bev_hw: tuple[int, int],
                        img_hw: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (grid [Hb,Wb,2] in [-1,1] for grid_sample, valid [Hb,Wb]).

    BEV contract: row r -> x = x_max - (r+0.5)*dx, col c ->
    y = y_max - (c+0.5)*dy (design doc 4.4).
    """
    H_bev, W_bev = bev_hw
    H_img, W_img = img_hw
    x_min, x_max, y_min, y_max = bev_range
    device = K.device
    r = torch.arange(H_bev, device=device, dtype=torch.float32)
    c = torch.arange(W_bev, device=device, dtype=torch.float32)
    x = x_max - (r + 0.5) * ((x_max - x_min) / H_bev)      # [H_bev]
    y = y_max - (c + 0.5) * ((y_max - y_min) / W_bev)      # [W_bev]
    xs = x.view(H_bev, 1).expand(H_bev, W_bev)
    ys = y.view(1, W_bev).expand(H_bev, W_bev)
    ones = torch.ones_like(xs)
    zeros = torch.zeros_like(xs)
    # ego homogeneous [4, H*W]
    pts = torch.stack([xs, ys, zeros, ones], dim=-1).view(-1, 4).t()
    T = T_C_E.float()
    P = (K.float() @ T[:3, :])                       # [3,4]
    uvw = P @ pts                                    # [3, H*W]
    z = uvw[2]
    in_front = z > 1e-6
    u = uvw[0] / z.clamp_min(1e-6)
    v = uvw[1] / z.clamp_min(1e-6)
    grid = torch.stack([2 * u / W_img - 1, 2 * v / H_img - 1], dim=-1)
    grid = grid.view(H_bev, W_bev, 2)
    valid = (in_front & (u >= 0) & (u <= W_img) & (v >= 0) & (v <= H_img))
    valid = valid.view(H_bev, W_bev)
    return grid, valid


class GroundPlaneBevEncoder(nn.Module):
    def __init__(self, in_channels: int, bev_hw: tuple[int, int],
                 bev_range: tuple[float, float, float, float],
                 img_hw: tuple[int, int]) -> None:
        super().__init__()
        self.bev_hw = bev_hw
        self.bev_range = bev_range
        self.img_hw = img_hw
        self.out_channels = in_channels
        # depthwise separable neck (8.4)
        self.neck = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1,
                      groups=in_channels),
            nn.GroupNorm(min(8, in_channels), in_channels), nn.SiLU(),
            nn.Conv2d(in_channels, in_channels, 1), nn.GroupNorm(
                min(8, in_channels), in_channels), nn.SiLU())
        self._pe: torch.Tensor | None = None

    def position_encoding(self, device: torch.device) -> torch.Tensor:
        """2D sinusoidal encoding from metric BEV cell positions [HW, C]."""
        H, W = self.bev_hw
        C = self.out_channels
        if self._pe is None or self._pe.device != device:
            x_min, x_max, y_min, y_max = self.bev_range
            r = torch.arange(H, device=device, dtype=torch.float32)
            c = torch.arange(W, device=device, dtype=torch.float32)
            x = x_max - (r + 0.5) * ((x_max - x_min) / H)
            y = y_max - (c + 0.5) * ((y_max - y_min) / W)
            xs = x.view(H, 1).expand(H, W).reshape(-1)
            ys = y.view(1, W).expand(H, W).reshape(-1)
            half = C // 4  # split C among x/y sin/cos
            pe = []
            for coords in (xs, ys):
                for k in range(half):
                    w = 1.0 / (10000 ** (2 * k / max(1, 2 * half)))
                    pe.append(torch.sin(coords * w))
                    pe.append(torch.cos(coords * w))
            pe_t = torch.stack(pe, dim=-1)                     # [HW, 4*half]
            if pe_t.shape[1] < C:
                pe_t = F.pad(pe_t, (0, C - pe_t.shape[1]))
            self._pe = pe_t[:, :C]
        return self._pe

    def forward(self, image_features: torch.Tensor, K: torch.Tensor,
                T_C_E: torch.Tensor,
                camera_valid: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """image_features [B,C,Hf,Wf] -> (tokens [B,HW,C], valid [B,HW]).

        grid_sample runs in FP32 outside autocast (design doc 8.3).
        """
        B, C, Hf, Wf = image_features.shape
        tokens = []
        valids = []
        for b in range(B):
            key = _GRID_CACHE.key(K[b], T_C_E[b], (Hf, Wf),
                                  self.img_hw, self.bev_range, self.bev_hw)
            hit = _GRID_CACHE.get(key, image_features.device)
            if hit is None:
                grid, valid = build_sampling_grid(
                    K[b], T_C_E[b], self.bev_range, self.bev_hw, self.img_hw)
                _GRID_CACHE.put(key, grid, valid)
            else:
                grid, valid = hit
            feat = image_features[b:b + 1].float()
            sampled = F.grid_sample(
                feat, grid.unsqueeze(0), mode="bilinear",
                padding_mode="zeros", align_corners=False)
            tokens.append(sampled[0])          # [C,H,W]
            v = valid
            if camera_valid is not None and not bool(camera_valid[b]):
                v = torch.zeros_like(v)
            valids.append(v)
        stacked = torch.stack(tokens, dim=0)    # [B,C,H,W] float32
        bev = self.neck(stacked)
        B2, C2, H2, W2 = bev.shape
        flat = bev.flatten(2).transpose(1, 2)   # [B, HW, C] row-major r*W+c
        valid = torch.stack(valids, dim=0).flatten(1)  # [B, HW]
        if not bool(valid.any()):
            raise RuntimeError(
                "no valid BEV cells: check camera / BEV range configuration")
        return flat, valid
