"""32-query transformer decoder + three heads (design doc 8.5-8.7)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

NO_OBJECT_CLASS = 6
NUM_CLASSES = 6
NUM_QUERIES = 32


class PreLNDecoderLayer(nn.Module):
    def __init__(self, d: int, heads: int, ffn_dim: int, dropout: float,
                 use_self_attention: bool = True) -> None:
        super().__init__()
        self.use_self_attention = use_self_attention
        self.ln_self = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, heads, dropout=dropout,
                                               batch_first=True)
        self.ln_cross = nn.LayerNorm(d)
        self.cross_attn = nn.MultiheadAttention(d, heads, dropout=dropout,
                                                batch_first=True)
        self.ln_ffn = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn_dim), nn.GELU(),
                                 nn.Linear(ffn_dim, d))

    def forward(self, h: torch.Tensor, q: torch.Tensor,
                memory: torch.Tensor, p: torch.Tensor,
                memory_valid: torch.Tensor) -> torch.Tensor:
        if self.use_self_attention:
            a = self.ln_self(h)
            h = h + self.self_attn(a + q, a + q, a, need_weights=False)[0]
        a = self.ln_cross(h)
        h = h + self.cross_attn(
            a + q, memory + p, memory,
            key_padding_mask=~memory_valid, need_weights=False)[0]
        h = h + self.ffn(self.ln_ffn(h))
        return h


class ArrowHead(nn.Module):
    def __init__(self, d: int = 64, num_queries: int = NUM_QUERIES,
                 layers: int = 2, heads: int = 4, ffn_dim: int = 128,
                 dropout: float = 0.0, use_self_attention: bool = True) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.query_embed = nn.Parameter(
            torch.randn(num_queries, d) * 0.02)
        self.memory_ln = nn.LayerNorm(d)
        self.layers = nn.ModuleList([
            PreLNDecoderLayer(d, heads, ffn_dim, dropout,
                              use_self_attention=use_self_attention)
            for _ in range(layers)])
        self.final_ln = nn.LayerNorm(d)
        self.cls_branch = nn.Linear(d, NUM_CLASSES + 1)  # 6 + no-object
        self.center_branch = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2), nn.Sigmoid())
        self.orient_branch = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.LayerNorm)):
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, bev_features: torch.Tensor,
                bev_valid: torch.Tensor,
                position_encoding: torch.Tensor
                ) -> dict[str, torch.Tensor]:
        """bev_features [B,HW,C] (pre-LN applied inside), valid [B,HW]."""
        memory = self.memory_ln(bev_features)
        p = position_encoding.unsqueeze(0)
        B = memory.shape[0]
        q = self.query_embed.unsqueeze(0).expand(B, -1, -1)
        h = memory.new_zeros(B, self.num_queries, memory.shape[-1])
        for layer in self.layers:
            h = layer(h, q, memory, p, bev_valid)
        h = self.final_ln(h)
        return {
            "pred_logits": self.cls_branch(h),          # [B,32,7]
            "pred_center_norm": self.center_branch(h),  # [B,32,2] in [0,1]
            "pred_direction_raw": self.orient_branch(h),  # [B,32,2]
        }


def safe_normalize(v: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize orientation vectors; flag near-zero ones as invalid."""
    norm = v.float().norm(dim=-1)
    valid = norm > eps
    out = v.float() / norm.clamp_min(eps).unsqueeze(-1)
    return out, valid


def direction_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
