"""Full model: backbone -> BEV encoder -> decoder head (design doc 8.7)."""
from __future__ import annotations

import torch
import torch.nn as nn

from arrow_toy.models.bev_encoder import GroundPlaneBevEncoder
from arrow_toy.models.backbone import TinyCNN
from arrow_toy.models.decoder import ArrowHead


class ArrowToyModel(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        m = cfg.get("model", cfg)
        channels = m.get("channels", [16, 32, 64, 64])
        hidden = m.get("hidden_dim", 64)
        bev_hw = tuple(m.get("bev_size_hw", [64, 32]))
        bev_range = tuple(m.get("bev_range_xy", [2.0, 18.0, -4.0, 4.0]))
        input_hw = tuple(cfg.get("dataset", cfg)
                         .get("input_size_hw", [192, 320]))
        self.backbone = TinyCNN(channels)
        # lite profile: channels [16,24,48,48] -> BEV neck matches out_ch
        self.bev_encoder = GroundPlaneBevEncoder(
            self.backbone.out_channels, bev_hw, bev_range,
            (input_hw[0], input_hw[1]))
        self.arrow_head = ArrowHead(
            d=hidden,
            num_queries=m.get("num_queries", 32),
            layers=m.get("decoder_layers", 2),
            heads=m.get("attention_heads", 4),
            ffn_dim=m.get("ffn_dim", 128),
            dropout=m.get("dropout", 0.0),
            use_self_attention=m.get("use_self_attention", True))
        # adapter when backbone channels != hidden dim (ablation D)
        if self.backbone.out_channels != hidden:
            self.adapter = nn.Conv2d(self.backbone.out_channels, hidden, 1)
            self.bev_encoder = GroundPlaneBevEncoder(
                hidden, bev_hw, bev_range, (input_hw[0], input_hw[1]))
        else:
            self.adapter = None
        self.hidden = hidden
        # optional auxiliary dense heatmap head on the SAME BEV tokens
        # (experiment F: auxiliary dense supervision to accelerate the
        # sparse head's convergence; primary head remains 32-query)
        self.aux_dense = None
        if m.get("aux_dense", False):
            from arrow_toy.models.dense_head import DenseHeatmapHead
            self.aux_dense = DenseHeatmapHead(
                hidden, bev_hw, bev_range)

    def forward(self, images: torch.Tensor, K: torch.Tensor,
                T_C_E: torch.Tensor,
                camera_valid: torch.Tensor | None = None
                ) -> dict[str, torch.Tensor]:
        feats = self.backbone(images)
        if self.adapter is not None:
            feats = self.adapter(feats)
        bev_feats, bev_valid = self.bev_encoder(feats, K, T_C_E,
                                                camera_valid)
        pe = self.bev_encoder.position_encoding(bev_feats.device)
        assert pe.shape[-1] == bev_feats.shape[-1], \
            f"position encoding C={pe.shape[-1]} != bev C={bev_feats.shape[-1]}"
        out = self.arrow_head(bev_feats, bev_valid, pe)
        if self.aux_dense is not None:
            out["aux_dense"] = self.aux_dense(bev_feats)
        return out


def count_parameters(model: nn.Module) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total}
