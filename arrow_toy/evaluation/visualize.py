"""Prediction overlays for fixed samples (design doc 11.3).

Draws camera-image re-projections and BEV views with GT / predictions /
ignore reasons. Saves PNGs into <run>/figures/.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw

from arrow_toy.data.dataset import ArrowCacheDataset, make_collate
from arrow_toy.geometry.projection import project_points
from arrow_toy.models.model import ArrowToyModel
from arrow_toy.utils.config import load_yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
CLASS_COLORS = [(230, 57, 70), (29, 53, 87), (42, 157, 143),
                (231, 111, 81), (94, 96, 206), (188, 71, 73)]


def draw_bev(size=(320, 640), bev_range=(2., 18., -4., 4.)) -> Image.Image:
    # image: W=640 (y span), H=320 (x span); row0=far top, col0=left
    img = Image.new("RGB", (size[1], size[0]), (24, 26, 30))
    d = ImageDraw.Draw(img)
    x_min, x_max, y_min, y_max = bev_range
    for gx in range(int(x_min), int(x_max) + 1, 2):
        r = (x_max - gx) / (x_max - x_min) * size[0]
        d.line([(0, r), (size[1], r)], fill=(45, 48, 54), width=1)
    for gy in range(int(y_min), int(y_max) + 1, 2):
        c = (y_max - gy) / (y_max - y_min) * size[1]
        d.line([(c, 0), (c, size[0])], fill=(45, 48, 54), width=1)
    return img


def bev_px(x, y, bev_range, size=(320, 640)):
    x_min, x_max, y_min, y_max = bev_range
    r = (x_max - x) / (x_max - x_min) * size[0]
    c = (y_max - y) / (y_max - y_min) * size[1]
    return c, r


def visualize_sample(sample_dir: pathlib.Path, pred: dict,
                     conf: float, out_path: pathlib.Path,
                     split_ds=None, index=None) -> None:
    labels = json.loads((sample_dir / "labels.json").read_text())
    rgb = Image.open(sample_dir / "rgb" / "front.png").convert("RGB")
    bev = draw_bev()
    db, dv = ImageDraw.Draw(rgb), ImageDraw.Draw(bev)
    bev_range = (labels["bev_range"]["x_min"], labels["bev_range"]["x_max"],
                 labels["bev_range"]["y_min"], labels["bev_range"]["y_max"])
    K = labels["cameras"]["front"]["K"]
    T = labels["cameras"]["front"]["T_C_E"]
    # GT
    for a in labels["arrows"]:
        x, y = a["center_ego"][0], a["center_ego"][1]
        col = CLASS_COLORS[a["class_id"]]
        if a["arrow_valid"]:
            c, r = bev_px(x, y, bev_range)
            dv.ellipse([c - 6, r - 6, c + 6, r + 6], outline=col, width=3)
            yaw = a["yaw_rad"]
            dv.line([c, r, c + 18 * math.cos(yaw) * -1,
                     r + 18 * math.sin(yaw) * -1], fill=col, width=2)
            uv, _ok = project_points([[x, y, 0.0]], K, T)
            u, v = uv[0]
            if math.isfinite(u) and math.isfinite(v):
                db.ellipse([u - 5, v - 5, u + 5, v + 5], outline=col, width=2)
        else:
            c, r = bev_px(x, y, bev_range)
            dv.rectangle([c - 5, r - 5, c + 5, r + 5], outline=(120, 120, 120))
            dv.text((c + 8, r - 6), a["ignore_reason"], fill=(160, 160, 160))
    # predictions
    if pred:
        for p, cls, cn, dxy in zip(pred["probs"], pred["cls"],
                                   pred["center_norm"], pred["dir"]):
            if p < conf:
                continue
            col = CLASS_COLORS[cls % 6]
            x = bev_range[0] + cn[0] * (bev_range[1] - bev_range[0])
            y = bev_range[2] + cn[1] * (bev_range[3] - bev_range[2])
            c, r = bev_px(x, y, bev_range)
            dv.ellipse([c - 9, r - 9, c + 9, r + 9], outline=col, width=2)
            dv.line([c, r, c + 22 * -dxy[0], r + 22 * -dxy[1]], fill=col, width=2)
            uv, _ok = project_points([[x, y, 0.0]], K, T)
            u, v = uv[0]
            if math.isfinite(u) and math.isfinite(v):
                db.ellipse([u - 8, v - 8, u + 8, v + 8], outline=col, width=2)
    W = rgb.width + bev.width + 12
    canvas = Image.new("RGB", (W, max(rgb.height, bev.height)), (10, 10, 10))
    canvas.paste(rgb, (0, 0))
    canvas.paste(bev, (rgb.width + 12, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main(argv=None) -> int:
    import torch
    from torch.utils.data import DataLoader
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--count", type=int, default=24)
    ap.add_argument("--checkpoint", default="best.pt")
    args = ap.parse_args(argv)

    run_dir = pathlib.Path(args.run_dir)
    cfg = load_yaml(run_dir / "resolved_config.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ArrowToyModel(cfg).to(device)
    ckpt = torch.load(run_dir / args.checkpoint, map_location=device,
                      weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    root = pathlib.Path(cfg["dataset"]["root"])
    ds = ArrowCacheDataset(root, args.split)
    res = json.loads((run_dir / "result.json").read_text())
    conf = res["metrics"].get("chosen_conf_threshold", 0.5)
    ids = (root / "splits" / f"{args.split}.txt").read_text().split()
    # spread sample choice deterministically
    idxs = list(range(0, len(ids), max(1, len(ids) // args.count)))[:args.count]
    out_dir = run_dir / "figures"
    loader = DataLoader(torch.utils.data.Subset(ds, idxs), batch_size=8,
                        collate_fn=make_collate)
    k = 0
    with torch.no_grad():
        for batch in loader:
            bdev = "cuda" if device == "cuda" else "cpu"
            batch = {kk: (v.to(bdev) if torch.is_tensor(v) else v)
                     for kk, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=bdev == "cuda"):
                pred = model(batch["image"], batch["K"], batch["T_C_E"])
            probs = pred["pred_logits"].float().softmax(-1)[..., :6]
            bp, bc = probs.max(-1)
            for b in range(batch["image"].shape[0]):
                sid = batch["sample_ids"][b]
                p = {
                    "probs": bp[b].tolist(), "cls": bc[b].tolist(),
                    "center_norm": pred["pred_center_norm"][b].tolist(),
                    "dir": (pred["pred_direction_raw"][b]
                            / pred["pred_direction_raw"][b].norm(
                                dim=-1, keepdim=True).clamp_min(1e-6)).tolist(),
                }
                visualize_sample(root / "samples" / sid, p, conf,
                                 out_dir / f"{args.split}_{k:02d}_{sid}.png")
                k += 1
    print(f"saved {k} figures to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
