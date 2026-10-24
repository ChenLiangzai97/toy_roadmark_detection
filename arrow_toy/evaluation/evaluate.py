"""Evaluation CLI: run detection metrics on a chosen split for a run dir.

    python -m arrow_toy.evaluation.evaluate --run-dir runs/baseline_s42 --split test
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from arrow_toy.data.dataset import ArrowCacheDataset, make_collate
from arrow_toy.evaluation.metrics import match_and_score, summarize
from arrow_toy.geometry.templates import template_for_class, world_bbox
from arrow_toy.models.model import ArrowToyModel
from arrow_toy.utils.config import load_yaml


def ignore_regions(labels: dict, margin_m: float):
    """BEV boxes of ignored arrows inflated by the distance threshold
    (design 11.2). Predictions inside are removed from P/R/AP counting."""
    regions = []
    bev = labels["bev_range"]
    for a in labels["arrows"]:
        if a["arrow_valid"] or a["ignore_reason"] in (None, "outside_bev",
                                                      "outside_camera"):
            continue
        t = template_for_class(a["class_id"])
        x0, x1, y0, y1 = world_bbox(t, a["center_ego"][:2], a["yaw_rad"],
                                    a["scale"])
        regions.append((x0 - margin_m, x1 + margin_m,
                        y0 - margin_m, y1 + margin_m))
    return regions


def in_regions(x: float, y: float, regions) -> bool:
    return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, x1, y0, y1 in regions)

REPO = pathlib.Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--split", default="test_clean")
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--output", default=None)
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
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"],
                        shuffle=False, num_workers=2, collate_fn=make_collate)
    thresholds = cfg["eval"]["center_thresholds_m"]
    bev = cfg["model"]["bev_range_xy"]
    # confidence threshold FIXED from the run's val selection (no test tuning)
    chosen = None
    res_path = run_dir / "result.json"
    if res_path.exists():
        chosen = json.loads(res_path.read_text()).get("metrics", {}).get(
            "chosen_conf_threshold")
    scores = {"preds": []}
    n_suppressed = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=device == "cuda"):
                pred = model(batch["image"], batch["K"], batch["T_C_E"])
            targets = {k: batch[k] for k in
                       ("cls", "center", "dir", "mask",
                        "orientation_valid", "allow_no_object")}
            # 11.2: drop predictions inside ignored-arrow regions (per
            # sample, using the ORIGINAL labels, class not required)
            thr = thresholds[0]
            probs = pred["pred_logits"].float().softmax(-1)
            for b in range(batch["image"].shape[0]):
                labels = json.loads(
                    (root / "samples" / batch["sample_ids"][b]
                     / "labels.json").read_text())
                regions = ignore_regions(labels, thr)
                if not regions:
                    continue
                x_min, x_max, y_min, y_max = bev
                cn = pred["pred_center_norm"][b]
                xs = x_min + cn[:, 0] * (x_max - x_min)
                ys = y_min + cn[:, 1] * (y_max - y_min)
                keep = torch.ones(xs.shape[0], dtype=torch.bool,
                                  device=xs.device)
                for qi in range(xs.shape[0]):
                    if in_regions(float(xs[qi]), float(ys[qi]), regions):
                        keep[qi] = False
                        n_suppressed += 1
                pred["pred_logits"][b] = torch.where(
                    keep.unsqueeze(-1), pred["pred_logits"][b],
                    torch.full_like(pred["pred_logits"][b], -1e9))
            match_and_score(pred, targets, thresholds, bev, scores)
    metrics = summarize(scores, thresholds, select_conf=(chosen is None))
    if chosen is not None:
        # recompute at the frozen val-chosen confidence
        metrics_sel = summarize(scores, thresholds, select_conf=True)
        metrics["chosen_conf_threshold"] = chosen
        metrics["note"] = ("confidence threshold frozen from val; "
                           "metrics@0.5 use it")
        del metrics_sel
    # attach sample ids to predictions
    out = {
        "run": str(run_dir),
        "split": args.split,
        "checkpoint": args.checkpoint,
        "metrics": metrics,
        "ignored_region_suppressed_predictions": n_suppressed,
    }
    out_path = pathlib.Path(
        args.output or run_dir / f"eval_{args.split}.json")
    out_path.write_text(json.dumps(out, indent=1))
    print(json.dumps(metrics, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
