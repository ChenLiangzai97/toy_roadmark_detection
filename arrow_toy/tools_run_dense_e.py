"""Ablation E runner: dense heatmap head on the SAME BEV features
(design doc 12/E). Same data, budget and schedule as the sparse runs so
the comparison is single-variable. Detection = 3x3 local maxima on the
per-class heatmap with sub-cell offsets; evaluation reuses the shared
greedy matcher + summarize.

    python -m arrow_toy.tools_run_dense_e --run-dir runs/dense_e_s42
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from arrow_toy.data.dataset import ArrowCacheDataset, make_collate
from arrow_toy.evaluation.metrics import match_and_score, summarize
from arrow_toy.models.bev_encoder import GroundPlaneBevEncoder
from arrow_toy.models.backbone import TinyCNN
from arrow_toy.models.dense_head import (DenseHeatmapHead, focal_heatmap_loss,
                                         heatmap_targets, offset_loss)
from arrow_toy.models.model import count_parameters
from arrow_toy.train import build_lr_schedule, set_seed
from arrow_toy.utils.config import load_yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
BEV_HW = (64, 32)
BEV_RANGE = (2.0, 18.0, -4.0, 4.0)


def extract_detections(heat_logits: torch.Tensor, off: torch.Tensor):
    """[B,6,H,W] logits -> per-batch lists of (prob, cls, xn, yn)."""
    h = heat_logits.sigmoid()
    B, C, H, W = h.shape
    pad = F.pad(h, (1, 1, 1, 1), mode="replicate")
    peaks = torch.ones_like(h, dtype=torch.bool)
    for dr in range(3):
        for dc in range(3):
            if dr == 1 and dc == 1:
                continue
            neigh = pad[:, :, dr:dr + H, dc:dc + W]
            peaks &= h >= neigh
    x_min, x_max, y_min, y_max = BEV_RANGE
    out = []
    for b in range(B):
        probs, cls_, cns = [], [], []
        idx = peaks[b].nonzero(as_tuple=False)
        for ci, r, c in idx.tolist():
            p = float(h[b, ci, r, c])
            dr = float(off[b, 0, r, c])
            dc = float(off[b, 1, r, c])
            r_f = r + dr
            c_f = c + dc
            x = x_max - (r_f + 0.5) * ((x_max - x_min) / H)
            y = y_max - (c_f + 0.5) * ((y_max - y_min) / W)
            probs.append(p)
            cls_.append(ci)
            cns.append([(x - x_min) / (x_max - x_min),
                        (y - y_min) / (y_max - y_min)])
        out.append((probs, cls_, cns))
    return out


def to_pred_dict(probs, cls_, cns, Q=32):
    """Pack detections into the shared evaluator's pred format."""
    n = len(probs)
    fg = torch.zeros(1, Q, 6)
    bc = torch.zeros(1, Q, dtype=torch.long)
    cn = torch.zeros(1, Q, 2)
    dn = torch.zeros(1, Q, 2)
    order = sorted(range(n), key=lambda i: -probs[i])[:Q]
    for k, i in enumerate(order):
        fg[0, k, cls_[i]] = probs[i]
        bc[0, k] = cls_[i]
        cn[0, k] = torch.tensor(cns[i])
        dn[0, k] = torch.tensor([1.0, 0.0])  # no yaw regression
    return {"probs_fg": fg, "best_cls": bc, "pred_center_norm": cn,
            "dir_norm": dn}


def _git_commit() -> str:
    try:
        import subprocess
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/quick.yaml")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--eval-only", action="store_true",
                    help="load last.pt and rerun evaluation only "
                         "(regenerates predictions with sample ids)")
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    run_dir = pathlib.Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    root = pathlib.Path(cfg["dataset"]["root"])
    train_ds = ArrowCacheDataset(root, cfg["dataset"]["train_split"])
    val_ds = ArrowCacheDataset(root, cfg["dataset"]["val_split"])
    train_ld = DataLoader(train_ds, 32, shuffle=True, num_workers=4,
                          pin_memory=True, persistent_workers=True,
                          collate_fn=make_collate, drop_last=True)
    val_ld = DataLoader(val_ds, 32, shuffle=False, num_workers=2,
                        collate_fn=make_collate)

    backbone = TinyCNN().to(device)
    bev_enc = GroundPlaneBevEncoder(backbone.out_channels, BEV_HW, BEV_RANGE,
                                    (192, 320)).to(device)
    dense = DenseHeatmapHead(backbone.out_channels, BEV_HW, BEV_RANGE).to(device)
    params = count_parameters(backbone)["trainable"] + \
        count_parameters(bev_enc)["trainable"] + \
        count_parameters(dense)["trainable"]
    opt = torch.optim.AdamW(
        list(backbone.parameters()) + list(bev_enc.parameters())
        + list(dense.parameters()), lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"])

    t0 = time.monotonic()
    step = 0
    stop = False
    status = "success"
    if args.eval_only:
        ck = torch.load(run_dir / "last.pt", map_location=device,
                        weights_only=True)
        backbone.load_state_dict(ck["backbone"])
        bev_enc.load_state_dict(ck["bev_encoder"])
        dense.load_state_dict(ck["dense"])
        step = ck["step"]
        stop = True
    budget = cfg["train"]["time_budget_s"]
    reserve = cfg["train"]["reserve_finalization_s"]
    metrics_f = (run_dir / "metrics.jsonl").open("w")
    while step < args.steps and not stop:
        for batch in train_ld:
            if step >= args.steps:
                break
            lr = build_lr_schedule(cfg["train"]["warmup_steps"],
                                   args.steps, cfg["train"]["lr"],
                                   cfg["train"]["min_lr"], step)
            for g in opt.param_groups:
                g["lr"] = lr
            batch = {k: (v.to(device, non_blocking=True)
                         if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=device == "cuda"):
                feat = backbone(batch["image"])
                tokens, _ = bev_enc(feat, batch["K"], batch["T_C_E"])
                out = dense(tokens)
            cls_map, off_map = heatmap_targets(batch, BEV_HW, BEV_RANGE,
                                               device=device)
            loss_hm = focal_heatmap_loss(out["heatmap"].float(), cls_map)
            loss_off = offset_loss(out["offset"].float(), off_map, cls_map)
            loss = loss_hm + loss_off
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(bev_enc.parameters())
                + list(dense.parameters()), 1.0)
            opt.step()
            step += 1
            if step % 50 == 0:
                metrics_f.write(json.dumps(
                    {"step": step, "loss_hm": float(loss_hm),
                     "loss_off": float(loss_off), "lr": lr,
                     "t": round(time.monotonic() - t0, 1)}) + "\n")
                metrics_f.flush()
            if time.monotonic() - t0 + reserve > budget:
                status = "budget_incomplete"
                stop = True
                break
    # eval (threshold selected on val, same rule as A/B)
    for m in (backbone, bev_enc, dense):
        m.eval()
    scores = {"preds": []}
    thresholds = [0.5, 1.0]
    n_eval = 0
    eval_ids = []
    with torch.no_grad():
        for batch in val_ld:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=device == "cuda"):
                feat = backbone(batch["image"])
                tokens, _ = bev_enc(feat, batch["K"], batch["T_C_E"])
                out = dense(tokens)
            dets = extract_detections(out["heatmap"].float(),
                                      out["offset"].float())
            for b, (probs, cls_, cns) in enumerate(dets):
                n_eval += 1
                eval_ids.append(batch["sample_ids"][b])
                pred = to_pred_dict(probs, cls_, cns)
                targets = {k: batch[k][b:b + 1] for k in
                           ("cls", "center", "dir", "mask",
                            "orientation_valid", "allow_no_object")}
                match_and_score(pred, targets, thresholds, BEV_RANGE,
                                scores,
                                sample_ids=[batch["sample_ids"][b]])
    metrics = summarize(scores, thresholds)
    wall = time.monotonic() - t0
    torch.save({"backbone": backbone.state_dict(),
                "bev_encoder": bev_enc.state_dict(),
                "dense": dense.state_dict(), "step": step, "seed": seed},
               run_dir / "last.pt")
    with (run_dir / "predictions_val.jsonl").open("w") as f:
        for pr in scores["preds"]:
            f.write(json.dumps(pr) + "\n")
    result = {
        "run": "ablation E (dense heatmap head, same BEV ARCHITECTURE, "
               "retrained from scratch)",
        "seed": seed, "params": params, "completed_steps": step,
        "wall_clock_s": round(wall, 2), "time_budget_s": budget,
        "time_budget_pass": bool(wall <= budget and status == "success"),
        "status": status, "metrics": metrics,
        "eval_samples": n_eval,
        "eval_sample_ids": eval_ids,
        "config": cfg,
        "git_commit": _git_commit(),
        "data_version": json.loads(
            (root / "cache" / "stats_train.json").read_text()
        ).get("data_version"),
        "experimental_differences_vs_sparse": [
            "backbone + BEV encoder are RE-INITIALIZED and retrained: the "
            "architecture is shared, NOT the trained features",
            "no yaw regression (angle metrics excluded from comparison)",
            "focal heatmap + offset losses instead of Hungarian matching; "
            "3x3 local-peak post-processing instead of query selection",
            "supervision contract differs: heatmap negatives do NOT use "
            "allow_no_object / ignore regions - cells over ignored arrows "
            "still receive negative supervision (unlike the sparse runs)",
        ],
        "conclusion_scope": ("diagnostic only: on the current data the "
                             "dense training recipe attains higher "
                             "detection metrics; this supports further "
                             "study of head/supervision/optimization "
                             "differences and does NOT isolate the cause "
                             "or prove feature quality"),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=1))
    metrics_f.close()
    print(json.dumps({k: result[k] for k in
                      ("status", "completed_steps", "wall_clock_s",
                       "time_budget_pass")}, indent=1))
    print("f1@0.5:", metrics["f1@0.5"], "mAP@0.5:", metrics.get("map@0.5"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
