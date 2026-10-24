"""Independent detection metrics (design doc 11.1/11.2).

Matching for metrics is NOT the training Hungarian: predictions are ranked
by confidence, matched to same-class valid GT within the distance
threshold. Ignore-region suppression is applied only for evaluation of
frames that contain ignored arrows.
"""
from __future__ import annotations

import math

import torch

NUM_CLASSES = 6
NO_OBJECT = 6


def _denorm(center_norm, bev_range):
    x_min, x_max, y_min, y_max = bev_range
    x = x_min + center_norm[..., 0] * (x_max - x_min)
    y = y_min + center_norm[..., 1] * (y_max - y_min)
    return x, y


def ang_err_deg(a: float, b: float) -> float:
    d = math.atan2(math.sin(a - b), math.cos(a - b))
    return abs(math.degrees(d))


def match_and_score(pred: dict, targets: dict, thresholds_m: list[float],
                    bev_range, out: dict) -> None:
    """Accumulate per-batch TP/FP/FN and errors for each threshold."""
    logits = pred["pred_logits"].float()
    probs = logits.softmax(-1)                    # [B,Q,7]
    B, Q = probs.shape[:2]
    feat = probs[..., :NUM_CLASSES]               # foreground classes only
    best_prob, best_cls = feat.max(-1)            # [B,Q]
    centers = pred["pred_center_norm"].float()
    raw_dir = pred["pred_direction_raw"].float()
    dir_norm = raw_dir / raw_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    for b in range(B):
        valid = targets["mask"][b] > 0
        gt_cls = targets["cls"][b][valid]
        gt_ctr = targets["center"][b][valid]
        order = best_prob[b].argsort(descending=True)
        for thr in thresholds_m:
            # same-class greedy matching by descending confidence
            used_gt = set()
            rec = {"tp": 0, "fp": 0, "fn": 0, "center_err": [], "ang": [],
                   "dup": 0}
            for qi in order.tolist():
                p = float(best_prob[b][qi])
                c = int(best_cls[b][qi])
                # denormalize
                x_min, x_max, y_min, y_max = bev_range
                px = x_min + float(centers[b][qi][0]) * (x_max - x_min)
                py = y_min + float(centers[b][qi][1]) * (y_max - y_min)
                # candidate gts: same class, unused, within thr
                cand = None
                for gi in range(int(valid.sum())):
                    if gi in used_gt or int(gt_cls[gi]) != c:
                        continue
                    gx = x_min + float(gt_ctr[gi][0]) * (x_max - x_min)
                    gy = y_min + float(gt_ctr[gi][1]) * (y_max - y_min)
                    if math.hypot(px - gx, py - gy) <= thr:
                        cand = gi
                        break
                if cand is not None:
                    used_gt.add(cand)
                    rec["tp"] += 1
                    gx = x_min + float(gt_ctr[cand][0]) * (x_max - x_min)
                    gy = y_min + float(gt_ctr[cand][1]) * (y_max - y_min)
                    rec["center_err"].append(math.hypot(px - gx, py - gy))
                    yaw = math.atan2(float(targets["dir"][b][valid][cand][1]),
                                     float(targets["dir"][b][valid][cand][0]))
                    pyaw = math.atan2(float(dir_norm[b][qi][1]),
                                      float(dir_norm[b][qi][0]))
                    rec["ang"].append(ang_err_deg(pyaw, yaw))
                elif p >= 0.5:
                    rec["fp"] += 1
            rec["fn"] = int(valid.sum()) - len(used_gt)
            key = f"thr_{thr}"
            out.setdefault(key, []).append(rec)
        # empty-frame false positives (threshold = first)
        if int(valid.sum()) == 0:
            out["empty_n"] += 1
            out["empty_fp"] += int((best_prob[b] >= 0.5).sum())
        out["preds"].append({
            "sample": None,
            "probs": [round(float(v), 4) for v in best_prob[b]],
            "cls": [int(c) for c in best_cls[b]],
            "center_norm": [[round(float(x), 4), round(float(y), 4)]
                            for x, y in centers[b]],
            "dir": [[round(float(x), 4), round(float(y), 4)]
                    for x, y in dir_norm[b]],
        })


def summarize(scores: dict, thresholds_m: list[float]) -> dict:
    out = {}
    for thr in thresholds_m:
        recs = scores.get(f"thr_{thr}", [])
        tp = sum(r["tp"] for r in recs)
        fp = sum(r["fp"] for r in recs)
        fn = sum(r["fn"] for r in recs)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if prec + rec else 0.0
        errs = [e for r in recs for e in r["center_err"]]
        angs = [a for r in recs for a in r["ang"]]
        out[f"precision@{thr}"] = prec
        out[f"recall@{thr}"] = rec
        out[f"f1@{thr}"] = f1
        out[f"center_err_m@{thr}"] = (sum(errs) / len(errs)) if errs else None
        out[f"ang_err_deg@{thr}"] = (sum(angs) / len(angs)) if angs else None
        out[f"matched@{thr}"] = tp
        out[f"recall_only@{thr}"] = rec
    if scores.get("empty_n"):
        out["empty_frame_fp_avg"] = scores["empty_fp"] / scores["empty_n"]
    return out
