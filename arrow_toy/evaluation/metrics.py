"""Independent detection metrics (design doc 11.1/11.2).

Greedy same-class matching by descending confidence at each distance
threshold is threshold-cutoff invariant, so the pass records
(p, matched, errors) once per query and the summary sweeps the
CONFIDENCE candidates. The F1 confidence threshold is SELECTED on val
from the preset candidates (never on test).
"""
from __future__ import annotations

import math

import torch

NUM_CLASSES = 6
NO_OBJECT = 6
CONF_CANDIDATES = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def ang_err_deg(a: float, b: float) -> float:
    d = math.atan2(math.sin(a - b), math.cos(a - b))
    return abs(math.degrees(d))


def match_and_score(pred: dict, targets: dict, thresholds_m: list[float],
                    bev_range, out: dict) -> None:
    """Accumulate per-frame greedy match results for each distance
    threshold. Confidence sweep happens in summarize()."""
    logits = pred["pred_logits"].float()
    probs = logits.softmax(-1)                    # [B,Q,7]
    B, Q = probs.shape[:2]
    fg = probs[..., :NUM_CLASSES]
    best_prob, best_cls = fg.max(-1)              # [B,Q]
    centers = pred["pred_center_norm"].float()
    raw_dir = pred["pred_direction_raw"].float()
    dir_norm = raw_dir / raw_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    x_min, x_max, y_min, y_max = bev_range
    for b in range(B):
        valid = targets["mask"][b] > 0
        n_gt = int(valid.sum())
        gt_cls = targets["cls"][b][valid]
        gt_ctr = targets["center"][b][valid]
        order = best_prob[b].argsort(descending=True)
        for thr in thresholds_m:
            recs = []
            used_gt = set()
            for qi in order.tolist():
                p = float(best_prob[b][qi])
                c = int(best_cls[b][qi])
                px = x_min + float(centers[b][qi][0]) * (x_max - x_min)
                py = y_min + float(centers[b][qi][1]) * (y_max - y_min)
                cand = None
                best_d = None
                for gi in range(n_gt):
                    if gi in used_gt or int(gt_cls[gi]) != c:
                        continue
                    gx = x_min + float(gt_ctr[gi][0]) * (x_max - x_min)
                    gy = y_min + float(gt_ctr[gi][1]) * (y_max - y_min)
                    d = math.hypot(px - gx, py - gy)
                    if d <= thr and (best_d is None or d < best_d):
                        cand, best_d = gi, d
                if cand is not None:
                    used_gt.add(cand)
                    gx = x_min + float(gt_ctr[cand][0]) * (x_max - x_min)
                    gy = y_min + float(gt_ctr[cand][1]) * (y_max - y_min)
                    yaw = math.atan2(float(targets["dir"][b][valid][cand][1]),
                                     float(targets["dir"][b][valid][cand][0]))
                    pyaw = math.atan2(float(dir_norm[b][qi][1]),
                                      float(dir_norm[b][qi][0]))
                    recs.append({"p": p, "matched": True,
                                 "err": math.hypot(px - gx, py - gy),
                                 "ang": ang_err_deg(pyaw, yaw)})
                else:
                    recs.append({"p": p, "matched": False})
            out.setdefault(f"thr_{thr}", []).append(
                {"recs": recs, "n_gt": n_gt})
        out["empty_n"] = out.get("empty_n", 0) + (1 if n_gt == 0 else 0)
        out["preds"].append({
            "sample": None,
            "probs": [round(float(v), 4) for v in best_prob[b]],
            "cls": [int(c) for c in best_cls[b]],
            "center_norm": [[round(float(x), 4), round(float(y), 4)]
                            for x, y in centers[b]],
            "dir": [[round(float(x), 4), round(float(y), 4)]
                    for x, y in dir_norm[b]],
        })


def summarize(scores: dict, thresholds_m: list[float],
              select_conf: bool = True) -> dict:
    """Compute P/R/F1 etc. for each distance threshold; the confidence
    threshold is chosen from CONF_CANDIDATES by best F1 at the FIRST
    distance threshold (val-selection semantics)."""
    out = {}
    chosen_conf = 0.5
    if select_conf:
        best_f1 = -1.0
        for c in CONF_CANDIDATES:
            tp = fp = fn = 0
            for fr in scores.get(f"thr_{thresholds_m[0]}", []):
                for r in fr["recs"]:
                    if r["p"] >= c:
                        if r["matched"]:
                            tp += 1
                        else:
                            fp += 1
                fn += fr["n_gt"] - sum(1 for r in fr["recs"] if r["matched"])
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if prec + rec else 0.0
            if f1 > best_f1:
                best_f1 = f1
                chosen_conf = c
    out["chosen_conf_threshold"] = chosen_conf

    for thr in thresholds_m:
        frames = scores.get(f"thr_{thr}", [])
        for label, conf in (("sel", chosen_conf), ("p50", 0.5)):
            tp = fp = fn = 0
            errs, angs = [], []
            for fr in frames:
                matched_gt = 0
                for r in fr["recs"]:
                    if r["matched"]:
                        matched_gt += 1
                        if r["p"] >= conf:
                            tp += 1
                            errs.append(r["err"])
                            angs.append(r["ang"])
                    elif r["p"] >= conf:
                        fp += 1
                fn += fr["n_gt"] - matched_gt
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if prec + rec else 0.0
            out[f"precision@{thr}_{label}"] = prec
            out[f"recall@{thr}_{label}"] = rec
            out[f"f1@{thr}_{label}"] = f1
            out[f"center_err_m@{thr}_{label}"] = (
                sum(errs) / len(errs)) if errs else None
            out[f"ang_err_deg@{thr}_{label}"] = (
                sum(angs) / len(angs)) if angs else None
        # canonical keys (selected threshold) for downstream selection
        out[f"precision@{thr}"] = out[f"precision@{thr}_sel"]
        out[f"recall@{thr}"] = out[f"recall@{thr}_sel"]
        out[f"f1@{thr}"] = out[f"f1@{thr}_sel"]
        out[f"center_err_m@{thr}"] = out[f"center_err_m@{thr}_sel"]
        out[f"ang_err_deg@{thr}"] = out[f"ang_err_deg@{thr}_sel"]

    # empty-frame FP at the chosen confidence
    if scores.get("empty_n"):
        fp = 0
        conf = chosen_conf
        for fr in scores.get(f"thr_{thresholds_m[0]}", []):
            if fr["n_gt"] == 0:
                fp += sum(1 for r in fr["recs"] if r["p"] >= conf)
        out["empty_frame_fp_avg"] = fp / scores["empty_n"]
    return out
