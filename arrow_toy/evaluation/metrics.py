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


def _pt_seg_dist(px: float, py: float, ax: float, ay: float,
                 bx: float, by: float) -> float:
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _pt_outline_dist(px: float, py: float,
                     poly: list[tuple[float, float]]) -> float:
    d = min(_pt_seg_dist(px, py, poly[i][0], poly[i][1],
                         poly[(i + 1) % len(poly)][0], poly[(i + 1) % len(poly)][1])
            for i in range(len(poly)))
    return d


def match_and_score(pred: dict, targets: dict, thresholds_m: list[float],
                    bev_range, out: dict,
                    excluded: torch.Tensor | None = None,
                    sample_ids: list[str] | None = None,
                    ignore_outlines_fn=None) -> None:
    """Accumulate per-frame greedy match results for each distance
    threshold. Confidence sweep happens in summarize().

    pred accepts either raw model outputs (pred_logits/center/direction)
    or precomputed foreground probabilities via pred["probs_fg"] /
    pred["best_cls"] (used by offline recomputation - never fabricate
    logits). `excluded` [B,Q] removes queries from ALL counting.
    `ignore_outlines_fn(b, thr)` returns world-space polygon outlines of
    ignored arrows for sample b; UNMATCHED predictions within thr of an
    outline are suppressed (removed from FP counting, design 11.2 -
    applied AFTER valid-GT matching, per distance threshold)."""
    if "probs_fg" in pred:
        fg = pred["probs_fg"].float()
        best_prob, best_cls_t = fg.max(-1)
        best_cls = best_cls_t
    else:
        probs = pred["pred_logits"].float().softmax(-1)
        fg = probs[..., :NUM_CLASSES]
        best_prob, best_cls_t = fg.max(-1)
        best_cls = best_cls_t
    B, Q = best_prob.shape[:2]
    centers = pred["pred_center_norm"].float()
    if "dir_norm" in pred:
        dir_norm = pred["dir_norm"].float()
    else:
        raw_dir = pred["pred_direction_raw"].float()
        dir_norm = raw_dir / raw_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    x_min, x_max, y_min, y_max = bev_range
    for b in range(B):
        valid = targets["mask"][b] > 0
        n_gt = int(valid.sum())
        gt_cls = targets["cls"][b][valid]
        gt_ctr = targets["center"][b][valid]
        order = best_prob[b].argsort(descending=True)
        b_excl = (excluded[b] > 0).tolist() if excluded is not None \
            else [False] * Q
        for thr in thresholds_m:
            outlines = (ignore_outlines_fn(b, thr)
                        if ignore_outlines_fn else [])
            recs = []
            used_gt = set()
            for qi in order.tolist():
                if b_excl[qi]:
                    continue  # removed from matching AND counting
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
                    recs.append({"p": p, "cls": c, "matched": True,
                                 "err": math.hypot(px - gx, py - gy),
                                 "ang": ang_err_deg(pyaw, yaw)})
                else:
                    sup = any(_pt_outline_dist(px, py, o) <= thr
                              for o in outlines)
                    if sup:
                        out["suppressed"] = out.get("suppressed", 0) + 1
                    recs.append({"p": p, "cls": c, "matched": False,
                                 "sup": sup})
            out.setdefault(f"thr_{thr}", []).append(
                {"recs": recs, "n_gt": n_gt,
                 "gt_cls": [int(v) for v in gt_cls.tolist()]})
        out["empty_n"] = out.get("empty_n", 0) + (1 if n_gt == 0 else 0)
        out["preds"].append({
            "sample": sample_ids[b] if sample_ids else None,
            "probs": [float(v) for v in best_prob[b]],
            "cls": [int(c) for c in best_cls[b]],
            "center_norm": [[float(x), float(y)] for x, y in centers[b]],
            "dir": [[float(x), float(y)] for x, y in dir_norm[b]],
        })


def _counts_at_conf(frames: list[dict], conf: float,
                    cls_filter: int | None = None) -> tuple[int, int, int,
                                                            list[float],
                                                            list[float]]:
    """TP/FP/FN + matched errors at a confidence cutoff.

    A GT is detected (TP) only when matched by a query with p >= conf;
    GT matched only below the cutoff is FN (regression-tested).
    """
    tp = fp = fn = 0
    errs, angs = [], []
    for fr in frames:
        matched_above = 0
        for r in fr["recs"]:
            if r.get("sup"):
                continue  # suppressed by 11.2: outside P/R counting
            if cls_filter is not None and r["cls"] != cls_filter:
                continue
            if r["matched"]:
                if r["p"] >= conf:
                    tp += 1
                    matched_above += 1
                    errs.append(r["err"])
                    angs.append(r["ang"])
            elif r["p"] >= conf:
                fp += 1
        if cls_filter is None:
            fn += fr["n_gt"] - matched_above
        else:
            fn += sum(1 for c in fr.get("gt_cls", [])
                      if c == cls_filter) - matched_above
    return tp, fp, fn, errs, angs


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if prec + rec else 0.0
    return prec, rec, f1


def summarize(scores: dict, thresholds_m: list[float],
              select_conf: bool = True,
              frozen_conf: float | None = None) -> dict:
    """Compute P/R/F1 etc. for each distance threshold.

    select_conf=True: choose the confidence threshold from
    CONF_CANDIDATES by best F1 at the FIRST distance threshold (val
    selection semantics, design 11.1). frozen_conf: use this threshold
    (test evaluation must freeze from val, never tune here)."""
    out = {}
    frames0 = scores.get(f"thr_{thresholds_m[0]}", [])
    if frozen_conf is not None:
        chosen_conf = float(frozen_conf)
        out["chosen_conf_threshold"] = chosen_conf
        out["conf_frozen_from_val"] = True
    else:
        chosen_conf = 0.5
        if select_conf:
            best_f1 = -1.0
            for c in CONF_CANDIDATES:
                tp, fp, fn, _, _ = _counts_at_conf(frames0, c)
                _, _, f1 = _prf(tp, fp, fn)
                if f1 > best_f1:
                    best_f1 = f1
                    chosen_conf = c
        out["chosen_conf_threshold"] = chosen_conf

    for thr in thresholds_m:
        frames = scores.get(f"thr_{thr}", [])
        for label, conf in (("sel", chosen_conf), ("p50", 0.5)):
            tp, fp, fn, errs, angs = _counts_at_conf(frames, conf)
            prec, rec, f1 = _prf(tp, fp, fn)
            out[f"precision@{thr}_{label}"] = prec
            out[f"recall@{thr}_{label}"] = rec
            out[f"f1@{thr}_{label}"] = f1
            out[f"center_err_m@{thr}_{label}"] = (
                sum(errs) / len(errs)) if errs else None
            out[f"ang_err_deg@{thr}_{label}"] = (
                sum(angs) / len(angs)) if angs else None
        # canonical keys (selected/frozen threshold)
        out[f"precision@{thr}"] = out[f"precision@{thr}_sel"]
        out[f"recall@{thr}"] = out[f"recall@{thr}_sel"]
        out[f"f1@{thr}"] = out[f"f1@{thr}_sel"]
        out[f"center_err_m@{thr}"] = out[f"center_err_m@{thr}_sel"]
        out[f"ang_err_deg@{thr}"] = out[f"ang_err_deg@{thr}_sel"]
        # macro F1: per-class F1 averaged over classes with GT (11.1)
        per_class_f1 = []
        for c in range(NUM_CLASSES):
            n_gt_c = sum(1 for fr in frames
                         for g in fr.get("gt_cls", []) if g == c)
            if n_gt_c == 0:
                per_class_f1.append(None)
                continue
            tp_c, fp_c, fn_c, _, _ = _counts_at_conf(frames, chosen_conf,
                                                    cls_filter=c)
            _, _, f1_c = _prf(tp_c, fp_c, fn_c)
            per_class_f1.append(f1_c)
        vals = [v for v in per_class_f1 if v is not None]
        out[f"macro_f1@{thr}"] = (sum(vals) / len(vals)) if vals else 0.0
        out[f"per_class_f1@{thr}"] = [
            round(v, 4) if v is not None else None for v in per_class_f1]

    # empty-frame FP at the chosen confidence
    if scores.get("empty_n"):
        fp = 0
        conf = chosen_conf
        for fr in scores.get(f"thr_{thresholds_m[0]}", []):
            if fr["n_gt"] == 0:
                fp += sum(1 for r in fr["recs"]
                          if r["p"] >= conf and not r.get("sup"))
        out["empty_frame_fp_avg"] = fp / scores["empty_n"]
    if scores.get("suppressed"):
        out["suppressed_predictions"] = scores["suppressed"]

    # per-class AP with 101-point interpolation (design 11.1); ranking by
    # confidence, TPs are matched predictions of that class
    for thr in thresholds_m:
        frames = scores.get(f"thr_{thr}", [])
        gt_per_class = [0] * NUM_CLASSES
        for fr in frames:
            for c in fr.get("gt_cls", []):
                gt_per_class[c] += 1
        ap_per_class = []
        for c in range(NUM_CLASSES):
            if gt_per_class[c] == 0:
                ap_per_class.append(None)
                continue
            preds = sorted(
                (r for fr in frames for r in fr["recs"]
                 if r["cls"] == c and not r.get("sup")),
                key=lambda r: -r["p"])
            tp_cum = 0
            fp_cum = 0
            pts = []
            for r in preds:
                if r["matched"]:
                    tp_cum += 1
                else:
                    fp_cum += 1
                pts.append((tp_cum / gt_per_class[c],
                            tp_cum / (tp_cum + fp_cum)))
            # 101-point interpolated precision
            ap = 0.0
            for k in range(101):
                r_k = k / 100.0
                p_int = max((p for r, p in pts if r >= r_k), default=0.0)
                ap += p_int / 101.0
            ap_per_class.append(ap)
        valid = [a for a in ap_per_class if a is not None]
        out[f"ap_per_class@{thr}"] = [round(a, 4) if a is not None else None
                                      for a in ap_per_class]
        out[f"map@{thr}"] = (sum(valid) / len(valid)) if valid else 0.0
        out[f"zero_gt_classes@{thr}"] = sum(
            1 for a in ap_per_class if a is None)
    return out
