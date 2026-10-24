"""Recompute metrics for finished runs from saved predictions (no GPU).

Rebuilds the greedy match from predictions_val.jsonl + dataset targets and
rewrites result.json metrics with the FIXED evaluator (FN now counts GT
matched only below the confidence cutoff).

    python -m arrow_toy.tools_recompute_metrics --runs runs/baseline_s42 ...
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from arrow_toy.evaluation.metrics import match_and_score, summarize  # noqa: E402


def recompute(run_dir: pathlib.Path) -> dict:
    import yaml
    cfg = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
    root = REPO / cfg["dataset"]["root"]
    split = cfg["dataset"]["val_split"]
    preds = [json.loads(l) for l in
             (run_dir / "predictions_val.jsonl").read_text().splitlines()
             if l.strip()]
    tgt = np.load(root / "cache" / f"targets_{split}.npz")
    thresholds = cfg["eval"]["center_thresholds_m"]
    bev = cfg["model"]["bev_range_xy"]
    # overfit's val ran with val_limit=32 -> only the first 32 samples
    n = len(preds)
    limit = len(tgt["sample_ids"])
    assert n <= limit, (n, limit)
    scores = {"preds": []}
    for b, p in enumerate(preds[:limit]):
        logits = torch.full((1, 32, 7), -8.0)
        for qi, (pr, cls) in enumerate(zip(p["probs"], p["cls"])):
            logits[0, qi, cls] = float(np.log(max(pr, 1e-6) /
                                              max(1 - pr, 1e-6)))
        pred = {
            "pred_logits": logits,
            "pred_center_norm": torch.tensor([p["center_norm"]]),
            "pred_direction_raw": torch.tensor([p["dir"]]),
        }
        targets = {
            "cls": torch.from_numpy(tgt["class_ids"][b:b + 1].astype(np.int64)),
            "center": torch.from_numpy(tgt["centers_norm"][b:b + 1]),
            "dir": torch.stack([torch.cos(torch.from_numpy(
                tgt["yaws"][b:b + 1].astype(np.float32))),
                torch.sin(torch.from_numpy(
                    tgt["yaws"][b:b + 1].astype(np.float32)))], dim=-1),
            "mask": torch.from_numpy(tgt["masks"][b:b + 1]),
            "orientation_valid": torch.from_numpy(
                tgt["masks"][b:b + 1]),
            "allow_no_object": torch.from_numpy(
                tgt["allow_no_object"][b:b + 1].astype(bool)),
        }
        match_and_score(pred, targets, thresholds, bev, scores)
    metrics = summarize(scores, thresholds)
    res_path = run_dir / "result.json"
    res = json.loads(res_path.read_text())
    old = {k: res["metrics"].get(k) for k in
           ("f1@0.5", "precision@0.5", "recall@0.5")}
    res["metrics"] = metrics
    res["metrics_recomputed"] = {
        "reason": "FN bug fix: GT matched only below the confidence cutoff"
                  " now counts as FN",
        "before": old,
        "after": {k: metrics.get(k) for k in
                  ("f1@0.5", "precision@0.5", "recall@0.5")},
    }
    res_path.write_text(json.dumps(res, indent=1))
    return {"run": run_dir.name, "before": old,
            "after": {k: metrics.get(k) for k in
                      ("f1@0.5", "precision@0.5", "recall@0.5",
                       "chosen_conf_threshold")}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    args = ap.parse_args(argv)
    for r in args.runs:
        print(json.dumps(recompute(pathlib.Path(r)), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
