"""Evaluation CLI: run detection metrics on a chosen split for a run dir.

    python -m arrow_toy.evaluation.evaluate --run-dir runs/baseline_s42 --split test_clean

Confidence threshold is FROZEN from the run's val selection (never tuned
here). Ignore-region suppression follows design 11.2: applied AFTER
valid-GT matching, per distance threshold, using the actual template
outlines inflated by that threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import torch
from torch.utils.data import DataLoader

from arrow_toy.data.dataset import ArrowCacheDataset, make_collate
from arrow_toy.evaluation.metrics import match_and_score, summarize
from arrow_toy.geometry.templates import template_for_class, world_polygons
from arrow_toy.models.model import ArrowToyModel
from arrow_toy.utils.config import load_yaml

IGNORE_FOR_REGIONS = {"occluded", "truncated", "too_small",
                      "no_visual_evidence", "overlapping_arrow"}


def data_version(root: pathlib.Path) -> str:
    """Hash of manifest + audit binding the dataset version."""
    h = hashlib.sha256()
    h.update((root / "manifest.json").read_bytes())
    h.update((root / "audit.json").read_bytes())
    return h.hexdigest()[:16]


def make_ignore_outlines_fn(labels_by_id: dict, sample_ids: list[str]):
    """Returns fn(b, thr) -> list of world polygon outlines for ignored
    arrows of sample b (design 11.2)."""
    cache: dict[tuple[str, float], list] = {}

    def fn(b: int, thr: float) -> list:
        sid = sample_ids[b]
        key = (sid, thr)
        if key in cache:
            return cache[key]
        outlines = []
        labels = labels_by_id.get(sid)
        if labels:
            for a in labels["arrows"]:
                if a["arrow_valid"] or a["ignore_reason"] not in IGNORE_FOR_REGIONS:
                    continue
                t = template_for_class(a["class_id"])
                polys = world_polygons(t, a["center_ego"][:2],
                                       a["yaw_rad"], a["scale"])
                outlines.extend(polys)
        cache[key] = outlines
        return outlines

    return fn


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
    ids = (root / "splits" / f"{args.split}.txt").read_text().split()
    labels_by_id = {}
    for sid in ids:
        labels_by_id[sid] = json.loads(
            (root / "samples" / sid / "labels.json").read_text())
    outlines_fn = make_ignore_outlines_fn(labels_by_id, ds.sample_ids)
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"],
                        shuffle=False, num_workers=2, collate_fn=make_collate)
    thresholds = cfg["eval"]["center_thresholds_m"]
    bev = cfg["model"]["bev_range_xy"]

    # threshold policy: VAL selects fresh (it is the selection set);
    # any other split must freeze from the run's val evaluation and
    # NEVER tune here (design 11.1)
    frozen = None
    if args.split != cfg["dataset"]["val_split"]:
        val_eval = run_dir / "eval_val.json"
        if val_eval.exists():
            frozen = json.loads(val_eval.read_text())["metrics"].get(
                "chosen_conf_threshold")
        else:
            res_path = run_dir / "result.json"
            if res_path.exists():
                frozen = json.loads(res_path.read_text()).get(
                    "metrics", {}).get("chosen_conf_threshold")
        if frozen is None:
            raise RuntimeError(
                "no val-selected confidence threshold available; evaluate "
                "the val split first")
    scores = {"preds": []}
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
            match_and_score(pred, targets, thresholds, bev, scores,
                            sample_ids=batch["sample_ids"],
                            ignore_outlines_fn=outlines_fn)
    metrics = summarize(scores, thresholds,
                        frozen_conf=frozen if frozen is not None else None)
    out = {
        "run": str(run_dir),
        "split": args.split,
        "checkpoint": args.checkpoint,
        "data_version": data_version(root),
        "git_commit": _git_commit(),
        "metrics": metrics,
    }
    out_path = pathlib.Path(
        args.output or run_dir / f"eval_{args.split}.json")
    out_path.write_text(json.dumps(out, indent=1))
    # full-precision per-query predictions with sample ids
    with (run_dir / f"predictions_{args.split}.jsonl").open("w") as f:
        for p in scores["preds"]:
            f.write(json.dumps(p) + "\n")
    print(json.dumps(metrics, indent=1))
    return 0


def _git_commit() -> str:
    try:
        import subprocess
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
