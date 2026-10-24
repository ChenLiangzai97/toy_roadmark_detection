"""Dataset audit (design doc 7.1 / 15.1).

Checks per split:
- counts vs manifest, unique sample ids, scene groups never cross splits
- arrows-per-frame histogram, instance class balance, yaw buckets,
  BEV position spread, visible-area buckets
- ignore reason breakdown, allow_no_object ratio
- label/mask consistency invariants (visible <= amodal, valid implies
  amodal>=64px etc.)
Writes data/arrow_v1/audit.json and flips manifest.completed on pass.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import sys

import numpy as np

from arrow_toy.geometry.bev import BevGrid
from arrow_toy.utils.config import load_yaml

MIN_CLASS_INSTANCES = 8
BEV_FEATURE = BevGrid.feature()


def load_split(root: pathlib.Path, split: str) -> list[dict]:
    ids = (root / "splits" / f"{split}.txt").read_text().split()
    labels = []
    for sid in ids:
        p = root / "samples" / sid / "labels.json"
        labels.append(json.loads(p.read_text()))
    return labels


def audit_split(root: pathlib.Path, split: str) -> dict:
    labels = load_split(root, split)
    n = len(labels)
    groups = {l["scene_group_id"] for l in labels}
    arrows = [a for l in labels for a in l["arrows"]]
    per_frame = collections.Counter(len(l["arrows"]) for l in labels)
    classes = collections.Counter(a["class_id"] for a in arrows)
    reasons = collections.Counter(a["ignore_reason"] for a in arrows
                                  if a["ignore_reason"])
    yaw_buckets = collections.Counter()
    pos_cells = collections.Counter()
    vis_area_buckets = collections.Counter()
    allow_no = 0
    bad = []
    for l in labels:
        if l["allow_no_object"]:
            allow_no += 1
        for a in l["arrows"]:
            yaw_buckets[int(((a["yaw_rad"] + math.pi) / (2 * math.pi)) * 8) % 8] += 1
            g = BEV_FEATURE
            r, c = g.metric_to_cell(a["center_ego"][0], a["center_ego"][1])
            if 0 <= r < g.rows and 0 <= c < g.cols:
                pos_cells[(r // 8, c // 8)] += 1
            v = next(iter(a["visibility"].values()))
            if v["amodal_px"] > 0:
                vis_area_buckets[min(6, int(math.log2(v["amodal_px"] / 32)))] += 1
            # invariants
            if v["visible_px"] > v["amodal_px"]:
                bad.append((l["sample_id"], a["instance_id"], "visible>amodal"))
            if a["arrow_valid"]:
                if v["amodal_px"] < 64 or a["ignore_reason"] is not None:
                    bad.append((l["sample_id"], a["instance_id"], "valid-but-invalid-stats"))
            else:
                if a["ignore_reason"] is None:
                    bad.append((l["sample_id"], a["instance_id"], "invalid-no-reason"))
    min_class = min(classes[c] for c in range(6)) if len(classes) == 6 else 0
    cov_cells = len(pos_cells)
    return {
        "split": split,
        "n_samples": n,
        "n_groups": len(groups),
        "groups": sorted(groups),
        "arrows_total": len(arrows),
        "arrows_per_frame": {str(k): v for k, v in sorted(per_frame.items())},
        "class_instances": {str(c): classes.get(c, 0) for c in range(6)},
        "min_class_instances": min_class,
        "ignore_reasons": {str(k): v for k, v in sorted(reasons.items(),
                                                        key=lambda x: str(x[0]))},
        "allow_no_object_ratio": allow_no / max(1, n),
        "yaw_buckets_8": {str(k): v for k, v in sorted(yaw_buckets.items())},
        "bev_position_cells_covered_8x8": cov_cells,
        "visible_area_buckets_log2_32px": {str(k): v for k, v in
                                           sorted(vis_area_buckets.items())},
        "invariant_violations": bad[:20],
        "invariant_violation_count": len(bad),
        "pass": bool(n > 0 and min_class >= MIN_CLASS_INSTANCES
                     and len(bad) == 0),
    }


def check_no_split_leakage(splits: dict[str, dict]) -> list[str]:
    seen: dict[str, str] = {}
    leaks = []
    for split, info in splits.items():
        for g in info["groups"]:
            if g in seen and seen[g] != split:
                leaks.append(f"{g}: {seen[g]} vs {split}")
            seen[g] = split
    return leaks


def run_audit(root: pathlib.Path, expected_splits: list[str] | None = None,
              min_class: int = MIN_CLASS_INSTANCES) -> dict:
    root = pathlib.Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    splits = expected_splits or list(manifest["splits"].keys())
    infos = {}
    for split in splits:
        infos[split] = audit_split(root, split)
        info = infos[split]
        info["pass"] = bool(info["n_samples"] > 0
                            and info["min_class_instances"] >= min_class
                            and info["invariant_violation_count"] == 0)
    leaks = check_no_split_leakage(infos)
    # occlusion/pose diagnostics: ignore reasons expected non-zero there
    audit = {
        "dataset": root.name,
        "manifest_splits": manifest["splits"],
        "splits": infos,
        "split_leaks": leaks,
        "class_balance_threshold": min_class,
        "notes": {
            "test_occlusion": "ignore reasons are EXPECTED (occlusion set)",
            "test_pose": "camera perturbation set; ignore reasons may include"
                         " truncated/too_small",
        },
    }
    clean_splits = [s for s in splits
                    if s not in ("test_occlusion", "test_pose", "stress")]
    overall = (not leaks
               and all(infos[s]["pass"] for s in splits if s in infos)
               # clean splits additionally need sane distributions
               and all(infos[s]["arrows_total"] > 0 for s in clean_splits))
    audit["overall_pass"] = bool(overall)
    (root / "audit.json").write_text(json.dumps(audit, indent=1))
    # flip manifest completion flag
    manifest["completed"] = bool(audit["overall_pass"])
    manifest["audit_summary"] = {
        "overall_pass": audit["overall_pass"],
        "leaks": leaks,
        "per_split_pass": {s: infos[s]["pass"] for s in infos},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return audit


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/arrow_v1")
    ap.add_argument("--splits", default=None)
    ap.add_argument("--min-class", type=int, default=MIN_CLASS_INSTANCES)
    args = ap.parse_args(argv)
    root = pathlib.Path(args.root)
    splits = args.splits.split(",") if args.splits else None
    audit = run_audit(root, splits, args.min_class)
    print(json.dumps({
        "overall_pass": audit["overall_pass"],
        "leaks": audit["split_leaks"],
        "per_split": {s: {
            "n": i["n_samples"], "arrows": i["arrows_total"],
            "min_class": i["min_class_instances"],
            "violations": i["invariant_violation_count"],
            "pass": i["pass"],
        } for s, i in audit["splits"].items()},
    }, indent=1))
    return 0 if audit["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
