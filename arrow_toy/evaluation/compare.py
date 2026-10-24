"""Compare runs into a single table (design doc 12).

    python -m arrow_toy.evaluation.compare --runs runs/baseline_s42 \
        runs/no_sa_s42 --output runs/comparison
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]


def collect(run: pathlib.Path) -> dict:
    res = json.loads((run / "result.json").read_text())
    env = json.loads((run / "environment.json").read_text())
    cfg = json.loads((run / "resolved_config.json").read_text())
    # prefer the checkpoint re-inference eval (fixed evaluator) over the
    # training-time metrics when present
    m = res["metrics"]
    eval_val = run / "eval_val.json"
    if eval_val.exists():
        m = json.loads(eval_val.read_text())["metrics"]
    clean = run / "eval_test_clean.json"
    occ = run / "eval_test_occlusion.json"
    clean_m = json.loads(clean.read_text())["metrics"] if clean.exists() else {}
    occ_m = json.loads(occ.read_text())["metrics"] if occ.exists() else {}
    return {
        "run": run.name,
        "seed": res.get("seed", cfg.get("seed")),
        "params": env["params"]["trainable"],
        "self_attention": cfg["model"].get("use_self_attention", True),
        "completed_steps": res["completed_steps"],
        "train_samples": None,
        "data_version": env.get("data_version"),
        "f1@0.5": round(m.get("f1@0.5", 0.0), 4),
        "map@0.5": round(m.get("map@0.5", 0.0), 4) if m.get("map@0.5") is not None else None,
        "macro_f1@0.5": round(m.get("macro_f1@0.5", 0.0), 4),
        "chosen_conf": m.get("chosen_conf_threshold"),
        "test_clean_f1@0.5": round(clean_m.get("f1@0.5", 0.0), 4) if clean_m else None,
        "test_occlusion_f1@0.5": round(occ_m.get("f1@0.5", 0.0), 4) if occ_m else None,
        "center_err_m@0.5": m.get("center_err_m@0.5"),
        "ang_err_deg@0.5": m.get("ang_err_deg@0.5"),
        "empty_frame_fp": m.get("empty_frame_fp_avg"),
        "wall_clock_s": res["wall_clock_s"],
        "peak_mem_gb": res.get("peak_mem_gb"),
        "status": res["status"],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--output", default="runs/comparison")
    args = ap.parse_args(argv)

    rows = []
    for r in args.runs:
        p = pathlib.Path(r)
        stats = p.parent / "cache" / f"stats_{p.name}.json"
        row = collect(p)
        cache_stats = pathlib.Path(cfg_root(p)) / "cache" / \
            f"stats_train.json"
        if cache_stats.exists():
            row["train_samples"] = json.loads(
                cache_stats.read_text())["n"]
        rows.append(row)

    out_dir = pathlib.Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps(rows, indent=1))
    # markdown table
    cols = ["run", "seed", "params", "self_attention", "completed_steps",
            "f1@0.5", "map@0.5", "macro_f1@0.5", "chosen_conf",
            "test_clean_f1@0.5", "test_occlusion_f1@0.5",
            "center_err_m@0.5", "ang_err_deg@0.5",
            "empty_frame_fp", "wall_clock_s", "peak_mem_gb", "status"]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "---|" * len(cols)]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c)) for c in cols) + " |")
    (out_dir / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def cfg_root(run: pathlib.Path) -> str:
    cfg = json.loads((run / "resolved_config.json").read_text())
    return str(REPO / cfg["dataset"]["root"])


if __name__ == "__main__":
    sys.exit(main())
