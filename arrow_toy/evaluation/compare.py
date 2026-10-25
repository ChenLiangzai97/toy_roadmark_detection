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

    cols = ["run", "seed", "params", "self_attention", "completed_steps",
            "f1@0.5", "map@0.5", "macro_f1@0.5", "chosen_conf",
            "test_clean_f1@0.5", "test_occlusion_f1@0.5",
            "center_err_m@0.5", "ang_err_deg@0.5",
            "empty_frame_fp", "wall_clock_s", "peak_mem_gb", "status"]

    def md_table(rs):
        lines = ["| " + " | ".join(cols) + " |",
                 "|" + "---|" * len(cols)]
        for row in rs:
            lines.append("| " + " | ".join(str(row.get(c)) for c in cols)
                         + " |")
        return lines

    # fair fixed-step pairing: main table = completed 1500-step runs in
    # complete A/B seed pairs; everything else is diagnostic (design 10.4)
    import statistics
    main_rows, diag_rows = [], []
    complete = {r["run"]: r for r in rows
                if r["completed_steps"] == 1500 and r["status"] == "success"}
    seeds = sorted({r["seed"] for r in complete.values()})
    for sd in seeds:
        a = complete.get(f"baseline_s{sd}")
        b = complete.get(f"no_sa_s{sd}")
        if a and b:
            main_rows.extend([a, b])
    main_names = {r["run"] for r in main_rows}
    diag_rows = [r for r in rows if r["run"] not in main_names]

    lines = ["# A/B comparison", "",
             "## Main table (complete 1500-step A/B pairs only)"]
    lines += md_table(main_rows)
    for grp, name in (("baseline", "A (self-attention ON)"),
                      ("no_sa", "B (self-attention OFF)")):
        f1s = [r["f1@0.5"] for r in main_rows
               if r["run"].startswith(grp)]
        if len(f1s) >= 2:
            lines += ["",
                      f"{name}: mean f1@0.5 = {statistics.mean(f1s):.4f} "
                      f"+- {statistics.stdev(f1s):.4f} (n={len(f1s)})"]
        elif f1s:
            lines += ["", f"{name}: f1@0.5 = {f1s[0]} (n=1)"]
    lines += ["", "## Diagnostic table (truncated / variant runs - NOT "
              "eligible for fixed-step structure conclusions)"]
    lines += md_table(diag_rows)
    (out_dir / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def cfg_root(run: pathlib.Path) -> str:
    cfg = json.loads((run / "resolved_config.json").read_text())
    return str(REPO / cfg["dataset"]["root"])


if __name__ == "__main__":
    sys.exit(main())
