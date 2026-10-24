"""Training entry (design doc 10).

    python -m arrow_toy.train --config configs/overfit.yaml --run-dir runs/overfit
    python -m arrow_toy.train --config configs/quick.yaml --run-dir runs/base_s42
    python -m arrow_toy.train --config configs/quick.yaml \
        --override configs/ablations/no_query_sa.yaml --run-dir runs/no_sa_s42

Time budget: wall clock from process start to val+checkpoint+metrics done,
<= time_budget_s. Status in result.json: success | budget_incomplete |
failed_nonfinite | failed_data.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys
import time

import numpy as np
import torch

# fixed input shapes -> cudnn autotune; TF32 allowed for fp32 matmuls
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from torch.utils.data import DataLoader

from arrow_toy.data.dataset import ArrowCacheDataset, make_collate
from arrow_toy.losses.criterion import arrow_criterion, hungarian_match
from arrow_toy.models.model import ArrowToyModel, count_parameters
from arrow_toy.utils.config import deep_merge, load_yaml, write_json

REPO = pathlib.Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_lr_schedule(warmup: int, total: int, base_lr: float, min_lr: float,
                      step: int) -> float:
    """Linear warmup then cosine to min_lr."""
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    t = min(1.0, t)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))


def evaluate(model, loader, device, cfg, max_samples: int | None = None):
    """Fixed val: loss parts + detection metrics via the shared evaluator."""
    from arrow_toy.evaluation.metrics import match_and_score
    model.eval()
    losses = {"loss_cls": 0.0, "loss_center": 0.0, "loss_ori": 0.0}
    n = 0
    all_scores = {"tp": [], "fp": [], "fn": 0, "center_err": [],
                  "ang_err": [], "dup": 0, "empty_fp": 0, "empty_n": 0,
                  "preds": []}
    thresholds = cfg["eval"]["center_thresholds_m"]
    bev = cfg["model"]["bev_range_xy"]
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=batch["image"].is_cuda):
                pred = model(batch["image"], batch["K"], batch["T_C_E"])
            targets = {k: batch[k] for k in
                       ("cls", "center", "dir", "mask", "orientation_valid",
                        "allow_no_object")}
            matches = hungarian_match(
                pred, targets,
                cls_weight=cfg["loss"]["class_weight"],
                center_weight=cfg["loss"]["center_weight"])
            loss_dict, _ = arrow_criterion(
                pred, targets, matches,
                class_weight=cfg["loss"]["class_weight"],
                center_weight=cfg["loss"]["center_weight"],
                orientation_weight=cfg["loss"]["orientation_weight"],
                no_object_weight=cfg["loss"]["no_object_class_weight"])
            for k in losses:
                losses[k] += float(loss_dict[k])
            n += 1
            match_and_score(pred, targets, thresholds, bev, all_scores)
            if max_samples and n * loader.batch_size >= max_samples:
                break
    for k in losses:
        losses[k] /= max(1, n)
    from arrow_toy.evaluation.metrics import summarize
    metrics = summarize(all_scores, thresholds)
    metrics.update({f"val_{k}": v for k, v in losses.items()})
    return metrics, all_scores


def main(argv=None) -> int:
    t_start = time.monotonic()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", default=None)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    cfg = load_yaml(args.config)
    if args.override:
        cfg = deep_merge(cfg, load_yaml(args.override))
    run_dir = pathlib.Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "resolved_config.yaml" if False else run_dir / "resolved_config.json", cfg)
    # also yaml for readability
    import yaml
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg))

    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = args.device or cfg["train"].get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("[train] cuda unavailable; falling back to cpu (recorded)")
        device = "cpu"

    # data
    root = pathlib.Path(cfg["dataset"]["root"])
    train_ds = ArrowCacheDataset(root, cfg["dataset"]["train_split"])
    val_ds = ArrowCacheDataset(root, cfg["dataset"]["val_split"])
    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["train"]["num_workers"], pin_memory=True,
        persistent_workers=cfg["train"]["num_workers"] > 0,
        collate_fn=make_collate, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=make_collate)

    model = ArrowToyModel(cfg).to(device)
    n_params = count_parameters(model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    max_steps = cfg["train"]["max_steps"]
    budget = cfg["train"]["time_budget_s"]
    reserve = cfg["train"]["reserve_finalization_s"]
    amp_dtype = cfg["train"].get("amp_dtype", "bfloat16")
    use_amp = device == "cuda" and amp_dtype == "bfloat16"

    env = {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "params": n_params,
        "git_commit": _git_commit(),
    }
    write_json(run_dir / "environment.json", env)

    metrics_path = run_dir / "metrics.jsonl"
    metrics_f = metrics_path.open("w")
    status = "success"
    completed_steps = 0
    best_metric = -1.0
    log_every = cfg["train"]["log_every"]
    eval_every = cfg["train"]["eval_every"]
    nonfinite_info = None

    step = 0
    stop_reason = None
    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break
            lr = build_lr_schedule(cfg["train"]["warmup_steps"], max_steps,
                                   cfg["train"]["lr"], cfg["train"]["min_lr"],
                                   step)
            for g in opt.param_groups:
                g["lr"] = lr
            batch = {k: (v.to(device, non_blocking=True)
                         if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=use_amp):
                pred = model(batch["image"], batch["K"], batch["T_C_E"])
            targets = {k: batch[k] for k in
                       ("cls", "center", "dir", "mask", "orientation_valid",
                        "allow_no_object")}
            matches = hungarian_match(
                pred, targets,
                cls_weight=cfg["loss"]["class_weight"],
                center_weight=cfg["loss"]["center_weight"])
            loss_dict, stats = arrow_criterion(
                pred, targets, matches,
                class_weight=cfg["loss"]["class_weight"],
                center_weight=cfg["loss"]["center_weight"],
                orientation_weight=cfg["loss"]["orientation_weight"],
                no_object_weight=cfg["loss"]["no_object_class_weight"])
            loss = loss_dict["loss_total"]
            if not torch.isfinite(loss):
                status = "failed_nonfinite"
                nonfinite_info = {
                    "step": step,
                    "sample_ids": batch["sample_ids"],
                    "loss": {k: float(v) for k, v in loss_dict.items()},
                }
                write_json(run_dir / "nonfinite.json", nonfinite_info)
                stop_reason = "nonfinite_loss"
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["train"]["grad_clip_norm"])
            if not torch.isfinite(grad_norm):
                status = "failed_nonfinite"
                write_json(run_dir / "nonfinite.json", {
                    "step": step, "reason": "grad_norm",
                    "grad_norm": float(grad_norm),
                    "sample_ids": batch["sample_ids"]})
                stop_reason = "nonfinite_grad"
                break
            opt.step()
            completed_steps = step + 1
            step += 1

            if step % log_every == 0:
                rec = {"step": step,
                       "loss_cls": float(loss_dict["loss_cls"]),
                       "loss_center": float(loss_dict["loss_center"]),
                       "loss_ori": float(loss_dict["loss_ori"]),
                       "lr": lr, "n_pos": stats["n_pos"],
                       "grad_norm": float(grad_norm),
                       "t": round(time.monotonic() - t_start, 1)}
                metrics_f.write(json.dumps(rec) + "\n")
                metrics_f.flush()

            elapsed = time.monotonic() - t_start
            if step % eval_every == 0 and step < max_steps:
                # mid-run validation only if budget clearly allows it
                if elapsed + 2 * reserve < budget:
                    metrics, _ = evaluate(model, val_loader, device, cfg)
                    metrics_f.write(json.dumps(
                        {"step": step, "val": True, **{
                            k: v for k, v in metrics.items()
                            if isinstance(v, (int, float))}}) + "\n")
                    metrics_f.flush()
                    model.train()
            if elapsed + reserve > budget and step < max_steps:
                status = "budget_incomplete"
                stop_reason = "time_budget"
                break
        if stop_reason:
            break

    # final validation + save
    t_val0 = time.monotonic()
    metrics, scores = evaluate(model, val_loader, device, cfg,
                               max_samples=cfg["eval"].get("val_limit"))
    val_seconds = time.monotonic() - t_val0
    main_metric = metrics.get("f1@0.5", 0.0)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "step": completed_steps,
        "config": cfg,
        "val_metrics": metrics,
        "seed": seed,
    }
    torch.save(ckpt, run_dir / "last.pt")
    if main_metric >= best_metric:
        best_metric = main_metric
        torch.save(ckpt, run_dir / "best.pt")
    write_json(run_dir / "predictions_val.jsonl" if False else run_dir / "predictions_val.json",
               {"note": "per-query raw predictions in predictions_val.jsonl",
                "preds": scores["preds"][:2000]})
    # full per-query lines
    with (run_dir / "predictions_val.jsonl").open("w") as f:
        for p in scores["preds"]:
            f.write(json.dumps(p) + "\n")

    wall = time.monotonic() - t_start
    peak_mem = (torch.cuda.max_memory_allocated() / 2**30
                if device == "cuda" else 0.0)
    result = {
        "status": status,
        "completed_steps": completed_steps,
        "max_steps": max_steps,
        "wall_clock_s": round(wall, 2),
        "time_budget_s": budget,
        "time_budget_pass": bool(wall <= budget and status == "success"),
        "val_seconds": round(val_seconds, 2),
        "peak_mem_gb": round(peak_mem, 3),
        "metrics": metrics,
        "main_metric": "f1@0.5",
        "params": n_params,
        "seed": seed,
        "stop_reason": stop_reason,
    }
    write_json(run_dir / "result.json", result)
    metrics_f.close()
    print(json.dumps({k: result[k] for k in
                      ("status", "completed_steps", "wall_clock_s",
                       "time_budget_pass", "peak_mem_gb")}, indent=1))
    print("val:", json.dumps({k: round(v, 4) for k, v in metrics.items()
                              if isinstance(v, (int, float))}, indent=1))
    return 0 if status == "success" else 1


def _git_commit() -> str:
    try:
        import subprocess
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                             capture_output=True, text=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
