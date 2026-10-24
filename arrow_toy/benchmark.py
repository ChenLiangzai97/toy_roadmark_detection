"""Standalone performance benchmark (design doc 10.4).

10 warmup + 30 timed steps of the full train iteration (fwd, match,
loss, bwd, optimizer) with CUDA events; reports median/P90 step time,
samples/s and peak memory. Also times one validation pass.

    python -m arrow_toy.benchmark --config configs/quick.yaml \
        --output runs/benchmark.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

import torch
from torch.utils.data import DataLoader

from arrow_toy.data.dataset import ArrowCacheDataset, make_collate
from arrow_toy.losses.criterion import arrow_criterion, hungarian_match
from arrow_toy.models.model import ArrowToyModel, count_parameters
from arrow_toy.utils.config import load_yaml

REPO = pathlib.Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/quick.yaml")
    ap.add_argument("--output", default="runs/benchmark.json")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args(argv)

    cfg = load_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = REPO / cfg["dataset"]["root"]
    ds = ArrowCacheDataset(root, cfg["dataset"]["train_split"])
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"],
                        shuffle=True, num_workers=4, pin_memory=True,
                        collate_fn=make_collate, drop_last=True,
                        persistent_workers=True)
    model = ArrowToyModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])

    it = iter(loader)
    times = []
    use_amp = device == "cuda"
    torch.cuda.reset_peak_memory_stats()
    for step in range(args.warmup + args.steps):
        batch = next(it)
        batch = {k: (v.to(device, non_blocking=True)
                     if torch.is_tensor(v) else v) for k, v in batch.items()}
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            pred = model(batch["image"], batch["K"], batch["T_C_E"])
        targets = {k: batch[k] for k in
                   ("cls", "center", "dir", "mask", "orientation_valid",
                    "allow_no_object")}
        matches = hungarian_match(pred, targets)
        losses, _ = arrow_criterion(pred, targets, matches)
        opt.zero_grad(set_to_none=True)
        losses["loss_total"].backward()
        opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if step >= args.warmup:
            times.append(dt)

    # validation timing (single pass, capped)
    val_ds = ArrowCacheDataset(root, cfg["dataset"]["val_split"])
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"],
                            shuffle=False, num_workers=2, collate_fn=make_collate)
    model.eval()
    t0 = time.perf_counter()
    n_val = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=use_amp):
                pred = model(batch["image"], batch["K"], batch["T_C_E"])
            targets = {k: batch[k] for k in
                       ("cls", "center", "dir", "mask", "orientation_valid",
                        "allow_no_object")}
            matches = hungarian_match(pred, targets)
            arrow_criterion(pred, targets, matches)
            n_val += len(batch["image"])
            if i >= 8:
                break
    val_s = time.perf_counter() - t0

    out = {
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "torch": torch.__version__,
        "params": count_parameters(model),
        "batch_size": cfg["train"]["batch_size"],
        "steps_timed": len(times),
        "step_time_median_s": round(statistics.median(times), 4),
        "step_time_p90_s": round(statistics.quantiles(times, n=10)[8], 4),
        "samples_per_s": round(cfg["train"]["batch_size"] / statistics.median(times), 1),
        "peak_mem_allocated_gb": round(
            torch.cuda.max_memory_allocated() / 2**30, 3) if device == "cuda" else 0,
        "val_probe_seconds": round(val_s, 2),
        "val_probe_samples": n_val,
        "budget_check": {
            "max_steps": cfg["train"]["max_steps"],
            "time_budget_s": cfg["train"]["time_budget_s"],
            "projected_train_seconds": round(
                statistics.median(times) * cfg["train"]["max_steps"], 1),
        },
    }
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
