"""Training cache builder (design doc 10.3).

Produces under <root>/cache/:
  images_<split>.npy    uint8 [N, H, W, 3] (front camera RGB)
  calib_<split>.npz     K [N,3,3], T_C_E [N,4,4] float32
  targets_<split>.npz   class_ids, centers_norm [N,32,2] padded, yaws,
                        masks [N,32] (1 valid, 0 pad), allow_no_object,
                        group_ids for debugging (not a model input)
Only RGB + calibration + small tensors are cached; masks/scenes are NOT
parsed per training step. The projection grid cache is a single tensor
per unique calibration (front camera is fixed across the dataset).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from PIL import Image

NUM_QUERIES = 32


def build_cache(root: pathlib.Path, split: str, camera_id: str = "front",
                max_queries: int = NUM_QUERIES) -> dict:
    root = pathlib.Path(root)
    out_dir = root / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = (root / "splits" / f"{split}.txt").read_text().split()
    images = []
    Ks, Ts = [], []
    cls, ctr, yaw, mask, ano = [], [], [], [], []
    n_over_capacity = 0
    for sid in ids:
        sdir = root / "samples" / sid
        labels = json.loads((sdir / "labels.json").read_text())
        img = np.asarray(Image.open(sdir / "rgb" / f"{camera_id}.png")
                         .convert("RGB"), dtype=np.uint8)
        images.append(img)
        cal = labels["cameras"][camera_id]
        Ks.append(np.asarray(cal["K"], dtype=np.float32))
        Ts.append(np.asarray(cal["T_C_E"], dtype=np.float32))
        valid = [a for a in labels["arrows"] if a["arrow_valid"]]
        if len(valid) > max_queries:
            n_over_capacity += 1
            raise RuntimeError(
                f"{sid}: {len(valid)} valid arrows > {max_queries} queries")
        bev = labels["bev_range"]
        x_min, x_max = bev["x_min"], bev["x_max"]
        y_min, y_max = bev["y_min"], bev["y_max"]
        rows = np.zeros((max_queries,), dtype=np.int64)
        c = np.zeros((max_queries, 2), dtype=np.float32)
        yw = np.zeros((max_queries,), dtype=np.float32)
        m = np.zeros((max_queries,), dtype=np.float32)
        for i, a in enumerate(valid):
            rows[i] = a["class_id"]
            xn = (a["center_ego"][0] - x_min) / (x_max - x_min)
            yn = (a["center_ego"][1] - y_min) / (y_max - y_min)
            c[i] = (xn, yn)  # order [x_n, y_n]
            yw[i] = a["yaw_rad"]
            m[i] = 1.0
        cls.append(rows)
        ctr.append(c)
        yaw.append(yw)
        mask.append(m)
        ano.append(labels["allow_no_object"])
    images = np.stack(images) if images else np.zeros((0,), dtype=np.uint8)
    np.save(out_dir / f"images_{split}.npy", images)
    np.savez(out_dir / f"calib_{split}.npz",
             K=np.stack(Ks) if Ks else np.zeros((0, 3, 3), np.float32),
             T_C_E=np.stack(Ts) if Ts else np.zeros((0, 4, 4), np.float32))
    np.savez(out_dir / f"targets_{split}.npz",
             class_ids=np.stack(cls), centers_norm=np.stack(ctr),
             yaws=np.stack(yaw), masks=np.stack(mask),
             allow_no_object=np.asarray(ano, dtype=bool),
             sample_ids=np.asarray(ids))
    stats = {
        "split": split, "n": len(ids), "shape": list(images.shape),
        "n_valid_arrows": int(sum(int(m.sum()) for m in mask)),
        "allow_no_object_true": int(sum(1 for a in ano if a)),
    }
    (out_dir / f"stats_{split}.json").write_text(json.dumps(stats, indent=1))
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/arrow_v1")
    ap.add_argument("--splits", default="overfit,train,val,test_clean,"
                                        "test_occlusion,test_pose")
    ap.add_argument("--camera", default="front")
    args = ap.parse_args(argv)
    root = pathlib.Path(args.root)
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest.get("completed"):
        print("manifest.completed is false; run audit first")
        return 2
    for split in args.splits.split(","):
        if not (root / "splits" / f"{split}.txt").exists():
            print(f"skip missing split {split}")
            continue
        stats = build_cache(root, split, args.camera)
        print(f"[cache] {split}: {stats['n']} samples, images "
              f"{stats['shape']}, valid arrows {stats['n_valid_arrows']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
