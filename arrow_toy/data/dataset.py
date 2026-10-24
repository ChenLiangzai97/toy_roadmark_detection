"""Torch dataset over the prepared cache (design doc 10.3).

Reads images_<split>.npy via mmap + calib/targets npz. Only RGB,
calibration and small tensors are touched per step. Normalization is
fixed (x-0.5)/0.5 per design 8.2.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import torch
from torch.utils.data import Dataset


class ArrowCacheDataset(Dataset):
    def __init__(self, root: str | pathlib.Path, split: str,
                 camera_id: str = "front") -> None:
        self.root = pathlib.Path(root)
        self.split = split
        cache = self.root / "cache"
        self.images = np.load(cache / f"images_{split}.npy", mmap_mode="r")
        calib = np.load(cache / f"calib_{split}.npz")
        tgt = np.load(cache / f"targets_{split}.npz")
        self.K = calib["K"].astype(np.float32)
        self.T_C_E = calib["T_C_E"].astype(np.float32)
        self.cls = tgt["class_ids"].astype(np.int64)
        self.center = tgt["centers_norm"].astype(np.float32)
        self.yaws = tgt["yaws"].astype(np.float32)
        self.mask = tgt["masks"].astype(np.float32)
        self.allow_no_object = tgt["allow_no_object"].astype(bool)
        self.sample_ids = [str(s) for s in tgt["sample_ids"]]
        self.orientation_valid = self.mask  # valid arrows have valid yaw

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, i: int):
        img = np.asarray(self.images[i], dtype=np.float32) / 255.0
        img = (img - 0.5) / 0.5                       # [H,W,3]
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        return {
            "image": img,
            "K": torch.from_numpy(self.K[i]),
            "T_C_E": torch.from_numpy(self.T_C_E[i]),
            "cls": torch.from_numpy(self.cls[i]),
            "center": torch.from_numpy(self.center[i]),
            "yaw": torch.from_numpy(self.yaws[i]),
            "dir": torch.stack([torch.cos(torch.from_numpy(self.yaws[i])),
                                torch.sin(torch.from_numpy(self.yaws[i]))],
                               dim=-1),
            "mask": torch.from_numpy(self.mask[i]),
            "orientation_valid": torch.from_numpy(self.orientation_valid[i]),
            "allow_no_object": torch.tensor(bool(self.allow_no_object[i])),
            "sample_id": self.sample_ids[i],
        }


def make_collate(batch: list[dict]) -> dict:
    out = {
        "image": torch.stack([b["image"] for b in batch]),
        "K": torch.stack([b["K"] for b in batch]),
        "T_C_E": torch.stack([b["T_C_E"] for b in batch]),
        "cls": torch.stack([b["cls"] for b in batch]),
        "center": torch.stack([b["center"] for b in batch]),
        "dir": torch.stack([b["dir"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "orientation_valid": torch.stack(
            [b["orientation_valid"] for b in batch]),
        "allow_no_object": torch.stack(
            [b["allow_no_object"] for b in batch]),
        "sample_ids": [b["sample_id"] for b in batch],
    }
    return out
