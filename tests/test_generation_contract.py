"""Regression tests for the P0/P1 review blockers.

1. allow_no_object semantics (design 6.6): empty frame -> true; clean frame
   with valid arrows -> true; any ignored arrow in BEV -> false.
2. Split sample ids must be globally unique (write_splits prefixes).
3. Occlusion label evidence via the real browser pipeline: none / partial /
   full occlusion produce the right supervision flags.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import random
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from arrow_toy.data import generate as gen  # noqa: E402
from arrow_toy.data.labels import build_labels, decode_instance_ids  # noqa: E402
from arrow_toy.render import write_splits  # noqa: E402

DIST = REPO / "web" / "dist" / "render.html"
CHROME = pathlib.Path.home() / ".cache/ms-playwright/chromium-1248/chrome-linux64/chrome"
HAVE_BROWSER = DIST.exists() and CHROME.exists()


def _base_scene(arrows: list[dict]) -> dict:
    cfg = gen.GenConfig()
    layout = gen._make_layout("t", random.Random(3), cfg)
    s = gen.make_scene("t", layout, 0, 42, cfg, n_arrows=0)
    s["arrows"] = arrows
    s["objects"] = []
    return s


def _arrow(iid: int, cls: int, x: float, y: float, yaw: float = 0.0) -> dict:
    return {"instance_id": iid, "template_id": "straight_v1", "class_id": cls,
            "center_ego": [x, y, 0.0], "yaw_rad": yaw, "scale": 1.0,
            "lane_id": None}


def _fake_masks(n: int = 120) -> np.ndarray:
    m = np.zeros((192, 320), dtype=bool)
    m[90:100, 100:106] = True  # 60 px
    m[95:105, 103:109] = True  # +60 px -> 120 total (>= 64)
    return m


def test_allow_no_object_three_cases():
    # case 1: truly empty frame, complete labels -> True
    scene = _base_scene([])
    labels = build_labels("s", scene, "front",
                          np.zeros((192, 320), dtype=np.int32), {})
    assert labels["allow_no_object"] is True

    # case 2: clean frame with a valid arrow -> True (background supervision
    # allowed on unmatched queries)
    scene = _base_scene([_arrow(1, 0, 8.0, 0.0)])
    vis = np.zeros((192, 320), dtype=np.int32)
    vis[_fake_masks()] = 1
    labels = build_labels("s", scene, "front", vis, {1: _fake_masks()})
    assert labels["arrows"][0]["arrow_valid"] is True
    assert labels["allow_no_object"] is True

    # case 3: one valid + one ignored (occluded) arrow in BEV -> False
    scene = _base_scene([_arrow(1, 0, 8.0, 0.0), _arrow(2, 1, 12.0, 1.5)])
    vis = np.zeros((192, 320), dtype=np.int32)
    vis[_fake_masks()] = 1  # only arrow 1 visible
    amodal = {1: _fake_masks(),
              2: np.roll(_fake_masks(), 20)}  # arrow 2 fully occluded
    labels = build_labels("s", scene, "front", vis, amodal)
    assert labels["arrows"][0]["arrow_valid"] is True
    assert labels["arrows"][1]["arrow_valid"] is False
    assert labels["arrows"][1]["ignore_reason"] == "occluded"
    assert labels["allow_no_object"] is False


def test_split_ids_unique_and_no_dir_collision(tmp_path):
    scenes_by_split = {"train": [{"x": 1}] * 3, "val": [{"x": 1}] * 2}
    prefixes = {"train": "train_", "val": "val_"}
    write_splits(tmp_path, scenes_by_split, prefixes)
    all_ids = []
    for split, n in [("train", 3), ("val", 2)]:
        ids = (tmp_path / "splits" / f"{split}.txt").read_text().split()
        assert len(ids) == n
        assert ids[0] == f"{split}_000001"
        all_ids.extend(ids)
    assert len(all_ids) == len(set(all_ids))
    # sample directories cannot collide because ids differ
    dirs = [tmp_path / "samples" / i for i in all_ids]
    assert len({str(d) for d in dirs}) == len(dirs)


def test_gitignore_does_not_exclude_source_modules():
    gi = (REPO / ".gitignore").read_text()
    bad = [l for l in gi.splitlines()
           if l.strip().rstrip("/") in ("data", "runs") and not l.startswith("/")]
    assert not bad, f"unanchored ignore rules exclude source dirs: {bad}"
    assert (REPO / "arrow_toy" / "data" / "labels.py").exists()


@pytest.mark.skipif(not HAVE_BROWSER, reason="web/dist or chromium not available")
def test_occlusion_label_evidence(tmp_path):
    """None / partial / full occlusion through the real renderer."""
    arrow = _arrow(1, 0, 8.0, 1.5)
    scenes = []
    for name, box in [
        ("none", None),
        ("partial", {"object_id": "occ", "shape": "box",
                     "size_abc": [0.9, 0.9, 2.6],
                     "center_ego": [4.0, 1.05, 0.0], "yaw_rad": 0.0,
                     "color": "#8a8f96"}),
        ("full", {"object_id": "occ", "shape": "box",
                  "size_abc": [2.0, 2.0, 2.6],
                  "center_ego": [4.0, 0.9, 0.0], "yaw_rad": 0.0,
                  "color": "#8a8f96"}),
    ]:
        s = _base_scene([dict(arrow)])
        s["scene_id"] = f"occ_{name}"
        if box:
            s["objects"] = [box]
        scenes.append(s)

    tmp_path.mkdir(parents=True, exist_ok=True)
    items = [{
        "sample_id": s["scene_id"], "scene": s, "camera_ids": ["front"],
        "passes": ["visible_instance", "amodal_instance"],
        "out_dir": str(tmp_path / s["scene_id"]),
        "bev_range": s["bev_range"],
    } for s in scenes]
    payload = json.dumps({"job_id": "occ", "scenes": items,
                          "render_url": DIST.resolve().as_uri()})
    env = dict(os.environ)
    env.setdefault("ARROW_CHROME_BIN", str(CHROME))
    p = subprocess.run(["npx", "tsx", "scripts/render_batch.ts"],
                       cwd=str(REPO / "web"), input=payload,
                       capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr[-2000:]

    stats = {}
    for s in scenes:
        d = tmp_path / s["scene_id"]
        vis = decode_instance_ids(d / "masks" / "front_visible.png")
        am = np.asarray(Image.open(
            d / "masks" / "amodal" / "front" / "1.png").convert("L"),
            dtype=np.uint8) > 127
        labels = build_labels(s["scene_id"], s, "front", vis, {1: am})
        a = labels["arrows"][0]
        v = a["visibility"]["front"]
        stats[s["scene_id"]] = (v["amodal_px"], v["visible_px"],
                                a["arrow_valid"], a["ignore_reason"],
                                labels["allow_no_object"])

    am_none, vis_none, valid_none, reason_none, ano_none = stats["occ_none"]
    am_part, vis_part, valid_part, reason_part, ano_part = stats["occ_partial"]
    am_full, vis_full, valid_full, reason_full, ano_full = stats["occ_full"]

    # no occluder: fully visible, valid, background supervision allowed
    assert valid_none is True and reason_none is None
    assert vis_none == am_none and ano_none is True
    # full occlusion: visible drops to ~0, ignored, background blocked
    assert vis_full <= max(1, int(0.02 * am_full)) or vis_full < 10
    assert valid_full is False and reason_full == "occluded"
    assert ano_full is False
    # partial occlusion: some pixels visible but still ignored (any confirmed
    # occlusion -> ignore); amodal area identical across the three renders
    assert 0 < vis_part < am_part
    assert valid_part is False and reason_part == "occluded"
    assert ano_part is False
    assert am_none == am_part == am_full, \
        "amodal area must not depend on occluders"
