"""End-to-end CLI regression for the data generation entry point.

Runs `python -m arrow_toy.render` with the repo default config on tiny
train/val splits and verifies index files, sample dirs, labels and masks
are all present and consistent. This covers the exact entry chain a user
invokes (the unit tests only cover the pieces).
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
DIST = REPO / "web" / "dist" / "render.html"
HAVE_BROWSER = DIST.exists()


@pytest.mark.skipif(not HAVE_BROWSER, reason="web/dist not built")
def test_render_cli_end_to_end(tmp_path):
    out_root = tmp_path / "mini"
    cmd = [
        sys.executable, "-m", "arrow_toy.render",
        "--config", "configs/data_v1.yaml",
        "--splits", "train,val",
        "--limit", "1",
        "--workers", "1",
        "--out-root", str(out_root),
    ]
    proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "labels written" in proc.stdout

    for split in ("train", "val"):
        ids = (out_root / "splits" / f"{split}.txt").read_text().split()
        assert ids == [f"{split}_000001"]
        sdir = out_root / "samples" / ids[0]
        # rgb + masks + labels + scene all present
        assert (sdir / "rgb" / "front.png").exists()
        assert (sdir / "masks" / "front_visible.png").exists()
        assert (sdir / "masks" / "bev_instance.png").exists()
        labels = json.loads((sdir / "labels.json").read_text())
        scene = json.loads((sdir / "scene.json").read_text())
        assert labels["sample_id"] == ids[0]
        assert labels["schema_version"] == 1
        assert labels["scene_group_id"] == scene["scene_group_id"]
        assert labels["training_camera_ids"] == ["front"]
        assert isinstance(labels["allow_no_object"], bool)
        n_arrows = len(scene["arrows"])
        assert len(labels["arrows"]) == n_arrows
        for a in labels["arrows"]:
            v = a["visibility"]["front"]
            assert 0 <= v["visible_px"] <= v["amodal_px"]
            if a["arrow_valid"]:
                assert a["ignore_reason"] is None
                assert v["amodal_px"] >= 64
            else:
                assert a["ignore_reason"] is not None
    manifest = json.loads((out_root / "manifest.json").read_text())
    assert manifest["splits"] == {"train": 1, "val": 1}
    assert manifest["completed"] is False


def test_render_cli_dry_run(tmp_path):
    proc = subprocess.run([
        sys.executable, "-m", "arrow_toy.render",
        "--config", "configs/data_v1.yaml",
        "--splits", "train,val", "--limit", "1", "--dry-run",
        "--out-root", str(tmp_path / "dry"),
    ], cwd=str(REPO), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "dry-run complete" in proc.stdout
    assert (tmp_path / "dry" / "dry" / "train" / "scenes.jsonl").exists()
