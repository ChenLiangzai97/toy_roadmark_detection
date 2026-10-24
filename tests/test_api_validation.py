"""API regression: all three scene endpoints reject invalid scenes with 400
BEFORE any rendering starts (P2 review final item).

Covers: unknown template_id (server used to accept), duplicate
instance_id, zero focal. Verifies preview/export/job-create each return
400 synchronously and that no render worker output is produced.
"""
from __future__ import annotations

import json
import pathlib
import random
import socket
import subprocess
import sys
import time

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _valid_scene() -> dict:
    from arrow_toy.data import generate as gen
    cfg = gen.GenConfig()
    layout = gen._make_layout("api_t", random.Random(3), cfg)
    return gen.make_scene("api_t", layout, 0, 42, cfg, n_arrows=1)


def _start_server() -> tuple[subprocess.Popen, str]:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    proc = subprocess.Popen(
        [sys.executable, "-m", "arrow_toy.editor_server",
         "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    import urllib.request
    base = f"http://127.0.0.1:{port}"
    for _ in range(50):
        time.sleep(0.3)
        try:
            urllib.request.urlopen(f"{base}/api/health", timeout=2)
            return proc, base
        except Exception:
            if proc.poll() is not None:
                raise RuntimeError("server died")
    proc.terminate()
    raise RuntimeError("server did not start")


def _post(base: str, path: str, payload: dict):
    import urllib.error
    import urllib.request
    req = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


BAD_MUTATIONS = [
    ("unknown_template",
     lambda s: s["arrows"][0].__setitem__("template_id", "does_not_exist")),
    ("dup_instance_id",
     lambda s: s["arrows"].append({**s["arrows"][0],
                                   "instance_id": s["arrows"][0]["instance_id"]})),
    ("zero_focal",
     lambda s: s["cameras"][0]["K"].__setitem__(0, [0, 0, 160])),
]


def test_api_400_before_render(tmp_path):
    proc, base = _start_server()
    try:
        for name, mutate in BAD_MUTATIONS:
            scene = _valid_scene()
            mutate(scene)
            # preview
            code, body = _post(base, "/api/render-sample-preview",
                               {"scene": scene, "camera_id": "front"})
            assert code == 400, (name, code, body[:200])
            # export
            code, body = _post(base, "/api/render-sample",
                               {"scene": scene, "camera_id": "front"})
            assert code == 400, (name, code, body[:200])
            # job create: must 400 synchronously (no job_id returned)
            code, body = _post(base, "/api/render-jobs", {
                "dataset_name": f"bad_{name}", "count": 2, "seed": 1,
                "mode": "variants", "scene": scene})
            assert code == 400, (name, code, body[:200])
            assert b"job_id" not in body
        # no render output for any bad dataset
        jobs_dir = REPO / "data" / "editor_jobs"
        for name, _ in BAD_MUTATIONS:
            assert not (jobs_dir / f"bad_{name}").exists(), name
        # sanity: the unmutated scene still passes validation at the entry
        scene = _valid_scene()
        code, body = _post(base, "/api/render-jobs", {
            "dataset_name": "api_ok_probe", "count": 1, "seed": 1,
            "mode": "variants", "scene": scene})
        assert code == 200 and b"job_id" in body, (code, body[:200])
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        import shutil
        ok_dir = REPO / "data" / "editor_jobs" / "api_ok_probe"
        if ok_dir.exists():
            shutil.rmtree(ok_dir, ignore_errors=True)
