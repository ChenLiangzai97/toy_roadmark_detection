"""Local editor server (design doc 3.2 / 14.3).

Serves the built editor + render pages and exposes the generation API:
  GET  /api/health
  POST /api/render-sample          current scene -> zip (rgb/masks/labels)
  POST /api/render-sample-preview  rgb + bev instance (base64 rgba) for UI
  POST /api/render-jobs            batch generation job
  GET  /api/render-jobs/{id}       status
  POST /api/render-jobs/{id}/cancel

Run: python -m arrow_toy.editor_server --host 127.0.0.1 --port 8765
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import pathlib
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from arrow_toy.data import generate as gen
from arrow_toy.data.labels import build_labels, decode_instance_ids
from PIL import Image

from arrow_toy.render import find_chrome, postprocess_labels, render_scenes
from arrow_toy.utils.config import load_yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
WEB_DIST = REPO / "web" / "dist"
DEFAULT_CFG_PATH = REPO / "configs" / "data_v1.yaml"

app = FastAPI(title="arrow-toy editor server")


class RenderSampleRequest(BaseModel):
    scene: dict
    camera_id: str = "front"


class PreviewRequest(BaseModel):
    scene: dict
    camera_id: str = "front"


class RenderJobRequest(BaseModel):
    dataset_name: str
    count: int = 16
    seed: int = 42
    mode: str = "variants"     # variants: randomize current scene; random: full synth
    scene: dict | None = None


JOBS: dict[str, dict] = {}
_JOB_LOCK = threading.Lock()


def _run_worker(items: list[dict], out_base: pathlib.Path) -> None:
    """Invoke the node worker synchronously for a small batch."""
    import os
    payload = json.dumps({
        "job_id": "srv", "scenes": items,
        "render_url": (WEB_DIST / "render.html").resolve().as_uri(),
    })
    env = dict(os.environ)
    chrome = find_chrome()
    if chrome:
        env.setdefault("ARROW_CHROME_BIN", chrome)
    proc = subprocess.run(
        ["npx", "tsx", str(REPO / "web" / "scripts" / "render_batch.ts")],
        cwd=str(REPO / "web"), input=payload, capture_output=True, text=True,
        env=env)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-2000:])
    for line in proc.stdout.splitlines():
        msg = json.loads(line)
        if msg.get("kind") == "error":
            raise RuntimeError(str(msg))


@app.get("/api/health")
def health():
    return {
        "service": "arrow_toy.editor_server",
        "schema_version": 1,
        "render_worker_available": bool(find_chrome()) and WEB_DIST.exists(),
    }


@app.post("/api/render-sample-preview")
def render_sample_preview(req: PreviewRequest):
    """Render rgb + bev passes for the editor preview panels."""
    scene = req.scene
    cam_id = req.camera_id
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="arrow_prev_"))
    try:
        items = [{
        "sample_id": "preview", "scene": scene, "camera_ids": [cam_id],
        "passes": ["rgb", "bev_instance"],
            "out_dir": str(tmp / "preview"),
            "bev_range": scene["bev_range"],
        }]
        _run_worker(items, tmp)
        from PIL import Image
        rgb = np.asarray(Image.open(tmp / "preview" / "rgb" / f"{cam_id}.png")
                         .convert("RGB"), dtype=np.uint8)
        bev = np.asarray(Image.open(tmp / "preview" / "masks" / "bev_instance.png")
                         .convert("RGB"), dtype=np.uint8)

        def b64(arr):
            return base64.b64encode(arr.tobytes()).decode()

        return {"rgb": {"w": rgb.shape[1], "h": rgb.shape[0], "b64": b64(rgb)},
                "bev": {"w": bev.shape[1], "h": bev.shape[0], "b64": b64(bev)}}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.post("/api/render-sample")
def render_sample(req: RenderSampleRequest):
    """Export the current scene as one labeled sample zip."""
    scene = req.scene
    cam_id = req.camera_id
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="arrow_export_"))
    try:
        items = [{
            "sample_id": "cur", "scene": scene, "camera_ids": [cam_id],
            "passes": ["rgb", "visible_instance", "amodal_instance",
                       "bev_instance"],
            "out_dir": str(tmp / "cur"),
            "bev_range": scene["bev_range"],
        }]
        _run_worker(items, tmp)
        visible = decode_instance_ids(tmp / "cur" / "masks" / f"{cam_id}_visible.png")
        amodal = {}
        for a in scene["arrows"]:
            m = np.asarray(Image.open(
                tmp / "cur" / "masks" / "amodal" / cam_id / f"{a['instance_id']}.png"
            ).convert("L"), dtype=np.uint8) > 127
            amodal[a["instance_id"]] = m
        labels = build_labels("current", scene, cam_id, visible, amodal)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for f in (tmp / "cur").rglob("*"):
                if f.is_file():
                    z.write(f, f.relative_to(tmp / "cur"))
            z.writestr("labels.json", json.dumps(labels, indent=1))
            z.writestr("scene.json", json.dumps(scene, indent=1))
        buf.seek(0)
        return Response(
            content=buf.getvalue(), media_type="application/zip",
            headers={"Content-Disposition":
                     f'attachment; filename="{scene.get("scene_id", "sample")}_sample.zip"'})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _make_variants(scene: dict, count: int, seed: int) -> list[dict]:
    """Randomize style/light/objects of the current scene; keep geometry."""
    rng = random.Random(seed)
    out = []
    for i in range(count):
        s = json.loads(json.dumps(scene))
        s["scene_id"] = f"{scene['scene_id']}_v{i:03d}"
        s["scene_group_id"] = scene["scene_group_id"]
        s["seed"] = rng.randint(0, 2**31 - 1)
        s["ground"]["color"] = rng.choice(gen.GROUND_COLORS)
        s["render"]["light"]["direction_ego"] = [
            rng.uniform(-0.8, 0.2), rng.uniform(-0.8, 0.8), rng.uniform(0.7, 1.3)]
        s["render"]["light"]["intensity"] = rng.uniform(0.8, 1.3)
        s["render"]["light"]["ambient"] = rng.uniform(0.35, 0.6)
        out.append(s)
    return out


def _job_thread(job_id: str, req: RenderJobRequest):
    out_root = REPO / "data" / "editor_jobs" / req.dataset_name
    try:
        with _JOB_LOCK:
            JOBS[job_id]["status"] = "running"
        if req.mode == "variants" and req.scene:
            scenes = _make_variants(req.scene, req.count, req.seed)
        else:
            cfg = load_yaml(DEFAULT_CFG_PATH)
            from arrow_toy.render import build_scenes_for_config
            all_scenes = build_scenes_for_config(
                {**cfg, "splits": {"train": req.count}}, ["train"])
            scenes = all_scenes["train"]
        prefix = f"job{int(time.time())}_"
        res = render_scenes(scenes, out_root, "job", 2, None,
                            id_prefix=prefix)
        if res["samples"] != len(scenes):
            raise RuntimeError(f"rendered {res['samples']}/{len(scenes)}")
        postprocess_labels(scenes, out_root, id_prefix=prefix)
        with _JOB_LOCK:
            JOBS[job_id].update(status="completed", done=len(scenes),
                                total=len(scenes), output_dir=str(out_root))
    except Exception as e:  # noqa: BLE001 - job isolation
        with _JOB_LOCK:
            JOBS[job_id].update(status="failed", error=str(e)[:500])


@app.post("/api/render-jobs")
def create_render_job(req: RenderJobRequest):
    job_id = f"job_{int(time.time() * 1000):x}"
    with _JOB_LOCK:
        JOBS[job_id] = {"job_id": job_id, "status": "queued",
                        "done": 0, "total": req.count}
    threading.Thread(target=_job_thread, args=(job_id, req), daemon=True).start()
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/render-jobs/{job_id}")
def get_render_job(job_id: str):
    with _JOB_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return job


@app.post("/api/render-jobs/{job_id}/cancel")
def cancel_render_job(job_id: str):
    with _JOB_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    if job["status"] in ("queued", "running"):
        job["status"] = "cancelled"
    return job


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    if not WEB_DIST.exists():
        print("web/dist missing; run npm --prefix web run build first")
        return 2
    import uvicorn
    app.mount("/", StaticFiles(directory=str(WEB_DIST), html=True), name="web")
    print(f"serving editor on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
