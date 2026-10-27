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

SCENE_SCHEMA = json.loads(
    (REPO / "schemas" / "scene_v1.schema.json").read_text())

# arrow template ids that actually exist (shared with the front-end
# validator, which derives them from the same assets)
_TEMPLATE_IDS = {p.stem for p in (REPO / "assets" / "arrows").glob("*.json")}


def _validate_scene(scene: dict) -> None:
    """Schema + semantic validation; called SYNCHRONOUSLY at every scene
    API entry so invalid scenes fail with 400 before any rendering."""
    import jsonschema
    try:
        jsonschema.validate(scene, SCENE_SCHEMA)
    except jsonschema.ValidationError as e:
        raise HTTPException(400, f"invalid scene: {e.message}")
    # semantic checks mirrored from web/src/shared/validate.ts
    seen = set()
    for a in scene.get("arrows", []):
        iid = a["instance_id"]
        if iid in seen:
            raise HTTPException(400, f"invalid scene: duplicate "
                                     f"instance_id {iid}")
        seen.add(iid)
        if a["template_id"] not in _TEMPLATE_IDS:
            raise HTTPException(400, f"invalid scene: unknown template_id "
                                     f"{a['template_id']}")
    for c in scene.get("cameras", []):
        K = c["K"]
        if not (K[0][0] > 0 and K[1][1] > 0):
            raise HTTPException(400, "invalid scene: focal length <= 0")
        if not (0 < c["near"] < c["far"]):
            raise HTTPException(400, "invalid scene: need 0 < near < far")


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
    _validate_scene(req.scene)
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

        def b64_rgba(arr):
            h, w = arr.shape[:2]
            rgba = np.dstack([arr, np.full((h, w, 1), 255, dtype=np.uint8)])
            return base64.b64encode(rgba.tobytes()).decode()

        return {"rgb": {"w": rgb.shape[1], "h": rgb.shape[0], "b64": b64_rgba(rgb)},
                "bev": {"w": bev.shape[1], "h": bev.shape[0], "b64": b64_rgba(bev)}}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.post("/api/render-sample")
def render_sample(req: RenderSampleRequest):
    """Export the current scene as one labeled sample zip."""
    _validate_scene(req.scene)
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

    def cancelled() -> bool:
        with _JOB_LOCK:
            return JOBS[job_id].get("status") == "cancelled"

    try:
        with _JOB_LOCK:
            JOBS[job_id]["status"] = "running"
        if req.mode == "variants" and req.scene is not None:
            scenes = _make_variants(req.scene, req.count, req.seed)
        else:
            cfg = load_yaml(DEFAULT_CFG_PATH)
            from arrow_toy.render import build_scenes_for_config
            all_scenes = build_scenes_for_config(
                {**cfg, "splits": {"train": req.count}}, ["train"])
            scenes = all_scenes["train"]
        prefix = f"job{int(time.time())}_"
        # render in small chunks so cancel requests take effect quickly;
        # sample numbering is GLOBAL across chunks (start_index offsets the
        # per-chunk enumerate) so later chunks never overwrite earlier ones.
        CHUNK = 4
        rendered: list[dict] = []
        for i in range(0, len(scenes), CHUNK):
            if cancelled():
                break
            chunk = scenes[i:i + CHUNK]
            res = render_scenes(chunk, out_root, "job", 2, None,
                                id_prefix=prefix, start_index=i + 1)
            if res["samples"] != len(chunk):
                raise RuntimeError(
                    f"rendered {res['samples']}/{len(chunk)} in chunk")
            rendered.extend(chunk)
            with _JOB_LOCK:
                JOBS[job_id]["done"] = len(rendered)
        if cancelled():
            with _JOB_LOCK:
                JOBS[job_id].update(status="cancelled",
                                    output_dir=str(out_root))
            return
        postprocess_labels(rendered, out_root, id_prefix=prefix,
                           start_index=1)
        with _JOB_LOCK:
            JOBS[job_id].update(status="completed", done=len(rendered),
                                total=len(scenes), output_dir=str(out_root))
    except Exception as e:  # noqa: BLE001 - job isolation
        with _JOB_LOCK:
            JOBS[job_id].update(status="failed", error=str(e)[:500])


class DetectRequest(BaseModel):
    run: str
    checkpoint: str = "best.pt"
    split: str = "val"
    sample_id: str


@app.get("/api/detect-runs")
def detect_runs():
    """List runs usable by the detection workspace (result.json + ckpt)."""
    import traceback
    runs_root = REPO / "runs"
    out = []
    if runs_root.exists():
        for d in sorted(runs_root.iterdir()):
            if not d.is_dir():
                continue
            res = d / "result.json"
            if not (res.exists() and (d / "best.pt").exists()):
                continue
            try:
                r = json.loads(res.read_text())
                out.append({
                    "run": d.name,
                    "status": r.get("status"),
                    "f1": (r.get("metrics") or {}).get("f1@0.5"),
                    "conf": (r.get("metrics") or {}).get(
                        "chosen_conf_threshold"),
                    "commit": (json.loads(
                        (d / "environment.json").read_text())
                        if (d / "environment.json").exists() else {}
                    ).get("git_commit"),
                })
            except Exception:
                traceback.print_exc()
    return {"runs": out}


@app.post("/api/detect")
def detect(req: DetectRequest):
    """Run a trained model on ONE dataset sample; returns predictions,
    ground truth and the camera frame for frontend overlays."""
    import base64
    import numpy as np
    import torch
    from PIL import Image
    from arrow_toy.models.model import ArrowToyModel
    from arrow_toy.utils.config import load_yaml

    run_dir = REPO / "runs" / req.run
    if not run_dir.exists():
        raise HTTPException(404, f"unknown run {req.run}")
    cfg = load_yaml(run_dir / "resolved_config.yaml")
    root = pathlib.Path(cfg["dataset"]["root"])
    sdir = root / "samples" / req.sample_id
    if not sdir.exists():
        raise HTTPException(404, f"unknown sample {req.sample_id}")
    labels = json.loads((sdir / "labels.json").read_text())
    cam_id = labels["training_camera_ids"][0]
    cal = labels["cameras"][cam_id]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ArrowToyModel(cfg).to(device)
    ck = torch.load(run_dir / req.checkpoint, map_location=device,
                    weights_only=True)
    model.load_state_dict(ck["model"])
    model.eval()
    img = np.asarray(Image.open(sdir / "rgb" / f"{cam_id}.png")
                     .convert("RGB"), dtype=np.float32) / 255.0
    img = (img - 0.5) / 0.5
    batch = {
        "image": torch.from_numpy(img.transpose(2, 0, 1))[None].to(device),
        "K": torch.tensor([cal["K"]], dtype=torch.float32, device=device),
        "T_C_E": torch.tensor([cal["T_C_E"]], dtype=torch.float32,
                              device=device),
    }
    with torch.no_grad():
        out = model(batch["image"], batch["K"], batch["T_C_E"])
    probs = out["pred_logits"].float().softmax(-1)[0, :, :6]
    best_p, best_c = probs.max(-1)
    centers = out["pred_center_norm"].float()[0]
    raw_dir = out["pred_direction_raw"].float()[0]
    dn = raw_dir / raw_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    res_path = run_dir / "result.json"
    conf = None
    if res_path.exists():
        conf = json.loads(res_path.read_text()).get("metrics", {}).get(
            "chosen_conf_threshold")
    rgb_png = (sdir / "rgb" / f"{cam_id}.png").read_bytes()
    return {
        "sample_id": req.sample_id,
        "camera": {"K": cal["K"], "T_C_E": cal["T_C_E"],
                   "width": cal["width"], "height": cal["height"]},
        "rgb_png_b64": base64.b64encode(rgb_png).decode(),
        "bev_range": labels["bev_range"],
        "conf_threshold": conf,
        "predictions": [
            {"prob": float(best_p[i]), "cls": int(best_c[i]),
             "center_norm": [float(centers[i][0]), float(centers[i][1])],
             "dir": [float(dn[i][0]), float(dn[i][1])]}
            for i in range(best_p.shape[0])
        ],
        "gt": [
            {"class_id": a["class_id"],
             "center_ego": a["center_ego"], "yaw_rad": a["yaw_rad"],
             "arrow_valid": a["arrow_valid"],
             "ignore_reason": a["ignore_reason"]}
            for a in labels["arrows"]
        ],
    }


@app.post("/api/render-jobs")
def create_render_job(req: RenderJobRequest):
    if req.mode == "variants" and req.scene is not None:
        _validate_scene(req.scene)  # synchronous 400, before job creation
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
