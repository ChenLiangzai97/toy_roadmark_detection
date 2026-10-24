"""Data generation entry: scene synthesis -> Node render workers -> labels.

    python -m arrow_toy.render --config configs/data_v1.yaml
    python -m arrow_toy.render --config configs/data_v1.yaml --splits overfit --dry-run

Pipeline per split: generate scenes (seeded), validate schema, render passes
(rgb / visible / amodal / bev) via Playwright workers, then compute labels
from masks and write labels.json per sample. Splits and manifest are written
at the end; audit is a separate command.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

from arrow_toy.data import generate as gen
from arrow_toy.data.labels import build_labels, decode_instance_ids
from arrow_toy.utils.config import load_yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
SCHEMA_SCENE = REPO / "schemas" / "scene_v1.schema.json"
SCHEMA_LABELS = REPO / "schemas" / "sample_labels_v1.schema.json"
RENDER_HTML = REPO / "web" / "dist" / "render.html"
WORKER = REPO / "web" / "scripts" / "render_batch.ts"


def validate_json(instance, schema_path: pathlib.Path) -> None:
    import jsonschema
    schema = json.loads(schema_path.read_text())
    jsonschema.validate(instance, schema)


def build_scenes_for_config(cfg: dict, splits: list[str]) -> dict[str, list[dict]]:
    g = gen.GenConfig(
        bev_range=tuple(cfg["scene"]["default_bev_range"]),
        scale_range=tuple(cfg["arrows"]["scale_range"]),
        arrow_margin_m=cfg["arrows"]["arrow_margin_m"],
        n_lanes_range=tuple(cfg["lane_random"]["n_lanes_range"]),
        lane_width_range=tuple(cfg["lane_random"]["width_range"]),
        road_heading_deg_range=(-40.0, 40.0),
        fov_v_deg=cfg["scene"]["default_camera"]["fov_v_deg"],
        cam_width=cfg["scene"]["default_camera"]["width"],
        cam_height=cfg["scene"]["default_camera"]["height"],
        cam_position=tuple(cfg["scene"]["default_camera"]["position_ego"]),
        pitch_down_deg=cfg["scene"]["default_camera"]["pitch_down_deg"],
        ground_size=tuple(cfg["scene"]["ground_size"]),
    )
    master = cfg["seed"]
    out: dict[str, list[dict]] = {}
    for split in splits:
        t0 = time.time()
        if split == "overfit":
            scenes = gen.generate_split("overfit", cfg["splits"]["overfit"],
                                        master, g, variants_per_layout=1,
                                        simple=True)
        elif split == "train":
            scenes = gen.generate_split("train", cfg["splits"]["train"], master,
                                        g, variants_per_layout=4,
                                        class_swap=cfg["fixed_pose_swap_samples"])
        elif split == "val":
            scenes = gen.generate_split("val", cfg["splits"]["val"], master, g)
        elif split == "test_clean":
            scenes = gen.generate_split("test_clean", cfg["splits"]["test_clean"],
                                        master, g)
        elif split == "test_occlusion":
            scenes = gen.generate_split(
                "test_occlusion", cfg["splits"]["test_occlusion"], master, g,
                variants_per_layout=4,
                occlusion_modes=cfg["occlusion_set_mix"])
        elif split == "test_pose":
            scenes = gen.generate_split("test_pose", cfg["splits"]["test_pose"],
                                        master, g, variants_per_layout=4,
                                        pose_perturb=True)
        elif split == "stress":
            scenes = []
            srng = __import__("random").Random(master ^ 0x57)
            occlusion_cycle = [None]
            for k in range(cfg["stress_samples"]["count"]):
                n = srng.randint(*cfg["stress_samples"]["arrows"])
                layout = gen._make_layout(f"stress_layout_{k:03d}",
                                          __import__("random").Random(
                                              master + k), g)
                s = gen.make_scene("stress", layout, 0, master, g,
                                   n_arrows=n, stress=True)
                scenes.append(s)
        else:
            raise SystemExit(f"unknown split {split}")
        out[split] = scenes
        print(f"[render] {split}: {len(scenes)} scenes "
              f"({time.time() - t0:.1f}s synthesis)", flush=True)
    return out


def render_scenes(scenes: list[dict], out_root: pathlib.Path, split: str,
                  n_workers: int, render_url: str | None) -> dict:
    """Spawn node workers; returns summary stats."""
    import numpy as np

    split_dir = out_root / "samples"
    split_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for i, scene in enumerate(scenes, start=1):
        sample_id = f"{i:06d}"
        out_dir = split_dir / sample_id
        item = {
            "sample_id": sample_id,
            "scene": scene,
            "camera_ids": [scene["cameras"][0]["camera_id"]],
            "passes": ["rgb", "visible_instance", "amodal_instance",
                       "bev_instance"],
            "out_dir": str(out_dir),
            "bev_range": scene["bev_range"],
        }
        jobs.append(item)

    url = render_url or (RENDER_HTML.resolve().as_uri())
    # chunk jobs across workers
    chunks = [jobs[i::n_workers] for i in range(n_workers)]
    procs = []
    t0 = time.time()
    for w, chunk in enumerate(chunks):
        if not chunk:
            continue
        payload = json.dumps({"job_id": f"{split}_w{w}", "scenes": chunk,
                              "render_url": url})
        procs.append(subprocess.Popen(
            ["npx", "tsx", str(WORKER)], cwd=str(REPO / "web"),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True))
        procs[-1]._payload = payload  # type: ignore[attr-defined]
    results = {"samples": 0, "errors": []}
    for p in procs:
        out, err = p.communicate(p._payload, timeout=3600)  # type: ignore
        for line in out.splitlines():
            if not line.strip():
                continue
            msg = json.loads(line)
            if msg["kind"] == "sample_done":
                results["samples"] += 1
            elif msg["kind"] == "error":
                results["errors"].append(msg)
        if p.returncode != 0:
            results["errors"].append({"kind": "fatal", "message": err[-2000:]})
    print(f"[render] {split}: {results['samples']}/{len(jobs)} rendered "
          f"in {time.time() - t0:.1f}s, errors={len(results['errors'])}",
          flush=True)
    if results["errors"]:
        for e in results["errors"][:5]:
            print("[render] error:", e, flush=True)
    return results


def postprocess_labels(scenes: list[dict], out_root: pathlib.Path,
                       camera_id: str = "front") -> int:
    """Read masks, compute labels, write labels.json + verify ID legality."""
    samples_dir = out_root / "samples"
    n = 0
    for i, scene in enumerate(scenes, start=1):
        sample_id = f"{i:06d}"
        sdir = samples_dir / sample_id
        visible_ids = decode_instance_ids(sdir / "masks" / f"{camera_id}_visible.png")
        assigned = {a["instance_id"] for a in scene["arrows"]} | {0}
        found = set(np.unique(visible_ids).tolist())
        illegal = found - assigned
        if illegal:
            raise RuntimeError(
                f"sample {sample_id}: illegal instance ids {illegal}")
        amodal_masks = {}
        for a in scene["arrows"]:
            p = sdir / "masks" / "amodal" / camera_id / f"{a['instance_id']}.png"
            arr = np.asarray(__import__("PIL.Image", fromlist=["Image"])
                             .open(p).convert("L"), dtype=np.uint8)
            amodal_masks[a["instance_id"]] = arr > 127
        labels = build_labels(sample_id, scene, camera_id, visible_ids,
                              amodal_masks)
        (sdir / "labels.json").write_text(json.dumps(labels, indent=1))
        # keep scene json next to labels
        (sdir / "scene.json").write_text(json.dumps(scene, indent=1))
        n += 1
    return n


def write_splits(out_root: pathlib.Path, scenes_by_split: dict[str, list[dict]]) -> None:
    splits_dir = out_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split, scenes in scenes_by_split.items():
        ids = [f"{i:06d}" for i in range(1, len(scenes) + 1)]
        (splits_dir / f"{split}.txt").write_text("\n".join(ids) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_v1.yaml")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--splits", default=None,
                    help="comma list; default all from config")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--render-url", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="synthesize + validate scenes only, no rendering")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap scenes per split (debug)")
    args = ap.parse_args(argv)

    cfg = load_yaml(args.config)
    out_root = pathlib.Path(args.out_root or cfg["output_root"])
    splits = (args.splits.split(",") if args.splits
              else list(cfg["splits"].keys()))
    n_workers = args.workers or cfg["render"]["workers"]

    if not RENDER_HTML.exists() and not args.dry_run:
        print("[render] web/dist/render.html missing; run npm --prefix web run build first")
        return 2

    scenes_by_split: dict[str, list[dict]] = {}
    for split in splits:
        scenes = build_scenes_for_config(cfg, [split])[split]
        if args.limit:
            scenes = scenes[: args.limit]
        for s in scenes:
            validate_json(s, SCHEMA_SCENE)
        scenes_by_split[split] = scenes
        print(f"[render] {split}: {len(scenes)} scenes schema-valid", flush=True)
    if args.dry_run:
        for split, scenes in scenes_by_split.items():
            outd = out_root / "dry" / split
            outd.mkdir(parents=True, exist_ok=True)
            (outd / "scenes.jsonl").write_text(
                "\n".join(json.dumps(s) for s in scenes))
        print("[render] dry-run complete")
        return 0

    for split, scenes in scenes_by_split.items():
        res = render_scenes(scenes, out_root, split, n_workers, args.render_url)
        if res["samples"] != len(scenes):
            print(f"[render] {split}: incomplete render "
                  f"({res['samples']}/{len(scenes)}); aborting")
            return 3
        n = postprocess_labels(scenes, out_root)
        print(f"[render] {split}: {n} labels written", flush=True)

    write_splits(out_root, scenes_by_split)
    manifest = {
        "dataset": "arrow_v1",
        "config": cfg,
        "master_seed": cfg["seed"],
        "generator": "arrow_toy.render",
        "generator_version": "0.1.0",
        "splits": {k: len(v) for k, v in scenes_by_split.items()},
        "render_url": args.render_url or RENDER_HTML.resolve().as_uri(),
        "completed": False,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("[render] all splits done (manifest completed=false until audit passes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
