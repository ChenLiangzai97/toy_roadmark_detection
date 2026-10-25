"""P2 editor acceptance through the real UI + live editor server.

Covers the full flow from the design doc gate: empty scene -> place
arrow/object/lane line -> add camera -> save JSON -> fresh page import ->
identical scene -> batch job (start, complete) and job cancel; plus
undo/redo and import validation of an unknown schema version.
"""
from __future__ import annotations

import json
import math
import pathlib
import socket
import subprocess
import sys
import time

import pytest
import numpy as np
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parents[1]
from conftest import BROWSER, SKIP_REASON, browser_path

DIST = REPO / "web" / "dist"
CHROME = browser_path()
HAVE = (DIST / "index.html").exists() and BROWSER is not None


def _stats(page) -> dict:
    txt = page.locator("h4:has-text('stats') + div").inner_text()
    out = {}
    for part in txt.replace("\n", ",").split(","):
        if ":" in part:
            k, v = part.split(":", 1)
            out[k.strip()] = int(v.strip())
    return out


def _mutate(scene: dict, fn) -> dict:
    s = json.loads(json.dumps(scene))
    fn(s)
    return s


def _offline_preview(scene: dict, out_png: pathlib.Path) -> None:
    """Render the front-camera rgb of `scene` through the offline worker."""
    import os
    import subprocess as sp
    tmp = out_png.parent / "offline_probe"
    items = [{"sample_id": "probe", "scene": scene, "camera_ids": ["front"],
              "passes": ["rgb"], "out_dir": str(tmp),
              "bev_range": scene["bev_range"]}]
    payload = json.dumps({"job_id": "probe", "scenes": items,
                          "render_url": (REPO / "web" / "dist" / "render.html").resolve().as_uri()})
    env = dict(os.environ)
    env.setdefault("ARROW_CHROME_BIN", str(CHROME))
    p = sp.run(["npx", "tsx", "scripts/render_batch.ts"],
               cwd=str(REPO / "web"), input=payload, capture_output=True,
               text=True, env=env)
    assert p.returncode == 0, p.stderr[-1500:]
    # the editor preview requests the FIRST camera; saved scene has 2 ->
    # worker wrote both; compare against the front camera render
    src = tmp / "rgb" / "front.png"
    import shutil
    shutil.copy(src, out_png)


def _load_rgba(png: pathlib.Path) -> np.ndarray:
    rgb = np.asarray(Image.open(png).convert("RGB"), dtype=np.uint8)
    return np.dstack([rgb, np.full((*rgb.shape[:2], 1), 255, np.uint8)])


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.mark.skipif(not HAVE, reason="web/dist or chromium not available")
def test_editor_full_flow(tmp_path):
    from playwright.sync_api import sync_playwright

    port = _free_port()
    server = subprocess.Popen(
        [sys.executable, "-m", "arrow_toy.editor_server",
         "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    for _ in range(50):
        time.sleep(0.3)
        try:
            import urllib.request
            urllib.request.urlopen(f"{base}/api/health", timeout=2)
            break
        except Exception:
            if server.poll() is not None:
                raise RuntimeError("editor server died")
    else:
        server.terminate()
        raise RuntimeError("editor server did not start")

    job_dirs = [REPO / "data" / "editor_jobs" / "flow_job",
                REPO / "data" / "editor_jobs" / "cancel_job"]
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True, executable_path=str(browser_path()))
            page = browser.new_page(
                viewport={"width": 1400, "height": 900},
                accept_downloads=True)
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"{base}/index.html")
            page.wait_for_selector("text=arrow-toy", timeout=15000)
            assert _stats(page) == {"arrows": 0, "objects": 0, "lines": 0,
                                    "lanes": 0, "cameras": 1}

            canvas = page.locator("canvas").first
            box = canvas.bounding_box()
            assert box
            cx, cy = box["x"] + box["width"] * 0.55, box["y"] + box["height"] * 0.55

            # place an arrow and an object
            page.click("button:has-text('+arrow')")
            page.mouse.click(cx, cy)
            page.wait_for_function(
                "() => document.body.innerText.includes('arrows: 1')")
            page.click("button:has-text('+box')")
            page.mouse.click(cx - 200, cy + 80)
            page.wait_for_function(
                "() => document.body.innerText.includes('objects: 1')")

            # undo / redo
            page.click("button[title='undo']")
            page.wait_for_function(
                "() => document.body.innerText.includes('objects: 0')")
            page.click("button[title='redo']")
            page.wait_for_function(
                "() => document.body.innerText.includes('objects: 1')")

            # draw a lane line (2 points) and a lane centerline
            page.click("button:has-text('lane line')")
            page.mouse.click(cx, cy - 150)
            page.mouse.click(cx + 150, cy - 150)
            page.click("button:has-text('finish solid')")
            page.wait_for_function(
                "() => document.body.innerText.includes('lines: 1')")
            page.click("button:has-text('lane (center)')")
            page.mouse.click(cx - 100, cy - 80)
            page.mouse.click(cx + 100, cy - 80)
            page.click("button:has-text('finish lane')")
            page.wait_for_function(
                "() => document.body.innerText.includes('lanes: 1')")

            # add a second training camera
            page.click("button:has-text('+camera')")
            page.wait_for_function(
                "() => document.body.innerText.includes('cameras: 2')")

            # camera preview renders through the live server
            page.wait_for_function(
                "() => { const c = document.querySelectorAll('canvas')[1];"
                " return c && c.width === 320 && c.height === 192; }",
                timeout=30000)

            # save scene json via download
            with page.expect_download() as dl:
                page.click("button:has-text('save')")
            saved_path = pathlib.Path(dl.value.path())
            saved = json.loads(saved_path.read_text())
            assert saved["schema_version"] == 1
            assert len(saved["arrows"]) == 1
            assert saved["arrows"][0]["template_id"] == "straight_v1"
            assert len(saved["objects"]) == 1
            assert len(saved["lane_lines"]) == 1
            assert len(saved["lanes"]) == 1
            assert len(saved["cameras"]) == 2
            for cam in saved["cameras"]:
                # sanity: fx=fy>0, principal point at image center
                assert cam["K"][0][0] == cam["K"][1][1] > 0
                assert cam["K"][0][2] == cam["width"] / 2
                assert cam["K"][1][2] == cam["height"] / 2

            # fresh page: import the saved scene, stats identical
            page2 = browser.new_page(viewport={"width": 1400, "height": 900})
            page2.goto(f"{base}/index.html")
            page2.wait_for_selector("text=arrow-toy")
            page2.once("filechooser", lambda fc: fc.set_files(str(saved_path)))
            page2.click("button:has-text('load')")
            page2.wait_for_function(
                "() => document.body.innerText.includes('arrows: 1') "
                "&& document.body.innerText.includes('objects: 1') "
                "&& document.body.innerText.includes('lines: 1') "
                "&& document.body.innerText.includes('lanes: 1') "
                "&& document.body.innerText.includes('cameras: 2')")
            assert _stats(page2) == {"arrows": 1, "objects": 1, "lines": 1,
                                     "lanes": 1, "cameras": 2}

            # import validation: bad scenes refused, state intact
            good = json.loads(saved_path.read_text())
            bad_cases = {
                "schema99": {"schema_version": 99},
                "missing_fields": {"schema_version": 1, "scene_id": "x"},
                "bad_matrix": _mutate(good, lambda s: s["T_E_W"].__setitem__(
                    0, [1, 2, 3])),
                "dup_instance_id": _mutate(good, lambda s: (
                    s["arrows"].__setitem__(0, {**s["arrows"][0],
                                                "instance_id":
                                                    s["arrows"][-1]["instance_id"]})
                    if len(s["arrows"]) > 1 else
                    s["arrows"].append({**s["arrows"][0],
                                        "instance_id": s["arrows"][0]["instance_id"]}))),
                "missing_seed": {k: v for k, v in good.items() if k != "seed"},
                "object_missing_fields": _mutate(good, lambda s: s["objects"].__setitem__(
                    0, {"object_id": "x", "shape": "box"})),
                "bad_focal": _mutate(good, lambda s: s["cameras"][0]["K"].__setitem__(
                    0, [0, 0, 160])),
                "unknown_template": _mutate(good, lambda s: s["arrows"][0].__setitem__(
                    "template_id", "banana_v9")),
            }
            for name, payload in bad_cases.items():
                f = tmp_path / f"bad_{name}.json"
                f.write_text(json.dumps(payload))
                page2.once("filechooser", lambda fc, f=f: fc.set_files(str(f)))
                page2.click("button:has-text('load')")
                page2.wait_for_function(
                    "() => document.body.innerText.includes('invalid scene')"
                    " || document.body.innerText.includes('unknown schema_version')",
                    timeout=10000)
                assert _stats(page2) == {"arrows": 1, "objects": 1, "lines": 1,
                                         "lanes": 1, "cameras": 2}, name

            # preview canvas pixels must EQUAL the offline render of the
            # same scene (same deterministic pipeline); force a refresh and
            # wait for completion so the canvas is not a stale frame
            preview_png = tmp_path / "preview_expected.png"
            _offline_preview(saved, preview_png)
            expected = _load_rgba(preview_png)
            page2.click("button:has-text('refresh')")
            page2.wait_for_function(
                "() => document.body.innerText.includes('preview updated')",
                timeout=30000)
            time.sleep(0.3)
            got = page2.evaluate(
                "() => { const c = document.querySelectorAll('canvas')[1];"
                " const ctx = c.getContext('2d');"
                " return Array.from(ctx.getImageData(0, 0, c.width, c.height).data); }")
            got = np.asarray(got, dtype=np.uint8)
            assert got.shape == expected.reshape(-1).shape, \
                (got.shape, expected.shape)
            assert bytes(got) == expected.tobytes(), \
                f"preview canvas differs from offline render: " \
                f"{np.abs(got.astype(int) - expected.reshape(-1).astype(int)).max()} max diff"

            # batch job: 9 samples -> crosses THREE chunks (4+4+1), must all
            # land on disk without overwrite (review fix #1)
            page2.fill("label:has-text('dataset') input", "flow_job")
            page2.locator("label:has-text('count') input").fill("9")
            page2.click("button:has-text('start job')")
            page2.wait_for_function(
                "() => document.body.innerText.includes('status: completed')",
                timeout=180000)
            out_dir = REPO / "data" / "editor_jobs" / "flow_job"
            samples = sorted((out_dir / "samples").iterdir())
            assert len(samples) == 9, [p.name for p in samples]
            for s in samples:
                assert (s / "labels.json").exists()
                assert (s / "rgb" / "front.png").exists()
                assert (s / "masks" / "front_visible.png").exists()

            # job cancel: start a larger job, cancel while queued/running
            page2.fill("label:has-text('dataset') input", "cancel_job")
            page2.locator("label:has-text('count') input").fill("60")
            page2.click("button:has-text('start job')")
            page2.wait_for_function(
                "() => ['queued','running'].some(s =>"
                " document.body.innerText.includes('status: ' + s))")
            page2.click("button:has-text('cancel')")
            page2.wait_for_function(
                "() => document.body.innerText.includes('status: cancelled')",
                timeout=60000)

            # ---- gizmo drag evidence: move actually changes the arrow,
            # undo restores EXACTLY, redo reproduces EXACTLY, save/reload
            # keeps the final state (review fix #3)
            def snap():
                return page2.evaluate(
                    "() => window.__sceneSnapshot()")

            s_before = snap()
            # click the arrow (it sits where placed) to attach the gizmo
            page2.mouse.click(cx, cy)
            time.sleep(0.6)
            # drag from the gizmo center (plane/XYZ handle) -- reliably
            # pickable; axis tips are too small for synthetic drags
            page2.mouse.move(cx, cy)
            page2.mouse.down()
            page2.mouse.move(cx + 114, cy + 38, steps=15)
            page2.mouse.up()
            time.sleep(0.6)
            s_dragged = snap()
            a0 = s_before["arrows"][0]
            a1 = s_dragged["arrows"][0]
            moved = math.dist(a0["center_ego"][:2], a1["center_ego"][:2])
            assert moved > 0.1, f"drag did not move the arrow ({moved:.3f} m)"

            page2.click("button[title='undo']")
            time.sleep(0.4)
            s_undo = snap()
            assert s_undo == s_before, "undo did not restore pre-drag scene"

            page2.click("button[title='redo']")
            time.sleep(0.4)
            s_redo = snap()
            assert s_redo == s_dragged, "redo did not reproduce dragged scene"

            # save the redone scene and reopen it in a fresh page: the arrow
            # must keep the dragged position
            with page2.expect_download() as dl2:
                page2.click("button:has-text('save')")
            p3 = pathlib.Path(dl2.value.path())
            reopened = json.loads(p3.read_text())
            assert math.dist(reopened["arrows"][0]["center_ego"][:2],
                             a1["center_ego"][:2]) < 1e-9

            # gizmo scale on an object: per-axis size change on disk after
            # save/reopen, undo restores exactly
            page3 = browser.new_page(viewport={"width": 1400, "height": 900})
            page3.goto(f"{base}/index.html")
            page3.wait_for_selector("text=arrow-toy")
            page3.once("filechooser", lambda fc: fc.set_files(str(p3)))
            page3.click("button:has-text('load')")
            page3.wait_for_function(
                "() => document.body.innerText.includes('loaded ')"
                " || document.body.innerText.includes('invalid scene')"
                " || document.body.innerText.includes('load failed')",
                timeout=10000)
            status_txt = page3.locator("div[style*='pointer-events']").first \
                .inner_text() if page3.locator("div[style*='pointer-events']").count() else ""
            assert "invalid" not in status_txt and "failed" not in status_txt, \
                f"page3 load failed: {status_txt}"
            snap3 = lambda: page3.evaluate(  # noqa: E731
                "() => window.__sceneSnapshot()")
            ob_before = snap3()["objects"][0]
            size_before = list(ob_before["size_abc"])
            # select the object (placed left-below the arrow), scale mode,
            # drag the X scale handle outward
            page3.mouse.click(cx - 200, cy + 80)
            time.sleep(0.6)
            page3.click("button:has-text('scale')")
            time.sleep(0.3)
            # X scale handle located at r=36px, 30 deg above horizontal
            # (probe-verified on this gizmo size / camera)
            hx = cx - 200 + 36 * math.cos(math.radians(30))
            hy = cy + 80 - 36 * math.sin(math.radians(30))
            page3.mouse.move(hx, hy)
            page3.mouse.down()
            page3.mouse.move(hx * 1 + 44, hy - 26, steps=15)
            page3.mouse.up()
            time.sleep(0.6)
            ob_after = snap3()["objects"][0]
            size_after = list(ob_after["size_abc"])
            assert size_after[0] > size_before[0] * 1.05, \
                f"scale drag did not grow x size: {size_before} -> {size_after}"
            assert abs(size_after[1] - size_before[1]) < 1e-6
            assert abs(size_after[2] - size_before[2]) < 1e-6
            # undo restores the exact pre-scale scene
            page3.click("button[title='undo']")
            time.sleep(0.4)
            assert snap3()["objects"][0]["size_abc"] == size_before

            assert not errors, errors
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=10)
        for d in job_dirs:
            if d.exists():
                import shutil
                shutil.rmtree(d, ignore_errors=True)
