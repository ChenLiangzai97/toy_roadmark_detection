"""P2 editor acceptance through the real UI + live editor server.

Covers the full flow from the design doc gate: empty scene -> place
arrow/object/lane line -> add camera -> save JSON -> fresh page import ->
identical scene -> batch job (start, complete) and job cancel; plus
undo/redo and import validation of an unknown schema version.
"""
from __future__ import annotations

import json
import pathlib
import socket
import subprocess
import sys
import time

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
DIST = REPO / "web" / "dist"
CHROME = pathlib.Path.home() / ".cache/ms-playwright/chromium-1248/chrome-linux64/chrome"
HAVE = (DIST / "index.html").exists() and CHROME.exists()


def _stats(page) -> dict:
    txt = page.locator("h4:has-text('stats') + div").inner_text()
    out = {}
    for part in txt.replace("\n", ",").split(","):
        if ":" in part:
            k, v = part.split(":", 1)
            out[k.strip()] = int(v.strip())
    return out


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
                headless=True, executable_path=str(CHROME))
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

            # import validation: unknown schema_version refused, state intact
            bad = tmp_path / "bad_scene.json"
            bad.write_text(json.dumps({"schema_version": 99}))
            page2.once("filechooser", lambda fc: fc.set_files(str(bad)))
            page2.click("button:has-text('load')")
            page2.wait_for_function(
                "() => document.body.innerText.includes('unknown schema_version')")
            assert _stats(page2)["arrows"] == 1  # unchanged

            # batch job from the current scene: start, wait for completion
            page2.fill("label:has-text('dataset') input", "flow_job")
            page2.locator("label:has-text('count') input").fill("2")
            page2.click("button:has-text('start job')")
            page2.wait_for_function(
                "() => document.body.innerText.includes('status: completed')",
                timeout=120000)
            out_dir = REPO / "data" / "editor_jobs" / "flow_job"
            samples = sorted((out_dir / "samples").iterdir())
            assert len(samples) == 2
            for s in samples:
                assert (s / "labels.json").exists()
                assert (s / "rgb" / "front.png").exists()

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

            assert not errors, errors
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=10)
        for d in job_dirs:
            if d.exists():
                import shutil
                shutil.rmtree(d, ignore_errors=True)
