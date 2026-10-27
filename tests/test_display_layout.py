"""Display-layer regressions (CSS-level, not canvas pixels): dock panels
do not overlap or clip, the camera preview keeps its intrinsic aspect
ratio at multiple resolutions, the BEV canvas keeps its native
far-up/left-left orientation (no rotation), and arrow thumbnails follow
the forward-up / left-left display convention."""
from __future__ import annotations

import json
import math
import pathlib
import socket
import subprocess
import sys
import time

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tests.conftest import BROWSER, _editor_cam_pixel, browser_path  # noqa: E402
from tests.test_editor_flow import _free_port  # noqa: E402

DIST = REPO / "web" / "dist"
HAVE = (DIST / "index.html").exists() and BROWSER is not None


def _rect(page, selector):
    return page.evaluate(
        """(sel) => { const r = document.querySelector(sel)
             .getBoundingClientRect();
             return {x: r.x, y: r.y, w: r.width, h: r.height}; }""",
        selector)


def _overlap(a, b):
    return not (a["x"] + a["w"] <= b["x"] or b["x"] + b["w"] <= a["x"]
                or a["y"] + a["h"] <= b["y"] or b["y"] + b["h"] <= a["y"])


@pytest.mark.skipif(not HAVE, reason="web/dist or browser missing")
def test_dock_display_layout():
    from playwright.sync_api import sync_playwright

    port = _free_port()
    server = subprocess.Popen(
        [sys.executable, "-m", "arrow_toy.editor_server",
         "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    import urllib.request
    for _ in range(50):
        time.sleep(0.3)
        try:
            urllib.request.urlopen(f"{base}/api/health", timeout=2)
            break
        except Exception:
            if server.poll() is not None:
                raise RuntimeError("editor server died")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True, executable_path=str(browser_path()))
            page = browser.new_page(viewport={"width": 1400, "height": 900})
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"{base}/index.html")
            page.wait_for_selector("text=arrow-toy", timeout=15000)
            # wait for the FIRST real preview render (canvas intrinsic
            # sizes are set by drawB64, not the 300x150 default)
            page.wait_for_function(
                "() => { const c = document.querySelector("
                "'[data-testid=preview-canvas]');"
                " const b = document.querySelector("
                "'[data-testid=bev-canvas]');"
                " return c && c.width === 320 && c.height === 192"
                " && b && b.width === 160 && b.height === 320; }",
                timeout=30000)
            time.sleep(1.0)

            def check_ratio(label):
                cv = _rect(page, "[data-testid=preview-canvas]")
                intrinsic = page.evaluate(
                    "() => { const c = document.querySelector("
                    "'[data-testid=preview-canvas]');"
                    " return c.width / c.height; }")
                displayed = cv["w"] / cv["h"]
                assert abs(displayed - intrinsic) / intrinsic < 0.02, (
                    label, intrinsic, displayed)

            # default 320x192 camera: displayed aspect == intrinsic
            check_ratio("default")

            # panels: preview box and BEV canvas must not overlap, and
            # both must be fully inside the viewport (no clipping)
            pb = _rect(page, "[data-testid=preview-box]")
            bev = _rect(page, "[data-testid=bev-canvas]")
            assert not _overlap(pb, bev), (pb, bev)
            vp_w, vp_h = 1400, 900
            for name, r in (("preview", pb), ("bev", bev)):
                assert r["x"] >= 0 and r["y"] >= 0, (name, r)
                assert r["x"] + r["w"] <= vp_w, (name, r)
                assert r["y"] + r["h"] <= vp_h, (name, r)

            # BEV: native orientation - displayed aspect must match the
            # intrinsic 160x320 (tall, NOT rotated to wide)
            bev_intrinsic = page.evaluate(
                "() => { const c = document.querySelector("
                "'[data-testid=bev-canvas]');"
                " return {w: c.width, h: c.height, "
                "transform: c.style.transform}; }")
            assert (bev_intrinsic["w"], bev_intrinsic["h"]) == (160, 320)
            assert bev["h"] > bev["w"], "BEV canvas must stay tall"
            assert bev_intrinsic["transform"] == "", (
                "BEV must not carry a CSS rotation")

            # arrow thumbnails: forward-UP convention
            thumbs = page.evaluate('''() => {
              const out = [];
              for (const cv of document.querySelectorAll("canvas")) {
                if (cv.width !== 44 || cv.height !== 44) continue;
                const ctx = cv.getContext("2d");
                const d = ctx.getImageData(0, 0, 44, 44).data;
                let minX = 99, maxX = -1, minY = 99, maxY = -1;
                for (let y = 0; y < 44; y++) for (let x = 0; x < 44; x++) {
                  const i = (y * 44 + x) * 4;
                  if (d[i] > 200) {
                    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
                    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
                  }
                }
                out.push({minX, maxX, minY, maxY,
                          w: maxX - minX, h: maxY - minY});
              }
              return out;
            }''')
            assert len(thumbs) >= 6, "expected 6 arrow thumbnails"
            straight, left, right = thumbs[0], thumbs[1], thumbs[2]
            # straight: vertical bar (forward up)
            assert straight["h"] > straight["w"] * 1.5, straight
            # left arrow's branch reaches further LEFT than straight's
            assert left["minX"] < straight["minX"] - 5, (left, straight)
            # right arrow's branch reaches further RIGHT
            assert right["maxX"] > straight["maxX"] + 5, (right, straight)

            # non-default camera resolution: ratio still follows the
            # intrinsic canvas (240x180 -> 4:3). Select the camera first
            # so the properties panel shows its fields.
            vb = page.locator("[data-testid=view3d]").bounding_box()
            cu, cv2_ = _editor_cam_pixel((0, 0, 1.7),
                                         (vb["width"], vb["height"]))
            page.mouse.click(vb["x"] + cu, vb["y"] + cv2_)
            time.sleep(0.8)
            page.locator("label:has-text('width px') input").fill("240")
            time.sleep(0.3)
            page.locator("label:has-text('height px') input").fill("180")
            time.sleep(0.5)
            page.click("button:has-text('刷新')")
            page.wait_for_function(
                "() => { const c = document.querySelector("
                "'[data-testid=preview-canvas]');"
                " return c && c.width === 240 && c.height === 180; }",
                timeout=30000)
            time.sleep(0.8)
            check_ratio("240x180")

            assert not errors, errors
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=10)
