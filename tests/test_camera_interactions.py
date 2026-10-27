"""Editor camera interactions: gizmo drag editing (position + aim) and
preview zoom (user feature request).

The camera's screen position in the 3D view is computed with the same
pinhole math the repo uses (editor camera: pos (-12,-12,14), lookAt
(10,0,0), up +z, vfov 50 deg), so the click targets are deterministic.
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

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from conftest import BROWSER, browser_path, _editor_cam_pixel  # noqa: E402
from tests.test_editor_flow import _free_port, _stats  # noqa: E402

DIST = REPO / "web" / "dist"
HAVE = (DIST / "index.html").exists() and BROWSER is not None




@pytest.mark.skipif(not HAVE, reason="web/dist or browser missing")
def test_camera_gizmo_and_preview_zoom():
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
            canvas = page.locator("canvas").first
            cb = canvas.bounding_box()
            cw, ch = cb["width"], cb["height"]

            snap = lambda: page.evaluate(  # noqa: E731
                "() => window.__sceneSnapshot()")

            # --- camera selection + gizmo drag ---
            # default camera sits at (0, 0, 1.7) in the 3D view
            u, v = _editor_cam_pixel((0, 0, 1.7), (cw, ch))
            px, py = cb["x"] + u, cb["y"] + v
            page.mouse.click(px, py)
            time.sleep(0.8)
            txt = page.inner_text("body")
            assert "camera front" in txt, "camera not selected"

            s_before = snap()
            # drag from the gizmo center (translate); move right/up in
            # screen space
            page.mouse.move(px, py)
            page.mouse.down()
            page.mouse.move(px + 120, py - 60, steps=15)
            page.mouse.up()
            time.sleep(0.8)
            s_after = snap()

            def cam_pos(sc):
                # position recovered from T_C_E inverse translation
                T = sc["cameras"][0]["T_C_E"]
                R = [[T[0][0], T[0][1], T[0][2]],
                     [T[1][0], T[1][1], T[1][2]],
                     [T[2][0], T[2][1], T[2][2]]]
                t = [T[0][3], T[1][3], T[2][3]]
                return [-sum(R[k][i] * t[k] for k in range(3))
                        for i in range(3)]

            moved = math.dist(cam_pos(s_before), cam_pos(s_after))
            assert moved > 0.15, f"camera drag did not move it ({moved:.3f} m)"

            # undo restores exactly
            page.click("button[title='undo']")
            time.sleep(0.5)
            assert snap() == s_before, "undo did not restore camera scene"

            # --- preview zoom ---
            preview = page.locator(
                "div[style*='aspect-ratio']").first
            canv2 = page.locator("canvas").nth(1)
            t0 = canv2.evaluate("el => el.style.transform")
            assert "scale(1)" in t0 or t0 == ""
            page.click("button[title='zoom in']")
            page.click("button[title='zoom in']")
            time.sleep(0.2)
            t1 = canv2.evaluate("el => el.style.transform")
            assert "scale(2.25)" in t1, t1
            # wheel zoom out then reset
            preview.hover()
            page.mouse.wheel(0, 120)
            time.sleep(0.2)
            t2 = canv2.evaluate("el => el.style.transform")
            assert "scale(1.875)" in t2 or "scale(1.9)" in t2, t2
            page.click("button[title='reset zoom/pan']")
            time.sleep(0.2)
            t3 = canv2.evaluate("el => el.style.transform")
            assert "scale(1)" in t3, t3
            # the preview wrapper exists and is visible
            assert preview.is_visible()

            assert not errors, errors
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=10)
