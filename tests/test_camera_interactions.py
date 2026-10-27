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


def _cam_pos_from_T(T):
    R = [[T[0][0], T[0][1], T[0][2]],
         [T[1][0], T[1][1], T[1][2]],
         [T[2][0], T[2][1], T[2][2]]]
    t = [T[0][3], T[1][3], T[2][3]]
    return [-sum(R[k][i] * t[k] for k in range(3)) for i in range(3)]




@pytest.mark.skipif(not HAVE, reason="web/dist or browser missing")
def test_scale_drag_absolute_and_deterministic():
    """Regression (review): gizmo scale must be ABSOLUTE from the drag
    start, not multiplied per objectChange (a single drag previously
    blew a box up to 5674x). Different mouse step counts must give the
    SAME final size."""
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
            canvas = page.locator("[data-testid=view3d]")
            cb = canvas.bounding_box()
            cw, ch = cb["width"], cb["height"]
            # place a box at a known spot and select it
            page.click("button:has-text('方盒')")
            bx, by = cb["x"] + cw * 0.5, cb["y"] + ch * 0.62
            page.mouse.click(bx, by)
            page.wait_for_function(
                "() => document.body.innerText.includes('objects: 1')")
            page.click("button:has-text('select')")
            time.sleep(0.4)
            # project the box center for a deterministic select click
            snap = lambda: page.evaluate(  # noqa: E731
                "() => window.__sceneSnapshot()")
            oc = snap()["objects"][0]["center_ego"]
            ou, ov = _editor_cam_pixel((oc[0], oc[1], oc[2]), (cw, ch))
            ox, oy = cb["x"] + ou, cb["y"] + ov
            page.mouse.click(ox, oy)
            time.sleep(0.8)
            page.click("button:has-text('scale')")
            time.sleep(0.4)
            s0 = snap()
            size0 = list(s0["objects"][0]["size_abc"])

            results = []
            for steps in (5, 20):
                # undo any previous gesture first
                if steps != 5:
                    page.click("button[title='undo']")
                    time.sleep(0.4)
                    assert snap()["objects"][0]["size_abc"] == size0
                hx = ox + 36 * math.cos(math.radians(30))
                hy = oy - 36 * math.sin(math.radians(30))
                page.mouse.move(hx, hy)
                page.mouse.down()
                page.mouse.move(hx + 44, hy - 26, steps=steps)
                page.mouse.up()
                time.sleep(0.8)
                sz = snap()["objects"][0]["size_abc"]
                results.append((steps, sz))
                # sanity: modest drag must NOT explode the size
                assert sz[0] < size0[0] * 5, (steps, size0, sz)
                assert abs(sz[1] - size0[1]) < 1e-6
                assert abs(sz[2] - size0[2]) < 1e-6
            # determinism: same gesture, different step counts, same size
            assert abs(results[0][1][0] - results[1][1][0]) < 1e-6, results
            # actually grew
            assert results[0][1][0] > size0[0] * 1.02, results
            # undo restores exactly
            page.click("button[title='undo']")
            time.sleep(0.4)
            assert snap()["objects"][0]["size_abc"] == size0
            page.click("button[title='redo']")
            time.sleep(0.4)
            assert snap()["objects"][0]["size_abc"] == results[0][1]
            assert not errors, errors
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=10)


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
            canvas = page.locator("[data-testid=view3d]")
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
            page.click("button[title='redo']")
            time.sleep(0.5)
            s_redo = snap()
            assert s_redo == s_after, "redo did not reproduce camera drag"

            # --- aim the camera by dragging the PREVIEW at 1x ---
            cal_before = json.loads(json.dumps(
                s_redo["cameras"][0]))
            pv = page.locator("[data-testid=preview-box]")
            box = pv.bounding_box()
            mx, my = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2

            def pose_of(T):
                # world-frame yaw/pitch from the optical axis (row 2)
                return (math.atan2(T[2][1], T[2][0]),
                        math.asin(-max(-1.0, min(1.0, T[2][2]))))

            yaw0, pitch0 = pose_of(cal_before["T_C_E"])

            # DEFAULT-pose round trip (world-frame contract): the panel's
            # displayed yaw/pitch decode the default as 0 deg / 12 deg,
            # NOT 90 deg / -12 deg (the old extraction bug)
            disp_yaw = float(page.locator(
                "label:has-text('yaw (deg)') input").input_value())
            disp_pitch = float(page.locator(
                "label:has-text('pitch down (deg)') input").input_value())
            assert abs(disp_yaw - 0.0) < 0.5, \
                f"default pose decoded yaw as {disp_yaw} deg (want 0)"
            assert abs(disp_pitch - 12.0) < 0.5, \
                f"default pose decoded pitch as {disp_pitch} deg (want 12)"

            # PANEL yaw edit -> optical axis (direct TS construction check):
            # set yaw to 0.6 rad (34.38 deg) and verify row 2 of T
            page.locator("label:has-text('yaw (deg)') input").fill("34.377468")
            time.sleep(0.6)
            yawed = snap()["cameras"][0]
            alpha = math.radians(12)
            f_expect = (math.cos(alpha) * math.cos(0.6),
                        math.cos(alpha) * math.sin(0.6),
                        -math.sin(alpha))
            f_got = yawed["T_C_E"][2][:3]
            for a_, b_ in zip(f_expect, f_got):
                assert a_ == pytest.approx(b_, abs=1e-6), (f_expect, f_got)
            page.click("button[title='undo']")
            time.sleep(0.5)
            assert snap()["cameras"][0]["T_C_E"] == cal_before["T_C_E"]

            # SMALL drag: continuity at 0.003 rad/px, no 90-degree jump
            page.mouse.move(mx, my)
            page.mouse.down()
            page.mouse.move(mx + 10, my + 2, steps=4)
            page.mouse.up()
            time.sleep(0.8)
            small = snap()["cameras"][0]
            yaw_s, pitch_s = pose_of(small["T_C_E"])
            dyaw = math.atan2(math.sin(yaw_s - yaw0), math.cos(yaw_s - yaw0))
            dpitch = pitch_s - pitch0
            assert abs(dyaw - (-10 * 0.003)) < 0.002, \
                f"small yaw drag moved {dyaw:.4f} rad (want -0.030)"
            assert abs(dpitch - 2 * 0.003) < 0.002, \
                f"small pitch drag moved {dpitch:.4f} rad (want 0.006)"
            page.click("button[title='undo']")
            time.sleep(0.5)

            # larger look-drag: heading changes, optical axis follows
            page.mouse.move(mx, my)
            page.mouse.down()
            page.mouse.move(mx + 60, my + 24, steps=8)
            page.mouse.up()
            time.sleep(0.8)
            s_look = snap()
            cal_after = s_look["cameras"][0]
            # rotation happened: T_C_E changed, K and position unchanged
            assert cal_after["T_C_E"] != cal_before["T_C_E"], \
                "look-drag did not rotate the camera"
            assert cal_after["K"] == cal_before["K"], "K must not change"
            pa = _cam_pos_from_T(cal_after["T_C_E"])
            pb = _cam_pos_from_T(cal_before["T_C_E"])
            assert math.dist(pa, pb) < 1e-9, "position must not change"
            # optical axis really rotated in the world frame
            yaw1, pitch1 = pose_of(cal_after["T_C_E"])
            big = math.atan2(math.sin(yaw1 - yaw0), math.cos(yaw1 - yaw0))
            assert abs(big - (-60 * 0.003)) < 0.02, \
                f"yaw moved {big:.3f} rad (want ~-0.18)"
            assert abs((pitch1 - pitch0) - 24 * 0.003) < 0.02
            # one undo returns to the pre-look state
            page.click("button[title='undo']")
            time.sleep(0.5)
            assert snap()["cameras"][0]["T_C_E"] == cal_before["T_C_E"]
            page.click("button[title='redo']")
            time.sleep(0.5)

            # --- save / reload keeps the edited calibration exactly ---
            with page.expect_download() as dl:
                page.click("button:has-text('保存')")
            saved = pathlib.Path(dl.value.path())
            saved_scene = json.loads(saved.read_text())
            assert saved_scene["cameras"][0]["T_C_E"] == \
                snap()["cameras"][0]["T_C_E"], "saved matrix drifted"

            # --- fullscreen viewer: open, modes, Esc ---
            page.click("button[title='open fullscreen viewer']")
            time.sleep(0.4)
            overlay = page.locator("div[style*='fixed']")
            assert overlay.first.is_visible()
            img = page.locator("img[alt='camera preview']")
            assert img.is_visible()
            page.click("button:has-text('原始 100%')")
            time.sleep(0.2)
            page.click("button:has-text('适应窗口')")
            time.sleep(0.2)
            page.keyboard.press("Escape")
            time.sleep(0.3)
            assert not img.is_visible(), "Esc must close the viewer"
            # calibration untouched by viewing
            assert snap()["cameras"][0]["T_C_E"] == \
                saved_scene["cameras"][0]["T_C_E"]

            # --- preview zoom ---
            preview = page.locator("[data-testid=preview-box]")
            canv2 = page.locator("[data-testid=preview-canvas]")
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
