"""UI phase 2: object library (incl. new sphere/cone shapes end-to-end
through the render pipeline) and the resizable side panel."""
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
from tests.conftest import BROWSER, browser_path  # noqa: E402
from tests.test_editor_flow import _free_port  # noqa: E402

DIST = REPO / "web" / "dist"
HAVE = (DIST / "index.html").exists() and BROWSER is not None


@pytest.mark.skipif(not HAVE, reason="web/dist or browser missing")
def test_object_library_and_panel_resize():
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
            page = browser.new_page(
                viewport={"width": 1400, "height": 900},
                accept_downloads=True)
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"{base}/index.html")
            page.wait_for_selector("text=arrow-toy", timeout=15000)
            canvas = page.locator("canvas").first
            cb = canvas.bounding_box()
            cx, cy = cb["x"] + cb["width"] * 0.5, cb["y"] + cb["height"] * 0.55

            snap = lambda: page.evaluate(  # noqa: E731
                "() => window.__sceneSnapshot()")

            # place a sphere and a cone from the library palette
            page.click("button:has-text('球体')")
            page.mouse.click(cx - 120, cy)
            page.wait_for_function(
                "() => document.body.innerText.includes('objects: 1')")
            page.click("button:has-text('锥体')")
            page.mouse.click(cx + 120, cy)
            page.wait_for_function(
                "() => document.body.innerText.includes('objects: 2')")
            s = snap()
            shapes = sorted(o["shape"] for o in s["objects"])
            assert shapes == ["cone", "sphere"], shapes
            sphere = next(o for o in s["objects"] if o["shape"] == "sphere")
            cone = next(o for o in s["objects"] if o["shape"] == "cone")
            assert sphere["size_abc"] == [1.2, 1.2, 1.2]
            assert cone["size_abc"] == [1.4, 1.4, 1.8]

            # save -> schema-valid scene with the new shapes
            with page.expect_download() as dl:
                page.click("button:has-text('save')")
            saved = json.loads(pathlib.Path(dl.value.path()).read_text())
            import jsonschema
            schema = json.loads(
                (REPO / "schemas" / "scene_v1.schema.json").read_text())
            jsonschema.validate(saved, schema)

            # export a labeled sample: the offline render pipeline must
            # handle the new shapes (they are occluders in the 3D scene)
            res = page.evaluate('''async () => {
              const r = await fetch('/api/render-sample', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({scene: window.__sceneSnapshot(),
                                      camera_id: 'front'}),
              });
              return {status: r.status, bytes: r.status === 200
                      ? (await r.blob()).size : 0};
            }''')
            assert res["status"] == 200 and res["bytes"] > 1000, res

            # resizable side panel: drag the splitter left/right
            def panel_width():
                return page.evaluate(
                    "() => { let best = 0;"
                    " for (const e of document.querySelectorAll('div')) {"
                    "  if (e.textContent && e.textContent.includes("
                    "'training camera preview')) {"
                    "   const r = e.getBoundingClientRect();"
                    "   if (r.width > 150 && (best === 0 || r.width < best))"
                    "     best = r.width; } } return Math.round(best); }")
            w0 = panel_width()
            splitter = page.locator("div[title='drag to resize panel']")
            sb = splitter.bounding_box()
            page.mouse.move(sb["x"] + 2, sb["y"] + 200)
            page.mouse.down()
            page.mouse.move(sb["x"] - 80, sb["y"] + 200, steps=8)
            page.mouse.up()
            time.sleep(0.3)
            w1 = panel_width()
            assert w1 > w0 + 60, (w0, w1)  # panel grew
            # and back: re-locate the splitter after it moved
            sb2 = page.locator("div[title='drag to resize panel']").bounding_box()
            page.mouse.move(sb2["x"] + 2, sb2["y"] + 200)
            page.mouse.down()
            page.mouse.move(sb2["x"] + 2 + 100, sb2["y"] + 200, steps=8)
            page.mouse.up()
            time.sleep(0.3)
            w2 = panel_width()
            assert w2 < w1 - 40, (w1, w2)

            assert not errors, errors
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=10)
