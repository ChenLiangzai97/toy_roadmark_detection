"""Detection results workspace: model/sample selection, RGB + BEV GT and
prediction overlays with legend and confidence (final UI phase)."""
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
HAVE_RUN = HAVE and (REPO / "runs" / "baseline_s42" / "best.pt").exists()


@pytest.mark.skipif(not HAVE_RUN, reason="dist/browser/trained run missing")
def test_detection_workspace():
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
        # --- API level ---
        runs = json.loads(urllib.request.urlopen(
            f"{base}/api/detect-runs", timeout=10).read())["runs"]
        names = [r["run"] for r in runs]
        assert "baseline_s42" in names, names
        req = urllib.request.Request(
            f"{base}/api/detect",
            data=json.dumps({"run": "baseline_s42", "split": "val",
                             "sample_id": "val_000001"}).encode(),
            headers={"Content-Type": "application/json"})
        det = json.loads(urllib.request.urlopen(req, timeout=60).read())
        assert det["sample_id"] == "val_000001"
        assert det["camera"]["width"] == 320
        assert len(det["rgb_png_b64"]) > 1000
        assert 0 < len(det["predictions"]) == 32
        for p in det["predictions"]:
            assert 0.0 <= p["prob"] <= 1.0
            assert 0 <= p["cls"] <= 5
        assert len(det["gt"]) >= 0
        assert det["conf_threshold"] is not None
        # bad sample -> 404
        req404 = urllib.request.Request(
            f"{base}/api/detect",
            data=json.dumps({"run": "baseline_s42", "split": "val",
                             "sample_id": "nope_999999"}).encode(),
            headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req404, timeout=10)
            raise AssertionError("404 expected")
        except urllib.error.HTTPError as e:
            assert e.code == 404

        # --- UI level ---
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True, executable_path=str(browser_path()))
            page = browser.new_page(viewport={"width": 1400, "height": 900})
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"{base}/index.html")
            page.wait_for_selector("text=arrow-toy", timeout=15000)
            page.click("button:has-text('检测结果')")
            page.wait_for_selector("[data-testid=detection-panel]")
            # runs list loads
            page.wait_for_function(
                "() => document.querySelector("
                "'[data-testid=detection-panel] select').options.length > 0",
                timeout=15000)
            # pick a specific run so counts are deterministic
            page.select_option("[data-testid=detection-panel] >> select >> nth=0",
                               "baseline_s42")
            # type a sample id and run detection
            page.locator("[data-testid=detection-panel] input").fill(
                "val_000001")
            page.click("button:has-text('推理')")
            page.wait_for_selector("[data-testid=detect-rgb]", timeout=30000)
            page.wait_for_function(
                "() => { const c = document.querySelector("
                "'[data-testid=detect-rgb]');"
                " return c && c.width === 320 && c.height === 192; }",
                timeout=30000)
            # overlays actually drawn: green GT stroke color and a class
            # color must appear among non-background pixels
            colors = page.evaluate('''() => {
              const c = document.querySelector('[data-testid=detect-rgb]');
              const d = c.getContext('2d').getImageData(0, 0, 320, 192).data;
              const set = new Set();
              for (let i = 0; i < d.length; i += 4)
                set.add(d[i] + ',' + d[i+1] + ',' + d[i+2]);
              return [...set];
            }''')
            # GT green stroke (#3dff6e with AA -> allow range)
            def near(c, t, tol=60):
                return all(abs(a - b) <= tol for a, b in zip(c, t))

            has_green = any(near(tuple(int(x) for x in c.split(",")),
                                 (61, 255, 110)) for c in colors)
            assert has_green, "GT green marker not drawn"
            # BEV canvas present with native orientation
            bev = page.evaluate(
                "() => { const c = document.querySelector("
                "'[data-testid=detect-bev]');"
                " return [c.width, c.height]; }")
            assert bev == [160, 320], bev
            # legend + confidence text present
            body = page.inner_text("body")
            assert "真值" in body and "置信度" in body
            assert "阈值" in body
            # next sample navigation
            page.click("button:has-text('下一帧')")
            page.wait_for_function(
                "() => document.body.innerText.includes('val_000002')",
                timeout=30000)
            assert not errors, errors
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=10)
