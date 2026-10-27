"""Shared test fixtures/browser discovery.

Browser resolution order (same rule as arrow_toy.render.find_chrome):
  1. $ARROW_CHROME_BIN (must exist)
  2. playwright cache globs (chromium*, chromium_headless_shell*)
Missing browser => browser tests SKIP by default with an explicit reason.
Acceptance mode: ARROW_REQUIRE_BROWSER=1 turns a missing browser into a
hard collection error so a green run proves the browser tests RAN.
"""
from __future__ import annotations

import os
import pathlib

import pytest

GLOBS = (
    ".cache/ms-playwright/chromium-*/chrome-linux*/chrome",
    ".cache/ms-playwright/chromium_headless_shell-*/"
    "chrome-headless-shell-linux*/chrome-headless-shell",
)


def find_browser() -> pathlib.Path | None:
    env = os.environ.get("ARROW_CHROME_BIN")
    if env:
        p = pathlib.Path(env)
        if p.exists():
            return p
        raise pytest.UsageError(
            f"ARROW_CHROME_BIN={env} does not exist; fix or unset it")
    home = pathlib.Path.home()
    for pat in GLOBS:
        hits = sorted(home.glob(pat), reverse=True)
        if hits:
            return hits[0]
    return None


BROWSER = find_browser()
REQUIRE_BROWSER = os.environ.get("ARROW_REQUIRE_BROWSER") == "1"

SKIP_REASON = (
    "no browser binary found: run `npx --prefix web playwright install "
    "chromium` or set ARROW_CHROME_BIN (acceptance mode: "
    "ARROW_REQUIRE_BROWSER=1 makes this an error instead of a skip)")


def pytest_configure(config):
    if REQUIRE_BROWSER and BROWSER is None:
        raise pytest.UsageError(
            "ARROW_REQUIRE_BROWSER=1 but no browser binary found; "
            "browser tests must run in acceptance mode")


def browser_path() -> pathlib.Path:
    """Browser for launched tests; asserts presence (callers are skipped
    when missing via the browser marker)."""
    assert BROWSER is not None, SKIP_REASON
    return BROWSER


def _editor_cam_pixel(point, canvas_wh):
    """Project a world point through the editor viewport camera
    (pos (-12,-12,14), lookAt (10,0,0), up +z, vfov 50 deg)."""
    import math
    W, H = canvas_wh
    pos = (-12.0, -12.0, 14.0)
    target = (10.0, 0.0, 0.0)
    up = (0.0, 0.0, 1.0)
    f = tuple((target[i] - pos[i]) for i in range(3))
    fl = math.sqrt(sum(v * v for v in f))
    f = tuple(v / fl for v in f)
    zc = tuple(-v for v in f)
    x = (up[1] * zc[2] - up[2] * zc[1],
         up[2] * zc[0] - up[0] * zc[2],
         up[0] * zc[1] - up[1] * zc[0])
    xl = math.sqrt(sum(v * v for v in x))
    x = tuple(v / xl for v in x)
    y = (zc[1] * x[2] - zc[2] * x[1],
         zc[2] * x[0] - zc[0] * x[2],
         zc[0] * x[1] - zc[1] * x[0])
    d = tuple(point[i] - pos[i] for i in range(3))
    X = sum(x[i] * d[i] for i in range(3))
    Y = sum(y[i] * d[i] for i in range(3))
    Z = sum(f[i] * d[i] for i in range(3))
    assert Z > 0
    fy = (H / 2) / math.tan(math.radians(50) / 2)
    u = fy * X / Z + W / 2
    v = -fy * Y / Z + H / 2
    return u, v
