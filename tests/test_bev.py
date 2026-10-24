"""BEV grid contract tests: corners, front/back/left/right semantics,
metric<->cell round trip, normalization round trip, resolutions."""
from __future__ import annotations

import math

from arrow_toy.geometry.bev import BevGrid


def test_corner_semantics():
    g = BevGrid.feature()
    # front = large x, left = large y. row0/col0 = front-left.
    x, y = g.cell_center(0, 0)
    assert x == g.x_max - 0.5 * g.dx
    assert y == g.y_max - 0.5 * g.dy
    # front-right: row 0, last col
    x, y = g.cell_center(0, g.cols - 1)
    assert y == g.y_min + 0.5 * g.dy
    # back-left: last row, col 0
    x, y = g.cell_center(g.rows - 1, 0)
    assert x == g.x_min + 0.5 * g.dx
    named = g.corners_metric()
    rc = g.corners_rc()
    for name, (r, c) in rc.items():
        x, y = g.cell_center(r, c)
        xr, yr = named[name]
        assert abs(x - xr) < g.dx
        assert abs(y - yr) < g.dy
    assert named["front_left"][0] > named["back_left"][0]
    assert named["front_left"][1] > named["front_right"][1]


def test_metric_cell_roundtrip():
    g = BevGrid.feature()
    for r in range(0, g.rows, 7):
        for c in range(0, g.cols, 5):
            x, y = g.cell_center(r, c)
            r2, c2 = g.metric_to_cell(x, y)
            assert (r2, c2) == (r, c)


def test_normalize_roundtrip():
    g = BevGrid.feature()
    for (x, y) in [(2.5, -3.7), (17.9, 3.9), (10.0, 0.0), (2.0, -4.0)]:
        xn, yn = g.normalize(x, y)
        assert 0.0 <= xn <= 1.0 and 0.0 <= yn <= 1.0
        x2, y2 = g.denormalize(xn, yn)
        assert abs(x2 - x) < 1e-12 and abs(y2 - y) < 1e-12
    # output order is [x_n, y_n], never [row, col]
    xn, yn = g.normalize(2.0, -4.0)
    assert xn == 0.0 and yn == 0.0
    xn, yn = g.normalize(18.0, 4.0)
    assert xn == 1.0 and yn == 1.0


def test_resolutions():
    f = BevGrid.feature()
    assert abs(f.dx - 0.25) < 1e-12 and abs(f.dy - 0.25) < 1e-12
    m = BevGrid.mask()
    assert abs(m.dx - 0.05) < 1e-12 and abs(m.dy - 0.05) < 1e-12
    assert (m.rows, m.cols) == (320, 160)


def test_contains_and_direction_axis_independence():
    g = BevGrid.feature()
    assert g.contains(10.0, 0.0)
    assert not g.contains(1.9, 0.0)
    assert not g.contains(10.0, 4.1)
    # x axis (forward) maps to rows; y axis (left) maps to cols
    r_near, _ = g.metric_to_cell(3.0, 0.0)
    r_far, _ = g.metric_to_cell(17.0, 0.0)
    assert r_far < r_near
    _, c_left = g.metric_to_cell(10.0, 3.5)
    _, c_right = g.metric_to_cell(10.0, -3.5)
    assert c_left < c_right
