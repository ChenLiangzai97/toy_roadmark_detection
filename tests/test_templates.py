"""Template geometry tests: design-doc straight vertices, mirror semantics,
centering, yaw semantics, containment."""
from __future__ import annotations

import math
import pathlib

import pytest

from arrow_toy.geometry.templates import (
    CLASS_NAMES, load_all_templates, load_template, template_for_class,
    world_bbox, world_polygons, direction_xy, point_in_template_world,
    yaw_from_direction,
)

ASSETS = pathlib.Path(__file__).resolve().parents[1] / "assets" / "arrows"


def test_all_templates_present():
    ts = load_all_templates(ASSETS)
    assert set(ts) == {"straight_v1", "left_v1", "right_v1",
                       "straight_left_v1", "straight_right_v1", "left_right_v1"}
    for tid, t in ts.items():
        assert t.class_name == CLASS_NAMES[t.class_id]
        assert t.entry_axis_local == (1.0, 0.0)
        assert t.anchor_local == (0.0, 0.0)


def test_straight_matches_design_doc_vertices():
    t = load_template("straight_v1", ASSETS)
    expected = [[-1.5, -0.12], [0.5, -0.12], [0.5, -0.48], [1.5, 0.0],
                [0.5, 0.48], [0.5, 0.12], [-1.5, 0.12]]
    got = [list(p) for p in t.polygons[0]]
    assert len(t.polygons) == 1
    for e, g in zip(expected, got):
        assert abs(e[0] - g[0]) < 1e-9 and abs(e[1] - g[1]) < 1e-9


def test_right_is_mirror_of_left():
    lt = load_template("left_v1", ASSETS)
    rt = load_template("right_v1", ASSETS)
    assert lt.class_id == 1 and rt.class_id == 2
    assert sorted((x, -y) for poly in rt.polygons for x, y in poly) == \
        sorted((x, y) for poly in lt.polygons for x, y in poly)


def test_templates_are_bbox_centered():
    for t in load_all_templates(ASSETS).values():
        xs = [x for poly in t.polygons for x, _ in poly]
        ys = [y for poly in t.polygons for _, y in poly]
        assert abs((min(xs) + max(xs)) / 2) < 1e-9
        assert abs((min(ys) + max(ys)) / 2) < 1e-9


def test_yaw_semantics_independent_of_class():
    # A RIGHT arrow at yaw=0 keeps class RIGHT; yaw is geometric only.
    rt = template_for_class(2)
    polys = world_polygons(rt, (10.0, 0.0), 0.0, 1.0)
    # entry axis along +x: template rod should span towards -x from center
    xs = [x for poly in polys for x, _ in poly]
    assert min(xs) == pytest.approx(10.0 - rt.canonical_size[0] / 2)
    assert rt.class_name == "RIGHT"
    # direction at yaw=0 is +x
    assert direction_xy(0.0) == (1.0, 0.0)
    assert yaw_from_direction(0.0, 1.0) == pytest.approx(math.pi / 2)


def test_placement_transform():
    t = load_template("straight_v1", ASSETS)
    polys = world_polygons(t, (10.0, 2.0), math.pi / 2, 1.0)
    # rotate so entry axis (+a, template +x direction) points to +y (left)
    tip_local = (1.5, 0.0)
    import math as m
    c, s = m.cos(m.pi / 2), m.sin(m.pi / 2)
    tip = (10 + c * 1.5 - s * 0.0, 2 + s * 1.5 + c * 0.0)
    xs = [x for poly in polys for x, _ in poly]
    ys = [y for poly in polys for _, y in poly]
    assert max(ys) == pytest.approx(tip[1])
    bbox = world_bbox(t, (10.0, 2.0), math.pi / 2, 1.0)
    assert bbox[1] - bbox[0] == pytest.approx(0.96)  # width now along x


def test_containment():
    t = load_template("straight_v1", ASSETS)
    assert point_in_template_world(10.0, 0.0, t, (10.0, 0.0), 0.0, 1.0)
    assert not point_in_template_world(10.0, 0.9, t, (10.0, 0.0), 0.0, 1.0)
    assert not point_in_template_world(8.0, 0.0, t, (10.0, 0.0), 0.0, 1.0)

