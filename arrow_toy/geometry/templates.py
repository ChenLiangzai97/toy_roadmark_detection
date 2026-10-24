"""Arrow template loading and geometry (design doc section 6).

A template is a set of ground-plane polygons in local coords (a forward along
the entry axis, b to the left of it), centered so the canonical bbox center is
the anchor (origin). "Union" is represented as the set of primitives: rendering
draws all polygons as one instance; containment is any-polygon.
"""
from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass
from typing import Sequence

ASSETS_DIR = pathlib.Path(__file__).resolve().parents[2] / "assets" / "arrows"

CLASS_NAMES = ["STRAIGHT", "LEFT", "RIGHT", "STRAIGHT_LEFT", "STRAIGHT_RIGHT",
               "LEFT_RIGHT"]
NUM_CLASSES = 6
NO_OBJECT_CLASS = 6
ALLOWED_TURNS_SLR = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1],
                     [0, 1, 1]]
TEMPLATE_VERSION = "v1"


@dataclass(frozen=True)
class ArrowTemplate:
    template_id: str
    class_id: int
    class_name: str
    entry_axis_local: tuple[float, float]
    anchor_local: tuple[float, float]
    canonical_size: tuple[float, float]
    polygons: tuple[tuple[tuple[float, float], ...], ...]

    def local_polygons_scaled(self, scale: float
                              ) -> list[list[tuple[float, float]]]:
        return [[(scale * x, scale * y) for x, y in poly] for poly in self.polygons]


def _load(path: pathlib.Path) -> ArrowTemplate:
    d = json.loads(path.read_text(encoding="utf-8"))
    polys = tuple(tuple(tuple(p) for p in prim["points_ab"])
                  for prim in d["primitives"])
    return ArrowTemplate(
        template_id=d["template_id"],
        class_id=d["class_id"],
        class_name=d["class_name"],
        entry_axis_local=tuple(d["entry_axis_local"]),
        anchor_local=tuple(d["anchor_local"]),
        canonical_size=(d["canonical_size"]["length_a"],
                        d["canonical_size"]["width_b"]),
        polygons=polys,
    )


_CACHE: dict[str, ArrowTemplate] = {}


def load_template(template_id: str, assets_dir: pathlib.Path = ASSETS_DIR
                  ) -> ArrowTemplate:
    key = f"{assets_dir}|{template_id}"
    if key not in _CACHE:
        _CACHE[key] = _load(assets_dir / f"{template_id}.json")
    return _CACHE[key]


def load_all_templates(assets_dir: pathlib.Path = ASSETS_DIR
                       ) -> dict[str, ArrowTemplate]:
    out: dict[str, ArrowTemplate] = {}
    for path in sorted(assets_dir.glob("*.json")):
        t = _load(path)
        out[t.template_id] = t
    return out


def template_for_class(class_id: int,
                       assets_dir: pathlib.Path = ASSETS_DIR) -> ArrowTemplate:
    name = CLASS_NAMES[class_id].lower()
    return load_template(f"{name}_v1", assets_dir)


# ---------- world-space placement ----------

def world_polygons(template: ArrowTemplate, center_xy: Sequence[float],
                   yaw: float, scale: float
                   ) -> list[list[tuple[float, float]]]:
    """Place template on z=0: p_world = center + R(yaw) @ (scale * p_local)."""
    c, s = math.cos(yaw), math.sin(yaw)
    cx, cy = center_xy[0], center_xy[1]
    out = []
    for poly in template.local_polygons_scaled(scale):
        out.append([(cx + c * x - s * y, cy + s * x + c * y) for x, y in poly])
    return out


def world_bbox(template: ArrowTemplate, center_xy: Sequence[float], yaw: float,
               scale: float) -> tuple[float, float, float, float]:
    xs, ys = [], []
    for poly in world_polygons(template, center_xy, yaw, scale):
        for x, y in poly:
            xs.append(x)
            ys.append(y)
    return min(xs), max(xs), min(ys), max(ys)


def direction_xy(yaw: float) -> tuple[float, float]:
    """Entry-axis direction (tail -> junction) in ego frame; yaw=0 -> +x."""
    return math.cos(yaw), math.sin(yaw)


def yaw_from_direction(dx: float, dy: float) -> float:
    return math.atan2(dy, dx)


def point_in_template_world(x: float, y: float, template: ArrowTemplate,
                            center_xy: Sequence[float], yaw: float,
                            scale: float) -> bool:
    for poly in world_polygons(template, center_xy, yaw, scale):
        if _point_in_polygon(x, y, poly):
            return True
    return False


def _point_in_polygon(x: float, y: float,
                      poly: Sequence[tuple[float, float]]) -> bool:
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            x_int = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_int:
                inside = not inside
        j = i
    return inside


def inflate_bbox_world(template: ArrowTemplate, center_xy: Sequence[float],
                       yaw: float, scale: float, margin: float
                       ) -> tuple[float, float, float, float]:
    x0, x1, y0, y1 = world_bbox(template, center_xy, yaw, scale)
    return x0 - margin, x1 + margin, y0 - margin, y1 + margin
