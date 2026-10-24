"""Generate the six canonical arrow template JSON files under assets/arrows/.

Single source of truth: assets/arrows/*.json consumed by both TypeScript and
Python. This script writes them deterministically; run after editing shapes:
    python tools/gen_templates.py
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "arrows"

# ---- primitive builders (local frame: a forward, b left, meters) ----

def rect(a0: float, a1: float, b0: float, b1: float) -> list[list[float]]:
    return [[a0, b0], [a1, b0], [a1, b1], [a0, b1]]


def tri(p0, p1, p2):
    return [list(p0), list(p1), list(p2)]


def bbox(primitives):
    xs, ys = [], []
    for p in primitives:
        for x, y in p:
            xs.append(x)
            ys.append(y)
    return min(xs), max(xs), min(ys), max(ys)


def center_primitives(primitives):
    x0, x1, y0, y1 = bbox(primitives)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    out = [[[x - cx, y - cy] for x, y in p] for p in primitives]
    return out, (cx, cy), (x1 - x0, y1 - y0)


def build_template(template_id: str, class_id: int, class_name: str,
                   primitives, description: str) -> dict:
    centered, _, size = center_primitives(primitives)
    return {
        "schema_version": 1,
        "template_id": template_id,
        "class_id": class_id,
        "class_name": class_name,
        "entry_axis_local": [1.0, 0.0],
        "anchor_local": [0.0, 0.0],
        "canonical_size": {"length_a": round(size[0], 6), "width_b": round(size[1], 6)},
        "primitives": [
            {"type": "polygon", "points_ab": [[round(x, 6), round(y, 6)] for x, y in p]}
            for p in centered
        ],
        "description": description,
    }


def straight() -> dict:
    # Exact vertices from the design doc (bbox center already at origin).
    pts = [[-1.5, -0.12], [0.5, -0.12], [0.5, -0.48], [1.5, 0.0],
           [0.5, 0.48], [0.5, 0.12], [-1.5, 0.12]]
    return build_template("straight_v1", 0, "STRAIGHT", [pts],
                          "entry rod + forward arrowhead")

def left() -> dict:
    prim = [
        rect(-1.5, 0.0, -0.12, 0.12),          # entry rod
        rect(-0.12, 0.12, 0.0, 0.5),           # bend / branch to +b (left)
        tri((-0.48, 0.5), (0.48, 0.5), (0.0, 1.5)),  # left arrowhead
    ]
    return build_template("left_v1", 1, "LEFT", prim,
                          "entry rod + 90-degree bend toward +b with arrowhead")

def right() -> dict:
    prim = [
        rect(-1.5, 0.0, -0.12, 0.12),
        rect(-0.12, 0.12, -0.5, 0.0),          # branch to -b (right)
        tri((-0.48, -0.5), (0.48, -0.5), (0.0, -1.5)),
    ]
    return build_template("right_v1", 2, "RIGHT", prim,
                          "mirror of left_v1 along local b axis")

def straight_left() -> dict:
    prim = [
        rect(-1.5, 0.3, -0.12, 0.12),          # shared entry rod
        tri((0.3, -0.48), (0.3, 0.48), (1.5, 0.0)),   # forward arrowhead
        rect(-0.12, 0.12, 0.0, 0.5),           # left branch
        tri((-0.48, 0.5), (0.48, 0.5), (0.0, 1.5)),   # left arrowhead
    ]
    return build_template("straight_left_v1", 3, "STRAIGHT_LEFT", prim,
                          "shared entry rod with forward and left arrowheads")

def straight_right() -> dict:
    prim = [
        rect(-1.5, 0.3, -0.12, 0.12),
        tri((0.3, -0.48), (0.3, 0.48), (1.5, 0.0)),
        rect(-0.12, 0.12, -0.5, 0.0),          # right branch
        tri((-0.48, -0.5), (0.48, -0.5), (0.0, -1.5)),
    ]
    return build_template("straight_right_v1", 4, "STRAIGHT_RIGHT", prim,
                          "shared entry rod with forward and right arrowheads")

def left_right() -> dict:
    prim = [
        rect(-1.5, 0.0, -0.12, 0.12),          # shared entry rod
        rect(-0.12, 0.12, 0.0, 0.5),           # left branch
        tri((-0.48, 0.5), (0.48, 0.5), (0.0, 1.5)),
        rect(-0.12, 0.12, -0.5, 0.0),          # right branch
        tri((-0.48, -0.5), (0.48, -0.5), (0.0, -1.5)),
    ]
    return build_template("left_right_v1", 5, "LEFT_RIGHT", prim,
                          "shared entry rod with left and right arrowheads")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    templates = [straight(), left(), right(), straight_left(),
                 straight_right(), left_right()]
    # sanity: right is the exact b-mirror of left after centering (order-insensitive)
    lt = [p["points_ab"] for p in left()["primitives"]]
    rt = [p["points_ab"] for p in right()["primitives"]]
    assert len(lt) == len(rt)
    for pl, pr in zip(lt, rt):
        assert sorted(map(tuple, pl)) == sorted((x, -y) for x, y in pr), \
            "right != mirror(left)"
    for t in templates:
        path = OUT / f"{t['template_id']}.json"
        path.write_text(json.dumps(t, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)} size={t['canonical_size']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
