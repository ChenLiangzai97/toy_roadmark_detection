"""Deterministic scene synthesis for the arrow toy dataset (design doc 7).

All randomness flows from a seeded random.Random (CPython Mersenne Twister);
the master seed and per-scene seeds are recorded in scene JSON and manifest.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from arrow_toy.geometry.bev import BEV_RANGE_DEFAULT
from arrow_toy.geometry.projection import (
    Intrinsics, default_camera_T_C_E, intrinsics_from_fov_v, project_points,
)
from arrow_toy.geometry.templates import (
    ALLOWED_TURNS_SLR, CLASS_NAMES, template_for_class, world_polygons,
)

GROUND_COLORS = ["#3d423c", "#40454f", "#4a4a44", "#37403b"]
LINE_COLOR = "#d9d9d9"
ARROW_COLOR = "#f2f2f2"


def _rng(master_seed: int, *keys: str) -> random.Random:
    h = master_seed
    for k in keys:
        h = (h * 1000003 + _stable_str(k)) & 0xFFFFFFFFFFFF
    return random.Random(h)


def _stable_str(s: str) -> int:
    return int.from_bytes(s.encode("utf-8"), "big") & 0xFFFFFFFF


@dataclass
class GenConfig:
    bev_range: tuple[float, float, float, float] = BEV_RANGE_DEFAULT
    scale_range: tuple[float, float] = (0.85, 1.15)
    arrow_margin_m: float = 0.6
    n_lanes_range: tuple[int, int] = (2, 4)
    lane_width_range: tuple[float, float] = (3.0, 3.8)
    road_heading_deg_range: tuple[float, float] = (-40.0, 40.0)
    free_arrow_fraction: float = 0.3
    objects_offroad_range: tuple[int, int] = (0, 6)
    fov_v_deg: float = 60.0
    cam_width: int = 320
    cam_height: int = 192
    cam_position: tuple[float, float, float] = (0.0, 0.0, 1.7)
    pitch_down_deg: float = 12.0
    near: float = 0.1
    far: float = 100.0
    ground_size: tuple[float, float] = (60.0, 40.0)


def default_camera(cfg: GenConfig, rng: Optional[random.Random] = None,
                   pose_perturb: bool = False) -> dict:
    if pose_perturb and rng is not None:
        pos = [cfg.cam_position[0],
               cfg.cam_position[1],
               rng.uniform(1.3, 2.2)]
        pitch = math.radians(rng.uniform(6.0, 18.0))
        yaw = math.radians(rng.uniform(-10.0, 10.0))
    else:
        pos = list(cfg.cam_position)
        pitch = math.radians(cfg.pitch_down_deg)
        yaw = 0.0
    T = _T_C_E_with_yaw(pitch, yaw, pos)
    K = intrinsics_from_fov_v(math.radians(cfg.fov_v_deg), cfg.cam_width,
                              cfg.cam_height)
    return {
        "camera_id": "front", "width": cfg.cam_width, "height": cfg.cam_height,
        "K": K.K(), "T_C_E": T, "near": cfg.near, "far": cfg.far,
        "position_ego": pos,  # metadata only; T_C_E is the source of truth
    }


def _T_C_E_with_yaw(pitch_down: float, yaw: float, pos: list[float]):
    """R_C_E = Ry(yaw) applied to the pitch-only default.

    Compose: first default pitch-down rotation R0, then rotate the camera
    about ego z by yaw: R = Rz(yaw) @ R0.
    """
    s, c = math.sin(pitch_down), math.cos(pitch_down)
    R0 = [[0.0, -1.0, 0.0], [-s, 0.0, -c], [c, 0.0, -s]]
    sy, cy = math.sin(yaw), math.cos(yaw)
    Rz = [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]]
    R = [[sum(Rz[i][k] * R0[k][j] for k in range(3)) for j in range(3)]
         for i in range(3)]
    t = [-sum(R[i][k] * pos[k] for k in range(3)) for i in range(3)]
    return [[R[i][0], R[i][1], R[i][2], t[i]] for i in range(3)] + \
        [[0.0, 0.0, 0.0, 1.0]]


@dataclass
class _Layout:
    layout_id: str
    road_heading: float                  # rad, lane tangent direction
    lanes: list[dict] = field(default_factory=list)
    lane_lines: list[dict] = field(default_factory=list)
    bound_arrow_slots: list[dict] = field(default_factory=list)  # x,y,yaw,lane_id
    free_slots: list[dict] = field(default_factory=list)


def _make_layout(layout_id: str, rng: random.Random, cfg: GenConfig) -> _Layout:
    n_lanes = rng.randint(*cfg.n_lanes_range)
    width = rng.uniform(*cfg.lane_width_range)
    heading = math.radians(rng.uniform(*cfg.road_heading_deg_range))
    y0 = rng.uniform(-2.0, 2.0)
    x_min, x_max, y_lo, y_hi = cfg.bev_range
    lanes = []
    for j in range(n_lanes):
        y_c = y0 + (j - (n_lanes - 1) / 2) * width
        if y_c < y_lo + 0.5 or y_c > y_hi - 0.5:
            continue
        lanes.append({
            "lane_id": f"lane_{j}",
            "centerline_ego": [
                [x_min - 6.0, y_c, 0.0],
                [x_min - 2.0 + 24.0 * math.cos(0), y_c, 0.0],
                [x_max + 8.0, y_c, 0.0],
            ],
            "width_m": width,
            "allowed_turns_slr": [0, 0, 0],
            "world_rule_known": False,
            "rule_source": None,
            "show_boundaries": False,
        })
    if not lanes:  # degenerate; force one center lane
        lanes.append({
            "lane_id": "lane_0",
            "centerline_ego": [[0.0, 0.0, 0.0], [26.0, 0.0, 0.0]],
            "width_m": width, "allowed_turns_slr": [0, 0, 0],
            "world_rule_known": False, "rule_source": None,
            "show_boundaries": False,
        })
    # heading rotates the whole road: lanes become polylines along heading
    cx, sy = math.cos(heading), math.sin(heading)
    for lane in lanes:
        base = lane["centerline_ego"]
        y_c = base[0][1]
        lane["centerline_ego"] = [
            [-4.0 + 0.0, y_c, 0.0],
            [10.0 * cx, y_c + 10.0 * sy, 0.0],
            [26.0 * cx, y_c + 26.0 * sy, 0.0],
        ]
    # lane lines at outer boundaries
    ys = [lane["centerline_ego"][0][1] for lane in lanes]
    width_avg = sum(l["width_m"] for l in lanes) / len(lanes)
    lane_lines = []
    for k, y in enumerate([min(ys) - width_avg / 2, max(ys) + width_avg / 2]):
        pts = [[-4.0 + 0.0, y, 0.0], [26.0 * cx, y + 26.0 * sy, 0.0]]
        lane_lines.append({
            "line_id": f"line_{k}", "points_ego": pts, "style": "solid",
            "width_m": 0.12, "color": LINE_COLOR,
        })
    for k in range(len(lanes) - 1):
        y_mid = (ys[k] + ys[k + 1]) / 2
        pts = [[0.0, y_mid, 0.0], [26.0 * cx, y_mid + 26.0 * sy, 0.0]]
        lane_lines.append({
            "line_id": f"dline_{k}", "points_ego": pts, "style": "dashed",
            "width_m": 0.10, "color": LINE_COLOR,
            "dash_length_m": 2.0, "gap_length_m": 2.0,
        })
    # arrow slots: 4-5 candidate positions along lanes
    x_min_bev, x_max_bev, y_lo_bev, y_hi_bev = cfg.bev_range
    bound_slots = []
    for lane in lanes:
        y_c = lane["centerline_ego"][0][1]
        for x in [4.5, 7.5, 10.5, 13.5, 16.5]:
            # project slot x onto the lane direction
            px = x
            py = y_c + (x) * sy if False else y_c + x * math.tan(heading) \
                if abs(math.tan(heading)) < 10 else y_c
            py = y_c + x * sy / cx if abs(cx) > 1e-6 else y_c
            bound_slots.append({"x": px, "y": py, "yaw": heading,
                                "lane_id": lane["lane_id"]})
    free_slots = []
    for x in [5.0, 8.0, 11.0, 14.0, 16.5]:
        free_slots.append({"x": x, "y": 0.0, "yaw": None, "lane_id": None})
    return _Layout(layout_id, heading, lanes, lane_lines, bound_slots, free_slots)


def _template_half_extent(class_id: int, scale: float) -> tuple[float, float]:
    t = template_for_class(class_id)
    la, wb = t.canonical_size
    return la * scale / 2, wb * scale / 2


def _fits_bev(class_id: int, cx: float, cy: float, yaw: float, scale: float,
              cfg: GenConfig) -> bool:
    from arrow_toy.geometry.templates import world_bbox
    t = template_for_class(class_id)
    x0, x1, y0, y1 = world_bbox(t, (cx, cy), yaw, scale)
    x_min, x_max, y_lo, y_hi = cfg.bev_range
    return x0 >= x_min and x1 <= x_max and y0 >= y_lo and y1 <= y_hi


def _bbox_overlap(a, b, margin: float) -> bool:
    return not (a[1] + margin < b[0] or b[1] + margin < a[0] or
                a[3] + margin < b[2] or b[3] + margin < a[2])


def _arrow_bbox(class_id, cx, cy, yaw, scale):
    from arrow_toy.geometry.templates import world_bbox
    t = template_for_class(class_id)
    x0, x1, y0, y1 = world_bbox(t, (cx, cy), yaw, scale)
    return (x0, x1, y0, y1)


def _object_occludes_arrow(obj: dict, arrow_box: tuple[float, float, float, float],
                           cam_pos, K: Intrinsics, T_C_E) -> bool:
    """Cheap conservative test: project object corners; if the projected bbox
    overlaps the arrow's projected bbox and the object is nearer, reject."""
    ox, oy, oz = obj["center_ego"]
    sx, sy, sz = obj["size_abc"][0], obj["size_abc"][1], obj["size_abc"][2]
    corners = [(ox - sx / 2, oy - sy / 2, oz - sz / 2), (ox + sx / 2, oy - sy / 2, oz - sz / 2),
               (ox - sx / 2, oy + sy / 2, oz - sz / 2), (ox + sx / 2, oy + sy / 2, oz - sz / 2),
               (ox - sx / 2, oy - sy / 2, oz + sz / 2), (ox + sx / 2, oy - sy / 2, oz + sz / 2),
               (ox - sx / 2, oy + sy / 2, oz + sz / 2), (ox + sx / 2, oy + sy / 2, oz + sz / 2)]
    uvs, valid = project_points(corners, K, T_C_E)
    if not all(valid):
        return False
    us = [u for u, _ in uvs]
    vs = [v for _, v in uvs]
    obj_box = (min(us), max(us), min(vs), max(vs))
    au, av = project_points(
        [[arrow_box[0], arrow_box[2], 0.0], [arrow_box[1], arrow_box[3], 0.0]],
        K, T_C_E)[0]
    arr_box = (min(au) - 2, max(au) + 2, min(av) - 2, max(av) + 2)
    return _bbox_overlap(obj_box, arr_box, margin=0.0)


def _projected_size_ok(class_id, cx, cy, yaw, scale, K, T_C_E, W, H):
    """Exact check on projected template polygons: total shoelace area >= 64 px
    and projected bbox short side >= 4 px, all vertices inside the image."""
    t = template_for_class(class_id)
    polys = world_polygons(t, (cx, cy), yaw, scale)
    uvs_all = []
    for poly in polys:
        uvs, valid = project_points([[x, y, 0.0] for x, y in poly], K, T_C_E)
        if not all(valid):
            return False, 0.0, 0.0
        uvs_all.append(uvs)
        for u, v in uvs:
            if u < 0 or u > W or v < 0 or v > H:
                return False, 0.0, 0.0
    area = 0.0
    us = [u for uvs in uvs_all for u, _ in uvs]
    vs = [v for uvs in uvs_all for _, v in uvs]
    for uvs in uvs_all:
        n = len(uvs)
        s = 0.0
        for i in range(n):
            u1, v1 = uvs[i]
            u2, v2 = uvs[(i + 1) % n]
            s += u1 * v2 - u2 * v1
        area += abs(s) / 2.0
    w = max(us) - min(us)
    h = max(vs) - min(vs)
    return (area >= 64.0 and min(w, h) >= 4.0), area, min(w, h)


def _pick_class_balanced(rng: random.Random, used: list[int],
                         split_counts: list[int] | None = None) -> int:
    counts = [used.count(c) + (split_counts[c] if split_counts else 0)
              for c in range(6)]
    m = min(counts)
    candidates = [c for c in range(6) if counts[c] == m]
    return rng.choice(candidates)


def make_scene(split: str, layout: _Layout, variant_idx: int, master_seed: int,
               cfg: GenConfig, *, n_arrows: int | None = None,
               occlusion_mode: str | None = None, pose_perturb: bool = False,
               class_swap_of: list[dict] | None = None,
               stress: bool = False,
               class_counter: list[int] | None = None) -> dict:
    """Build one SceneV1 from a layout with per-variant randomization.

    class_counter (len 6) balances instance classes ACROSS the whole split,
    not per scene; required by design doc 7.4 (instance-level balance).
    """
    rng = _rng(master_seed, split, layout.layout_id, f"v{variant_idx}")
    scene_id = f"{split}_{layout.layout_id}_v{variant_idx:02d}"
    cam = default_camera(cfg, rng, pose_perturb=pose_perturb)
    K = Intrinsics(cam["K"][0][0], cam["K"][1][1], cam["K"][0][2], cam["K"][1][2])
    T_C_E = cam["T_C_E"]

    arrows: list[dict] = []
    used_boxes: list[tuple] = []
    used_classes: list[int] = []

    def try_place(class_id, slot, scale):
        cx, cy = slot["x"], slot["y"]
        yaw = slot["yaw"]
        if yaw is None:
            yaw = rng.uniform(-math.pi, math.pi)
        if not _fits_bev(class_id, cx, cy, yaw, scale, cfg):
            return None
        box = _arrow_bbox(class_id, cx, cy, yaw, scale)
        for b2 in used_boxes:
            if _bbox_overlap(box, b2, cfg.arrow_margin_m):
                return None
        ok, _, _ = _projected_size_ok(class_id, cx, cy, yaw, scale, K, T_C_E,
                                      cfg.cam_width, cfg.cam_height)
        if not ok:
            return None
        used_boxes.append(box)
        return {"box": box, "cx": cx, "cy": cy, "yaw": yaw,
                "class_id": class_id, "scale": scale}

    if class_swap_of:
        for entry in class_swap_of:
            r = try_place(entry["class_id"],
                          {"x": entry["cx"], "y": entry["cy"], "yaw": entry["yaw"]},
                          entry["scale"])
            if r:
                arrows.append(r)
    else:
        if n_arrows is None:
            u = rng.random()
            if u < 0.15:
                n = 0
            elif u < 0.50:
                n = 1
            elif u < 0.80:
                n = 2
            elif u < 0.95:
                n = 3
            else:
                n = 4
        else:
            n = n_arrows
        slots = list(layout.bound_arrow_slots) * 3
        rng.shuffle(slots)
        free_slots = list(layout.free_slots) * 3
        rng.shuffle(free_slots)
        split_counts = class_counter if class_counter is not None else \
            [0, 0, 0, 0, 0, 0]
        for k in range(n):
            class_id = _pick_class_balanced(rng, used_classes, split_counts)
            used_classes.append(class_id)
            split_counts[class_id] += 1
            placed = None
            free = rng.random() < cfg.free_arrow_fraction
            pool = free_slots if free and free_slots else slots
            while pool:
                slot = pool.pop()
                scale = rng.uniform(*cfg.scale_range)
                placed = try_place(class_id, slot, scale)
                if placed:
                    break
            if not placed and free:
                while slots:
                    slot = slots.pop()
                    scale = rng.uniform(*cfg.scale_range)
                    placed = try_place(class_id, slot, scale)
                    if placed:
                        break
            if placed:
                arrows.append(placed)

    # assign instance ids deterministically
    arrows.sort(key=lambda r: (r["cy"], r["cx"]))
    scene_arrows = []
    for i, r in enumerate(arrows, start=1):
        scene_arrows.append({
            "instance_id": i,
            "template_id": f"{CLASS_NAMES[r['class_id']].lower()}_v1",
            "class_id": r["class_id"],
            "center_ego": [r["cx"], r["cy"], 0.0],
            "yaw_rad": r["yaw"],
            "scale": r["scale"],
            "lane_id": None,
        })

    # bind arrows to nearest lane (bound slots keep their lane)
    for r, sa in zip(arrows, scene_arrows):
        if r.get("lane_id"):
            sa["lane_id"] = r["lane_id"]
        else:
            best, best_d = None, 1e9
            for lane in layout.lanes:
                ly = lane["centerline_ego"][0][1]
                d = abs(sa["center_ego"][1] - ly)
                if d < best_d and d < 2.5:
                    best, best_d = lane["lane_id"], d
            sa["lane_id"] = best

    # lane rules from bound arrows (world truth known by construction)
    for lane in layout.lanes:
        rule = [0, 0, 0]
        known = False
        for sa in scene_arrows:
            if sa["lane_id"] == lane["lane_id"]:
                for d in range(3):
                    rule[d] = max(rule[d], ALLOWED_TURNS_SLR[sa["class_id"]][d])
                known = True
        if known:
            lane["allowed_turns_slr"] = rule
            lane["world_rule_known"] = True
            lane["rule_source"] = "ARROWS"
        else:
            lane["allowed_turns_slr"] = [0, 0, 0]
            lane["world_rule_known"] = False
            lane["rule_source"] = None

    # objects
    objects: list[dict] = []
    cam_pos = list(cam["position_ego"])
    if occlusion_mode:
        objects = _occlusion_objects(rng, scene_arrows, occlusion_mode, K, T_C_E, cfg)
    elif not stress:
        n_obj = rng.randint(*cfg.objects_offroad_range)
        for k in range(n_obj):
            for _attempt in range(8):
                side = rng.choice([-1, 1])
                oy = side * rng.uniform(5.0, 9.0)
                ox = rng.uniform(3.0, 20.0)
                shape = rng.choice(["box", "cylinder", "car"])
                size = [rng.uniform(0.5, 3.0), rng.uniform(0.5, 2.5),
                        rng.uniform(0.5, 2.2)]
                if shape == "car":
                    size = [rng.uniform(2.5, 4.5), rng.uniform(1.4, 1.9),
                            rng.uniform(1.3, 1.8)]
                obj = {
                    "object_id": f"obj_{k}", "shape": shape,
                    "size_abc": size,
                    "center_ego": [ox, oy, 0.0],
                    "yaw_rad": rng.uniform(-math.pi, math.pi),
                    "color": rng.choice(["#6f7378", "#8a8f96", "#5d6268",
                                         "#9aa0a8", "#7d8288"]),
                }
                if any(_object_occludes_arrow(obj, a["box"], cam_pos, K, T_C_E)
                       for a in arrows):
                    continue
                objects.append(obj)
                break

    cam_out = {k: v for k, v in cam.items() if k != "position_ego"}
    scene = {
        "schema_version": 1,
        "scene_id": scene_id,
        "scene_group_id": layout.layout_id,
        "seed": rng.randint(0, 2**31 - 1),
        "coordinate_frame": "ego_x_forward_y_left_z_up",
        "units": {"length": "m", "angle": "rad"},
        "T_E_W": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "bev_range": {"x_min": cfg.bev_range[0], "x_max": cfg.bev_range[1],
                      "y_min": cfg.bev_range[2], "y_max": cfg.bev_range[3]},
        "ground": {"size_x": cfg.ground_size[0], "size_y": cfg.ground_size[1],
                   "color": rng.choice(GROUND_COLORS), "texture": None},
        "objects": objects,
        "lane_lines": layout.lane_lines,
        "lanes": layout.lanes,
        "arrows": scene_arrows,
        "cameras": [cam_out],
        "render": {
            "light": {
                "direction_ego": [rng.uniform(-0.8, 0.2), rng.uniform(-0.8, 0.8),
                                  rng.uniform(0.7, 1.3)],
                "intensity": rng.uniform(0.8, 1.3),
                "ambient": rng.uniform(0.35, 0.6),
            },
            "arrow_color": ARROW_COLOR,
            "background": "#101418",
        },
        "randomization": {"variant_idx": variant_idx, "split": split,
                          "occlusion_mode": occlusion_mode,
                          "pose_perturb": pose_perturb, "stress": stress},
        "arrow_template_version": "v1",
    }
    return scene


def _occlusion_objects(rng, scene_arrows, mode, K, T_C_E, cfg):
    """Place tall boxes between camera and arrows to create the four occlusion
    situations (design doc 7.3)."""
    cam = cfg.cam_position
    out = []
    if not scene_arrows:
        return out
    if mode == "fully_occluded":
        targets = scene_arrows
        full = True
    elif mode == "partially_occluded":
        targets = scene_arrows[:1]
        full = False
    elif mode == "one_visible_one_occluded":
        targets = scene_arrows[-1:]
        full = True
    elif mode == "all_occluded":
        targets = scene_arrows
        full = True
    else:
        raise ValueError(mode)
    for k, arrow in enumerate(targets):
        ax, ay = arrow["center_ego"][0], arrow["center_ego"][1]
        t = rng.uniform(0.35, 0.6)
        ox = t * ax
        oy = t * ay + rng.uniform(-0.3, 0.3)
        if full:
            sx = max(0.8, abs(ay - oy) + 0.8)
            sy = sx
        else:
            sx = rng.uniform(0.5, 0.8)
            sy = 0.6
        out.append({
            "object_id": f"occ_{k}", "shape": "box",
            "size_abc": [sx, sy, 2.6],
            "center_ego": [ox, oy, 0.0],
            "yaw_rad": 0.0, "color": "#8a8f96",
        })
    return out


def generate_split(split: str, count: int, master_seed: int, cfg: GenConfig,
                   variants_per_layout: int = 4,
                   occlusion_modes: list[str] | None = None,
                   pose_perturb: bool = False, simple: bool = False,
                   class_swap: int = 0, stress: bool = False
                   ) -> list[dict]:
    """Generate `count` scenes for a split; layout groups stay within split."""
    scenes: list[dict] = []
    n_layouts = max(1, count // variants_per_layout)
    layout_rng = _rng(master_seed, split, "layouts")
    occlusion_cycle = occlusion_modes or [None]
    split_class_counts = [0, 0, 0, 0, 0, 0]
    made = 0
    li = 0
    while made < count:
        layout = _make_layout(f"{split}_layout_{li:05d}", layout_rng, cfg)
        for v in range(variants_per_layout):
            if made >= count:
                break
            mode = occlusion_cycle[made % len(occlusion_cycle)]
            scene = make_scene(
                split, layout, v, master_seed, cfg,
                n_arrows=(1 if simple else None),
                occlusion_mode=mode, pose_perturb=pose_perturb, stress=stress,
                class_counter=split_class_counts)
            scenes.append(scene)
            made += 1
        li += 1
    # class-swap controls: same layout, arrows same pose but permuted classes
    if class_swap:
        base = [s for s in scenes if s["arrows"]]
        swap_rng = _rng(master_seed, split, "class_swap")
        for k in range(class_swap):
            src = base[k % len(base)]
            layout_id = src["scene_group_id"] + f"_cs{k:02d}"
            entries = [{"class_id": (a["class_id"] + 1 + swap_rng.randint(0, 4)) % 6,
                        "cx": a["center_ego"][0], "cy": a["center_ego"][1],
                        "yaw": a["yaw_rad"], "scale": a["scale"]}
                       for a in src["arrows"]]
            scene = make_scene(split, _Layout(layout_id, 0.0), 0, master_seed,
                               cfg, class_swap_of=entries,
                               class_counter=split_class_counts)
            scene["scene_group_id"] = layout_id
            scene["scene_id"] = f"{split}_{layout_id}_v00"
            scenes.append(scene)
    return scenes
