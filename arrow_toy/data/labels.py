"""Label construction from rendered masks + geometry (design doc 6.5, 6.6).

Computes per-arrow per-camera visibility stats, arrow_valid / ignore_reason
and sample-level allow_no_object. Called after the Node worker wrote PNGs.
"""
from __future__ import annotations

import math
import pathlib
from typing import Optional

import numpy as np
from PIL import Image

from arrow_toy.geometry.projection import project_points
from arrow_toy.geometry.templates import template_for_class, world_polygons

MIN_AMODAL_PX = 64
MIN_BBOX_SIDE_PX = 4.0


def decode_instance_ids(png_path: pathlib.Path) -> np.ndarray:
    """PNG -> int32 instance id map (id = R + 256G + 65536B)."""
    arr = np.asarray(Image.open(png_path).convert("RGB"), dtype=np.uint8)
    ids = (arr[..., 0].astype(np.int32) + 256 * arr[..., 1].astype(np.int32)
           + 65536 * arr[..., 2].astype(np.int32))
    return ids


def _bbox_of_mask(mask: np.ndarray) -> Optional[dict]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return {"x0": float(xs.min()), "y0": float(ys.min()),
            "x1": float(xs.max()), "y1": float(ys.max())}


def visibility_for_arrow(scene: dict, arrow: dict, camera_id: str,
                         visible_ids: np.ndarray, amodal_mask: np.ndarray,
                         occlusion_tol_px: int = 0) -> dict:
    """All per-camera visibility fields from masks + geometry."""
    cal = next(c for c in scene["cameras"] if c["camera_id"] == camera_id)
    K, T = cal["K"], cal["T_C_E"]
    W, H = cal["width"], cal["height"]

    amodal_px = int(np.count_nonzero(amodal_mask))
    instance_id = arrow["instance_id"]
    visible_px = int(np.count_nonzero(visible_ids == instance_id))
    occluded_px = max(0, amodal_px - visible_px)

    # geometric checks on the amodal outline
    t = template_for_class(arrow["class_id"])
    polys = world_polygons(t, arrow["center_ego"][:2], arrow["yaw_rad"],
                           arrow["scale"])
    verts = [p for poly in polys for p in poly]
    uvs, valid_z = project_points([[v[0], v[1], 0.0] for v in verts], K, T)
    in_front = all(valid_z)
    truncated = (not in_front) or any(
        (not math.isnan(u)) and (u < 0 or u > W or v < 0 or v > H)
        for u, v in uvs
    )
    bbox_px = _bbox_of_mask(amodal_mask) or {"x0": 0.0, "y0": 0.0,
                                             "x1": -1.0, "y1": -1.0}
    visible_ratio = (visible_px / amodal_px) if amodal_px > 0 else 0.0
    occluded = occluded_px > occlusion_tol_px
    return {
        "amodal_px": amodal_px,
        "visible_px": visible_px,
        "occluded_px": occluded_px,
        "truncated": bool(truncated),
        "in_front": bool(in_front),
        "in_image": bool(amodal_px > 0),
        "bbox_px": bbox_px,
        "visible_ratio": float(min(1.0, visible_ratio)),
        "occluded": bool(occluded),
    }


def arrow_supervision(scene: dict, arrow: dict, vis: dict,
                      bev_range: dict) -> tuple[bool, Optional[str]]:
    """Returns (arrow_valid, ignore_reason) per design doc 6.5."""
    if not vis["in_front"]:
        return False, "outside_camera"
    if vis["truncated"]:
        return False, "truncated"
    # center + full outline inside BEV range (metric check)
    t = template_for_class(arrow["class_id"])
    from arrow_toy.geometry.templates import world_bbox
    x0, x1, y0, y1 = world_bbox(t, arrow["center_ego"][:2], arrow["yaw_rad"],
                                arrow["scale"])
    if not (x0 >= bev_range["x_min"] and x1 <= bev_range["x_max"]
            and y0 >= bev_range["y_min"] and y1 <= bev_range["y_max"]):
        return False, "outside_bev"
    if not vis["in_image"] or vis["amodal_px"] < MIN_AMODAL_PX:
        return False, "too_small"
    bb = vis["bbox_px"]
    if min(bb["x1"] - bb["x0"], bb["y1"] - bb["y0"]) + 1 < MIN_BBOX_SIDE_PX:
        return False, "too_small"
    if vis["occluded"]:
        return False, "occluded"
    return True, None


def build_labels(sample_id: str, scene: dict, camera_id: str,
                 visible_ids: np.ndarray,
                 amodal_masks: dict[int, np.ndarray],
                 occlusion_tol_px: int = 0) -> dict:
    """Full SampleLabelsV1 for one sample / training camera."""
    bev = scene["bev_range"]
    arrows_out = []
    any_ignored_in_bev = False
    for arrow in scene["arrows"]:
        amodal = amodal_masks.get(arrow["instance_id"])
        if amodal is None:
            raise ValueError(f"missing amodal mask for {arrow['instance_id']}")
        vis = visibility_for_arrow(scene, arrow, camera_id, visible_ids, amodal,
                                   occlusion_tol_px)
        valid, reason = arrow_supervision(scene, arrow, vis, bev)
        if not valid and reason in ("occluded", "truncated", "too_small",
                                    "no_visual_evidence"):
            any_ignored_in_bev = True
        arrows_out.append({
            "instance_id": arrow["instance_id"],
            "class_id": arrow["class_id"],
            "template_id": arrow["template_id"],
            "center_ego": list(arrow["center_ego"]),
            "yaw_rad": arrow["yaw_rad"],
            "direction_xy": [math.cos(arrow["yaw_rad"]),
                             math.sin(arrow["yaw_rad"])],
            "scale": arrow["scale"],
            "lane_id": arrow.get("lane_id"),
            "arrow_valid": bool(valid),
            "orientation_valid": bool(valid),
            "ignore_reason": reason,
            "visibility": {camera_id: vis},
        })
    # Design 6.6: allow_no_object=false ONLY when ignored arrows exist in the
    # BEV region (occluded/truncated/too_small/no_visual_evidence), which
    # would otherwise receive negative background supervision. Clean frames
    # with valid arrows still train background on unmatched queries; truly
    # empty frames with complete labels train background normally.
    allow_no_object = not any_ignored_in_bev
    # lanes
    lanes_out = []
    for lane in scene["lanes"]:
        bound = [a for a in arrows_out if a["lane_id"] == lane["lane_id"]]
        valid_bound = [a for a in bound if a["arrow_valid"]]
        conflicting = _rules_conflict(valid_bound)
        current_valid = bool(valid_bound) and not conflicting
        lanes_out.append({
            "lane_id": lane["lane_id"],
            "centerline_ego": [list(p) for p in lane["centerline_ego"]],
            "width_m": lane["width_m"],
            "allowed_turns_slr": list(lane["allowed_turns_slr"]),
            "allowed_turns_valid_slr": (
                _valid_turns_mask(valid_bound) if current_valid
                else [False, False, False]),
            "world_rule_known": bool(lane["world_rule_known"]),
            "current_rule_valid": current_valid,
            "rule_source": lane.get("rule_source"),
        })
    cal = next(c for c in scene["cameras"] if c["camera_id"] == camera_id)
    return {
        "schema_version": 1,
        "sample_id": sample_id,
        "scene_group_id": scene["scene_group_id"],
        "coordinate_frame": "ego_x_forward_y_left_z_up",
        "units": {"length": "m", "angle": "rad"},
        "bev_range": dict(bev),
        "training_camera_ids": [camera_id],
        "allow_no_object": bool(allow_no_object),
        "cameras": {camera_id: {
            "width": cal["width"], "height": cal["height"],
            "K": cal["K"], "T_C_E": cal["T_C_E"],
            "near": cal["near"], "far": cal["far"],
        }},
        "arrows": arrows_out,
        "lanes": lanes_out,
    }


def _rules_conflict(arrows: list[dict]) -> bool:
    """Bound arrows disagree on any single direction -> conflict (13)."""
    if len(arrows) <= 1:
        return False
    from arrow_toy.geometry.templates import ALLOWED_TURNS_SLR
    masks = [ALLOWED_TURNS_SLR[a["class_id"]] for a in arrows]
    for d in range(3):
        vals = {m[d] for m in masks}
        if len(vals) > 1:
            return True
    return False


def _valid_turns_mask(valid_bound: list[dict]) -> list[bool]:
    from arrow_toy.geometry.templates import ALLOWED_TURNS_SLR
    out = [False, False, False]
    for a in valid_bound:
        for d in range(3):
            out[d] = out[d] or bool(ALLOWED_TURNS_SLR[a["class_id"]][d])
    return out


IGNORE_NEG_REASONS = {"occluded", "truncated", "too_small", "no_visual_evidence"}


def frame_has_ignored_in_bev(labels: dict) -> bool:
    return any(a["ignore_reason"] in IGNORE_NEG_REASONS
               for a in labels["arrows"])
