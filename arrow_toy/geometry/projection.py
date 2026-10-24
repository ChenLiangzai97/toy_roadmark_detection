"""Coordinate, camera and BEV contract implementations (design doc section 4).

All lengths in meters, all angles in radians. Matrices in JSON and here are
row-major nested lists. Frames:
  E (ego/world): x forward, y left, z up (right-handed)
  C (OpenCV):    X right, Y down, Z along optical axis
  G (three.js):  x right, y up, camera looks along local -z
T_<target>_<source> naming convention throughout.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

Mat4 = list[list[float]]
Mat3 = list[list[float]]

S_E_C_TO_G: Mat4 = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
"""Maps OpenCV camera coords to three.js camera coords (flip y and z)."""


# ---------- small matrix helpers (pure python, no numpy dependency) ----------

def mat4_mul(a: Mat4, b: Mat4) -> Mat4:
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)]
            for i in range(4)]


def mat4_identity() -> Mat4:
    return [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]


def mat4_transpose_rotation(m: Mat4) -> Mat4:
    """Transpose the 3x3 rotation block, keep translation column zeroed."""
    out = mat4_identity()
    for i in range(3):
        for j in range(3):
            out[i][j] = m[j][i]
    return out


def mat4_inverse_rigid(m: Mat4) -> Mat4:
    """Inverse of a rigid transform: [R|t]^-1 = [R^T | -R^T t]."""
    out = mat4_transpose_rotation(m)
    t = [m[i][3] for i in range(3)]
    for i in range(3):
        out[i][3] = -sum(out[i][k] * t[k] for k in range(3))
    return out


def transform_point(T: Mat4, p: Sequence[float]) -> list[float]:
    x, y, z = p[0], p[1], p[2]
    return [T[0][0] * x + T[0][1] * y + T[0][2] * z + T[0][3],
            T[1][0] * x + T[1][1] * y + T[1][2] * z + T[1][3],
            T[2][0] * x + T[2][1] * y + T[2][2] * z + T[2][3],
            T[3][0] * x + T[3][1] * y + T[3][2] * z + T[3][3]]


def T_G_E_from_T_C_E(T_C_E: Mat4) -> Mat4:
    return mat4_mul(S_E_C_TO_G, T_C_E)


def T_C_E_from_T_G_E(T_G_E: Mat4) -> Mat4:
    return mat4_mul(S_E_C_TO_G, T_G_E)  # S is its own inverse


# ---------- camera intrinsics ----------

@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def K(self) -> Mat3:
        return [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]


def intrinsics_from_fov_v(fov_v_rad: float, width: int, height: int,
                          cx: float | None = None, cy: float | None = None) -> Intrinsics:
    """fx=fy derived from vertical FOV; principal point defaults to image center.

    Pixel centers are at half-integers, so the center of a WxH image is
    [W/2, H/2].
    """
    fy = height / (2.0 * math.tan(fov_v_rad / 2.0))
    return Intrinsics(fx=fy, fy=fy,
                      cx=(width / 2.0) if cx is None else cx,
                      cy=(height / 2.0) if cy is None else cy)


# ---------- default camera ----------

def default_camera_T_C_E(pitch_down_rad: float = math.radians(12.0),
                         position_ego: Sequence[float] = (0.0, 0.0, 1.7)) -> Mat4:
    """R_C_E rows per design doc: [0,-1,0], [-sin a,0,-cos a], [cos a,0,-sin a]."""
    s, c = math.sin(pitch_down_rad), math.cos(pitch_down_rad)
    R = [[0.0, -1.0, 0.0],
         [-s, 0.0, -c],
         [c, 0.0, -s]]
    t = [-sum(R[i][k] * position_ego[k] for k in range(3)) for i in range(3)]
    return [[R[0][0], R[0][1], R[0][2], t[0]],
            [R[1][0], R[1][1], R[1][2], t[1]],
            [R[2][0], R[2][1], R[2][2], t[2]],
            [0.0, 0.0, 0.0, 1.0]]


# ---------- projection ----------

def project_points(points_ego: Iterable[Sequence[float]], K: Mat3 | Intrinsics,
                   T_C_E: Mat4) -> tuple[list[tuple[float, float]], list[bool]]:
    """Project ego-frame points to continuous pixel coords.

    Returns (uv, valid) where valid requires Z_C > 0. Caller checks image
    bounds; both are reported separately in labels.
    """
    if isinstance(K, Intrinsics):
        fx, fy, cx, cy = K.fx, K.fy, K.cx, K.cy
    else:
        fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    uv: list[tuple[float, float]] = []
    valid: list[bool] = []
    for p in points_ego:
        q = transform_point(T_C_E, p)
        X, Y, Z = q[0], q[1], q[2]
        ok = Z > 1e-9
        if ok:
            uv.append((fx * X / Z + cx, fy * Y / Z + cy))
        else:
            uv.append((float("nan"), float("nan")))
        valid.append(ok)
    return uv, valid


def opengl_projection_matrix(K: Mat3 | Intrinsics, width: int, height: int,
                             near: float, far: float) -> Mat4:
    """OpenGL-style projection matrix P mapping three.js camera coords to clip
    space, equivalent to the pinhole model for skew-free K (design doc 4.2)."""
    if isinstance(K, Intrinsics):
        fx, fy, cx, cy = K.fx, K.fy, K.cx, K.cy
    else:
        fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    W, H = float(width), float(height)
    n, f = near, far
    return [[2 * fx / W, 0.0, 1 - 2 * cx / W, 0.0],
            [0.0, 2 * fy / H, 2 * cy / H - 1, 0.0],
            [0.0, 0.0, -(f + n) / (f - n), -2 * f * n / (f - n)],
            [0.0, 0.0, -1.0, 0.0]]


def project_point_opengl(points_ego: Iterable[Sequence[float]], P: Mat4,
                         T_G_E: Mat4, width: int, height: int
                         ) -> tuple[list[tuple[float, float]], list[bool]]:
    """Project via the OpenGL matrix path (used to cross-check the pinhole path
    and to mirror what three.js does internally)."""
    uv, valid = [], []
    W, H = float(width), float(height)
    for p in points_ego:
        g = transform_point(T_G_E, p)
        x, y, z, w = transform_point(P, g)
        ok = w > 1e-9  # w = -z_G; positive means in front of the camera
        if ok:
            ndc_x, ndc_y = x / w, y / w
            uv.append(((ndc_x + 1.0) * W / 2.0, (1.0 - ndc_y) * H / 2.0))
        else:
            uv.append((float("nan"), float("nan")))
        valid.append(bool(ok))
    return uv, valid
