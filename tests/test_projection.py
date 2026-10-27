"""Projection contract tests (design doc 4.2, 15.1).

Covers:
- Python pinhole vs its own OpenGL-matrix path
- JS (real three.js PerspectiveCamera via tsx) vs Python pinhole, <=0.5 px
  on >=20 points per camera, including a NON-CENTER principal point and
  several poses (left/right, near/far, different pitches/yaws)
- left/right and near/far image orientation semantics
- rigid transform inverse round trips
"""
from __future__ import annotations

import json
import math
import pathlib
import subprocess
import sys

import pytest

from arrow_toy.geometry.projection import (
    Intrinsics, default_camera_T_C_E, intrinsics_from_fov_v,
    mat4_identity, mat4_inverse_rigid, mat4_mul, opengl_projection_matrix,
    project_point_opengl, project_points, T_C_E_from_T_G_E, T_G_E_from_T_C_E,
    transform_point,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
WEB = REPO / "web"


def _cameras_for_check():
    """Camera variants: default + offset principal point + poses."""
    W, H = 320, 192
    default_K = intrinsics_from_fov_v(math.radians(60), W, H)
    default_T = default_camera_T_C_E()
    offset_K = Intrinsics(fx=180.0, fy=170.0, cx=140.0, cy=80.0)
    yawed_T = default_camera_T_C_E(pitch_down_rad=math.radians(5))
    return [
        {"name": "default", "K": default_K.K(), "T_C_E": default_T},
        {"name": "offset_principal_point", "K": offset_K.K(), "T_C_E": default_T},
        {"name": "low_pitch", "K": default_K.K(), "T_C_E": yawed_T},
    ]


def _points_set_a(n=24):
    pts = []
    for i in range(n):
        x = 3.0 + 14.0 * (i / (n - 1))
        y = -3.5 + 7.0 * ((i * 7) % n) / (n - 1)
        z = 0.0
        pts.append([x, y, z])
    return pts


def test_rigid_inverse_roundtrip():
    T = default_camera_T_C_E(math.radians(12), (0.4, -0.2, 1.9))
    p = [5.0, 1.0, 0.3]
    q = transform_point(T, p)
    back = transform_point(mat4_inverse_rigid(T), q[:3])
    assert all(abs(back[i] - p[i]) < 1e-12 for i in range(3))
    prod = mat4_mul(T, mat4_inverse_rigid(T))
    for i in range(4):
        for j in range(4):
            assert abs(prod[i][j] - mat4_identity()[i][j]) < 1e-12


def test_TC_E_TG_E_roundtrip():
    T_C_E = default_camera_T_C_E()
    T2 = T_C_E_from_T_G_E(T_G_E_from_T_C_E(T_C_E))
    for i in range(4):
        for j in range(4):
            assert abs(T2[i][j] - T_C_E[i][j]) < 1e-12
    p = [10.0, 2.0, 0.0]
    pc = transform_point(T_C_E, p)
    pg = transform_point(T_G_E_from_T_C_E(T_C_E), p)
    # S = diag(1,-1,-1): y and z flip
    assert pg[0] == pytest.approx(pc[0])
    assert pg[1] == pytest.approx(-pc[1])
    assert pg[2] == pytest.approx(-pc[2])


def test_pinhole_vs_opengl_matrix_python():
    W, H = 320, 192
    for cam in _cameras_for_check():
        K, T = cam["K"], cam["T_C_E"]
        P = opengl_projection_matrix(K, W, H, 0.1, 100.0)
        pts = _points_set_a()
        uv1, valid1 = project_points(pts, K, T)
        uv2, valid2 = project_point_opengl(pts, P, T_G_E_from_T_C_E(T), W, H)
        assert valid1 == valid2
        assert valid1 == valid2
        for (u1, v1), (u2, v2), ok in zip(uv1, uv2, valid2):
            if ok:
                assert abs(u1 - u2) < 1e-6 and abs(v1 - v2) < 1e-6


def test_image_orientation_semantics():
    """Left ego point lands left of principal point; nearer ground point
    lands lower in the image (larger v) for a downlooking camera."""
    K = intrinsics_from_fov_v(math.radians(60), 320, 192)
    T = default_camera_T_C_E()
    ul, _ = project_points([[10.0, 2.0, 0.0]], K, T)
    ur, _ = project_points([[10.0, -2.0, 0.0]], K, T)
    cx = K.cx
    assert ul[0][0] < cx < ur[0][0]
    vn, _ = project_points([[4.0, 0.0, 0.0]], K, T)
    vf, _ = project_points([[16.0, 0.0, 0.0]], K, T)
    assert vn[0][1] > vf[0][1]
    # behind camera -> invalid
    _, valid = project_points([[-5.0, 0.0, 0.0]], K, T)
    assert valid[0] is False


def test_principal_point_offset_moves_projection():
    Kc = intrinsics_from_fov_v(math.radians(60), 320, 192)
    Ko = Intrinsics(Kc.fx, Kc.fy, cx=140.0, cy=80.0)
    T = default_camera_T_C_E()
    p = [[10.0, 1.5, 0.0]]
    c, _ = project_points(p, Kc, T)
    o, _ = project_points(p, Ko, T)
    assert o[0][0] - c[0][0] == pytest.approx(140.0 - 160.0, abs=1e-9)
    assert o[0][1] - c[0][1] == pytest.approx(80.0 - 96.0, abs=1e-9)


@pytest.mark.skipif(not (WEB / "node_modules").exists(), reason="npm install not run")
def test_threejs_vs_python_projection():
    """Real three.js projection vs Python pinhole: max error <= 0.5 px."""
    W, H = 320, 192
    pts = _points_set_a(24)
    for cam in _cameras_for_check():
        payload = json.dumps({
            "points": pts, "K": cam["K"], "T_C_E": cam["T_C_E"],
            "width": W, "height": H, "near": 0.1, "far": 100.0,
        })
        proc = subprocess.run(
            [sys.executable, "-c", "pass"] and ["npx", "tsx", "scripts/projection_check.ts"],
            input=payload, capture_output=True, text=True, cwd=str(WEB),
        )
        assert proc.returncode == 0, proc.stderr
        out = json.loads(proc.stdout)
        uv_py, valid_py = project_points(pts, cam["K"], cam["T_C_E"])
        assert out["valid"] == valid_py, f"valid mismatch for {cam['name']}"
        errs = [
            math.hypot(a[0] - b[0], a[1] - b[1])
            for a, b in zip(uv_py, out["uv"])
        ]
        assert max(errs) <= 0.5, f"{cam['name']}: max err {max(errs):.4f}px"


def test_camera_yaw_pitch_world_frame():
    """World-frame yaw/pitch contract (review): the optical axis (row 2
    of R_C_E) must rotate in the world X-Y plane with yaw, and the
    default pose must match the design-doc R exactly at yaw=0."""
    from arrow_toy.data.generate import _T_C_E_with_yaw
    alpha = math.radians(12)
    pos = [0.3, -0.2, 1.7]
    T0 = _T_C_E_with_yaw(alpha, 0.0, pos)
    T1 = _T_C_E_with_yaw(alpha, 0.6, pos)
    # design-doc default rows at yaw=0
    assert T0[0][:3] == pytest.approx([0, -1, 0])
    assert T0[1][:3] == pytest.approx(
        [-math.sin(alpha), 0, -math.cos(alpha)])
    assert T0[2][:3] == pytest.approx(
        [math.cos(alpha), 0, -math.sin(alpha)])
    # optical axes must DIFFER between yaw 0 and 0.6 (the old Rz@R0
    # composition left them identical - a camera-frame roll)
    f0 = T0[2][:3]
    f1 = T1[2][:3]
    assert f0 != pytest.approx(f1), "yaw did not change the optical axis"
    # forward stays at the same elevation; heading rotates by 0.6
    assert math.asin(-f1[2]) == pytest.approx(alpha)
    assert math.atan2(f1[1], f1[0]) == pytest.approx(0.6)
    # round trip: extraction is the inverse of construction
    def pose_of(T):
        return math.atan2(T[2][1], T[2][0]), math.asin(-T[2][2])
    for yaw in (-0.7, -0.1, 0.0, 0.35, 1.9, math.pi - 0.01):
        T = _T_C_E_with_yaw(alpha, yaw, pos)
        y2, a2 = pose_of(T)
        assert y2 == pytest.approx(yaw, abs=1e-12)
        assert a2 == pytest.approx(alpha, abs=1e-12)
