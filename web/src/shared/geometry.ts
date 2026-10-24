// Coordinate transforms mirroring arrow_toy/geometry/projection.py.
// Frames: E ego (x fwd, y left, z up), C OpenCV, G three.js camera.
import type { Mat3, Mat4, Vec3 } from "./types";

export const S_C_TO_G: Mat4 = [
  [1, 0, 0, 0],
  [0, -1, 0, 0],
  [0, 0, -1, 0],
  [0, 0, 0, 1],
];

export function mat4Mul(a: Mat4, b: Mat4): Mat4 {
  const out: Mat4 = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
  for (let i = 0; i < 4; i++)
    for (let j = 0; j < 4; j++)
      for (let k = 0; k < 4; k++) out[i][j] += a[i][k] * b[k][j];
  return out;
}

export function mat4Identity(): Mat4 {
  return [
    [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
  ];
}

export function transformPoint(T: Mat4, p: Vec3 | number[]): number[] {
  const [x, y, z] = p;
  return [
    T[0][0] * x + T[0][1] * y + T[0][2] * z + T[0][3],
    T[1][0] * x + T[1][1] * y + T[1][2] * z + T[1][3],
    T[2][0] * x + T[2][1] * y + T[2][2] * z + T[2][3],
    T[3][0] * x + T[3][1] * y + T[3][2] * z + T[3][3],
  ];
}

export function T_G_E_from_T_C_E(T_C_E: Mat4): Mat4 {
  return mat4Mul(S_C_TO_G, T_C_E);
}

export function T_C_E_from_T_G_E(T_G_E: Mat4): Mat4 {
  return mat4Mul(S_C_TO_G, T_G_E);
}

export interface Intrinsics {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
}

export function intrinsicsFromFovV(
  fovVRad: number, width: number, height: number,
  cx?: number, cy?: number
): Intrinsics {
  const fy = height / (2 * Math.tan(fovVRad / 2));
  return {
    fx: fy, fy,
    cx: cx ?? width / 2,
    cy: cy ?? height / 2,
  };
}

export function KFromIntrinsics(K: Intrinsics): Mat3 {
  return [
    [K.fx, 0, K.cx],
    [0, K.fy, K.cy],
    [0, 0, 1],
  ];
}

/** Default camera per design doc: pitch down alpha at position t. */
export function defaultCameraTCE(
  pitchDownRad = (12 * Math.PI) / 180,
  positionEgo: Vec3 = [0, 0, 1.7]
): Mat4 {
  const s = Math.sin(pitchDownRad), c = Math.cos(pitchDownRad);
  const R = [
    [0, -1, 0],
    [-s, 0, -c],
    [c, 0, -s],
  ];
  const t = [0, 1, 2].map((i) =>
    -(R[i][0] * positionEgo[0] + R[i][1] * positionEgo[1] + R[i][2] * positionEgo[2])
  );
  return [
    [R[0][0], R[0][1], R[0][2], t[0]],
    [R[1][0], R[1][1], R[1][2], t[1]],
    [R[2][0], R[2][1], R[2][2], t[2]],
    [0, 0, 0, 1],
  ];
}

/** Pinhole projection of ego points; returns NaN uv for Z<=0. */
export function projectPoints(
  pointsEgo: (Vec3 | number[])[],
  K: Intrinsics | Mat3,
  T_C_E: Mat4
): { uv: [number, number][]; valid: boolean[] } {
  const fx = Array.isArray(K) ? K[0][0] : K.fx;
  const fy = Array.isArray(K) ? K[1][1] : K.fy;
  const cx = Array.isArray(K) ? K[0][2] : K.cx;
  const cy = Array.isArray(K) ? K[1][2] : K.cy;
  const uv: [number, number][] = [];
  const valid: boolean[] = [];
  for (const p of pointsEgo) {
    const q = transformPoint(T_C_E, p);
    const X = q[0], Y = q[1], Z = q[2];
    if (Z > 1e-9) {
      uv.push([fx * (X / Z) + cx, fy * (Y / Z) + cy]);
      valid.push(true);
    } else {
      uv.push([NaN, NaN]);
      valid.push(false);
    }
  }
  return { uv, valid };
}

/** OpenGL projection matrix equivalent to pinhole K (design doc 4.2). */
export function openglProjectionMatrix(
  K: Intrinsics | Mat3, width: number, height: number,
  near: number, far: number
): Mat4 {
  const fx = Array.isArray(K) ? K[0][0] : K.fx;
  const fy = Array.isArray(K) ? K[1][1] : K.fy;
  const cx = Array.isArray(K) ? K[0][2] : K.cx;
  const cy = Array.isArray(K) ? K[1][2] : K.cy;
  const W = width, H = height, n = near, f = far;
  return [
    [(2 * fx) / W, 0, 1 - (2 * cx) / W, 0],
    [0, (2 * fy) / H, (2 * cy) / H - 1, 0],
    [0, 0, -(f + n) / (f - n), (-2 * f * n) / (f - n)],
    [0, 0, -1, 0],
  ];
}
