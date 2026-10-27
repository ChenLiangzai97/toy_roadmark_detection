// Pure helpers for the detection workspace overlays (unit-testable).
export interface Cam {
  K: number[][];
  T_C_E: number[][];
  width: number;
  height: number;
}

/** Project a ground point (z=0) through K/T_C_E; null if behind the
 *  camera or non-finite. */
export function projGround(c: Cam, x: number, y: number): [number, number] | null {
  const T = c.T_C_E;
  const Xc = T[0][0] * x + T[0][1] * y + T[0][3];
  const Yc = T[1][0] * x + T[1][1] * y + T[1][3];
  const Zc = T[2][0] * x + T[2][1] * y + T[2][3];
  if (Zc <= 1e-6) return null;
  const u = c.K[0][0] * Xc / Zc + c.K[0][2];
  const v = c.K[1][1] * Yc / Zc + c.K[1][2];
  if (!isFinite(u) || !isFinite(v)) return null;
  return [u, v];
}

/** Heading segment for an overlay: center and center+len*dir are BOTH
 *  projected through the camera (a naive image-space offset drew the
 *  wrong axis). Null when either endpoint is invalid. */
export function headingSegment(c: Cam, x: number, y: number,
                               dir: [number, number],
                               len = 0.6): [[number, number], [number, number]] | null {
  const a = projGround(c, x, y);
  const b = projGround(c, x + len * dir[0], y + len * dir[1]);
  if (!a || !b) return null;
  return [a, b];
}
