// Calibration -> three.js PerspectiveCamera setup (design doc 4.2).
import * as THREE from "three";
import type { CameraCalibration, Mat4 } from "./types";

export function mat4FromRows(rows: Mat4): THREE.Matrix4 {
  const m = rows;
  return new THREE.Matrix4().set(
    m[0][0], m[0][1], m[0][2], m[0][3],
    m[1][0], m[1][1], m[1][2], m[1][3],
    m[2][0], m[2][1], m[2][2], m[2][3],
    m[3][0], m[3][1], m[3][2], m[3][3]
  );
}

export function rowsFromMat4(m: THREE.Matrix4): Mat4 {
  const e = m.elements; // column-major internally
  return [
    [e[0], e[4], e[8], e[12]],
    [e[1], e[5], e[9], e[13]],
    [e[2], e[6], e[10], e[14]],
    [e[3], e[7], e[11], e[15]],
  ];
}

/** Set camera.matrixWorldInverse = T_G_E and projectionMatrix = P(K,n,f).
 *  matrixAutoUpdate is disabled so lookAt/updateMatrixWorld cannot override
 *  the calibration. */
export function setPerspectiveFromCalibration(
  camera: THREE.PerspectiveCamera, cal: CameraCalibration
): void {
  camera.up.set(0, 0, 1);
  camera.matrixAutoUpdate = false;
  const S = new THREE.Matrix4().set(
    1, 0, 0, 0,
    0, -1, 0, 0,
    0, 0, -1, 0,
    0, 0, 0, 1
  );
  const T_G_E = S.clone().multiply(mat4FromRows(cal.T_C_E));
  camera.matrixWorldInverse.copy(T_G_E);
  camera.matrixWorld.copy(T_G_E.clone().invert());

  const K = cal.K, W = cal.width, H = cal.height;
  const fx = K[0][0], fy = K[1][1], cx = K[0][2], cy = K[1][2];
  const n = cal.near, f = cal.far;
  camera.projectionMatrix.set(
    (2 * fx) / W, 0, 1 - (2 * cx) / W, 0,
    0, (2 * fy) / H, (2 * cy) / H - 1, 0,
    0, 0, -(f + n) / (f - n), (-2 * f * n) / (f - n),
    0, 0, -1, 0
  );
  camera.projectionMatrixInverse.copy(camera.projectionMatrix.clone().invert());
}
