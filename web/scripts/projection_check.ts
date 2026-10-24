// Project a fixed set of world points through a real three.js PerspectiveCamera
// driven by the calibration contract (T_G_E + OpenGL P), and dump JSON to
// stdout for the pytest cross-check against the Python pinhole model.
//
// Input JSON on stdin: { points: [[x,y,z],...], K: [[..]], T_C_E: [[..]],
//                        width, height, near, far }
// Output JSON on stdout: { uv: [[u,v],...], valid: [bool,...] }
import * as THREE from "three";

interface CheckInput {
  points: [number, number, number][];
  K: number[][];
  T_C_E: number[][];
  width: number;
  height: number;
  near: number;
  far: number;
}

function mat4FromRows(rows: number[][]): THREE.Matrix4 {
  // Three.js Matrix4.set takes row-major arguments.
  const m = rows;
  return new THREE.Matrix4().set(
    m[0][0], m[0][1], m[0][2], m[0][3],
    m[1][0], m[1][1], m[1][2], m[1][3],
    m[2][0], m[2][1], m[2][2], m[2][3],
    m[3][0], m[3][1], m[3][2], m[3][3]
  );
}

function openglP(K: number[][], width: number, height: number,
                 near: number, far: number): THREE.Matrix4 {
  const fx = K[0][0], fy = K[1][1], cx = K[0][2], cy = K[1][2];
  const W = width, H = height, n = near, f = far;
  return new THREE.Matrix4().set(
    (2 * fx) / W, 0, 1 - (2 * cx) / W, 0,
    0, (2 * fy) / H, (2 * cy) / H - 1, 0,
    0, 0, -(f + n) / (f - n), (-2 * f * n) / (f - n),
    0, 0, -1, 0
  );
}

async function main() {
  const raw = await new Promise<string>((resolve, reject) => {
    let buf = "";
    process.stdin.setEncoding("utf-8");
    process.stdin.on("data", (d) => (buf += d));
    process.stdin.on("end", () => resolve(buf));
    process.stdin.on("error", reject);
  });
  const input = JSON.parse(raw) as CheckInput;

  THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

  // T_G_E = S @ T_C_E with S = diag(1,-1,-1,1)
  const S = new THREE.Matrix4().set(
    1, 0, 0, 0,
    0, -1, 0, 0,
    0, 0, -1, 0,
    0, 0, 0, 1
  );
  const T_G_E = S.clone().multiply(mat4FromRows(input.T_C_E));

  const camera = new THREE.PerspectiveCamera();
  camera.up.set(0, 0, 1);
  // world -> camera; prevent updateMatrixWorld from overriding via rotation.
  camera.matrixAutoUpdate = false;
  camera.matrixWorldInverse.copy(T_G_E);
  camera.matrixWorld.copy(T_G_E.clone().invert());
  camera.projectionMatrix.copy(
    openglP(input.K, input.width, input.height, input.near, input.far)
  );
  camera.projectionMatrixInverse.copy(camera.projectionMatrix.clone().invert());

  const uv: [number, number][] = [];
  const valid: boolean[] = [];
  for (const p of input.points) {
    const v = new THREE.Vector3(p[0], p[1], p[2]).project(camera);
    // project() maps to NDC; points behind camera produce |v|>1 / flipped w.
    const behind = v.z > 1 || v.z < -1;
    if (behind) {
      uv.push([NaN, NaN]);
      valid.push(false);
      continue;
    }
    uv.push([((v.x + 1) * input.width) / 2, ((1 - v.y) * input.height) / 2]);
    valid.push(true);
  }
  process.stdout.write(JSON.stringify({ uv, valid }));
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
