// Shared scene construction for editor preview, render page and anything else.
// Single implementation for ground/objects/lane lines/lanes/arrows so the
// editor and offline renderer can never diverge (design doc section 2).
import * as THREE from "three";
import type { CameraCalibration, SceneV1, SceneArrow } from "./types";
import { TEMPLATES, worldPolygons } from "./templates";
import { mat4FromRows, setPerspectiveFromCalibration } from "./camera_utils";

export const ARROW_Z = 0.0;       // arrows live exactly on z=0
export const LINE_Z = 0.0;        // lane markings also on the ground plane

export interface BuiltScene {
  root: THREE.Group;
  arrowMeshes: THREE.Mesh[];      // one mesh per arrow instance
  objectMeshes: THREE.Object3D[]; // occluders (boxes, cylinders, cars)
  arrowGroup: THREE.Group;
  objectGroup: THREE.Group;
  markingGroup: THREE.Group;      // lane lines + lane boundaries (decorative)
}

function arrowMaterialRGB(colorHex: string): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({
    color: new THREE.Color(colorHex),
    polygonOffset: true,
    polygonOffsetFactor: -2,
    polygonOffsetUnits: -2,
    roughness: 0.9,
    metalness: 0.0,
  });
}

function polygonMesh(polys: [number, number][], material: THREE.Material,
                     z: number): THREE.Mesh {
  const shape = new THREE.Shape();
  let first = true;
  for (const [x, y] of polys) {
    if (first) { shape.moveTo(x, y); first = false; }
    else shape.lineTo(x, y);
  }
  shape.closePath();
  const geom = new THREE.ShapeGeometry(shape);
  const mesh = new THREE.Mesh(geom, material);
  mesh.position.z = z;
  return mesh;
}

export function buildArrowMesh(arrow: SceneArrow,
                               material: THREE.Material): THREE.Mesh {
  const t = TEMPLATES[arrow.template_id];
  if (!t) throw new Error(`unknown template ${arrow.template_id}`);
  const polys = worldPolygons(t, [arrow.center_ego[0], arrow.center_ego[1]],
                              arrow.yaw_rad, arrow.scale);
  const groupParts: THREE.Mesh[] = polys.map((p) => polygonMesh(p, material, ARROW_Z));
  // merge into a single mesh so per-instance material swaps stay simple
  const merged = mergeMeshes(groupParts);
  merged.userData = { kind: "arrow", instance_id: arrow.instance_id };
  return merged;
}

function mergeMeshes(meshes: THREE.Mesh[]): THREE.Mesh {
  if (meshes.length === 1) return meshes[0];
  const geoms = meshes.map((m) => m.geometry.clone());
  const geom = mergeGeometriesXY(geoms);
  return new THREE.Mesh(geom, meshes[0].material);
}

function mergeGeometriesXY(geoms: THREE.BufferGeometry[]): THREE.BufferGeometry {
  let total = 0;
  for (const g of geoms) total += g.getAttribute("position").count;
  const pos = new Float32Array(total * 3);
  const nor = new Float32Array(total * 3);
  const uv = new Float32Array(total * 2);
  const idx: number[] = [];
  let vo = 0;
  for (const g of geoms) {
    const p = g.getAttribute("position");
    const n = g.getAttribute("normal");
    const u = g.getAttribute("uv");
    for (let i = 0; i < p.count; i++) {
      pos[(vo + i) * 3] = p.getX(i);
      pos[(vo + i) * 3 + 1] = p.getY(i);
      pos[(vo + i) * 3 + 2] = p.getZ(i);
      nor[(vo + i) * 3] = n ? n.getX(i) : 0;
      nor[(vo + i) * 3 + 1] = n ? n.getY(i) : 0;
      nor[(vo + i) * 3 + 2] = n ? n.getZ(i) : 1;
      uv[(vo + i) * 2] = u ? u.getX(i) : 0;
      uv[(vo + i) * 2 + 1] = u ? u.getY(i) : 0;
    }
    const gi = g.getIndex();
    if (gi) {
      for (let i = 0; i < gi.count; i++) idx.push(gi.getX(i) + vo);
    } else {
      for (let i = 0; i < p.count; i++) idx.push(i + vo);
    }
    vo += p.count;
  }
  const out = new THREE.BufferGeometry();
  out.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  out.setAttribute("normal", new THREE.BufferAttribute(nor, 3));
  out.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  out.setIndex(idx);
  return out;
}

function buildObject(obj: SceneV1["objects"][number]): THREE.Object3D {
  const g = new THREE.Group();
  const color = obj.color ?? "#7a7f85";
  const mat = new THREE.MeshStandardMaterial({
    color: new THREE.Color(color), roughness: 0.8, metalness: 0.1,
  });
  const size = obj.size_abc ?? [1.0, 1.0, 1.0];
  if (obj.shape === "box") {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(size[0], size[1], size[2]), mat);
    mesh.position.z = size[2] / 2;
    g.add(mesh);
  } else if (obj.shape === "cylinder") {
    const mesh = new THREE.Mesh(
      new THREE.CylinderGeometry(size[0] / 2, size[0] / 2, size[2], 16), mat);
    mesh.rotation.x = Math.PI / 2; // cylinder axis (default y) -> z
    mesh.position.z = size[2] / 2;
    g.add(mesh);
  } else { // car: body + cabin
    const [L, W, H] = size;
    const bodyH = H * 0.6;
    const body = new THREE.Mesh(new THREE.BoxGeometry(L, W, bodyH), mat);
    body.position.z = bodyH / 2;
    const cabinMat = new THREE.MeshStandardMaterial({
      color: new THREE.Color(color).multiplyScalar(0.7), roughness: 0.4,
    });
    const cabin = new THREE.Mesh(
      new THREE.BoxGeometry(L * 0.5, W * 0.85, H * 0.4), cabinMat);
    cabin.position.z = bodyH + (H * 0.4) / 2;
    cabin.position.x = -L * 0.05;
    g.add(body, cabin);
  }
  g.position.set(obj.center_ego[0], obj.center_ego[1], obj.center_ego[2]);
  g.rotation.z = obj.yaw_rad;
  g.userData = { kind: "object", object_id: obj.object_id };
  return g;
}

function flatSegment(p0: THREE.Vector3, p1: THREE.Vector3, width: number,
                     material: THREE.Material, z: number): THREE.Mesh {
  const dir = new THREE.Vector3().subVectors(p1, p0);
  const len = dir.length();
  if (len < 1e-9) return new THREE.Mesh(new THREE.BufferGeometry(), material);
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(len, width, 0.002), material);
  mesh.position.copy(p0).add(p1).multiplyScalar(0.5);
  mesh.position.z = z + 0.001;
  mesh.rotation.z = Math.atan2(dir.y, dir.x);
  return mesh;
}

function buildMarkings(scene: SceneV1): THREE.Group {
  const group = new THREE.Group();
  group.name = "markings";
  const lineMat = (color: string) => new THREE.MeshStandardMaterial({
    color: new THREE.Color(color), polygonOffset: true,
    polygonOffsetFactor: -1, polygonOffsetUnits: -1, roughness: 0.9,
  });
  for (const line of scene.lane_lines) {
    const mat = lineMat(line.color);
    const pts = line.points_ego.map((p) => new THREE.Vector3(p[0], p[1], p[2]));
    if (line.style === "solid") {
      for (let i = 0; i + 1 < pts.length; i++) {
        group.add(flatSegment(pts[i], pts[i + 1], line.width_m, mat, LINE_Z));
      }
    } else {
      const dash = line.dash_length_m ?? 2.0;
      const gap = line.gap_length_m ?? 2.0;
      for (let i = 0; i + 1 < pts.length; i++) {
        const a = pts[i], b = pts[i + 1];
        const segLen = a.distanceTo(b);
        const dir = new THREE.Vector3().subVectors(b, a).normalize();
        let t = 0;
        while (t < segLen) {
          const d = Math.min(dash, segLen - t);
          const p0 = a.clone().addScaledVector(dir, t);
          const p1 = a.clone().addScaledVector(dir, t + d);
          group.add(flatSegment(p0, p1, line.width_m, mat, LINE_Z));
          t += dash + gap;
        }
      }
    }
  }
  for (const lane of scene.lanes) {
    if (!lane.show_boundaries) continue;
    const mat = lineMat("#9fe08c");
    const pts = lane.centerline_ego.map((p) => new THREE.Vector3(p[0], p[1], p[2]));
    for (let i = 0; i + 1 < pts.length; i++) {
      const dir = new THREE.Vector3().subVectors(pts[i + 1], pts[i]).normalize();
      const n = new THREE.Vector3(-dir.y, dir.x, 0).multiplyScalar(lane.width_m / 2);
      group.add(flatSegment(
        pts[i].clone().add(n), pts[i + 1].clone().add(n), 0.05, mat, LINE_Z));
      group.add(flatSegment(
        pts[i].clone().sub(n), pts[i + 1].clone().sub(n), 0.05, mat, LINE_Z));
    }
  }
  return group;
}

export function buildScene(scene: SceneV1): BuiltScene {
  THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
  THREE.ColorManagement.enabled = false;
  const root = new THREE.Group();
  root.name = "scene_root";

  const groundMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color(scene.ground.color), roughness: 0.95, metalness: 0.0,
  });
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(scene.ground.size_x, scene.ground.size_y), groundMat);
  ground.position.set(scene.ground.size_x / 2 - 10, 0, -0.01); // centered-ish ahead
  ground.name = "ground";
  ground.userData = { kind: "ground" };
  root.add(ground);

  const markingGroup = buildMarkings(scene);
  root.add(markingGroup);

  const arrowGroup = new THREE.Group();
  arrowGroup.name = "arrows";
  const arrowMat = arrowMaterialRGB(scene.render?.arrow_color ?? "#f2f2f2");
  const arrowMeshes: THREE.Mesh[] = [];
  for (const arrow of scene.arrows) {
    const mesh = buildArrowMesh(arrow, arrowMat.clone());
    arrowGroup.add(mesh);
    arrowMeshes.push(mesh);
  }
  root.add(arrowGroup);

  const objectGroup = new THREE.Group();
  objectGroup.name = "objects";
  const objectMeshes: THREE.Object3D[] = [];
  for (const obj of scene.objects) {
    const node = buildObject(obj);
    objectGroup.add(node);
    objectMeshes.push(node);
  }
  root.add(objectGroup);

  const lightDir = scene.render?.light?.direction_ego ?? [-0.4, 0.6, 1.0];
  const dir = new THREE.DirectionalLight(
    0xffffff, scene.render?.light?.intensity ?? 1.0);
  dir.position.set(lightDir[0], lightDir[1], lightDir[2]);
  root.add(dir);
  const amb = new THREE.AmbientLight(0xffffff, scene.render?.light?.ambient ?? 0.45);
  root.add(amb);

  return { root, arrowMeshes, objectMeshes, arrowGroup, objectGroup, markingGroup };
}

export { mat4FromRows, setPerspectiveFromCalibration };
export type { CameraCalibration };
