// Focused SceneV1 validator mirroring schemas/scene_v1.schema.json.
// Returns an error string or null. Kept dependency-free; a pytest
// cross-checks its behavior against the JSON schema on mutations.
import type { SceneV1 } from "./types";

function isNum(x: unknown): x is number {
  return typeof x === "number" && isFinite(x);
}

function checkMat(m: unknown, n: number, what: string): string | null {
  if (!Array.isArray(m) || m.length !== n) return `${what}: expected ${n} rows`;
  for (const row of m) {
    if (!Array.isArray(row) || row.length !== n)
      return `${what}: expected ${n} cols`;
    for (const v of row) if (!isNum(v)) return `${what}: non-finite entry`;
  }
  return null;
}

export function validateSceneV1(s: unknown): string | null {
  if (typeof s !== "object" || s === null) return "scene: not an object";
  const o = s as Record<string, unknown>;
  if (o.schema_version !== 1) return `schema_version: ${String(o.schema_version)}`;
  for (const k of ["scene_id", "scene_group_id", "coordinate_frame",
                   "units", "T_E_W", "bev_range", "ground", "objects",
                   "lane_lines", "lanes", "arrows", "cameras", "render"]) {
    if (!(k in o)) return `missing field: ${k}`;
  }
  if (o.coordinate_frame !== "ego_x_forward_y_left_z_up")
    return `coordinate_frame: ${String(o.coordinate_frame)}`;
  const err = checkMat(o.T_E_W, 4, "T_E_W");
  if (err) return err;
  const bev = o.bev_range as Record<string, unknown>;
  for (const k of ["x_min", "x_max", "y_min", "y_max"]) {
    if (!isNum(bev?.[k])) return `bev_range.${k}: not a number`;
  }
  const xmin = bev.x_min as number, xmax = bev.x_max as number;
  const ymin = bev.y_min as number, ymax = bev.y_max as number;
  if (!(xmin < xmax) || !(ymin < ymax)) return "bev_range: empty range";
  const cams = o.cameras as Array<Record<string, unknown>>;
  if (!Array.isArray(cams) || cams.length < 1) return "cameras: need >= 1";
  const camIds = new Set<string>();
  for (const c of cams) {
    if (typeof c.camera_id !== "string" || !c.camera_id)
      return "camera.camera_id: invalid";
    if (camIds.has(c.camera_id)) return `camera_id duplicated: ${c.camera_id}`;
    camIds.add(c.camera_id as string);
    if (!Number.isInteger(c.width) || !Number.isInteger(c.height)
        || (c.width as number) < 16 || (c.height as number) < 16)
      return "camera resolution invalid";
    let e = checkMat(c.K, 3, `K(${c.camera_id})`);
    if (e) return e;
    e = checkMat(c.T_C_E, 4, `T_C_E(${c.camera_id})`);
    if (e) return e;
    const K = c.K as number[][];
    if (!(K[0][0] > 0) || !(K[1][1] > 0)) return "K: focal lengths must be > 0";
  }
  const seenIds = new Set<number>();
  const arrows = o.arrows as Array<Record<string, unknown>>;
  for (const a of arrows) {
    const iid = a.instance_id as number;
    if (!Number.isInteger(iid) || iid < 1)
      return `arrow.instance_id invalid: ${String(iid)}`;
    if (seenIds.has(iid)) return `arrow.instance_id duplicated: ${String(iid)}`;
    seenIds.add(iid);
    if (!Number.isInteger(a.class_id)
        || (a.class_id as number) < 0 || (a.class_id as number) > 5)
      return `arrow.class_id invalid: ${String(a.class_id)}`;
    const c = a.center_ego as unknown[];
    if (!Array.isArray(c) || c.length !== 3 || !c.every(isNum))
      return `arrow.center_ego invalid (id ${String(iid)})`;
    if (!isNum(a.yaw_rad) || !isNum(a.scale) || (a.scale as number) <= 0)
      return `arrow yaw/scale invalid (id ${String(iid)})`;
  }
  const objIds = new Set<string>();
  for (const ob of o.objects as Array<Record<string, unknown>>) {
    if (typeof ob.object_id !== "string" || !ob.object_id)
      return "object.object_id invalid";
    if (objIds.has(ob.object_id)) return `object_id duplicated: ${ob.object_id}`;
    objIds.add(ob.object_id as string);
    if (!["box", "cylinder", "car"].includes(String(ob.shape)))
      return `object.shape invalid: ${String(ob.shape)}`;
  }
  return null;
}

export type { SceneV1 };
