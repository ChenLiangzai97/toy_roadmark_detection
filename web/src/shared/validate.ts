// Schema-driven SceneV1 validation: reuses schemas/scene_v1.schema.json as
// the SINGLE source of truth (a subset validator of draft-07 sufficient for
// this schema), plus semantic checks the JSON grammar cannot express
// (unique instance ids, matrix finiteness, focal lengths, template ids).
import schemaJson from "../../../schemas/scene_v1.schema.json";
import type { SceneV1 } from "./types";
import { TEMPLATES } from "./templates";

type Schema = Record<string, unknown>;

function isNum(x: unknown): x is number {
  return typeof x === "number" && isFinite(x);
}

/** Draft-07 subset: type/required/properties/items/enum/const/minItems/
 *  maxItems/minimum/maximum/pattern/$ref/definitions/oneOf-like arrays. */
export function validateAgainstSchema(value: unknown, schema: Schema,
                                      path: string): string | null {
  if (typeof schema.$ref === "string") {
    const ref = schema.$ref.replace(/^#\/definitions\//, "");
    const def = (schemaJson.definitions as Record<string, Schema>)[ref];
    if (!def) return `${path}: unknown $ref ${schema.$ref}`;
    return validateAgainstSchema(value, def, path);
  }
  if (schema.type === "object" || schema.properties || schema.required) {
    if (typeof value !== "object" || value === null || Array.isArray(value))
      return `${path}: expected object`;
    const obj = value as Record<string, unknown>;
    for (const r of (schema.required as string[] | undefined) ?? []) {
      if (!(r in obj)) return `${path}: missing required field '${r}'`;
    }
    if (schema.additionalProperties === false && schema.properties) {
      const allowed = new Set(Object.keys(schema.properties as Schema));
      for (const k of Object.keys(obj))
        if (!allowed.has(k))
          return `${path}: unexpected field '${k}'`;
    }
    for (const [k, sub] of Object.entries(
      (schema.properties as Record<string, Schema> | undefined) ?? {})) {
      if (k in obj) {
        const err = validateAgainstSchema(obj[k], sub, `${path}.${k}`);
        if (err) return err;
      }
    }
    return null;
  }
  if (schema.type === "array" || schema.items) {
    if (!Array.isArray(value)) return `${path}: expected array`;
    const items = schema.items as Schema | undefined;
    const min = schema.minItems as number | undefined;
    const max = schema.maxItems as number | undefined;
    if (min !== undefined && value.length < min)
      return `${path}: needs >= ${min} items`;
    if (max !== undefined && value.length > max)
      return `${path}: needs <= ${max} items`;
    if (items)
      for (let i = 0; i < value.length; i++) {
        const err = validateAgainstSchema(value[i], items, `${path}[${i}]`);
        if (err) return err;
      }
    return null;
  }
  if (schema.enum) {
    const ok = (schema.enum as unknown[]).some((v) => v === value);
    return ok ? null : `${path}: ${JSON.stringify(value)} not in enum`;
  }
  if (schema.const !== undefined)
    return value === schema.const
      ? null : `${path}: expected const ${JSON.stringify(schema.const)}`;
  if (schema.type === "number" || schema.type === "integer") {
    if (!isNum(value)) return `${path}: expected number`;
    if (schema.type === "integer" && !Number.isInteger(value))
      return `${path}: expected integer`;
    if (schema.minimum !== undefined && value < (schema.minimum as number))
      return `${path}: below minimum`;
    if (schema.maximum !== undefined && value > (schema.maximum as number))
      return `${path}: above maximum`;
    if (schema.exclusiveMinimum !== undefined
        && value <= (schema.exclusiveMinimum as number))
      return `${path}: must be > ${(schema.exclusiveMinimum as number)}`;
    return null;
  }
  if (schema.type === "string") {
    if (typeof value !== "string") return `${path}: expected string`;
    if (schema.minLength !== undefined && value.length < (schema.minLength as number))
      return `${path}: string too short`;
    if (schema.pattern !== undefined
        && !new RegExp(schema.pattern as string).test(value))
      return `${path}: string fails pattern`;
    return null;
  }
  if (schema.type === "boolean")
    return typeof value === "boolean" ? null : `${path}: expected boolean`;
  if (schema.type === "null")
    return value === null ? null : `${path}: expected null`;
  if (Array.isArray(schema.type)) {
    const ok = (schema.type as string[]).some((t) =>
      validateAgainstSchema(value, { type: t }, path) === null);
    return ok ? null : `${path}: type mismatch`;
  }
  return null;
}

/** Semantic checks beyond the JSON grammar; mirrored in editor_server.py. */
export function semanticChecks(s: unknown): string | null {
  const o = s as SceneV1 & Record<string, unknown>;
  // unique arrow instance ids within the scene
  const seen = new Set<number>();
  for (const a of o.arrows) {
    if (seen.has(a.instance_id))
      return `arrows: duplicate instance_id ${a.instance_id}`;
    seen.add(a.instance_id);
    if (!TEMPLATES[a.template_id])
      return `arrows: unknown template_id ${a.template_id}`;
  }
  for (const m of [o.T_E_W, ...o.cameras.map((c) => c.T_C_E)]) {
    for (const row of m)
      for (const v of row)
        if (!isNum(v)) return "matrix: non-finite entry";
  }
  for (const c of o.cameras) {
    if (!(c.K[0][0] > 0) || !(c.K[1][1] > 0))
      return `K(${c.camera_id}): focal lengths must be > 0`;
    if (c.near <= 0 || c.far <= c.near)
      return `camera ${c.camera_id}: need 0 < near < far`;
  }
  const ids = new Set<string>();
  for (const ob of o.objects) {
    if (ids.has(ob.object_id)) return `objects: duplicate id ${ob.object_id}`;
    ids.add(ob.object_id);
  }
  return null;
}

export function validateSceneV1(s: unknown): string | null {
  const structErr = validateAgainstSchema(s, schemaJson as Schema, "scene");
  if (structErr) return structErr;
  return semanticChecks(s);
}

export type { SceneV1 };
