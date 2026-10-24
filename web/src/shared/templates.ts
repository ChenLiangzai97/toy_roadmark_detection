// Arrow template geometry; loads assets/arrows/*.json (same files Python reads).
import straight from "../../../assets/arrows/straight_v1.json";
import left from "../../../assets/arrows/left_v1.json";
import right from "../../../assets/arrows/right_v1.json";
import straightLeft from "../../../assets/arrows/straight_left_v1.json";
import straightRight from "../../../assets/arrows/straight_right_v1.json";
import leftRight from "../../../assets/arrows/left_right_v1.json";

export interface ArrowTemplate {
  template_id: string;
  class_id: number;
  class_name: string;
  entry_axis_local: [number, number];
  anchor_local: [number, number];
  canonical_size: { length_a: number; width_b: number };
  primitives: { type: "polygon"; points_ab: [number, number][] }[];
  description: string;
}

export const TEMPLATES: Record<string, ArrowTemplate> = {
  straight_v1: straight as unknown as ArrowTemplate,
  left_v1: left as unknown as ArrowTemplate,
  right_v1: right as unknown as ArrowTemplate,
  straight_left_v1: straightLeft as unknown as ArrowTemplate,
  straight_right_v1: straightRight as unknown as ArrowTemplate,
  left_right_v1: leftRight as unknown as ArrowTemplate,
};

export const TEMPLATE_IDS = Object.keys(TEMPLATES);

export function templateForClass(classId: number): ArrowTemplate {
  const tid = [
    "straight_v1", "left_v1", "right_v1",
    "straight_left_v1", "straight_right_v1", "left_right_v1",
  ][classId];
  if (!tid) throw new Error(`unknown class ${classId}`);
  return TEMPLATES[tid];
}

/** World polygons on z=0: p = center + R(yaw) @ (scale * p_local). */
export function worldPolygons(
  t: ArrowTemplate, centerXy: [number, number], yaw: number, scale: number
): [number, number][][] {
  const c = Math.cos(yaw), s = Math.sin(yaw);
  return t.primitives.map((prim) =>
    prim.points_ab.map(([x, y]) => [
      centerXy[0] + c * (scale * x) - s * (scale * y),
      centerXy[1] + s * (scale * x) + c * (scale * y),
    ] as [number, number])
  );
}

export function worldBbox(
  t: ArrowTemplate, centerXy: [number, number], yaw: number, scale: number
): { x0: number; x1: number; y0: number; y1: number } {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const poly of worldPolygons(t, centerXy, yaw, scale)) {
    for (const [x, y] of poly) {
      x0 = Math.min(x0, x); x1 = Math.max(x1, x);
      y0 = Math.min(y0, y); y1 = Math.max(y1, y);
    }
  }
  return { x0, x1, y0, y1 };
}

export function directionXy(yaw: number): [number, number] {
  return [Math.cos(yaw), Math.sin(yaw)];
}

export function pointInPolygon(
  x: number, y: number, poly: [number, number][]
): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if (yi > y !== yj > y) {
      const xInt = ((xj - xi) * (y - yi)) / (yj - yi) + xi;
      if (x < xInt) inside = !inside;
    }
  }
  return inside;
}

export function pointInTemplateWorld(
  x: number, y: number, t: ArrowTemplate,
  centerXy: [number, number], yaw: number, scale: number
): boolean {
  return worldPolygons(t, centerXy, yaw, scale).some((p) => pointInPolygon(x, y, p));
}
