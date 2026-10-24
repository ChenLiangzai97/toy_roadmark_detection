// SceneV1 / labels types shared by editor and render pages. Mirrors
// schemas/scene_v1.schema.json (single source of truth for field names).

export type Mat3 = number[][];
export type Mat4 = number[][];
export type Vec2 = [number, number];
export type Vec3 = [number, number, number];

export type ArrowClassName =
  | "STRAIGHT" | "LEFT" | "RIGHT"
  | "STRAIGHT_LEFT" | "STRAIGHT_RIGHT" | "LEFT_RIGHT";

export const CLASS_NAMES: ArrowClassName[] = [
  "STRAIGHT", "LEFT", "RIGHT", "STRAIGHT_LEFT", "STRAIGHT_RIGHT", "LEFT_RIGHT",
];
export const NUM_CLASSES = 6;
export const NO_OBJECT_CLASS = 6;
export const ALLOWED_TURNS_SLR: number[][] = [
  [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1],
];

export interface Ground {
  size_x: number;
  size_y: number;
  color: string;
  grid_color?: string | null;
  texture?: string | null;
}

export interface SceneObject {
  object_id: string;
  shape: "box" | "cylinder" | "car";
  size_abc?: Vec3;
  center_ego: Vec3;
  yaw_rad: number;
  color?: string;
}

export interface LaneLine {
  line_id: string;
  points_ego: Vec3[];
  style: "solid" | "dashed";
  width_m: number;
  color: string;
  dash_length_m?: number;
  gap_length_m?: number;
}

export interface Lane {
  lane_id: string;
  centerline_ego: Vec3[];
  width_m: number;
  allowed_turns_slr: number[];
  world_rule_known: boolean;
  rule_source?: string | null;
  show_boundaries?: boolean;
}

export interface SceneArrow {
  instance_id: number;
  template_id: string;
  class_id: number;
  center_ego: Vec3;
  yaw_rad: number;
  scale: number;
  lane_id?: string | null;
}

export interface CameraCalibration {
  camera_id: string;
  width: number;
  height: number;
  K: Mat3;
  T_C_E: Mat4;
  near: number;
  far: number;
}

export interface RenderSettings {
  light?: { direction_ego?: number[]; intensity?: number; ambient?: number };
  arrow_color?: string;
  background?: string;
  ambient?: number;
}

export interface SceneV1 {
  schema_version: 1;
  scene_id: string;
  scene_group_id: string;
  seed: number;
  coordinate_frame: "ego_x_forward_y_left_z_up";
  units: { length: "m"; angle: "rad" };
  T_E_W: Mat4;
  bev_range: { x_min: number; x_max: number; y_min: number; y_max: number };
  ground: Ground;
  objects: SceneObject[];
  lane_lines: LaneLine[];
  lanes: Lane[];
  arrows: SceneArrow[];
  cameras: CameraCalibration[];
  render: RenderSettings;
  randomization?: object | null;
  arrow_template_version?: string;
}

export interface ArrowVisibilityPerCamera {
  amodal_px: number;
  visible_px: number;
  occluded_px: number;
  truncated: boolean;
  in_front: boolean;
  in_image: boolean;
  bbox_px: { x0: number; y0: number; x1: number; y1: number };
  visible_ratio: number;
  occluded: boolean;
}

export interface LabelArrow {
  instance_id: number;
  class_id: number;
  template_id: string;
  center_ego: Vec3;
  yaw_rad: number;
  direction_xy: Vec2;
  scale: number;
  lane_id?: string | null;
  arrow_valid: boolean;
  orientation_valid: boolean;
  ignore_reason: string | null;
  visibility: Record<string, ArrowVisibilityPerCamera>;
}

export interface LabelLane {
  lane_id: string;
  centerline_ego: Vec3[];
  width_m: number;
  allowed_turns_slr: number[];
  allowed_turns_valid_slr?: boolean[];
  world_rule_known: boolean;
  current_rule_valid: boolean;
  rule_source?: string | null;
}

export interface SampleLabelsV1 {
  schema_version: 1;
  sample_id: string;
  scene_group_id: string;
  coordinate_frame: "ego_x_forward_y_left_z_up";
  units: { length: "m"; angle: "rad" };
  bev_range: { x_min: number; x_max: number; y_min: number; y_max: number };
  training_camera_ids: string[];
  allow_no_object: boolean;
  cameras?: Record<string, CameraCalibration>;
  arrows: LabelArrow[];
  lanes: LabelLane[];
}
