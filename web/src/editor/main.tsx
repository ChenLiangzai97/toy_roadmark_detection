// Scene editor (design doc section 5). Single-file app: toolbar, 3D view,
// properties panel, camera preview + BEV/label layer, undo/redo, JSON
// import/export, current-sample export and batch generation via /api.
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import type {
  CameraCalibration, Lane, LaneLine, SceneArrow, SceneObject, SceneV1,
} from "../shared/types";
import { CLASS_NAMES } from "../shared/types";
import { TEMPLATE_IDS, TEMPLATES } from "../shared/templates";
import { intrinsicsFromFovV, KFromIntrinsics, defaultCameraTCE } from "../shared/geometry";
import { ThreeViewport, type SelectionKind } from "./three_view";
import { headingSegment } from "../shared/detection";
import { validateSceneV1 } from "../shared/validate";

/* ------------------------------ scene helpers ----------------------------- */

let idCounter = 1;
const uid = (p: string) => `${p}_${Date.now().toString(36)}${(idCounter++).toString(36)}`;

function emptyScene(): SceneV1 {
  const K = intrinsicsFromFovV((60 * Math.PI) / 180, 320, 192);
  return {
    schema_version: 1,
    scene_id: uid("scene"),
    scene_group_id: uid("group"),
    seed: 42,
    coordinate_frame: "ego_x_forward_y_left_z_up",
    units: { length: "m", angle: "rad" },
    T_E_W: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    bev_range: { x_min: 2, x_max: 18, y_min: -4, y_max: 4 },
    ground: { size_x: 60, size_y: 40, color: "#3d423c", texture: null },
    objects: [],
    lane_lines: [],
    lanes: [],
    arrows: [],
    cameras: [{
      camera_id: "front", width: 320, height: 192, K: KFromIntrinsics(K),
      T_C_E: defaultCameraTCE(), near: 0.1, far: 100,
    }],
    render: {
      light: { direction_ego: [-0.4, 0.6, 1.0], intensity: 1.0, ambient: 0.45 },
      arrow_color: "#f2f2f2", background: "#101418",
    },
    randomization: null,
    arrow_template_version: "v1",
  };
}

/* --------------------------------- props UI -------------------------------- */

function NumField({ label, value, step = 0.1, onChange }: {
  label: string; value: number; step?: number;
  onChange: (v: number) => void;
}) {
  return (
    <label style={{ display: "flex", justifyContent: "space-between", gap: 6, marginBottom: 4 }}>
      <span>{label}</span>
      <input type="number" step={step} value={Number(value.toFixed(4))}
        style={{ width: 110 }}
        onChange={(e) => onChange(parseFloat(e.target.value) || 0)} />
    </label>
  );
}

function Row({ children }: { children: React.ReactNode }) {
  return <div style={{ display: "flex", gap: 6, marginBottom: 6, flexWrap: "wrap" }}>{children}</div>;
}

function Btn({ children, onClick, active, title, disabled }: {
  children: React.ReactNode; onClick: () => void; active?: boolean;
  title?: string; disabled?: boolean;
}) {
  return (
    <button title={title} onClick={onClick} disabled={disabled}
      style={{
        padding: "4px 8px", cursor: "pointer", fontSize: 12,
        background: active ? "#3a6ea5" : "#2a2f36", color: "#e8e8e8",
        border: "1px solid #444", borderRadius: 4,
      }}>{children}</button>
  );
}

/* -------------------------- detection workspace ---------------------------- */

const DET_CLASS_NAMES = ["直行", "左转", "右转", "直左", "直右", "左右"];
const DET_CLASS_COLORS = ["#e63946", "#29537a", "#2a9d8f",
                          "#e76f51", "#5e60ce", "#bc4749"];

function DetectionPanel(props: {
  runs: { run: string; f1?: number; conf?: number }[];
  run: string; setRun: (r: string) => void;
  split: string; setSplit: (s: string) => void;
  sample: string; setSample: (s: string) => void;
  result: {
    sample_id: string;
    camera: { K: number[][]; T_C_E: number[][]; width: number; height: number };
    rgb_png_b64: string;
    bev_range: { x_min: number; x_max: number; y_min: number; y_max: number };
    conf_threshold?: number; conf_source?: string;
    predictions: { prob: number; cls: number;
      center_norm: number[]; dir: number[] }[];
    gt: { class_id: number; center_ego: number[]; yaw_rad: number;
      arrow_valid: boolean; ignore_reason: string | null }[];
  } | null;
  busy: boolean; err: string;
  onDetect: () => void; onStep: (d: number) => void;
  onRefreshRuns: () => void;
}) {
  const rgbRef = useRef<HTMLCanvasElement>(null);
  const bevRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [showGT, setShowGT] = useState(true);
  const [showPred, setShowPred] = useState(true);

  useEffect(() => {
    const r = props.result;
    if (!r) return;
    // draw once the RGB image has decoded (the hoisted inner function
    // previously shadowed the real draw() and silently no-op'd)
    const img = new Image();
    img.onload = () => { imgRef.current = img; draw(); };
    img.src = "data:image/png;base64," + r.rgb_png_b64;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.result]);

  function draw() {
    const r = props.result;
    const img = imgRef.current;
    if (!r || !img) return;
    const cv = rgbRef.current;
    if (!cv) return;
    const W = r.camera.width, H = r.camera.height;
    cv.width = W; cv.height = H;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(img, 0, 0);
    const cam = r.camera;
    // ground truth: hollow green circles (+ gray for ignored)
    if (showGT) {
      for (const g of r.gt) {
        const pv = headingSegment(cam, g.center_ego[0], g.center_ego[1],
                                  [1, 0], 0); // center only
        if (!pv) continue;
        const [u, v] = pv[0];
        ctx.beginPath();
        ctx.arc(u, v, 7, 0, Math.PI * 2);
        ctx.strokeStyle = g.arrow_valid ? "#3dff6e" : "#8a8f96";
        ctx.lineWidth = 2;
        ctx.stroke();
        if (!g.arrow_valid && g.ignore_reason) {
          ctx.fillStyle = "#8a8f96";
          ctx.font = "8px sans-serif";
          ctx.fillText(g.ignore_reason.slice(0, 6), u + 9, v + 3);
        }
      }
    }
    // predictions above threshold: filled circles + heading + conf.
    // Heading lines project BOTH endpoints through K/T (a naive
    // image-space offset draws the wrong axis: world +x maps UP in the
    // default camera, not right); skip behind-camera/non-finite.
    const thr = r.conf_threshold ?? 0.5;
    if (showPred) {
      for (const p of r.predictions) {
        if (p.prob < thr) continue;
        const b = r.bev_range;
        const x = b.x_min + p.center_norm[0] * (b.x_max - b.x_min);
        const y = b.y_min + p.center_norm[1] * (b.y_max - b.y_min);
        const c0 = headingSegment(cam, x, y, [1, 0], 0);
        if (!c0) continue;
        const [u, v] = c0[0];
        const col = DET_CLASS_COLORS[p.cls % 6];
        ctx.beginPath();
        ctx.arc(u, v, 5, 0, Math.PI * 2);
        ctx.fillStyle = col;
        ctx.globalAlpha = 0.75;
        ctx.fill();
        ctx.globalAlpha = 1;
        // heading: BOTH endpoints projected (world +x draws UP in the
        // default camera, not right); skip invalid segments
        const seg = headingSegment(
          cam, x, y, [p.dir[0], p.dir[1]], 0.6);
        if (seg) {
          ctx.strokeStyle = col;
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.moveTo(seg[0][0], seg[0][1]);
          ctx.lineTo(seg[1][0], seg[1][1]);
          ctx.stroke();
        }
        ctx.fillStyle = col;
        ctx.font = "9px sans-serif";
        ctx.fillText(p.prob.toFixed(2), u + 7, v - 6);
      }
    }
    drawBev();
  }

  function drawBev() {
    const r = props.result;
    const cv = bevRef.current;
    if (!r || !cv) return;
    const BW = 160, BH = 320; // native BEV orientation: far on top
    cv.width = BW; cv.height = BH;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "#14171b";
    ctx.fillRect(0, 0, BW, BH);
    const b = r.bev_range;
    const toPx = (x: number, y: number): [number, number] => [
      (b.y_max - y) / (b.y_max - b.y_min) * BW,
      (b.x_max - x) / (b.x_max - b.x_min) * BH];
    if (showGT) {
      for (const g of r.gt) {
        const [px, py] = toPx(g.center_ego[0], g.center_ego[1]);
        ctx.beginPath();
        ctx.arc(px, py, 6, 0, Math.PI * 2);
        ctx.strokeStyle = g.arrow_valid ? "#3dff6e" : "#8a8f96";
        ctx.lineWidth = 2;
        ctx.stroke();
        // GT heading
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(px - 12 * Math.sin(g.yaw_rad),
                   py - 12 * Math.cos(g.yaw_rad));
        ctx.stroke();
      }
    }
    const thr = r.conf_threshold ?? 0.5;
    if (showPred) for (const p of r.predictions) {
      if (p.prob < thr) continue;
      const x = b.x_min + p.center_norm[0] * (b.x_max - b.x_min);
      const y = b.y_min + p.center_norm[1] * (b.y_max - b.y_min);
      const [px, py] = toPx(x, y);
      const col = DET_CLASS_COLORS[p.cls % 6];
      ctx.beginPath();
      ctx.arc(px, py, 5, 0, Math.PI * 2);
      ctx.fillStyle = col;
      ctx.globalAlpha = 0.8;
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(px - 14 * p.dir[1], py - 14 * p.dir[0]);
      ctx.strokeStyle = col;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }

  useEffect(() => {
    if (props.result && imgRef.current) draw();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.result, showGT, showPred]);

  const r = props.result;
  return (
    <div data-testid="detection-panel">
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap",
                    alignItems: "center", marginBottom: 6 }}>
        <select value={props.run} onChange={(e) => props.setRun(e.target.value)}>
          {props.runs.map((x) => (
            <option key={x.run} value={x.run}>
              {x.run}{x.f1 != null ? ` (f1 ${x.f1.toFixed(3)})` : ""}
            </option>
          ))}
        </select>
        <Btn onClick={props.onRefreshRuns}>刷新列表</Btn>
        <select value={props.split} onChange={(e) => props.setSplit(e.target.value)}>
          {["val", "test_clean", "test_occlusion", "test_pose"].map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
      </div>
      <div style={{ display: "flex", gap: 6, marginBottom: 6 }}>
        <Btn onClick={() => props.onStep(-1)} title="previous sample">← 上一帧</Btn>
        <input value={props.sample} onChange={(e) => props.setSample(e.target.value)}
          placeholder="sample id" style={{ flex: 1 }} />
        <Btn onClick={() => props.onStep(1)} title="next sample">下一帧 →</Btn>
      </div>
      <div style={{ display: "flex", gap: 10, alignItems: "center",
                    flexWrap: "wrap" }}>
        <Btn onClick={props.onDetect} disabled={props.busy ? true : undefined}>
          {props.busy ? "推理中…" : "推理"}
        </Btn>
        <label style={{ fontSize: 12 }}>
          <input type="checkbox" checked={showGT}
                 onChange={(e) => setShowGT(e.target.checked)} /> 真值
        </label>
        <label style={{ fontSize: 12 }}>
          <input type="checkbox" checked={showPred}
                 onChange={(e) => setShowPred(e.target.checked)} /> 预测
        </label>
      </div>
      {props.err && <div style={{ color: "#f66", fontSize: 12, marginTop: 4 }}>{props.err}</div>}
      {r && (
        <div style={{ marginTop: 8 }}>
          <div style={{ fontSize: 12, color: "#9ab", marginBottom: 4 }}>
            {r.sample_id} · 阈值 {r.conf_threshold ?? 0.5}
            {r.conf_source ? ` (${r.conf_source === "eval_val" ? "val评测冻结" : r.conf_source})` : ""} · 预测 {r.predictions.filter(
              (p: { prob: number }) => p.prob >= (r.conf_threshold ?? 0.5)).length} / GT {r.gt.length}
          </div>
          <canvas ref={rgbRef} data-testid="detect-rgb" style={{
            width: "100%", border: "1px solid #333",
            imageRendering: "pixelated", display: "block" }} />
          <div style={{ display: "flex", gap: 8, marginTop: 6 }}>
            <canvas ref={bevRef} data-testid="detect-bev" style={{
              height: 200, width: "auto", border: "1px solid #333",
              imageRendering: "pixelated" }} />
            <div style={{ fontSize: 11, lineHeight: 1.7 }}>
              <div style={{ color: "#3dff6e" }}>○ 真值 (有效)</div>
              <div style={{ color: "#8a8f96" }}>○ 真值 (忽略)</div>
              <div style={{ color: "#ddd" }}>● 预测 (按类别着色)</div>
              {DET_CLASS_NAMES.map((n, i) => (
                <div key={n} style={{ color: DET_CLASS_COLORS[i] }}>
                  ● {i} {n}
                </div>
              ))}
              <div style={{ color: "#9ab" }}>数字 = 置信度</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* --------------------------- fullscreen viewer ---------------------------- */

function FullscreenViewer({ url, onClose }: { url: string; onClose: () => void }) {
  const [mode, setMode] = useState<"fit" | "100">("fit");
  const [scale, setScale] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const panRef = useRef<{ x: number; y: number; px: number; py: number } | null>(null);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "0") { setMode("fit"); setScale(1); setPan({ x: 0, y: 0 }); }
      if (e.key === "1") { setMode("100"); setScale(1); setPan({ x: 0, y: 0 }); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  const zoomBy = (f: number) => { setMode("100"); setScale((z) => Math.min(16, Math.max(0.25, z * f))); };
  const wrapRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      setMode("100");
      setScale((z) => Math.min(16, Math.max(0.25, z * (e.deltaY < 0 ? 1.2 : 1 / 1.2))));
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);
  const imgStyle: React.CSSProperties = mode === "fit"
    ? { width: "100%", height: "100%", objectFit: "contain", imageRendering: "pixelated" }
    : { transform: `scale(${scale}) translate(${pan.x / scale}px, ${pan.y / scale}px)`,
        transformOrigin: "0 0", imageRendering: "pixelated",
        width: "auto", height: "auto", maxWidth: "none" };
  return (
    <div ref={wrapRef}
      style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.88)",
               zIndex: 1000, display: "flex", flexDirection: "column" }}
      onPointerDown={(e) => {
        if (mode !== "100") return;
        panRef.current = { x: e.clientX, y: e.clientY, px: pan.x, py: pan.y };
        (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
      }}
      onPointerMove={(e) => {
        const p = panRef.current;
        if (!p) return;
        setPan({ x: p.px + (e.clientX - p.x), y: p.py + (e.clientY - p.y) });
      }}
      onPointerUp={() => { panRef.current = null; }}>
      <div style={{ display: "flex", gap: 8, padding: 8, color: "#eee",
                    alignItems: "center", background: "#181c20" }}>
        <b>camera preview</b>
        <Btn active={mode === "fit"} onClick={() => { setMode("fit"); setScale(1); setPan({ x: 0, y: 0 }); }}>适应窗口</Btn>
        <Btn active={mode === "100"} onClick={() => { setMode("100"); setScale(1); setPan({ x: 0, y: 0 }); }}>原始 100%</Btn>
        <Btn onClick={() => zoomBy(1 / 1.25)}>−</Btn>
        <span style={{ minWidth: 40 }}>{mode === "fit" ? "fit" : `${(scale * 100).toFixed(0)}%`}</span>
        <Btn onClick={() => zoomBy(1.25)}>＋</Btn>
        <Btn onClick={onClose}>关闭 (Esc)</Btn>
      </div>
      <div style={{ flex: 1, overflow: "hidden", display: "flex",
                    alignItems: "center", justifyContent: "center" }}>
        <img src={url} style={imgStyle} alt="camera preview" draggable={false} />
      </div>
    </div>
  );
}

/* ------------------------------- thumbnails -------------------------------- */

function ArrowThumb({ classId, size = 44 }: { classId: number; size?: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    const tid = ["straight_v1", "left_v1", "right_v1",
      "straight_left_v1", "straight_right_v1", "left_right_v1"][classId];
    const t = TEMPLATES[tid];
    const pts: [number, number][] = t.primitives.flatMap(
      (p: { points_ab: [number, number][] }) => p.points_ab);
    const xs = pts.map((q: [number, number]) => q[0]);
    const ys = pts.map((q: [number, number]) => q[1]);
    const w = Math.max(...xs) - Math.min(...xs);
    const h = Math.max(...ys) - Math.min(...ys);
    const sc = (size - 8) / Math.max(w, h);
    const ox = size / 2 - (Math.min(...xs) + w / 2) * sc;
    const oy = size / 2 + (Math.min(...ys) + h / 2) * sc;
    ctx.clearRect(0, 0, size, size);
    ctx.fillStyle = "#e8e8e8";
    for (const prim of t.primitives) {
      ctx.beginPath();
      prim.points_ab.forEach(([a, b], i) => {
        // display convention: entry/forward (+a) points UP, left branch
        // (+b) points LEFT -> px = -b, py = -a
        const px = ox - b * sc;
        const py = oy - a * sc;
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      ctx.closePath();
      ctx.fill();
    }
  }, [classId, size]);
  return <canvas ref={ref} width={size} height={size}
    style={{ imageRendering: "pixelated", display: "block" }} />;
}

function ObjectThumb({ shape, size = 36 }: { shape: string; size?: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    const c = size / 2;
    ctx.clearRect(0, 0, size, size);
    ctx.strokeStyle = "#cfd6de";
    ctx.fillStyle = "rgba(159,168,180,0.35)";
    ctx.lineWidth = 1.5;
    const poly = (pts: number[][]) => {
      ctx.beginPath();
      pts.forEach(([x, y], i) => (i ? ctx.lineTo(c + x, c + y)
        : ctx.moveTo(c + x, c + y)));
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    };
    const r = size * 0.32;
    if (shape === "box" || shape === "car") {
      if (shape === "car") {
        poly([[-r, -r * 0.6], [r * 0.9, -r * 0.6], [r * 0.9, r * 0.6],
              [-r, r * 0.6]]);
        poly([[-r * 0.45, -r * 0.45], [r * 0.2, -r * 0.45],
              [r * 0.2, r * 0.45], [-r * 0.45, r * 0.45]]);
      } else {
        poly([[-r * 0.8, -r * 0.8], [r * 0.8, -r * 0.8],
              [r * 0.8, r * 0.8], [-r * 0.8, r * 0.8]]);
      }
    } else if (shape === "cylinder") {
      ctx.beginPath();
      ctx.ellipse(c, c - r * 0.5, r * 0.55, r * 0.2, 0, 0, Math.PI * 2);
      ctx.moveTo(c - r * 0.55, c - r * 0.5);
      ctx.lineTo(c - r * 0.55, c + r * 0.5);
      ctx.lineTo(c + r * 0.55, c + r * 0.5);
      ctx.lineTo(c + r * 0.55, c - r * 0.5);
      ctx.fill();
      ctx.stroke();
      ctx.beginPath();
      ctx.ellipse(c, c + r * 0.5, r * 0.55, r * 0.2, 0, 0, Math.PI * 2);
      ctx.stroke();
    } else if (shape === "sphere") {
      ctx.beginPath();
      ctx.arc(c, c, r * 0.8, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    } else if (shape === "cone") {
      poly([[0, -r * 0.9], [r * 0.75, r * 0.7], [-r * 0.75, r * 0.7]]);
    }
  }, [shape, size]);
  return <canvas ref={ref} width={size} height={size}
    style={{ display: "block" }} />;
}

/* ----------------------------- object library ------------------------------ */

interface LibraryEntry {
  label: string;
  shape: SceneObject["shape"];
  size: [number, number, number];
  color?: string;
}

const OBJECT_LIBRARY: LibraryEntry[] = [
  { label: "方盒", shape: "box", size: [1.2, 1.2, 1.5] },
  { label: "高箱", shape: "box", size: [1.5, 1.5, 3.0] },
  { label: "墙体", shape: "box", size: [4.0, 0.3, 1.5] },
  { label: "厢式车", shape: "box", size: [5.0, 2.0, 2.4], color: "#9aa0a8" },
  { label: "圆柱", shape: "cylinder", size: [1.0, 1.0, 1.6] },
  { label: "立杆", shape: "cylinder", size: [0.3, 0.3, 3.5] },
  { label: "圆桶", shape: "cylinder", size: [0.6, 0.6, 0.9], color: "#d2691e" },
  { label: "球体", shape: "sphere", size: [1.2, 1.2, 1.2] },
  { label: "锥体", shape: "cone", size: [1.4, 1.4, 1.8], color: "#d2691e" },
  { label: "轿车", shape: "car", size: [4.2, 1.8, 1.5] },
];

/* ---------------------------------- app ----------------------------------- */

type Tool = "select" | "add_object" | "add_arrow"
  | "draw_line" | "add_lane" | "add_camera";

function App() {
  const [scene, setScene] = useState<SceneV1>(() => emptyScene());
  const [sel, setSel] = useState<SelectionKind>(null);
  const [tool, setTool] = useState<Tool>("select");
  const [gizmoMode, setGizmoMode] = useState<"translate" | "rotate" | "scale">("translate");
  const [status, setStatus] = useState("ready");
  const [previewCam, setPreviewCam] = useState(0);
  const [showBevLayer, setShowBevLayer] = useState(true);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const panRef = useRef<{ x: number; y: number; px: number; py: number } | null>(null);
  // absolute sizes/scales at gizmo-drag start: TransformControls scale is
  // RELATIVE to proxy creation, and the proxy now persists through the
  // drag (deferred rebuild), so every objectChange must compute the
  // ABSOLUTE value from this snapshot (review: repeated multiplication
  // blew sizes up to 5674x)
  const dragStartRef = useRef<{
    objectSizes: Record<string, [number, number, number]>;
    arrowScales: Record<number, number>;
  } | null>(null);
  const previewWrap = useRef<HTMLDivElement>(null);
  const zoomBy = (f: number) => setZoom((z) => Math.min(8, Math.max(1, z * f)));
  const resetZoom = () => { setZoom(1); setPan({ x: 0, y: 0 }); };
  // native non-passive wheel listener: React's delegated wheel handler is
  // passive and was observed to miss events after unrelated re-renders
  useEffect(() => {
    const el = previewWrap.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      setZoom((z) => Math.min(8, Math.max(1, z * (e.deltaY < 0 ? 1.2 : 1 / 1.2))));
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);
  const [topView, setTopView] = useState(false);
  const [showGrid, setShowGrid] = useState(true);
  const [showAxes, setShowAxes] = useState(true);
  const [lineDraft, setLineDraft] = useState<number[][] | null>(null);
  const [laneDraft, setLaneDraft] = useState<number[][] | null>(null);
  const [arrowClass, setArrowClass] = useState(0);
  const [libIdx, setLibIdx] = useState(0);
  const [rightW, setRightW] = useState(280);

  const undoStack = useRef<string[]>([]);
  const redoStack = useRef<string[]>([]);
  const vpRef = useRef<ThreeViewport | null>(null);
  const viewDiv = useRef<HTMLDivElement>(null);
  const previewCanvas = useRef<HTMLCanvasElement>(null);
  const bevCanvas = useRef<HTMLCanvasElement>(null);
  const dirtyRef = useRef(true);

  const commit = useCallback((next: SceneV1 | ((s: SceneV1) => SceneV1)) => {
    setScene((prev) => {
      undoStack.current.push(JSON.stringify(prev));
      if (undoStack.current.length > 100) undoStack.current.shift();
      redoStack.current = [];
      const n = typeof next === "function" ? (next as (s: SceneV1) => SceneV1)(prev) : next;
      dirtyRef.current = true;
      return n;
    });
  }, []);

  const undo = () => {
    if (!undoStack.current.length) return;
    setScene((cur) => {
      redoStack.current.push(JSON.stringify(cur));
      const prev = JSON.parse(undoStack.current.pop()!);
      dirtyRef.current = true;
      return prev;
    });
  };
  const redo = () => {
    if (!redoStack.current.length) return;
    setScene((cur) => {
      undoStack.current.push(JSON.stringify(cur));
      const next = JSON.parse(redoStack.current.pop()!);
      dirtyRef.current = true;
      return next;
    });
  };

  /* -------------------------- viewport lifecycle --------------------------- */
  useEffect(() => {
    if (!viewDiv.current || vpRef.current) return;
    const vp = new ThreeViewport(viewDiv.current, {
      onSelect: (s) => { setSel(s); },
      onDragStart: () => {
        // one snapshot per drag gesture -> single-step undo (review fix #4)
        setScene((cur) => {
          undoStack.current.push(JSON.stringify(cur));
          if (undoStack.current.length > 100) undoStack.current.shift();
          redoStack.current = [];
          // absolute sizes at drag start for the scale write-back
          dragStartRef.current = {
            objectSizes: Object.fromEntries(
              cur.objects.map((o) => [o.object_id,
                [...(o.size_abc ?? [1, 1, 1])] as [number, number, number]])),
            arrowScales: Object.fromEntries(
              cur.arrows.map((a) => [a.instance_id, a.scale])),
          };
          return cur;
        });
      },
      onTransform: () => {
        // read proxy transform into scene (no undo push here; the drag-start
        // hook pushed one snapshot for the whole gesture)
        const t = vpRef.current?.getProxyTransform();
        if (!t || !selRef.current) return;
        const s = selRef.current;
        const dragged = !!vpRef.current?.wasDragged();
        if (!dragged) return;
        setScene((prev) => {
          const next = JSON.parse(JSON.stringify(prev)) as SceneV1;
          const ds = dragStartRef.current;
          if (s.kind === "object") {
            const o = next.objects.find((o) => o.object_id === (s as { id: string }).id);
            if (o) {
              o.center_ego = [t.pos[0], t.pos[1], t.pos[2]];
              o.yaw_rad = t.rotZ;
              const startSize = ds?.objectSizes[(s as { id: string }).id]
                ?? o.size_abc ?? [1, 1, 1];
              if (o.size_abc && t.scale.some((v) => Math.abs(v - 1) > 1e-9)) {
                // ABSOLUTE size = drag-start size * cumulative proxy scale
                // (proxy scale is relative to drag start, not per-event)
                const [sx, sy, sz] = t.scale;
                o.size_abc = [
                  Math.max(0.05, startSize[0] * sx),
                  Math.max(0.05, startSize[1] * sy),
                  Math.max(0.05, startSize[2] * sz),
                ];
              } else if (o.size_abc) {
                o.size_abc = startSize;
              }
            }
          } else if (s.kind === "arrow") {
            const a = next.arrows.find((a) => a.instance_id === (s as { instanceId: number }).instanceId);
            if (a) {
              a.center_ego = [t.pos[0], t.pos[1], 0];
              a.yaw_rad = t.rotZ;
              const startScale = ds?.arrowScales[
                (s as { instanceId: number }).instanceId] ?? a.scale;
              if (t.scale.some((v) => Math.abs(v - 1) > 1e-9)) {
                // uniform ABSOLUTE scale (aspect ratio preserved, 5.2)
                const g = Math.cbrt(t.scale[0] * t.scale[1] * t.scale[2]);
                a.scale = Math.max(0.1, startScale * g);
              } else {
                a.scale = startScale;
              }
            }
          } else if (s.kind === "lane_line_point") {
            const line = next.lane_lines.find((l) => l.line_id === (s as { lineId: string }).lineId);
            if (line) {
              const i = (s as { index: number }).index;
              line.points_ego[i] = [t.pos[0], t.pos[1], 0];
            }
          } else if (s.kind === "camera") {
            const camId = (s as { cameraId: string }).cameraId;
            const cal = next.cameras.find((c) => c.camera_id === camId);
            if (cal) {
              // translate -> position; rotate (proxy +x = view dir) ->
              // yaw/pitch; K unchanged (intrinsics stay panel-owned)
              const pos: [number, number, number] = [
                t.pos[0], t.pos[1], Math.max(0.2, t.pos[2])];
              const [fx, fy, fz] = t.forward;
              const horiz = Math.hypot(fx, fy);
              const yaw = Math.atan2(fy, fx);
              const pitch = -Math.atan2(fz, horiz); // down = positive
              cal.T_C_E = tceFromPitchYaw(pitch, yaw, pos);
            }
          }
          dirtyRef.current = true;
          return next;
        });
      },
      onGroundClick: (x, y) => {
        const t: Tool = toolRef.current;
        if (t === "add_object") {
          const entry = OBJECT_LIBRARY[arrowClassRef2.current];
          commit((prev) => {
            const next = JSON.parse(JSON.stringify(prev)) as SceneV1;
            const obj: SceneObject = {
              object_id: uid("obj"), shape: entry.shape,
              size_abc: entry.size, center_ego: [x, y, 0], yaw_rad: 0,
              color: entry.color ?? "#7a7f85",
            };
            next.objects.push(obj);
            return next;
          });
          setStatus(`added ${entry.label} at ${x.toFixed(1)},${y.toFixed(1)}`);
        } else if (t === "add_arrow") {
          commit((prev) => {
            const next = JSON.parse(JSON.stringify(prev)) as SceneV1;
            const cls = arrowClassRef.current;
            const iid = next.arrows.reduce((m, a) => Math.max(m, a.instance_id), 0) + 1;
            const a: SceneArrow = {
              instance_id: iid, template_id: TEMPLATE_IDS[cls], class_id: cls,
              center_ego: [x, y, 0], yaw_rad: 0, scale: 1, lane_id: null,
            };
            next.arrows.push(a);
            return next;
          });
          setStatus(`added ${CLASS_NAMES[arrowClassRef.current].toLowerCase()} arrow`);
        } else if (t === "draw_line") {
          setLineDraft((d) => (d ? [...d, [x, y]] : [[x, y]]));
        } else if (t === "add_lane") {
          setLaneDraft((d) => (d ? [...d, [x, y]] : [[x, y]]));
        }
      },
    });
    vpRef.current = vp;
  }, []);

  // e2e hook: full scene snapshot for tests (documented; not used by logic)
  useEffect(() => {
    (window as unknown as { __sceneSnapshot: () => SceneV1 }).__sceneSnapshot =
      () => JSON.parse(JSON.stringify(scene));
  }, [scene]);

  const selRef = useRef(sel); selRef.current = sel;
  const toolRef = useRef(tool); toolRef.current = tool;
  const arrowClassRef = useRef(arrowClass); arrowClassRef.current = arrowClass;
  const arrowClassRef2 = useRef(libIdx); arrowClassRef2.current = libIdx;

  useEffect(() => { vpRef.current?.setScene(scene); }, [scene]);
  useEffect(() => { vpRef.current?.setSelection(sel, scene); }, [sel, scene]);
  useEffect(() => { vpRef.current?.setGizmoMode(gizmoMode); }, [gizmoMode]);
  useEffect(() => { vpRef.current?.setPlaceMode(tool !== "select" && tool !== "add_camera"); }, [tool]);
  useEffect(() => { vpRef.current?.setTopView(topView); }, [topView]);

  /* ------------------------------ preview cam ------------------------------ */
  const [rightTab, setRightTab] = useState<"props" | "detect">("props");
  const [detRuns, setDetRuns] = useState<{ run: string; f1?: number;
    conf?: number }[]>([]);
  const [detRun, setDetRun] = useState("");
  const [detSplit, setDetSplit] = useState("val");
  const [detSample, setDetSample] = useState("");
  const [detResult, setDetResult] = useState<null | {
    sample_id: string; camera: { K: number[][]; T_C_E: number[][]; width: number; height: number };
    rgb_png_b64: string; bev_range: { x_min: number; x_max: number; y_min: number; y_max: number };
    conf_threshold?: number; conf_source?: string;
    predictions: { prob: number; cls: number; center_norm: number[];
      dir: number[] }[];
    gt: { class_id: number; center_ego: number[]; yaw_rad: number;
      arrow_valid: boolean; ignore_reason: string | null }[];
  }>(null);
  const [detIds, setDetIds] = useState<string[]>([]);
  const [detBusy, setDetBusy] = useState(false);
  const [detErr, setDetErr] = useState("");

  const loadDetRuns = useCallback(async () => {
    try {
      const r = await fetch("/api/detect-runs");
      const j = await r.json();
      setDetRuns(j.runs ?? []);
      if (j.runs?.length && !detRun) {
        setDetRun(j.runs[0].run);
        setDetSample("");
      }
    } catch { /* server offline */ }
  }, [detRun]);

  const runDetect = useCallback(async (sampleId?: string) => {
    if (!detRun) return;
    setDetBusy(true); setDetErr(""); setDetResult(null);
    try {
      const sid = (sampleId ?? detSample) || detIds[0] || "";
      const r = await fetch("/api/detect", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run: detRun, split: detSplit,
                               sample_id: sid }),
      });
      if (!r.ok) {
        setDetErr(`detect failed: HTTP ${r.status}`);
        return;
      }
      const j = await r.json();
      setDetResult(j);
      setDetSample(j.sample_id);
    } catch (e) {
      setDetErr(`detect error: ${e}`);
    } finally {
      setDetBusy(false);
    }
  }, [detRun, detSplit, detSample, detIds]);

  // stepping stays WITHIN the split's id list (clamped at the edges)
  const detStep = useCallback(async (delta: number) => {
    const cur = detResult?.sample_id ?? detSample;
    let next: string | undefined;
    if (detIds.length) {
      const i = detIds.indexOf(cur);
      if (i >= 0) {
        const j = Math.min(detIds.length - 1, Math.max(0, i + delta));
        next = detIds[j];
      } else {
        next = delta > 0 ? detIds[0] : detIds[detIds.length - 1];
      }
    } else {
      const m = cur.match(/^(.*?)(\d+)$/);
      if (!m) return;
      next = `${m[1]}${String(parseInt(m[2]) + delta).padStart(
        m[2].length, "0")}`;
    }
    if (next && next !== cur) await runDetect(next);
  }, [detResult, detSample, detIds, runDetect]);

  // run/split change: clear stale overlays, load the split's id list
  // and jump to its FIRST sample
  useEffect(() => {
    let cancelled = false;
    setDetResult(null);       // never show the old selection's overlay
    setDetSample("");
    (async () => {
      try {
        const r = await fetch(`/api/detect-samples?split=${detSplit}`);
        if (!r.ok) return;
        const j = await r.json();
        if (cancelled) return;
        setDetIds(j.ids ?? []);
        if (j.ids?.length) setDetSample(j.ids[0]);  // jump to first frame
      } catch { /* offline */ }
    })();
    return () => { cancelled = true; };
  }, [detSplit]);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [viewerOpen, setViewerOpen] = useState(false);
  const refreshPreview = useCallback(async () => {
    const camId = scene.cameras[previewCam]?.camera_id;
    if (!camId || !previewCanvas.current) return;
    try {
      const res = await fetch("/api/render-sample-preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scene, camera_id: camId }),
      });
      if (!res.ok) { setStatus("preview failed (server offline?)"); return; }
      const data = await res.json();
      const { rgb, bev } = data as { rgb: { w: number; h: number; b64: string }; bev?: { w: number; h: number; b64: string } };
      drawB64(previewCanvas.current, rgb);
      setPreviewUrl(previewCanvas.current.toDataURL("image/png"));
      if (bev && bevCanvas.current) drawB64(bevCanvas.current, bev, true);
      setStatus("preview updated");
    } catch (e) {
      setStatus(`preview error: ${e}`);
    }
  }, [scene, previewCam]);

  /** Camera "look" drag on the preview at 1x: dx -> yaw, dy -> pitch.
   *  Pushes ONE undo snapshot per gesture like the 3D gizmo. */
  const splitRef = useRef<{ x: number; w: number } | null>(null);
  const lookRef = useRef<{ x: number; y: number; yaw: number; pitch: number;
                           pushed: boolean } | null>(null);
  const lookDrag = (e: React.PointerEvent) => {
    if (zoom > 1) return; // zoomed: pointer drag pans instead
    const cal = scene.cameras[previewCam];
    if (!cal) return;
    const pose0 = cameraPoseFromT(cal.T_C_E);
    lookRef.current = {
      x: e.clientX, y: e.clientY,
      yaw: pose0.yaw,
      pitch: pose0.pitchDown,
      pushed: false,
    };
    (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
  };
  const lookMove = (e: React.PointerEvent) => {
    const st = lookRef.current;
    if (!st) return;
    if (!st.pushed) {
      // one undo snapshot per look gesture
      setScene((cur) => {
        undoStack.current.push(JSON.stringify(cur));
        if (undoStack.current.length > 100) undoStack.current.shift();
        redoStack.current = [];
        return cur;
      });
      st.pushed = true;
    }
    const cal = scene.cameras[previewCam];
    if (!cal) return;
    const dx = e.clientX - st.x, dy = e.clientY - st.y;
    const yaw = st.yaw - dx * 0.003;   // drag right -> look right
    const pitch = st.pitch + dy * 0.003; // drag down -> look down
    const pos = cameraPos(cal);
    // no per-move undo: the gesture pushed a single snapshot on start
    patchCameraPose(cal.camera_id, pitch, yaw, pos, false);
  };
  const lookEnd = () => { lookRef.current = null; };
  const patchCameraPose = (cameraId: string, pitchDeg: number,
                           yawRad: number, pos: [number, number, number],
                           withUndo = true) => {
    const apply = (prev: SceneV1): SceneV1 => {
      const next = JSON.parse(JSON.stringify(prev)) as SceneV1;
      const cal = next.cameras.find((c) => c.camera_id === cameraId);
      if (cal) cal.T_C_E = tceFromPitchYaw(pitchDeg, yawRad, pos);
      return next;
    };
    if (withUndo) commit(apply);
    else setScene(apply);  // gesture already pushed one snapshot
  };

  useEffect(() => {
    const t = setTimeout(refreshPreview, 600);
    return () => clearTimeout(t);
  }, [refreshPreview]);

  function drawB64(canvas: HTMLCanvasElement, img: { w: number; h: number; b64: string }, colorize = false) {
    const bin = atob(img.b64);
    const ctx = canvas.getContext("2d")!;
    canvas.width = img.w; canvas.height = img.h;
    const id = ctx.createImageData(img.w, img.h);
    for (let i = 0, j = 0; i < bin.length; i += 4, j += 4) {
      const r = bin.charCodeAt(i), g = bin.charCodeAt(i + 1), b = bin.charCodeAt(i + 2);
      if (colorize) {
        const idv = r + 256 * g + 65536 * b;
        if (idv === 0) { id.data[j] = 24; id.data[j + 1] = 26; id.data[j + 2] = 30; }
        else {
          const h = (idv * 47) % 360;
          const [r2, g2, b2] = hslToRgb(h / 360, 0.75, 0.55);
          id.data[j] = r2; id.data[j + 1] = g2; id.data[j + 2] = b2;
        }
      } else {
        id.data[j] = r; id.data[j + 1] = g; id.data[j + 2] = b;
      }
      id.data[j + 3] = 255;
    }
    ctx.putImageData(id, 0, 0);
  }

  function hslToRgb(h: number, s: number, l: number): [number, number, number] {
    const f = (n: number) => {
      const k = (n + h * 12) % 12;
      const a = s * Math.min(l, 1 - l);
      return Math.round(255 * (l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1))));
    };
    return [f(0), f(8), f(4)];
  }

  /* ------------------------------- file ops -------------------------------- */
  const saveJson = () => {
    const blob = new Blob([JSON.stringify(scene, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${scene.scene_id}.json`;
    a.click();
  };

  const loadJson = () => {
    const inp = document.createElement("input");
    inp.type = "file"; inp.accept = ".json";
    inp.onchange = async () => {
      const f = inp.files?.[0];
      if (!f) return;
      try {
        const parsed = JSON.parse(await f.text());
        const err = validateSceneV1(parsed);
        if (err) {
          setStatus(`invalid scene: ${err}; refusing`);
          return;
        }
        commit(parsed);
        setStatus(`loaded ${f.name}`);
      } catch (e) { setStatus(`load failed: ${e}`); }
    };
    inp.click();
  };

  const exportSample = async () => {
    setStatus("exporting sample...");
    try {
      const res = await fetch("/api/render-sample", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scene, camera_id: scene.cameras[0].camera_id }),
      });
      if (!res.ok) { setStatus(`export failed: ${res.status}`); return; }
      const blob = await res.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `${scene.scene_id}_sample.zip`;
      a.click();
      setStatus("sample exported");
    } catch (e) { setStatus(`export error: ${e}`); }
  };

  /* ------------------------------ batch panel ------------------------------ */
  const [batch, setBatch] = useState({
    dataset: "arrow_editor_batch", count: 16, seed: 42,
    mode: "variants" as "variants" | "random",
  });
  const [jobInfo, setJobInfo] = useState<{ job_id: string; status: string; done?: number; total?: number; output_dir?: string } | null>(null);

  const startBatch = async () => {
    const res = await fetch("/api/render-jobs", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_name: batch.dataset, count: batch.count, seed: batch.seed,
        mode: batch.mode, scene,
      }),
    });
    const j = await res.json();
    setJobInfo(j);
    pollJob(j.job_id);
  };
  const pollJob = (id: string) => {
    const iv = setInterval(async () => {
      const res = await fetch(`/api/render-jobs/${id}`);
      const j = await res.json();
      setJobInfo(j);
      if (["completed", "failed", "cancelled"].includes(j.status)) clearInterval(iv);
    }, 2000);
  };

  const cancelJob = async () => {
    if (!jobInfo) return;
    await fetch(`/api/render-jobs/${jobInfo.job_id}/cancel`, { method: "POST" });
    setJobInfo({ ...jobInfo, status: "cancelled" });
  };

  /* ------------------------------ selected UI ------------------------------ */
  const selPanel = useMemo(() => {
    if (!sel) return <div style={{ color: "#888" }}>nothing selected</div>;
    if (sel.kind === "arrow") {
      const a = scene.arrows.find((x) => x.instance_id === sel.instanceId)!;
      return (
        <div>
          <b>arrow #{a.instance_id}</b>
          <div style={{ marginBottom: 6 }}>
            <select value={a.class_id} onChange={(e) => {
              const cls = parseInt(e.target.value);
              commit((prev) => {
                const next = JSON.parse(JSON.stringify(prev)) as SceneV1;
                const t = next.arrows.find((x) => x.instance_id === a.instance_id)!;
                t.class_id = cls; t.template_id = TEMPLATE_IDS[cls];
                return next;
              });
            }}>
              {CLASS_NAMES.map((n: string, i: number) => <option key={n} value={i}>{i} {n}</option>)}
            </select>
          </div>
          <NumField label="x (m)" value={a.center_ego[0]} onChange={(v) => patchArrow(a.instance_id, { center_ego: [v, a.center_ego[1], 0] })} />
          <NumField label="y (m)" value={a.center_ego[1]} onChange={(v) => patchArrow(a.instance_id, { center_ego: [a.center_ego[0], v, 0] })} />
          <NumField label="yaw (rad)" value={a.yaw_rad} onChange={(v) => patchArrow(a.instance_id, { yaw_rad: v })} />
          <NumField label="scale (uniform)" value={a.scale} onChange={(v) => patchArrow(a.instance_id, { scale: Math.max(0.1, v) })} />
          <div>
            lane_id:
            <select value={a.lane_id ?? ""} onChange={(e) => patchArrow(a.instance_id, { lane_id: e.target.value || null })}>
              <option value="">(none / free)</option>
              {scene.lanes.map((l) => <option key={l.lane_id} value={l.lane_id}>{l.lane_id}</option>)}
            </select>
          </div>
          <Row><Btn onClick={() => deleteArrow(a.instance_id)}>delete</Btn>
            <Btn onClick={() => commit((prev) => {
              const next = JSON.parse(JSON.stringify(prev)) as SceneV1;
              const iid = next.arrows.reduce((m, x) => Math.max(m, x.instance_id), 0) + 1;
              next.arrows.push({ ...a, instance_id: iid, center_ego: [a.center_ego[0] + 1, a.center_ego[1], 0] });
              return next;
            })}>duplicate</Btn></Row>
        </div>
      );
    }
    if (sel.kind === "object") {
      const o = scene.objects.find((x) => x.object_id === sel.id)!;
      return (
        <div>
          <b>{o.shape} {o.object_id}</b>
          {(["x", "y", "z"] as const).map((k, i) => (
            <NumField key={k} label={`center ${k}`} value={o.center_ego[i]}
              onChange={(v) => patchObject(o.object_id, (oo) => { oo.center_ego[i] = v; })} />
          ))}
          {o.size_abc && (["len", "wid", "hgt"] as const).map((k, i) => (
            <NumField key={k} label={`size ${k}`} value={o.size_abc![i]}
              onChange={(v) => patchObject(o.object_id, (oo) => { oo.size_abc![i] = Math.max(0.05, v); })} />
          ))}
          <NumField label="yaw (rad)" value={o.yaw_rad}
            onChange={(v) => patchObject(o.object_id, (oo) => { oo.yaw_rad = v; })} />
          <div>color: <input type="color" value={o.color ?? "#7a7f85"}
            onChange={(e) => patchObject(o.object_id, (oo) => { oo.color = e.target.value; })} /></div>
          <Row><Btn onClick={() => deleteObject(o.object_id)}>delete</Btn>
            <Btn onClick={() => commit((prev) => {
              const next = JSON.parse(JSON.stringify(prev)) as SceneV1;
              next.objects.push({ ...o, object_id: uid("obj"), center_ego: [o.center_ego[0] + 1.5, o.center_ego[1], o.center_ego[2]] });
              return next;
            })}>duplicate</Btn></Row>
        </div>
      );
    }
    if (sel.kind === "lane_line_point") {
      const line = scene.lane_lines.find((l) => l.line_id === sel.lineId)!;
      const p = line.points_ego[sel.index];
      return (
        <div>
          <b>{line.line_id} point #{sel.index}</b>
          <NumField label="x" value={p[0]} onChange={(v) => patchLinePoint(line.line_id, sel.index, 0, v)} />
          <NumField label="y" value={p[1]} onChange={(v) => patchLinePoint(line.line_id, sel.index, 1, v)} />
          <Row><Btn onClick={() => deleteLinePoint(line.line_id, sel.index)}>delete point</Btn>
            <Btn onClick={() => deleteLine(line.line_id)}>delete line</Btn></Row>
        </div>
      );
    }
    if (sel.kind === "camera") {
      const cal = scene.cameras.find((c) => c.camera_id === sel.cameraId)!;
      return (
        <div>
          <b>camera {cal.camera_id}</b>
          <p style={{ color: "#888", fontSize: 12 }}>
            Pose is edited via position/pitch below; K is derived from vertical
            FOV and resolution (square pixels, principal point at center).
          </p>
          <NumField label="pos x" value={cameraPos(cal)[0]} onChange={(v) => patchCamera(cal.camera_id, { pos: [v, cameraPos(cal)[1], cameraPos(cal)[2]] })} />
          <NumField label="pos y" value={cameraPos(cal)[1]} onChange={(v) => patchCamera(cal.camera_id, { pos: [cameraPos(cal)[0], v, cameraPos(cal)[2]] })} />
          <NumField label="pos z (height)" value={cameraPos(cal)[2]} onChange={(v) => patchCamera(cal.camera_id, { pos: [cameraPos(cal)[0], cameraPos(cal)[1], v] })} />
          <NumField label="pitch down (deg)" value={(cameraPoseFromT(cal.T_C_E).pitchDown * 180) / Math.PI} step={1}
            onChange={(v) => patchCamera(cal.camera_id, { pitchDeg: v })} />
          <NumField label="yaw (deg)" value={(cameraPoseFromT(cal.T_C_E).yaw * 180) / Math.PI} step={1}
            onChange={(v) => patchCamera(cal.camera_id, { yawDeg: v })} />
          <NumField label="width px" value={cal.width} step={1} onChange={(v) => patchCamera(cal.camera_id, { width: Math.round(v) })} />
          <NumField label="height px" value={cal.height} step={1} onChange={(v) => patchCamera(cal.camera_id, { height: Math.round(v) })} />
          <NumField label="v-FOV (deg)" value={(2 * Math.atan2(cal.height / 2, cal.K[1][1]) * 180) / Math.PI} step={1}
            onChange={(v) => patchCamera(cal.camera_id, { fovDeg: v })} />
          <Row>
            <Btn onClick={() => commit((prev) => {
              const next = JSON.parse(JSON.stringify(prev)) as SceneV1;
              const src = next.cameras.find((c) => c.camera_id === cal.camera_id)!;
              next.cameras.push({ ...src, camera_id: uid("cam") });
              return next;
            })}>duplicate</Btn>
            {scene.cameras.length > 1 && <Btn onClick={() => commit((prev) => {
              const next = JSON.parse(JSON.stringify(prev)) as SceneV1;
              next.cameras = next.cameras.filter((c) => c.camera_id !== cal.camera_id);
              return next;
            })}>delete</Btn>}
          </Row>
        </div>
      );
    }
    return null;
  }, [sel, scene, commit]);

  /* ------------------------------ patch helpers ---------------------------- */
  function patchArrow(instanceId: number, patch: Partial<SceneArrow>) {
    commit((prev) => {
      const next = JSON.parse(JSON.stringify(prev)) as SceneV1;
      const a = next.arrows.find((x) => x.instance_id === instanceId)!;
      Object.assign(a, patch);
      return next;
    });
  }
  function patchObject(objectId: string, patch: (o: SceneObject) => void) {
    commit((prev) => {
      const next = JSON.parse(JSON.stringify(prev)) as SceneV1;
      const o = next.objects.find((x) => x.object_id === objectId)!;
      patch(o);
      return next;
    });
  }
  function patchLinePoint(lineId: string, index: number, axis: 0 | 1, v: number) {
    commit((prev) => {
      const next = JSON.parse(JSON.stringify(prev)) as SceneV1;
      const l = next.lane_lines.find((x) => x.line_id === lineId)!;
      const p = l.points_ego[index];
      p[axis] = v;
      if (axis === 0) l.points_ego[index] = [v, p[1], p[2]];
      return next;
    });
  }
  function patchCamera(cameraId: string, patch: Partial<{
    pos: [number, number, number]; pitchDeg: number; yawDeg: number;
    width: number; height: number; fovDeg: number;
  }>) {
    commit((prev) => {
      const next = JSON.parse(JSON.stringify(prev)) as SceneV1;
      const cal = next.cameras.find((c) => c.camera_id === cameraId)!;
      const cur = cameraPos(cal);
      const pos = patch.pos ?? cur;
      const pose0 = cameraPoseFromT(cal.T_C_E);
      const pitchDeg = patch.pitchDeg ?? (pose0.pitchDown * 180) / Math.PI;
      const yawDeg = patch.yawDeg ?? (pose0.yaw * 180) / Math.PI;
      const width = patch.width ?? cal.width;
      const height = patch.height ?? cal.height;
      const fovDeg = patch.fovDeg ?? (2 * Math.atan2(cal.height / 2, cal.K[1][1]) * 180) / Math.PI;
      const intr = intrinsicsFromFovV((fovDeg * Math.PI) / 180, width, height);
      const T = tceFromPitchYaw((pitchDeg * Math.PI) / 180, (yawDeg * Math.PI) / 180, pos);
      cal.width = width; cal.height = height;
      cal.K = KFromIntrinsics(intr);
      cal.T_C_E = T;
      return next;
    });
  }
  function deleteArrow(iid: number) {
    commit((prev) => ({ ...prev, arrows: prev.arrows.filter((a) => a.instance_id !== iid) }));
    setSel(null);
  }
  function deleteObject(oid: string) {
    commit((prev) => ({ ...prev, objects: prev.objects.filter((o) => o.object_id !== oid) }));
    setSel(null);
  }
  function deleteLinePoint(lineId: string, index: number) {
    commit((prev) => {
      const next = JSON.parse(JSON.stringify(prev)) as SceneV1;
      const l = next.lane_lines.find((x) => x.line_id === lineId)!;
      if (l.points_ego.length > 2) l.points_ego.splice(index, 1);
      else next.lane_lines = next.lane_lines.filter((x) => x.line_id !== lineId);
      return next;
    });
    setSel(null);
  }
  function deleteLine(lineId: string) {
    commit((prev) => ({ ...prev, lane_lines: prev.lane_lines.filter((l) => l.line_id !== lineId) }));
    setSel(null);
  }

  /* --------------------------------- render -------------------------------- */
  return (
    <div style={{ display: "grid",
                   gridTemplateRows: "42px 1fr 235px",
                   gridTemplateColumns: `200px 1fr 5px ${rightW}px`,
                   gridTemplateAreas: `"top top top top" "side view split panel" "dock dock split panel"`,
                   height: "100vh", color: "#e8e8e8", fontFamily: "sans-serif",
                   background: "#14171b" }}>
      {/* top toolbar */}
      <div style={{ gridArea: "top", display: "flex", alignItems: "center",
                    gap: 6, padding: "0 10px", borderBottom: "1px solid #2a2f36",
                    background: "#181c21" }}>
        <b style={{ marginRight: 8 }}>arrow-toy</b>
        <Btn onClick={undo} title="undo">↶ 撤销</Btn>
        <Btn onClick={redo} title="redo">↷ 重做</Btn>
        <Btn onClick={() => commit(emptyScene())}>新建</Btn>
        <Btn onClick={saveJson}>保存</Btn>
        <Btn onClick={loadJson}>导入</Btn>
        <span style={{ width: 1, height: 22, background: "#2a2f36" }} />
        <Btn active={topView} onClick={() => setTopView(!topView)}>俯视</Btn>
        <Btn active={showGrid} onClick={() => { setShowGrid(!showGrid);
          vpRef.current?.setHelpers(!showGrid, null); }}>网格</Btn>
        <Btn active={showAxes} onClick={() => { setShowAxes(!showAxes);
          vpRef.current?.setHelpers(null, !showAxes); }}>坐标轴</Btn>
        <Btn onClick={() => vpRef.current?.focusScene(scene)}>聚焦</Btn>
        <Btn onClick={() => vpRef.current?.resetView()}>复位视角</Btn>
        <span style={{ marginLeft: "auto", fontSize: 12, color: "#8aa" }}>{status}</span>
      </div>
      {/* left sidebar: tools */}
      <div style={{ gridArea: "side", borderRight: "1px solid #2a2f36", padding: 8, overflowY: "auto" }}>
        <div style={{ fontSize: 11, color: "#9ab", marginBottom: 4 }}>tools</div>
        <Row><Btn active={tool === "select"} onClick={() => setTool("select")}>select</Btn></Row>
        <div style={{ fontSize: 11, color: "#9ab", margin: "4px 0" }}>object library</div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 4 }}>
          {OBJECT_LIBRARY.map((entry, idx) => (
            <Btn key={entry.label}
                 active={tool === "add_object" && libIdx === idx}
                 onClick={() => { setTool("add_object"); setLibIdx(idx); }}
                 title={`${entry.shape} ${entry.size.join("x")}`}>
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <ObjectThumb shape={entry.shape} />
                <span style={{ fontSize: 11 }}>{entry.label}</span>
              </div>
            </Btn>
          ))}
        </div>
        <div style={{ fontSize: 11, color: "#9ab", margin: "4px 0" }}>arrow</div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 4 }}>
          {CLASS_NAMES.map((n: string, i: number) => (
            <Btn key={n} title={n}
                 active={tool === "add_arrow" && arrowClass === i}
                 onClick={() => { setTool("add_arrow"); setArrowClass(i); }}>
              <div style={{ display: "flex", flexDirection: "column",
                            alignItems: "center", gap: 2 }}>
                <ArrowThumb classId={i} />
                <span style={{ fontSize: 10 }}>{["直行", "左转", "右转",
                  "直左", "直右", "左右"][i]}</span>
              </div>
            </Btn>
          ))}
        </div>
        <Row><Btn active={tool === "draw_line"} onClick={() => { setTool("draw_line"); setLineDraft([]); }}>lane line</Btn></Row>
        {tool === "draw_line" && lineDraft && (
          <div>
            <div style={{ fontSize: 11 }}>click points; then:</div>
            <Btn onClick={() => {
              if (lineDraft.length >= 2) {
                commit((prev) => {
                  const next = JSON.parse(JSON.stringify(prev)) as SceneV1;
                  const line: LaneLine = {
                    line_id: uid("line"),
                    points_ego: lineDraft.map(([x, y]) => [x, y, 0]),
                    style: "solid", width_m: 0.12, color: "#d9d9d9",
                  };
                  next.lane_lines.push(line);
                  return next;
                });
              }
              setLineDraft(null); setTool("select");
            }}>finish solid</Btn>{" "}
            <Btn onClick={() => {
              if (lineDraft.length >= 2) {
                commit((prev) => {
                  const next = JSON.parse(JSON.stringify(prev)) as SceneV1;
                  next.lane_lines.push({
                    line_id: uid("line"),
                    points_ego: lineDraft.map(([x, y]) => [x, y, 0]),
                    style: "dashed", width_m: 0.1, color: "#d9d9d9",
                    dash_length_m: 2, gap_length_m: 2,
                  });
                  return next;
                });
              }
              setLineDraft(null); setTool("select");
            }}>finish dashed</Btn>
          </div>
        )}
        <Row><Btn active={tool === "add_lane"} onClick={() => { setTool("add_lane"); setLaneDraft([]); }}>lane (center)</Btn></Row>
        {tool === "add_lane" && laneDraft && (
          <div>
            <div style={{ fontSize: 11 }}>click 2+ points for centerline; then:</div>
            <Btn onClick={() => {
              if (laneDraft.length >= 2) {
                commit((prev) => {
                  const next = JSON.parse(JSON.stringify(prev)) as SceneV1;
                  const lane: Lane = {
                    lane_id: uid("lane"),
                    centerline_ego: laneDraft.map(([x, y]) => [x, y, 0]),
                    width_m: 3.4, allowed_turns_slr: [0, 0, 0],
                    world_rule_known: false, rule_source: null, show_boundaries: true,
                  };
                  next.lanes.push(lane);
                  return next;
                });
              }
              setLaneDraft(null); setTool("select");
            }}>finish lane</Btn>
          </div>
        )}
        <Row><Btn onClick={() => commit((prev) => {
          const next = JSON.parse(JSON.stringify(prev)) as SceneV1;
          const K = intrinsicsFromFovV((60 * Math.PI) / 180, 320, 192);
          next.cameras.push({
            camera_id: uid("cam"), width: 320, height: 192,
            K: KFromIntrinsics(K), T_C_E: defaultCameraTCE(), near: 0.1, far: 100,
          });
          return next;
        })}>+camera</Btn></Row>
        <hr style={{ borderColor: "#2a2f36" }} />
        <Row>
          <Btn active={gizmoMode === "translate"} onClick={() => setGizmoMode("translate")}>move</Btn>
          <Btn active={gizmoMode === "rotate"} onClick={() => setGizmoMode("rotate")}>rotate</Btn>
          <Btn active={gizmoMode === "scale"} onClick={() => setGizmoMode("scale")}>scale</Btn>
        </Row>
        <Row>
          <Btn active={showBevLayer} onClick={() => setShowBevLayer(!showBevLayer)}>BEV 层</Btn>
        </Row>
        <hr style={{ borderColor: "#2a2f36" }} />
        <div style={{ fontSize: 11, color: "#9ab" }}>ground</div>
        <input type="color" value={scene.ground.color} onChange={(e) => commit((prev) => ({ ...prev, ground: { ...prev.ground, color: e.target.value } }))} />
        <NumField label="size x" value={scene.ground.size_x} onChange={(v) => commit((prev) => ({ ...prev, ground: { ...prev.ground, size_x: v } }))} />
        <NumField label="size y" value={scene.ground.size_y} onChange={(v) => commit((prev) => ({ ...prev, ground: { ...prev.ground, size_y: v } }))} />
        <hr style={{ borderColor: "#2a2f36" }} />
        <Btn onClick={exportSample}>export sample (zip)</Btn>
      </div>

      {/* 3D view + drafts */}
      <div style={{ gridArea: "view", position: "relative" }}>
        <div ref={viewDiv} data-testid="view3d" style={{ width: "100%", height: "100%" }} />
        <div style={{ position: "absolute", top: 6, left: 8, fontSize: 12, color: "#9ab", pointerEvents: "none" }}>
          {lineDraft?.length ? `line pts: ${lineDraft.length}` : ""} {laneDraft?.length ? `lane pts: ${laneDraft.length}` : ""}
        </div>
      </div>

      {/* bottom dock: camera preview + BEV layer side by side */}
      <div style={{ gridArea: "dock", display: "flex", gap: 8, padding: "6px 8px",
                    borderTop: "1px solid #2a2f36", background: "#171b1f",
                    overflow: "hidden" }}>
        <div style={{ flex: "0 0 auto", display: "flex", flexDirection: "column", maxWidth: "45%" }}>
          <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap", marginBottom: 4 }}>
            <b style={{ fontSize: 12 }}>相机预览</b>
            <select value={previewCam} onChange={(e) => setPreviewCam(parseInt(e.target.value))}>
              {scene.cameras.map((c, i) => <option key={c.camera_id} value={i}>{c.camera_id}</option>)}
            </select>
            <Btn onClick={refreshPreview}>刷新</Btn>
            <Btn onClick={() => zoomBy(1 / 1.5)} title="zoom out">−</Btn>
            <span style={{ fontSize: 12, minWidth: 34, textAlign: "center" }}>{zoom.toFixed(1)}×</span>
            <Btn onClick={() => zoomBy(1.5)} title="zoom in">＋</Btn>
            <Btn onClick={resetZoom} title="reset zoom/pan">复位</Btn>
            <Btn onClick={() => setViewerOpen(true)} title="open fullscreen viewer">⤢ 大图</Btn>
          </div>
          <div
            ref={previewWrap}
            data-testid="preview-box"
            style={{ position: "relative", display: "inline-block",
                     height: 184, overflow: "hidden",
                     border: "1px solid #333",
                     cursor: zoom > 1 ? "grab" : "default" }}
            onPointerDown={(e) => {
              if (zoom <= 1) { lookDrag(e); return; }
              panRef.current = { x: e.clientX, y: e.clientY, px: pan.x, py: pan.y };
              (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
            }}
            onPointerMove={(e) => {
              const p = panRef.current;
              if (p) {
                setPan({ x: p.px + (e.clientX - p.x), y: p.py + (e.clientY - p.y) });
                return;
              }
              if (lookRef.current) lookMove(e);
            }}
            onPointerUp={() => { panRef.current = null; lookEnd(); }}
            onDoubleClick={resetZoom}
            title="拖拽: 1x 转向相机 / 放大后平移; 滚轮: 缩放"
          >
            <canvas ref={previewCanvas} data-testid="preview-canvas" style={{
              height: "100%", width: "auto", display: "block",
              imageRendering: "pixelated",
              transform: `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`,
              transformOrigin: "0 0",
            }} />
          </div>
          {viewerOpen && previewUrl && (
            <FullscreenViewer
              url={previewUrl}
              onClose={() => setViewerOpen(false)}
            />
          )}
        </div>
        {showBevLayer && <div style={{ flex: "0 0 auto", display: "flex",
                                       flexDirection: "column", paddingLeft: 8,
                                       borderLeft: "1px solid #2a2f36" }}>
          <div style={{ fontSize: 11, color: "#9ab", marginBottom: 4 }}>BEV (上=远, 左=左)</div>
          <canvas ref={bevCanvas} data-testid="bev-canvas" style={{
            height: 184, width: "auto", display: "block",
            imageRendering: "pixelated", border: "1px solid #333" }} />
        </div>}
        <div style={{ marginLeft: "auto", alignSelf: "end", fontSize: 11, color: "#667" }}>
          stats: arrows {scene.arrows.length} · objects {scene.objects.length} · lines {scene.lane_lines.length} · lanes {scene.lanes.length} · cameras {scene.cameras.length}
        </div>
      </div>

      {/* panel splitter (drag to resize the right panel) */}
      <div
        onPointerDown={(e) => {
          splitRef.current = { x: e.clientX, w: rightW };
          (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
        }}
        onPointerMove={(e) => {
          const p = splitRef.current;
          if (!p) return;
          setRightW(Math.min(560, Math.max(220, p.w - (e.clientX - p.x))));
        }}
        onPointerUp={() => { splitRef.current = null; }}
        style={{ gridArea: "split", cursor: "col-resize", background: "#22262c" }}
        title="drag to resize panel"
      />
      {/* right panel */}
      <div data-testid="props-panel" style={{ gridArea: "panel", borderLeft: "1px solid #2a2f36", padding: 8, overflowY: "auto" }}>
        <div style={{ display: "flex", gap: 4, marginBottom: 8 }}>
          <Btn active={rightTab === "props"} onClick={() => setRightTab("props")}>属性</Btn>
          <Btn active={rightTab === "detect"} onClick={() => { setRightTab("detect"); loadDetRuns(); }}>检测结果</Btn>
        </div>
        {rightTab === "props" && <>
        {selPanel}
        <h4 style={{ margin: "12px 0 6px" }}>batch generation</h4>
        </>}
        {rightTab === "detect" && (
          <DetectionPanel
            runs={detRuns} run={detRun} setRun={(r) => {
              setDetResult(null); setDetRun(r); }}
            split={detSplit} setSplit={setDetSplit}
            sample={detSample} setSample={setDetSample}
            result={detResult} busy={detBusy} err={detErr}
            onDetect={() => runDetect()} onStep={detStep}
            onRefreshRuns={loadDetRuns}
          />
        )}
        <label style={{ display: "flex", justifyContent: "space-between", gap: 6, marginBottom: 4 }}>
          <span>dataset</span>
          <input value={batch.dataset} onChange={(e) => setBatch({ ...batch, dataset: e.target.value })} style={{ width: 130 }} />
        </label>
        <NumField label="count" value={batch.count} step={1} onChange={(v) => setBatch({ ...batch, count: Math.max(1, Math.round(v)) })} />
        <NumField label="seed" value={batch.seed} step={1} onChange={(v) => setBatch({ ...batch, seed: Math.round(v) })} />
        <Row>
          <Btn active={batch.mode === "variants"} onClick={() => setBatch({ ...batch, mode: "variants" })}>from this scene</Btn>
          <Btn active={batch.mode === "random"} onClick={() => setBatch({ ...batch, mode: "random" })}>random</Btn>
        </Row>
        <Row>
          <Btn onClick={startBatch}>start job</Btn>
          {jobInfo && !["completed", "failed", "cancelled"].includes(jobInfo.status) && <Btn onClick={cancelJob}>cancel</Btn>}
        </Row>
        {jobInfo && (
          <div style={{ fontSize: 12 }}>
            status: {jobInfo.status} {jobInfo.done != null && `(${jobInfo.done}/${jobInfo.total})`}
            {jobInfo.output_dir && <div>output: {jobInfo.output_dir}</div>}
          </div>
        )}
        <h4 style={{ margin: "12px 0 6px" }}>stats</h4>
        <div style={{ fontSize: 12, color: "#9ab" }}>
          arrows: {scene.arrows.length}, objects: {scene.objects.length},<br />
          lines: {scene.lane_lines.length}, lanes: {scene.lanes.length},<br />
          cameras: {scene.cameras.length}
        </div>
      </div>
    </div>
  );
}

/* --------------------------- camera pose helpers -------------------------- */

function cameraPos(cal: CameraCalibration): [number, number, number] {
  // t_E = -R^T t_C with T_C_E = [R|t]
  const R = [
    [cal.T_C_E[0][0], cal.T_C_E[0][1], cal.T_C_E[0][2]],
    [cal.T_C_E[1][0], cal.T_C_E[1][1], cal.T_C_E[1][2]],
    [cal.T_C_E[2][0], cal.T_C_E[2][1], cal.T_C_E[2][2]],
  ];
  const t = [cal.T_C_E[0][3], cal.T_C_E[1][3], cal.T_C_E[2][3]];
  return [
    -(R[0][0] * t[0] + R[1][0] * t[1] + R[2][0] * t[2]),
    -(R[0][1] * t[0] + R[1][1] * t[1] + R[2][1] * t[2]),
    -(R[0][2] * t[0] + R[1][2] * t[1] + R[2][2] * t[2]),
  ];
}

function tceFromPitchYaw(pitchDown: number, yaw: number,
                         pos: [number, number, number]): number[][] {
  // WORLD-frame yaw/pitch: the optical axis (row 2 of R) points along
  // (cos a cos y, cos a sin y, -sin a) in ego coords. Rows are the
  // camera axes in ego (right / down / forward); yaw rotates the
  // heading in the world X-Y plane, pitchDown tilts it below horizon.
  // (A previous version composed Rz @ R0 which rotated in the CAMERA
  // frame - a roll that left the optical axis unchanged.)
  const sa = Math.sin(pitchDown), ca = Math.cos(pitchDown);
  const sy = Math.sin(yaw), cy = Math.cos(yaw);
  const R = [
    [sy, -cy, 0],                    // camera right in ego
    [-sa * cy, -sa * sy, -ca],       // camera down in ego
    [ca * cy, ca * sy, -sa],         // camera forward (optical axis)
  ];
  const t = [0, 1, 2].map((i) =>
    -(R[i][0] * pos[0] + R[i][1] * pos[1] + R[i][2] * pos[2]));
  return [...R.map((r, i) => [...r, t[i]]), [0, 0, 0, 1]];
}

/** Extract world-frame yaw/pitch from T_C_E (inverse of the above). */
function cameraPoseFromT(T: number[][]): { yaw: number; pitchDown: number } {
  const fx = T[2][0], fy = T[2][1], fz = T[2][2];
  return {
    yaw: Math.atan2(fy, fx),
    pitchDown: Math.asin(-fz),
  };
}

createRoot(document.getElementById("root")!).render(<App />);
