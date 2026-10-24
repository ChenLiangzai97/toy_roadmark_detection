// Scene editor (design doc section 5). Single-file app: toolbar, 3D view,
// properties panel, camera preview + BEV/label layer, undo/redo, JSON
// import/export, current-sample export and batch generation via /api.
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import type {
  CameraCalibration, Lane, LaneLine, SceneArrow, SceneObject, SceneV1,
} from "../shared/types";
import { CLASS_NAMES } from "../shared/types";
import { TEMPLATE_IDS } from "../shared/templates";
import { intrinsicsFromFovV, KFromIntrinsics, defaultCameraTCE } from "../shared/geometry";
import { ThreeViewport, type SelectionKind } from "./three_view";
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

function Btn({ children, onClick, active, title }: {
  children: React.ReactNode; onClick: () => void; active?: boolean; title?: string;
}) {
  return (
    <button title={title} onClick={onClick}
      style={{
        padding: "4px 8px", cursor: "pointer", fontSize: 12,
        background: active ? "#3a6ea5" : "#2a2f36", color: "#e8e8e8",
        border: "1px solid #444", borderRadius: 4,
      }}>{children}</button>
  );
}

/* ---------------------------------- app ----------------------------------- */

type Tool = "select" | "add_box" | "add_cylinder" | "add_car" | "add_arrow"
  | "draw_line" | "add_lane" | "add_camera";

function App() {
  const [scene, setScene] = useState<SceneV1>(() => emptyScene());
  const [sel, setSel] = useState<SelectionKind>(null);
  const [tool, setTool] = useState<Tool>("select");
  const [gizmoMode, setGizmoMode] = useState<"translate" | "rotate" | "scale">("translate");
  const [status, setStatus] = useState("ready");
  const [previewCam, setPreviewCam] = useState(0);
  const [showBevLayer, setShowBevLayer] = useState(true);
  const [topView, setTopView] = useState(false);
  const [lineDraft, setLineDraft] = useState<number[][] | null>(null);
  const [laneDraft, setLaneDraft] = useState<number[][] | null>(null);
  const [arrowClass, setArrowClass] = useState(0);

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
          if (s.kind === "object") {
            const o = next.objects.find((o) => o.object_id === (s as { id: string }).id);
            if (o) {
              o.center_ego = [t.pos[0], t.pos[1], t.pos[2]];
              o.yaw_rad = t.rotZ;
              if (t.scale !== 1 && o.size_abc) {
                // gizmo uniform scale multiplies per-axis size
                o.size_abc = o.size_abc.map((v) => Math.max(0.05, v * t.scale)) as [number, number, number];
              }
            }
          } else if (s.kind === "arrow") {
            const a = next.arrows.find((a) => a.instance_id === (s as { instanceId: number }).instanceId);
            if (a) {
              a.center_ego = [t.pos[0], t.pos[1], 0];
              a.yaw_rad = t.rotZ;
              if (t.scale !== 1) a.scale = Math.max(0.1, a.scale * t.scale);
            }
          } else if (s.kind === "lane_line_point") {
            const line = next.lane_lines.find((l) => l.line_id === (s as { lineId: string }).lineId);
            if (line) {
              const i = (s as { index: number }).index;
              line.points_ego[i] = [t.pos[0], t.pos[1], 0];
            }
          }
          dirtyRef.current = true;
          return next;
        });
      },
      onGroundClick: (x, y) => {
        const t: Tool = toolRef.current;
        if (t === "add_box" || t === "add_cylinder" || t === "add_car") {
          commit((prev) => {
            const next = JSON.parse(JSON.stringify(prev)) as SceneV1;
                    const size: [number, number, number] =
            t === "add_car" ? [4.0, 1.8, 1.5] : [1.2, 1.2, 1.5];
            const obj: SceneObject = {
              object_id: uid("obj"), shape: t === "add_box" ? "box" : t === "add_car" ? "car" : "cylinder",
              size_abc: size, center_ego: [x, y, 0], yaw_rad: 0, color: "#7a7f85",
            };
            next.objects.push(obj);
            return next;
          });
          setStatus(`added ${t} at ${x.toFixed(1)},${y.toFixed(1)}`);
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

  const selRef = useRef(sel); selRef.current = sel;
  const toolRef = useRef(tool); toolRef.current = tool;
  const arrowClassRef = useRef(arrowClass); arrowClassRef.current = arrowClass;

  useEffect(() => { vpRef.current?.setScene(scene); }, [scene]);
  useEffect(() => { vpRef.current?.setSelection(sel, scene); }, [sel, scene]);
  useEffect(() => { vpRef.current?.setGizmoMode(gizmoMode); }, [gizmoMode]);
  useEffect(() => { vpRef.current?.setPlaceMode(tool !== "select" && tool !== "add_camera"); }, [tool]);
  useEffect(() => { vpRef.current?.setTopView(topView); }, [topView]);

  /* ------------------------------ preview cam ------------------------------ */
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
      if (bev && bevCanvas.current) drawB64(bevCanvas.current, bev, true);
      setStatus("preview updated");
    } catch (e) {
      setStatus(`preview error: ${e}`);
    }
  }, [scene, previewCam]);

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
          <NumField label="pitch down (deg)" value={(-Math.asin(-cal.T_C_E[1][0]) * 180) / Math.PI} step={1}
            onChange={(v) => patchCamera(cal.camera_id, { pitchDeg: v })} />
          <NumField label="yaw (deg)" value={(Math.atan2(-cal.T_C_E[0][1], cal.T_C_E[0][0]) * 180) / Math.PI} step={1}
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
      const pitchDeg = patch.pitchDeg ?? (-Math.asin(-cal.T_C_E[1][0]) * 180) / Math.PI;
      const yawDeg = patch.yawDeg ?? (Math.atan2(-cal.T_C_E[0][1], cal.T_C_E[0][0]) * 180) / Math.PI;
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
    <div style={{ display: "grid", gridTemplateColumns: "190px 1fr 260px", height: "100vh", color: "#e8e8e8", fontFamily: "sans-serif", background: "#14171b" }}>
      {/* toolbar */}
      <div style={{ borderRight: "1px solid #2a2f36", padding: 8, overflowY: "auto" }}>
        <h3 style={{ margin: "4px 0" }}>arrow-toy</h3>
        <Row>
          <Btn onClick={undo} title="undo">↶</Btn>
          <Btn onClick={redo} title="redo">↷</Btn>
          <Btn onClick={() => commit(emptyScene())}>new</Btn>
        </Row>
        <Row>
          <Btn onClick={saveJson}>save</Btn>
          <Btn onClick={loadJson}>load</Btn>
        </Row>
        <hr style={{ borderColor: "#2a2f36" }} />
        <div style={{ fontSize: 11, color: "#9ab", marginBottom: 4 }}>tools</div>
        <Row><Btn active={tool === "select"} onClick={() => setTool("select")}>select</Btn></Row>
        <Row>
          <Btn active={tool === "add_box"} onClick={() => setTool("add_box")}>+box</Btn>
          <Btn active={tool === "add_cylinder"} onClick={() => setTool("add_cylinder")}>+cyl</Btn>
        </Row>
        <Row><Btn active={tool === "add_car"} onClick={() => setTool("add_car")}>+car</Btn></Row>
        <Row>
          <Btn active={tool === "add_arrow"} onClick={() => setTool("add_arrow")}>+arrow</Btn>
          <select value={arrowClass} onChange={(e) => setArrowClass(parseInt(e.target.value))}>
            {CLASS_NAMES.map((n: string, i: number) => <option key={n} value={i}>{n}</option>)}
          </select>
        </Row>
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
        </Row>
        <Row>
          <Btn active={topView} onClick={() => setTopView(!topView)}>top view</Btn>
          <Btn active={showBevLayer} onClick={() => setShowBevLayer(!showBevLayer)}>bev layer</Btn>
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
      <div style={{ position: "relative" }}>
        <div ref={viewDiv} style={{ width: "100%", height: "100%" }} />
        <div style={{ position: "absolute", top: 6, left: 8, fontSize: 12, color: "#9ab", pointerEvents: "none" }}>
          {status} {lineDraft?.length ? `| line pts: ${lineDraft.length}` : ""}
        </div>
      </div>

      {/* right panel */}
      <div style={{ borderLeft: "1px solid #2a2f36", padding: 8, overflowY: "auto" }}>
        <h4 style={{ margin: "2px 0 6px" }}>properties</h4>
        {selPanel}
        <h4 style={{ margin: "12px 0 6px" }}>training camera preview</h4>
        <select value={previewCam} onChange={(e) => setPreviewCam(parseInt(e.target.value))}>
          {scene.cameras.map((c, i) => <option key={c.camera_id} value={i}>{c.camera_id}</option>)}
        </select>{" "}
        <Btn onClick={refreshPreview}>refresh</Btn>
        <canvas ref={previewCanvas} style={{ width: "100%", border: "1px solid #333", marginTop: 6, imageRendering: "pixelated" }} />
        {showBevLayer && <>
          <div style={{ fontSize: 11, color: "#9ab", marginTop: 6 }}>BEV instance layer (row0=far, col0=left)</div>
          <canvas ref={bevCanvas} style={{ width: "100%", border: "1px solid #333", imageRendering: "pixelated" }} />
        </>}
        <h4 style={{ margin: "12px 0 6px" }}>batch generation</h4>
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
  // build T_C_E = [Rz(yaw) @ R0 | -R t]
  const s = Math.sin(pitchDown), c = Math.cos(pitchDown);
  const R0 = [[0, -1, 0], [-s, 0, -c], [c, 0, -s]];
  const sy = Math.sin(yaw), cy = Math.cos(yaw);
  const Rz = [[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]];
  const R = [0, 1, 2].map((i) =>
    [0, 1, 2].map((j) => Rz[i][0] * R0[0][j] + Rz[i][1] * R0[1][j] + Rz[i][2] * R0[2][j]));
  const t = [0, 1, 2].map((i) => -(R[i][0] * pos[0] + R[i][1] * pos[1] + R[i][2] * pos[2]));
  return [...R.map((r, i) => [...r, t[i]]), [0, 0, 0, 1]];
}

createRoot(document.getElementById("root")!).render(<App />);
