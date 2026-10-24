// window.toyRenderer.renderSample — deterministic offline rendering API used by
// the Playwright batch worker and the editor's export-current-sample feature.
// Design doc 5.4 with one documented deviation: passes return raw RGBA (base64)
// instead of PNG; the Node worker encodes exact PNGs with pngjs.
import * as THREE from "three";
import type { CameraCalibration, SceneV1 } from "../shared/types";
import { buildScene, type BuiltScene } from "../shared/scene_builder";
import { setPerspectiveFromCalibration } from "../shared/camera_utils";

export interface RgbaImage {
  width: number;
  height: number;
  b64: string; // raw RGBA bytes, top-down rows, no PNG container
}

export type PassName = "rgb" | "visible_instance" | "amodal_instance" | "bev_instance";

export interface RenderRequest {
  scene: SceneV1;
  cameraIds: string[];
  passes: PassName[];
  bevRange?: { x_min: number; x_max: number; y_min: number; y_max: number };
  bevMaskSize?: [number, number]; // [rows, cols] = [H, W]; default [320,160]
}

export interface RenderResult {
  cameras: Record<string, {
    rgbRgba?: RgbaImage;
    visibleInstanceRgba?: RgbaImage;
    amodalPerArrowRgba?: Record<string, RgbaImage>;
    calibration: CameraCalibration;
  }>;
  bevInstanceRgba?: RgbaImage;
}

function b64FromBytes(bytes: Uint8Array): string {
  let s = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    s += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(s);
}

function idColor(id: number): THREE.Color {
  const r = id & 255;
  const g = (id >> 8) & 255;
  const b = (id >> 16) & 255;
  return new THREE.Color(r / 255, g / 255, b / 255);
}

interface MatSwap { mesh: THREE.Mesh; prev: THREE.Material }

export class ToyRenderer {
  private renderer: THREE.WebGLRenderer;
  private target: THREE.WebGLRenderTarget;
  private scene3: THREE.Scene | null = null;
  private built: BuiltScene | null = null;
  private activeCamera: THREE.Camera | null = null;

  constructor() {
    THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
    THREE.ColorManagement.enabled = false;
    this.renderer = new THREE.WebGLRenderer({
      antialias: false,
      alpha: false,
      preserveDrawingBuffer: false,
      powerPreference: "high-performance",
    });
    this.renderer.setPixelRatio(1);
    this.renderer.autoClear = true;
    this.renderer.toneMapping = THREE.NoToneMapping;
    this.target = new THREE.WebGLRenderTarget(4, 4, {
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      depthBuffer: true,
      stencilBuffer: false,
      generateMipmaps: false,
    });
  }

  dispose(): void {
    this.disposeCurrentScene();
    this.target.dispose();
    this.renderer.dispose();
    this.renderer.forceContextLoss();
  }

  private disposeCurrentScene(): void {
    if (!this.scene3) return;
    this.scene3.traverse((obj) => {
      const mesh = obj as THREE.Mesh;
      if (mesh.geometry) mesh.geometry.dispose();
      const mat = mesh.material as THREE.Material | THREE.Material[] | undefined;
      if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
      else if (mat) mat.dispose();
    });
    this.scene3 = null;
    this.built = null;
  }

  private ensureScene(scene: SceneV1): BuiltScene {
    this.disposeCurrentScene();
    const s = new THREE.Scene();
    s.background = new THREE.Color(scene.render?.background ?? "#101418");
    const built = buildScene(scene);
    s.add(built.root);
    this.scene3 = s;
    this.built = built;
    return built;
  }

  private renderToTarget(width: number, height: number,
                         target?: THREE.WebGLRenderTarget): Uint8Array {
    const rt = target ?? this.target;
    if (rt.width !== width || rt.height !== height) {
      rt.setSize(width, height);
    }
    this.renderer.setRenderTarget(rt);
    this.renderer.clear();
    this.renderer.render(this.scene3!, this.activeCamera!);
    const buf = new Uint8Array(width * height * 4);
    this.renderer.readRenderTargetPixels(rt, 0, 0, width, height, buf);
    this.renderer.setRenderTarget(null);
    // GL readback is bottom-up; flip rows to top-down exactly once.
    const row = width * 4;
    const flipped = new Uint8Array(buf.length);
    for (let y = 0; y < height; y++) {
      flipped.set(
        buf.subarray((height - 1 - y) * row, (height - 1 - y) * row + row),
        y * row
      );
    }
    return flipped;
  }

  private cameraFor(cal: CameraCalibration): THREE.PerspectiveCamera {
    const cam = new THREE.PerspectiveCamera();
    setPerspectiveFromCalibration(cam, cal);
    return cam;
  }

  private setIdBackground(active: boolean): THREE.Color | null {
    if (!this.scene3) return null;
    const prev = this.scene3.background;
    this.scene3.background = active ? new THREE.Color(0x000000) : prev;
    return active ? (prev as THREE.Color | null) : null;
  }

  private idPassGroundSwap(): MatSwap | null {
    const ground = this.built!.root.getObjectByName("ground") as THREE.Mesh | null;
    if (!ground) return null;
    const prev = ground.material as THREE.Material;
    ground.material = new THREE.MeshBasicMaterial({ color: 0x000000 });
    return { mesh: ground, prev };
  }

  private setLightsVisible(v: boolean): void {
    this.built!.root.traverse((o) => {
      const l = o as THREE.Light;
      if (l.isLight) l.visible = v;
    });
  }

  renderSample(req: RenderRequest): RenderResult {
    const built = this.ensureScene(req.scene);
    const want = new Set(req.passes);
    const result: RenderResult = { cameras: {} };

    for (const cal of req.scene.cameras) {
      if (req.cameraIds.length && !req.cameraIds.includes(cal.camera_id)) continue;
      const camResult: RenderResult["cameras"][string] = { calibration: cal };
      this.activeCamera = this.cameraFor(cal);

      if (want.has("rgb")) {
        const bytes = this.renderToTarget(cal.width, cal.height);
        camResult.rgbRgba = {
          width: cal.width, height: cal.height, b64: b64FromBytes(bytes),
        };
      }

      if (want.has("amodal_instance")) {
        // Per arrow: hide occluders (objects + other arrows + markings),
        // render this arrow alone with its ID color. Ground stays for depth.
        const perArrow: Record<string, RgbaImage> = {};
        const swap = this.idPassGroundSwap();
        const prevBg = this.setIdBackground(true);
        this.setLightsVisible(false);
        built.objectGroup.visible = false;
        built.markingGroup.visible = false;
        try {
          for (const mesh of built.arrowMeshes) {
            const id = (mesh.userData as { instance_id: number }).instance_id;
            const idMat = new THREE.MeshBasicMaterial({ color: idColor(id) });
            const prevMat = mesh.material as THREE.Material;
            const prevVis = mesh.visible;
            mesh.material = idMat;
            const hidden: THREE.Object3D[] = [];
            for (const other of built.arrowMeshes) {
              if (other !== mesh && other.visible) {
                other.visible = false;
                hidden.push(other);
              }
            }
            try {
              const bytes = this.renderToTarget(cal.width, cal.height);
              perArrow[String(id)] = {
                width: cal.width, height: cal.height, b64: b64FromBytes(bytes),
              };
            } finally {
              mesh.material = prevMat;
              mesh.visible = prevVis;
              for (const o of hidden) o.visible = true;
              idMat.dispose();
            }
          }
        } finally {
          if (swap) swap.mesh.material = swap.prev;
          if (prevBg) this.scene3!.background = prevBg;
          this.setLightsVisible(true);
          built.objectGroup.visible = true;
          built.markingGroup.visible = true;
        }
        camResult.amodalPerArrowRgba = perArrow;
      }

      if (want.has("visible_instance")) {
        // All arrows with ID colors; occluders black but writing depth.
        const restores: (() => void)[] = [];
        const swap = this.idPassGroundSwap();
        const prevBg = this.setIdBackground(true);
        this.setLightsVisible(false);
        built.markingGroup.visible = false;
        try {
          for (const mesh of built.arrowMeshes) {
            const id = (mesh.userData as { instance_id: number }).instance_id;
            const idMat = new THREE.MeshBasicMaterial({ color: idColor(id) });
            const prevMat = mesh.material as THREE.Material;
            mesh.material = idMat;
            restores.push(() => {
              mesh.material = prevMat;
              idMat.dispose();
            });
          }
          for (const obj of built.objectMeshes) {
            obj.traverse((child) => {
              const mesh = child as THREE.Mesh;
              if (mesh.isMesh) {
                const prev = mesh.material as THREE.Material;
                mesh.material = new THREE.MeshBasicMaterial({ color: 0x000000 });
                restores.push(() => {
                  mesh.material = prev;
                });
              }
            });
          }
          const bytes = this.renderToTarget(cal.width, cal.height);
          camResult.visibleInstanceRgba = {
            width: cal.width, height: cal.height, b64: b64FromBytes(bytes),
          };
        } finally {
          restores.forEach((r) => r());
          if (swap) swap.mesh.material = swap.prev;
          if (prevBg) this.scene3!.background = prevBg;
          this.setLightsVisible(true);
          built.markingGroup.visible = true;
        }
      }

      result.cameras[cal.camera_id] = camResult;
    }

    if (want.has("bev_instance")) {
      const range = req.bevRange ?? {
        x_min: 2.0, x_max: 18.0, y_min: -4.0, y_max: 4.0,
      };
      const [rows, cols] = req.bevMaskSize ?? [320, 160];
      const cx = (range.x_min + range.x_max) / 2;
      const cy = (range.y_min + range.y_max) / 2;
      const half_w = (range.y_max - range.y_min) / 2;
      const half_h = (range.x_max - range.x_min) / 2;
      // Top-down ortho over the BEV range, NO mirrored frustum (mirroring
      // flips winding and backface-culls everything). With up=+x, the
      // lookAt frame is x_cam=(0,-1,0), y_cam=(1,0,0): image left (NDC -1)
      // is x_cam=-half_w == y_max (left), image top is y_cam=+half_h == x_max
      // (far). left<right keeps faces front-facing.
      const cam = new THREE.OrthographicCamera(
        -half_w, half_w, half_h, -half_h, -1, 10
      );
      cam.position.set(cx, cy, 5);
      cam.up.set(1, 0, 0); // image-up = +x (far); must not parallel view dir
      cam.lookAt(cx, cy, 0);
      const swap = this.idPassGroundSwap();
      const prevBg = this.setIdBackground(true);
      this.setLightsVisible(false);
      built.objectGroup.visible = false;
      built.markingGroup.visible = false;
      const swaps: (() => void)[] = [];
      for (const mesh of built.arrowMeshes) {
        const id = (mesh.userData as { instance_id: number }).instance_id;
        const idMat = new THREE.MeshBasicMaterial({ color: idColor(id) });
        mesh.material = idIdMaterialSwap(mesh, idMat, swaps);
      }
      const bevTarget = new THREE.WebGLRenderTarget(cols, rows, {
        minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter,
        format: THREE.RGBAFormat, type: THREE.UnsignedByteType,
        depthBuffer: true, stencilBuffer: false, generateMipmaps: false,
      });
      try {
        this.activeCamera = cam;
        const bytes = this.renderToTarget(cols, rows, bevTarget);
        result.bevInstanceRgba = {
          width: cols, height: rows, b64: b64FromBytes(bytes),
        };
      } finally {
        swaps.forEach((r) => r());
        if (swap) swap.mesh.material = swap.prev;
        if (prevBg) this.scene3!.background = prevBg;
        this.setLightsVisible(true);
        built.objectGroup.visible = true;
        built.markingGroup.visible = true;
      }
    }

    return result;
  }
}

function idIdMaterialSwap(mesh: THREE.Mesh, idMat: THREE.Material,
                          swaps: (() => void)[]): THREE.Material {
  const prev = mesh.material as THREE.Material;
  mesh.material = idMat;
  swaps.push(() => {
    mesh.material = prev;
    idMat.dispose();
  });
  return idMat;
}

declare global {
  interface Window {
    toyRenderer: {
      renderSample: (req: RenderRequest) => RenderResult;
      dispose?: () => void;
    };
  }
}

export function installToyRenderer(): void {
  const r = new ToyRenderer();
  window.toyRenderer = {
    renderSample: (req) => r.renderSample(req),
    dispose: () => r.dispose(),
  };
  const el = document.getElementById("status");
  if (el) el.textContent = "toyRenderer ready";
}
