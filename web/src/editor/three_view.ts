// 3D editor viewport: OrbitControls + TransformControls + ground picking.
// Renders from SceneV1 through the shared builder so the editor can never
// diverge from offline rendering (design doc section 2/5).
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";
import type { SceneV1 } from "../shared/types";
import { buildScene, type BuiltScene } from "../shared/scene_builder";

export type SelectionKind =
  | { kind: "object"; id: string }
  | { kind: "arrow"; instanceId: number }
  | { kind: "lane_line_point"; lineId: string; index: number }
  | { kind: "camera"; cameraId: string }
  | null;

export interface ViewportCallbacks {
  onSelect: (sel: SelectionKind) => void;
  onTransform: () => void; // gizmo moved the selection; caller reads object
  onGroundClick?: (x: number, y: number) => void;
}

const CAM_PREVIEW_COLORS = [0x4fc3f7, 0xffb74d, 0x81c784, 0xf06292];

export class ThreeViewport {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  gizmo: TransformControls;
  private built: BuiltScene | null = null;
  private frustums = new THREE.Group();
  private lineHandles = new THREE.Group();
  private sel: SelectionKind = null;
  private raycaster = new THREE.Raycaster();
  private groundPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
  private cb: ViewportCallbacks;
  private dom: HTMLElement;
  private placeMode: boolean = false;
  private downPos = { x: 0, y: 0 };

  constructor(dom: HTMLElement, cb: ViewportCallbacks) {
    this.dom = dom;
    this.cb = cb;
    THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
    THREE.ColorManagement.enabled = false;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(dom.clientWidth, dom.clientHeight);
    this.renderer.toneMapping = THREE.NoToneMapping;
    dom.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color("#181c20");
    const hemi = new THREE.HemisphereLight(0xffffff, 0x404040, 0.9);
    this.scene.add(hemi);
    const dir = new THREE.DirectionalLight(0xffffff, 0.9);
    dir.position.set(-6, 8, 12);
    this.scene.add(dir);
    // BEV range reference box
    const bevBox = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.PlaneGeometry(16, 8)),
      new THREE.LineBasicMaterial({ color: 0x66aaff })
    );
    bevBox.position.set(10, 0, 0.02);
    bevBox.name = "bev_range";
    this.scene.add(bevBox);

    this.camera = new THREE.PerspectiveCamera(
      50, dom.clientWidth / dom.clientHeight, 0.1, 500
    );
    this.camera.position.set(-12, -12, 14);
    this.camera.lookAt(10, 0, 0);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(10, 0, 0);

    this.gizmo = new TransformControls(this.camera, this.renderer.domElement);
    this.gizmo.setSize(0.7);
    const gizmoHelper = this.gizmo.getHelper?.() ?? (this.gizmo as unknown as THREE.Object3D);
    this.scene.add(gizmoHelper);
    this.gizmo.addEventListener("dragging-changed", (e) => {
      this.controls.enabled = !(e as unknown as { value: boolean }).value;
    });
    this.gizmo.addEventListener("objectChange", () => this.cb.onTransform());

    this.renderer.domElement.addEventListener("pointerdown", (e) => {
      this.downPos = { x: e.clientX, y: e.clientY };
    });
    this.renderer.domElement.addEventListener("pointerup", (e) => this.handleClick(e));

    window.addEventListener("resize", () => this.resize());
    this.animate();
  }

  setPlaceMode(on: boolean) {
    this.placeMode = on;
    this.renderer.domElement.style.cursor = on ? "crosshair" : "default";
  }

  resize() {
    const w = this.dom.clientWidth, h = this.dom.clientHeight;
    if (w === 0 || h === 0) return;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  private ndc(e: PointerEvent | MouseEvent) {
    const r = this.renderer.domElement.getBoundingClientRect();
    return new THREE.Vector2(
      ((e.clientX - r.left) / r.width) * 2 - 1,
      -((e.clientY - r.top) / r.height) * 2 + 1
    );
  }

  private handleClick(e: PointerEvent) {
    const moved = Math.hypot(e.clientX - this.downPos.x, e.clientY - this.downPos.y);
    if (moved > 4) return; // drag, not click
    this.raycaster.setFromCamera(this.ndc(e), this.camera);
    const hit = new THREE.Vector3();
    const hitsGround = this.raycaster.ray.intersectPlane(this.groundPlane, hit);
    if (this.gizmo.dragging) return;
    if (this.placeMode) {
      if (hitsGround) this.cb.onGroundClick?.(hit.x, hit.y);
      return;
    }
    // pick line handles first (small spheres)
    const handleHits = this.raycaster.intersectObjects(this.lineHandles.children, false);
    if (handleHits.length) {
      const ud = handleHits[0].object.userData as { lineId: string; index: number };
      this.cb.onSelect({ kind: "lane_line_point", lineId: ud.lineId, index: ud.index });
      return;
    }
    if (this.built) {
      const arrowHits = this.raycaster.intersectObjects(this.built.arrowMeshes, false);
      if (arrowHits.length) {
        const id = (arrowHits[0].object.userData as { instance_id: number }).instance_id;
        this.cb.onSelect({ kind: "arrow", instanceId: id });
        return;
      }
      const objHits = this.raycaster.intersectObjects(
        this.built.objectGroup.children, true);
      if (objHits.length) {
        let o: THREE.Object3D | null = objHits[0].object;
        while (o && o.userData.kind !== "object") o = o.parent;
        if (o) {
          this.cb.onSelect({ kind: "object", id: o.userData.object_id as string });
          return;
        }
      }
      const frustumHits = this.raycaster.intersectObjects(this.frustums.children, true);
      if (frustumHits.length) {
        let o: THREE.Object3D | null = frustumHits[0].object;
        while (o && !o.userData.camera_id) o = o.parent;
        if (o) this.cb.onSelect({ kind: "camera", cameraId: o.userData.camera_id as string });
        return;
      }
    }
    this.cb.onSelect(null);
  }

  /** Rebuild view from scene JSON. */
  setScene(scene: SceneV1) {
    if (this.built) {
      this.scene.remove(this.built.root);
      this.built.root.traverse((o) => {
        const m = o as THREE.Mesh;
        m.geometry?.dispose?.();
      });
    }
    this.built = buildScene(scene);
    this.scene.add(this.built.root);
    // ground visible in editor even at default dark bg
    this.updateFrustums(scene);
    this.updateLineHandles(scene);
    this.rebindGizmo(scene);
  }

  updateFrustums(scene: SceneV1) {
    this.scene.remove(this.frustums);
    this.frustums.traverse((o) => {
      const m = o as THREE.Mesh;
      m.geometry?.dispose?.();
    });
    this.frustums = new THREE.Group();
    scene.cameras.forEach((cal, i) => {
      const color = CAM_PREVIEW_COLORS[i % CAM_PREVIEW_COLORS.length];
      const g = new THREE.Group();
      // simple pyramid from camera position along its viewing direction
      const inv = new THREE.Matrix4().set(
        cal.T_C_E[0][0], cal.T_C_E[0][1], cal.T_C_E[0][2], cal.T_C_E[0][3],
        cal.T_C_E[1][0], cal.T_C_E[1][1], cal.T_C_E[1][2], cal.T_C_E[1][3],
        cal.T_C_E[2][0], cal.T_C_E[2][1], cal.T_C_E[2][2], cal.T_C_E[2][3],
        cal.T_C_E[3][0], cal.T_C_E[3][1], cal.T_C_E[3][2], cal.T_C_E[3][3]
      ).invert();
      const camPos = new THREE.Vector3(0, 0, 0).applyMatrix4(inv);
      // forward in E: R^T @ (0,0,1)
      const R = inv.elements;
      void R;
      const fwd = new THREE.Vector3(
        inv.elements[8], inv.elements[9], inv.elements[10]
      ).sub(camPos).normalize();
      // camera position from T_C_E^{-1} translation
      const zC = new THREE.Vector3(0, 0, 1).normalize();
      const rot = new THREE.Matrix4().extractRotation(inv);
      const fwdE = zC.clone().applyMatrix4(rot).normalize();
      void fwd;
      const upE = new THREE.Vector3(0, -1, 0).applyMatrix4(rot).normalize();
      const rightE = new THREE.Vector3(1, 0, 0).applyMatrix4(rot).normalize();
      const fovV = 2 * Math.atan2(cal.height / 2, cal.K[1][1]);
      const fovH = 2 * Math.atan2(cal.width / 2, cal.K[0][0]);
      const L = 4.0;
      const cFar = camPos.clone().addScaledVector(fwdE, L);
      const hHalf = Math.tan(fovH / 2) * L;
      const vHalf = Math.tan(fovV / 2) * L;
      const corners = [
        cFar.clone().addScaledVector(rightE, hHalf).addScaledVector(upE, vHalf),
        cFar.clone().addScaledVector(rightE, -hHalf).addScaledVector(upE, vHalf),
        cFar.clone().addScaledVector(rightE, -hHalf).addScaledVector(upE, -vHalf),
        cFar.clone().addScaledVector(rightE, hHalf).addScaledVector(upE, -vHalf),
      ];
      const pts: number[] = [];
      for (let k = 0; k < 4; k++) {
        pts.push(camPos.x, camPos.y, camPos.z,
                 corners[k].x, corners[k].y, corners[k].z);
        const k2 = (k + 1) % 4;
        pts.push(corners[k].x, corners[k].y, corners[k].z,
                 corners[k2].x, corners[k2].y, corners[k2].z);
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position",
        new THREE.Float32BufferAttribute(pts, 3));
      const lines = new THREE.LineSegments(geo,
        new THREE.LineBasicMaterial({ color }));
      g.add(lines);
      const marker = new THREE.Mesh(
        new THREE.SphereGeometry(0.12, 12, 8),
        new THREE.MeshBasicMaterial({ color })
      );
      marker.position.copy(camPos);
      g.add(marker);
      g.userData = { camera_id: cal.camera_id };
      this.frustums.add(g);
    });
    this.scene.add(this.frustums);
  }

  updateLineHandles(scene: SceneV1) {
    this.scene.remove(this.lineHandles);
    this.lineHandles.traverse((o) => {
      const m = o as THREE.Mesh;
      m.geometry?.dispose?.();
    });
    this.lineHandles = new THREE.Group();
    scene.lane_lines.forEach((line) => {
      line.points_ego.forEach((p, idx) => {
        const s = new THREE.Mesh(
          new THREE.SphereGeometry(0.09, 10, 8),
          new THREE.MeshBasicMaterial({ color: 0xffd54f })
        );
        s.position.set(p[0], p[1], p[2] + 0.02);
        s.userData = { lineId: line.line_id, index: idx };
        this.lineHandles.add(s);
      });
    });
    this.scene.add(this.lineHandles);
  }

  /** Attach gizmo to a temporary proxy object mirroring the selection. */
  private proxy: THREE.Object3D | null = null;
  rebindGizmo(scene: SceneV1) {
    const gizmoHelper = this.gizmo.getHelper?.() ?? (this.gizmo as unknown as THREE.Object3D);
    if (this.proxy) {
      this.scene.remove(this.proxy);
      this.proxy = null;
    }
    this.gizmo.detach();
    if (!this.sel) return;
    if (this.sel.kind === "object") {
      const obj = scene.objects.find((o) => o.object_id === this.sel!.kind === true ? false : o.object_id === (this.sel as { id: string }).id);
      if (!obj) return;
      const p = new THREE.Object3D();
      p.position.set(...obj.center_ego);
      p.rotation.z = obj.yaw_rad;
      this.attachProxy(p, gizmoHelper);
    } else if (this.sel.kind === "arrow") {
      const a = scene.arrows.find((x) => x.instance_id === (this.sel as { instanceId: number }).instanceId);
      if (!a) return;
      const p = new THREE.Object3D();
      p.position.set(a.center_ego[0], a.center_ego[1], 0);
      p.rotation.z = a.yaw_rad;
      this.attachProxy(p, gizmoHelper);
    } else if (this.sel.kind === "lane_line_point") {
      const line = scene.lane_lines.find(
        (l) => l.line_id === (this.sel as { lineId: string }).lineId);
      if (!line) return;
      const idx = (this.sel as { index: number }).index;
      const p = new THREE.Object3D();
      const pt = line.points_ego[idx];
      p.position.set(pt[0], pt[1], pt[2] + 0.02);
      p.userData = { lockZ: true };
      this.attachProxy(p, gizmoHelper);
    }
    // camera selection: no gizmo (edit numerically / via pose helper)
  }

  private attachProxy(p: THREE.Object3D, gizmoHelper: THREE.Object3D) {
    this.proxy = p;
    this.scene.add(p);
    this.gizmo.attach(p);
    void gizmoHelper;
  }

  setSelection(sel: SelectionKind, scene: SceneV1) {
    this.sel = sel;
    this.rebindGizmo(scene);
  }

  setGizmoMode(mode: "translate" | "rotate" | "scale") {
    this.gizmo.setMode(mode);
  }

  getProxyTransform(): { pos: [number, number, number]; rotZ: number; scale: number } | null {
    if (!this.proxy) return null;
    return {
      pos: [this.proxy.position.x, this.proxy.position.y, this.proxy.position.z],
      rotZ: this.proxy.rotation.z,
      scale: this.proxy.scale.x,
    };
  }

  setTopView(on: boolean) {
    if (on) {
      this.camera.position.set(10, 0.01, 30);
      this.camera.lookAt(10, 0, 0);
      this.controls.target.set(10, 0, 0);
    } else {
      this.camera.position.set(-12, -12, 14);
      this.camera.lookAt(10, 0, 0);
      this.controls.target.set(10, 0, 0);
    }
  }

  private animate = () => {
    requestAnimationFrame(this.animate);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };
}
