import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import type { CameraMode, DemoConfig, SimulationSnapshot, VisualizationSettings } from "../simulation/types";
import type { MujocoWorld } from "../simulation/mujocoWorld";
import { MujocoModelScene, mujocoToThree } from "./mujocoModelScene";

const CONTACT_MARKER_TTL = 0.45;
const TRAIL_MAX_POINTS = 180;
const CAMERA_DEBUG_ENABLED = false;

type ContactMarker = {
  mesh: THREE.Mesh;
  createdAt: number;
};

export class ThreeScene {
  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene: THREE.Scene;
  private readonly cameras: Record<Exclude<CameraMode, "four">, THREE.PerspectiveCamera | THREE.OrthographicCamera>;
  private readonly controls: OrbitControls;
  private readonly targetBand: THREE.Mesh;
  private readonly trailLine: THREE.Line;
  private readonly cameraDebug: HTMLPreElement | null = null;
  private readonly trailPositions = new Float32Array(TRAIL_MAX_POINTS * 3);
  private readonly markers: ContactMarker[] = [];
  private modelScene: MujocoModelScene | null = null;
  private width = 1;
  private height = 1;
  private trailPointCount = 0;
  private lastContactTime: number | null = null;
  private lastTrailResetSerial = -1;
  private lastMarkerResetSerial = -1;

  constructor(private readonly host: HTMLElement) {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0.15, 0.25, 0.35);
    this.scene.fog = new THREE.Fog(this.scene.background, 15, 25.5);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(1);
    this.renderer.setClearColor(0x263f59, 1);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.host.appendChild(this.renderer.domElement);
    if (CAMERA_DEBUG_ENABLED) {
      this.host.style.position = "relative";
      this.cameraDebug = createCameraDebugElement();
      this.host.appendChild(this.cameraDebug);
    }

    const target = mujocoToThree([0.45, 0, 0.72]);
    this.cameras = {
      free: perspectiveCamera(mujocoToThree([0.28, 1.887, 0.87]), target, 42),
      north: perspectiveCamera(mujocoToThree([0.45, -2.25, 1.0]), target),
      south: perspectiveCamera(mujocoToThree([0.45, 2.25, 1.0]), target),
      east: perspectiveCamera(mujocoToThree([2.2, 0, 1.0]), target),
      west: perspectiveCamera(mujocoToThree([-1.35, 0, 1.0]), target),
      top: orthographicCamera(mujocoToThree([0.45, 0, 3.0]), target)
    };

    this.controls = new OrbitControls(this.cameras.free, this.renderer.domElement);
    this.controls.target.copy(target);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.1;
    this.controls.panSpeed = 1.8;
    this.controls.zoomSpeed = 1.0;
    this.controls.update();

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x1b3146, 2.25));
    const key = new THREE.DirectionalLight(0xffffff, 1.55);
    key.position.copy(mujocoToThree([0, -3, 3]));
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.camera.near = 0.2;
    key.shadow.camera.far = 6;
    key.shadow.camera.left = -2.2;
    key.shadow.camera.right = 2.2;
    key.shadow.camera.top = 2.2;
    key.shadow.camera.bottom = -2.2;
    key.shadow.bias = -0.00035;
    this.scene.add(key.target);
    this.scene.add(key);

    const fill = new THREE.DirectionalLight(0xffffff, 1.15);
    fill.position.copy(mujocoToThree([-1.2, 1.4, 2.0]));
    this.scene.add(fill);

    const headlight = new THREE.DirectionalLight(0xffffff, 0.9);
    headlight.position.copy(mujocoToThree([1.4, -1.2, 2.0]));
    this.scene.add(headlight);

    this.targetBand = new THREE.Mesh(
      new THREE.BoxGeometry(1.6, 0.3, 1.6),
      new THREE.MeshBasicMaterial({
        color: 0x34d399,
        transparent: true,
        opacity: 0.15,
        depthWrite: false
      })
    );
    this.targetBand.visible = false;
    this.scene.add(this.targetBand);

    this.trailLine = new THREE.Line(
      createTrailGeometry(this.trailPositions),
      new THREE.LineBasicMaterial({ color: 0x7dd3fc, transparent: true, opacity: 0.86 })
    );
    this.trailLine.visible = false;
    this.scene.add(this.trailLine);

    this.resize();
  }

  loadWorld(world: MujocoWorld): void {
    const runtime = world.getRuntime();
    if (!runtime) {
      return;
    }

    this.modelScene?.dispose();
    this.modelScene = new MujocoModelScene(runtime);
    this.scene.add(this.modelScene.root);
  }

  dispose(): void {
    this.modelScene?.dispose();
    this.clearContactMarkers();
    this.targetBand.geometry.dispose();
    (this.targetBand.material as THREE.Material).dispose();
    this.trailLine.geometry.dispose();
    (this.trailLine.material as THREE.Material).dispose();
    this.controls.dispose();
    this.renderer.dispose();
    this.cameraDebug?.remove();
    this.host.removeChild(this.renderer.domElement);
  }

  resize(): void {
    const rect = this.host.getBoundingClientRect();
    this.width = Math.max(1, Math.floor(rect.width));
    this.height = Math.max(1, Math.floor(rect.height));
    this.renderer.setSize(this.width, this.height, false);

    Object.values(this.cameras).forEach((camera) => {
      if (camera instanceof THREE.PerspectiveCamera) {
        camera.aspect = this.width / this.height;
        camera.updateProjectionMatrix();
      } else {
        camera.left = -1.15;
        camera.right = 1.15;
        camera.top = 1.15 / (this.width / this.height);
        camera.bottom = -1.15 / (this.width / this.height);
        camera.updateProjectionMatrix();
      }
    });
  }

  update(
    snapshot: SimulationSnapshot,
    _world: MujocoWorld,
    visualization: VisualizationSettings,
    config: DemoConfig
  ): void {
    this.modelScene?.update();
    this.updateTrail(snapshot, visualization.trail);
    this.updateTargetBand(snapshot, visualization.targetBand, config);
    this.updateContactMarkers(snapshot, visualization.contactMarker);
  }

  render(mode: CameraMode): void {
    this.renderer.setScissorTest(false);
    this.renderer.clear();

    if (mode !== "four") {
      this.controls.enabled = mode === "free";
      if (mode === "free") {
        this.controls.update();
      }
      if (CAMERA_DEBUG_ENABLED) {
        this.updateCameraDebug(mode);
      }

      this.renderer.setViewport(0, 0, this.width, this.height);
      this.renderer.render(this.scene, this.cameras[mode]);
      return;
    }

    this.controls.enabled = false;
    if (CAMERA_DEBUG_ENABLED) {
      this.updateCameraDebug(mode);
    }
    this.renderer.setScissorTest(true);
    const views: Array<[Exclude<CameraMode, "free" | "top" | "four">, number, number]> = [
      ["north", 0, 1],
      ["south", 1, 1],
      ["east", 0, 0],
      ["west", 1, 0]
    ];
    const viewWidth = Math.floor(this.width / 2);
    const viewHeight = Math.floor(this.height / 2);

    for (const [cameraName, column, row] of views) {
      const x = column * viewWidth;
      const y = row * viewHeight;
      this.renderer.setViewport(x, y, viewWidth, viewHeight);
      this.renderer.setScissor(x, y, viewWidth, viewHeight);
      this.renderer.render(this.scene, this.cameras[cameraName]);
    }

    this.renderer.setScissorTest(false);
  }

  private updateCameraDebug(mode: CameraMode): void {
    if (!this.cameraDebug) {
      return;
    }

    if (mode !== "free") {
      this.cameraDebug.style.display = "none";
      return;
    }

    const camera = this.cameras.free as THREE.PerspectiveCamera;
    const cameraPosition = threeToMujoco(camera.position);
    const targetPosition = threeToMujoco(this.controls.target);
    this.cameraDebug.style.display = "block";
    this.cameraDebug.textContent = [
      "TEMP FREE VIEW CAMERA",
      `camera: [${formatVec3(cameraPosition)}]`,
      `target:  [${formatVec3(targetPosition)}]`,
      `fov:     ${formatNumber(camera.fov)}`,
      "",
      `free: perspectiveCamera(mujocoToThree([${formatVec3(cameraPosition)}]), target, ${formatNumber(camera.fov)})`
    ].join("\n");
  }

  private updateTrail(snapshot: SimulationSnapshot, visible: boolean): void {
    this.trailLine.visible = visible;
    if (!visible) {
      if (this.trailPointCount > 0 || this.lastTrailResetSerial !== snapshot.resetSerial) {
        this.resetTrailGeometry(snapshot.resetSerial);
      }
      return;
    }

    if (this.lastTrailResetSerial !== snapshot.resetSerial) {
      this.resetTrailGeometry(snapshot.resetSerial);
    }

    if (this.trailPointCount >= TRAIL_MAX_POINTS) {
      this.trailPositions.copyWithin(0, 3);
      this.trailPointCount = TRAIL_MAX_POINTS - 1;
    }

    const point = mujocoToThree(snapshot.ball.position);
    const offset = this.trailPointCount * 3;
    this.trailPositions[offset] = point.x;
    this.trailPositions[offset + 1] = point.y;
    this.trailPositions[offset + 2] = point.z;
    this.trailPointCount += 1;
    this.updateTrailGeometry();
  }

  private resetTrailGeometry(resetSerial: number): void {
    this.trailPointCount = 0;
    this.lastTrailResetSerial = resetSerial;
    this.updateTrailGeometry();
  }

  private updateTrailGeometry(): void {
    const position = this.trailLine.geometry.getAttribute("position") as THREE.BufferAttribute;
    position.needsUpdate = true;
    this.trailLine.geometry.setDrawRange(0, this.trailPointCount);
  }

  private updateTargetBand(snapshot: SimulationSnapshot, visible: boolean, config: DemoConfig): void {
    this.targetBand.visible = visible;
    if (!visible) {
      return;
    }

    const height = Math.max(0.01, config.heightTolerance * 2);
    this.targetBand.position.copy(
      mujocoToThree([
        snapshot.racketPosition[0],
        snapshot.racketPosition[1],
        snapshot.racketPosition[2] + config.targetHeight
      ])
    );
    this.targetBand.scale.set(1, height / 0.3, 1);
  }

  private updateContactMarkers(snapshot: SimulationSnapshot, visible: boolean): void {
    if (this.lastMarkerResetSerial !== snapshot.resetSerial) {
      this.clearContactMarkers();
      this.lastContactTime = null;
      this.lastMarkerResetSerial = snapshot.resetSerial;
    }

    if (visible && snapshot.lastContact && snapshot.lastContact.time !== this.lastContactTime) {
      const marker = new THREE.Mesh(
        new THREE.SphereGeometry(0.025, 18, 10),
        new THREE.MeshBasicMaterial({ color: 0xfacc15, transparent: true, opacity: 0.95 })
      );
      marker.position.copy(mujocoToThree(snapshot.lastContact.position));
      this.scene.add(marker);
      this.markers.push({ mesh: marker, createdAt: snapshot.time });
      this.lastContactTime = snapshot.lastContact.time;
    }

    for (let index = this.markers.length - 1; index >= 0; index -= 1) {
      const marker = this.markers[index];
      const age = snapshot.time - marker.createdAt;
      const material = marker.mesh.material as THREE.MeshBasicMaterial;
      material.opacity = visible ? Math.max(0, 1 - age / CONTACT_MARKER_TTL) : 0;
      marker.mesh.visible = visible;
      if (age > CONTACT_MARKER_TTL) {
        this.scene.remove(marker.mesh);
        marker.mesh.geometry.dispose();
        material.dispose();
        this.markers.splice(index, 1);
      }
    }
  }

  private clearContactMarkers(): void {
    for (const marker of this.markers) {
      this.scene.remove(marker.mesh);
      marker.mesh.geometry.dispose();
      (marker.mesh.material as THREE.Material).dispose();
    }
    this.markers.length = 0;
  }
}

function createTrailGeometry(positions: Float32Array): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry();
  const attribute = new THREE.BufferAttribute(positions, 3);
  attribute.setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute("position", attribute);
  geometry.setDrawRange(0, 0);
  return geometry;
}

function createCameraDebugElement(): HTMLPreElement {
  const element = document.createElement("pre");
  element.style.position = "absolute";
  element.style.left = "16px";
  element.style.bottom = "16px";
  element.style.zIndex = "6";
  element.style.margin = "0";
  element.style.padding = "10px 12px";
  element.style.border = "1px solid rgba(255, 255, 255, 0.16)";
  element.style.borderRadius = "7px";
  element.style.background = "rgba(17, 17, 17, 0.74)";
  element.style.color = "#f2f2f2";
  element.style.font = "12px/1.45 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace";
  element.style.pointerEvents = "none";
  element.style.whiteSpace = "pre-wrap";
  element.style.backdropFilter = "blur(10px)";
  return element;
}

function perspectiveCamera(position: THREE.Vector3, target: THREE.Vector3, fov = 45): THREE.PerspectiveCamera {
  const camera = new THREE.PerspectiveCamera(fov, 1, 0.001, 100);
  camera.position.copy(position);
  camera.lookAt(target);
  return camera;
}

function orthographicCamera(position: THREE.Vector3, target: THREE.Vector3): THREE.OrthographicCamera {
  const camera = new THREE.OrthographicCamera(-1.15, 1.15, 1.15, -1.15, 0.001, 100);
  camera.position.copy(position);
  camera.up.set(0, 0, -1);
  camera.lookAt(target);
  return camera;
}

function threeToMujoco(position: THREE.Vector3): [number, number, number] {
  return [position.x, -position.z, position.y];
}

function formatVec3(values: [number, number, number]): string {
  return values.map(formatNumber).join(", ");
}

function formatNumber(value: number): string {
  return value.toFixed(3).replace(/\.?0+$/, "");
}
