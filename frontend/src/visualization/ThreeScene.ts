import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import type { CameraMode, DemoConfig, SimulationSnapshot, VisualizationSettings, Vec3 } from "../simulation/types";
import type { MujocoWorld } from "../simulation/mujocoWorld";
import { MujocoModelScene, mujocoToThree } from "./mujocoModelScene";

const CONTACT_MARKER_TTL = 0.45;

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
  private readonly trailPoints: THREE.Vector3[] = [];
  private readonly markers: ContactMarker[] = [];
  private modelScene: MujocoModelScene | null = null;
  private width = 1;
  private height = 1;
  private lastContactTime: number | null = null;

  constructor(private readonly host: HTMLElement) {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0.15, 0.25, 0.35);
    this.scene.fog = new THREE.Fog(this.scene.background, 15, 25.5);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(1);
    this.renderer.setClearColor(0x263f59, 1);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    this.host.appendChild(this.renderer.domElement);

    const target = mujocoToThree([0.45, 0, 0.72]);
    this.cameras = {
      free: perspectiveCamera(mujocoToThree([1.8, -1.7, 1.25]), target),
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

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x1b3146, 1.15));
    const key = new THREE.SpotLight(0xffffff, 12.5, 12, 1.1, 0.45, 1.0);
    key.position.copy(mujocoToThree([0, -3, 3]));
    key.target.position.copy(target);
    key.castShadow = true;
    key.shadow.mapSize.width = 1024;
    key.shadow.mapSize.height = 1024;
    key.shadow.camera.near = 0.1;
    key.shadow.camera.far = 10;
    this.scene.add(key.target);
    this.scene.add(key);

    const headlight = new THREE.DirectionalLight(0xffffff, 1.4);
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
      new THREE.BufferGeometry(),
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
    this.controls.dispose();
    this.renderer.dispose();
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
    this.updateTrail(snapshot.ball.position, visualization.trail);
    this.updateTargetBand(visualization.targetBand, config);
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

      this.renderer.setViewport(0, 0, this.width, this.height);
      this.renderer.render(this.scene, this.cameras[mode]);
      return;
    }

    this.controls.enabled = false;
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

  private updateTrail(ballPosition: Vec3, visible: boolean): void {
    this.trailLine.visible = visible;
    if (!visible) {
      this.trailPoints.length = 0;
      this.replaceTrailGeometry([]);
      return;
    }

    this.trailPoints.push(mujocoToThree(ballPosition));
    if (this.trailPoints.length > 180) {
      this.trailPoints.shift();
    }
    this.replaceTrailGeometry(this.trailPoints);
  }

  private replaceTrailGeometry(points: THREE.Vector3[]): void {
    const previous = this.trailLine.geometry;
    this.trailLine.geometry = new THREE.BufferGeometry().setFromPoints(points);
    previous.dispose();
  }

  private updateTargetBand(visible: boolean, config: DemoConfig): void {
    this.targetBand.visible = visible;
    if (!visible) {
      return;
    }

    const height = Math.max(0.01, config.heightTolerance * 2);
    this.targetBand.position.copy(mujocoToThree([0.45, 0, config.targetHeight]));
    this.targetBand.scale.set(1, height / 0.3, 1);
  }

  private updateContactMarkers(snapshot: SimulationSnapshot, visible: boolean): void {
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
}

function perspectiveCamera(position: THREE.Vector3, target: THREE.Vector3): THREE.PerspectiveCamera {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.001, 100);
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
