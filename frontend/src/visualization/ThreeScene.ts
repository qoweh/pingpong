import * as THREE from "three";

import type { CameraMode, DemoConfig, SimulationSnapshot, VisualizationSettings, Vec3 } from "../simulation/types";
import type { MujocoWorld } from "../simulation/mujocoWorld";

const BODY_CHAIN = ["link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand"];
const CONTACT_MARKER_TTL = 0.45;

type ContactMarker = {
  mesh: THREE.Mesh;
  createdAt: number;
};

export class ThreeScene {
  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene: THREE.Scene;
  private readonly cameras: Record<Exclude<CameraMode, "four">, THREE.PerspectiveCamera | THREE.OrthographicCamera>;
  private readonly ball: THREE.Mesh;
  private readonly racketHeadPivot: THREE.Object3D;
  private readonly racketHandlePivot: THREE.Object3D;
  private readonly racketFallback: THREE.Object3D;
  private readonly targetBand: THREE.Mesh;
  private readonly trailLine: THREE.Line;
  private readonly trailPoints: THREE.Vector3[] = [];
  private readonly bodySpheres: THREE.Mesh[] = [];
  private readonly bodyLinks: THREE.Mesh[] = [];
  private readonly markers: ContactMarker[] = [];
  private readonly floor: THREE.Mesh;
  private width = 1;
  private height = 1;
  private lastContactTime: number | null = null;

  constructor(private readonly host: HTMLElement) {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x111111);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x111111, 1);
    this.renderer.shadowMap.enabled = true;
    this.host.appendChild(this.renderer.domElement);

    this.cameras = {
      free: perspectiveCamera([1.55, -1.6, 1.2], [0.35, 0, 0.72]),
      north: perspectiveCamera([0.35, -2.2, 0.95], [0.35, 0, 0.75]),
      south: perspectiveCamera([0.35, 2.2, 0.95], [0.35, 0, 0.75]),
      east: perspectiveCamera([2.1, 0, 0.95], [0.35, 0, 0.75]),
      west: perspectiveCamera([-1.3, 0, 0.95], [0.35, 0, 0.75]),
      top: orthographicCamera([0.35, 0, 3.0], [0.35, 0, 0.7])
    };

    this.floor = new THREE.Mesh(
      new THREE.PlaneGeometry(3, 3),
      new THREE.MeshStandardMaterial({ color: 0x1d1d1d, roughness: 0.9, metalness: 0.05 })
    );
    this.floor.receiveShadow = true;
    this.floor.rotation.x = 0;
    this.scene.add(this.floor);

    const grid = new THREE.GridHelper(3, 24, 0x444444, 0x242424);
    grid.rotation.x = Math.PI / 2;
    this.scene.add(grid);

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x1c1c1c, 1.1));
    const key = new THREE.DirectionalLight(0xffffff, 2.3);
    key.position.set(1.8, -2.2, 3.4);
    key.castShadow = true;
    this.scene.add(key);

    this.ball = new THREE.Mesh(
      new THREE.SphereGeometry(0.02, 32, 16),
      new THREE.MeshStandardMaterial({ color: 0xffa319, roughness: 0.42, metalness: 0.02 })
    );
    this.ball.castShadow = true;
    this.scene.add(this.ball);

    this.racketHeadPivot = new THREE.Object3D();
    const racketHead = new THREE.Mesh(
      new THREE.CylinderGeometry(0.084, 0.084, 0.012, 64),
      new THREE.MeshStandardMaterial({ color: 0xd3312a, roughness: 0.58 })
    );
    racketHead.rotation.x = Math.PI / 2;
    racketHead.castShadow = true;
    this.racketHeadPivot.add(racketHead);
    this.scene.add(this.racketHeadPivot);

    this.racketHandlePivot = new THREE.Object3D();
    const handle = new THREE.Mesh(
      new THREE.CylinderGeometry(0.012, 0.012, 0.18, 24),
      new THREE.MeshStandardMaterial({ color: 0x51341b, roughness: 0.7 })
    );
    handle.rotation.z = Math.PI / 2;
    handle.position.x = -0.095;
    handle.castShadow = true;
    this.racketHandlePivot.add(handle);
    this.racketHeadPivot.add(this.racketHandlePivot);

    this.racketFallback = new THREE.Object3D();
    this.scene.add(this.racketFallback);

    this.targetBand = new THREE.Mesh(
      new THREE.BoxGeometry(1.6, 1.6, 0.3),
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
      new THREE.LineBasicMaterial({ color: 0x7dd3fc, transparent: true, opacity: 0.8 })
    );
    this.trailLine.visible = false;
    this.scene.add(this.trailLine);

    for (let index = 0; index < BODY_CHAIN.length; index += 1) {
      const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(index === 0 ? 0.035 : 0.022, 16, 10),
        new THREE.MeshStandardMaterial({ color: index === BODY_CHAIN.length - 1 ? 0xe8e8e8 : 0x8fd0ff })
      );
      sphere.castShadow = true;
      this.bodySpheres.push(sphere);
      this.scene.add(sphere);

      if (index > 0) {
        const link = new THREE.Mesh(
          new THREE.CylinderGeometry(1, 1, 1, 12),
          new THREE.MeshStandardMaterial({ color: 0xd7dee5, roughness: 0.55 })
        );
        link.castShadow = true;
        this.bodyLinks.push(link);
        this.scene.add(link);
      }
    }

    this.resize();
  }

  dispose(): void {
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
        camera.left = -1.1;
        camera.right = 1.1;
        camera.top = 1.1 / (this.width / this.height);
        camera.bottom = -1.1 / (this.width / this.height);
        camera.updateProjectionMatrix();
      }
    });
  }

  update(
    snapshot: SimulationSnapshot,
    world: MujocoWorld,
    visualization: VisualizationSettings,
    config: DemoConfig
  ): void {
    this.ball.position.set(...snapshot.ball.position);
    this.updateRacket(world, snapshot);
    this.updateBodySkeleton(world);
    this.updateTrail(snapshot.ball.position, visualization.trail);
    this.updateTargetBand(visualization.targetBand, config);
    this.updateContactMarkers(snapshot, visualization.contactMarker);
  }

  render(mode: CameraMode): void {
    this.renderer.setScissorTest(false);
    this.renderer.clear();

    if (mode !== "four") {
      this.renderer.setViewport(0, 0, this.width, this.height);
      this.renderer.render(this.scene, this.cameras[mode]);
      return;
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

  private updateRacket(world: MujocoWorld, snapshot: SimulationSnapshot): void {
    const transform = world.getGeometryTransform("racket_head");
    if (transform) {
      applyMujocoTransform(this.racketHeadPivot, transform.position, transform.matrix);
      return;
    }

    this.racketHeadPivot.position.set(...snapshot.racketPosition);
    this.racketHeadPivot.rotation.set(0, 0, 0);
  }

  private updateBodySkeleton(world: MujocoWorld): void {
    const positions = BODY_CHAIN.map((name) => world.getBodyPosition(name));
    positions.forEach((position, index) => {
      this.bodySpheres[index].visible = Boolean(position);
      if (position) {
        this.bodySpheres[index].position.set(...position);
      }
    });

    for (let index = 1; index < positions.length; index += 1) {
      const previous = positions[index - 1];
      const current = positions[index];
      const link = this.bodyLinks[index - 1];
      link.visible = Boolean(previous && current);
      if (previous && current) {
        updateCylinderBetween(link, previous, current, 0.014);
      }
    }
  }

  private updateTrail(ballPosition: Vec3, visible: boolean): void {
    this.trailLine.visible = visible;
    if (!visible) {
      this.trailPoints.length = 0;
      this.replaceTrailGeometry([]);
      return;
    }

    this.trailPoints.push(new THREE.Vector3(...ballPosition));
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
    this.targetBand.position.set(0.35, 0, config.targetHeight);
    this.targetBand.scale.set(1, 1, height / 0.3);
  }

  private updateContactMarkers(snapshot: SimulationSnapshot, visible: boolean): void {
    if (visible && snapshot.lastContact && snapshot.lastContact.time !== this.lastContactTime) {
      const marker = new THREE.Mesh(
        new THREE.SphereGeometry(0.025, 18, 10),
        new THREE.MeshBasicMaterial({ color: 0xfacc15, transparent: true, opacity: 0.95 })
      );
      marker.position.set(...snapshot.lastContact.position);
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

function perspectiveCamera(position: Vec3, target: Vec3): THREE.PerspectiveCamera {
  const camera = new THREE.PerspectiveCamera(48, 1, 0.01, 20);
  camera.up.set(0, 0, 1);
  camera.position.set(...position);
  camera.lookAt(new THREE.Vector3(...target));
  return camera;
}

function orthographicCamera(position: Vec3, target: Vec3): THREE.OrthographicCamera {
  const camera = new THREE.OrthographicCamera(-1.1, 1.1, 1.1, -1.1, 0.01, 20);
  camera.up.set(0, 1, 0);
  camera.position.set(...position);
  camera.lookAt(new THREE.Vector3(...target));
  return camera;
}

function updateCylinderBetween(mesh: THREE.Mesh, start: Vec3, end: Vec3, radius: number): void {
  const a = new THREE.Vector3(...start);
  const b = new THREE.Vector3(...end);
  const direction = b.clone().sub(a);
  const length = Math.max(direction.length(), 0.001);
  mesh.position.copy(a.add(b).multiplyScalar(0.5));
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize());
  mesh.scale.set(radius, length, radius);
}

function applyMujocoTransform(object: THREE.Object3D, position: Vec3, xmat: number[]): void {
  const matrix = new THREE.Matrix4();
  matrix.set(
    xmat[0],
    xmat[1],
    xmat[2],
    position[0],
    xmat[3],
    xmat[4],
    xmat[5],
    position[1],
    xmat[6],
    xmat[7],
    xmat[8],
    position[2],
    0,
    0,
    0,
    1
  );
  matrix.decompose(object.position, object.quaternion, object.scale);
}
