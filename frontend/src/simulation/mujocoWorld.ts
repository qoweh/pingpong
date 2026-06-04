import type { MainModule, MjData, MjModel, MjVFS } from "@mujoco/mujoco";

import { loadMujocoAssets, loadAssetManifest } from "./assetLoader";
import { loadMujocoModule } from "./mujocoLoader";
import type {
  BallSpawnSettings,
  ContactEvent,
  DemoConfig,
  PlaybackState,
  SimulationSnapshot,
  Vec3
} from "./types";
import { ZERO_SNAPSHOT } from "./types";

const MODEL_ROOT = "/assets/mujoco";
const MODEL_FS_ROOT = "/pingpong_model";
const BALL_BODY_NAME = "ball";
const BALL_JOINT_NAME = "ball_joint";
const RACKET_SITE_NAME = "racket_center";

type MujocoIds = {
  ballBody: number;
  ballDofAdr: number;
  racketSite: number;
};

type LiveFrame = {
  type: "frame";
  episode: number;
  step: number;
  time: number;
  reset: boolean;
  terminated: boolean;
  truncated: boolean;
  failureReason: string | null;
  policyLoaded: boolean;
  policyMessage: string;
  state: {
    qpos: number[];
    qvel: number[];
    ctrl: number[];
    time: number;
  };
  ball: {
    position: Vec3;
    velocity: Vec3;
  };
  racketPosition: Vec3;
  contact: {
    event: boolean;
    count: number;
    last: ContactEvent | null;
  };
};

type LiveMessage =
  | LiveFrame
  | {
      type: "ready";
      config: Record<string, unknown>;
    };

export class MujocoWorld {
  private module: MainModule | null = null;
  private vfs: MjVFS | null = null;
  private model: MjModel | null = null;
  private data: MjData | null = null;
  private ids: MujocoIds | null = null;
  private socket: WebSocket | null = null;
  private latestFrame: LiveFrame | null = null;
  private liveReady = false;
  private liveConnected = false;
  private playback: PlaybackState = "paused";
  private contactCount = 0;
  private contactEvent = false;
  private lastContact: ContactEvent | null = null;
  private currentEpisode = -1;
  private resetSerial = 0;
  private lastVisualResetKey = "";
  private failureReason: string | null = null;
  private terminated = false;
  private truncated = false;
  private fallbackTime = 0;
  private fallbackBallPosition: Vec3 = [...ZERO_SNAPSHOT.ball.position] as Vec3;
  private fallbackBallVelocity: Vec3 = [0, 0, 0];
  private racketAnchor: Vec3 = [...ZERO_SNAPSHOT.racketPosition] as Vec3;
  private policyMessage = "Connecting to Python live backend";

  async initialize(_config: DemoConfig): Promise<void> {
    const [module, manifest] = await Promise.all([loadMujocoModule(), loadAssetManifest()]);

    this.module = module;
    this.vfs = new module.MjVFS();

    const modelRoot = manifest.modelRoot || MODEL_ROOT;

    this.ensureVirtualDirectory(MODEL_FS_ROOT);
    await loadMujocoAssets(manifest.files, modelRoot, (file, bytes) => {
      this.writeVirtualFile(`${MODEL_FS_ROOT}/${file}`, bytes);
    });

    this.model = this.loadModel(module, `${MODEL_FS_ROOT}/${manifest.scene}`, manifest.sceneFormat);
    this.data = new module.MjData(this.model);
    this.ids = this.resolveIds(module, this.model);
    this.resetLocalState();
    this.connectLiveBackend();
  }

  dispose(): void {
    this.socket?.close();
    this.data?.delete();
    this.model?.delete();
    this.vfs?.delete();
    this.socket = null;
    this.data = null;
    this.model = null;
    this.vfs = null;
    this.ids = null;
  }

  getRuntime(): { module: MainModule; model: MjModel; data: MjData } | null {
    if (!this.module || !this.model || !this.data) {
      return null;
    }

    return {
      module: this.module,
      model: this.model,
      data: this.data
    };
  }

  reset(): SimulationSnapshot {
    this.resetLocalState();
    this.sendCommand({ type: "reset" });
    return this.snapshot();
  }

  resetBall(settings: BallSpawnSettings): SimulationSnapshot {
    this.resetLocalState();
    this.sendCommand({
      type: "resetBall",
      xOffset: settings.xOffset,
      yOffset: settings.yOffset,
      height: settings.height,
      velocityZ: settings.velocityZ
    });
    return this.snapshot();
  }

  setPlayback(playback: PlaybackState): void {
    this.playback = playback;
    this.sendCommand({ type: "playback", playback });
  }

  step(_elapsedSeconds: number): SimulationSnapshot {
    if (!this.module || !this.model || !this.data || !this.ids) {
      return this.stepFallback(1 / 60);
    }

    if (this.latestFrame) {
      this.applyLiveFrame(this.latestFrame);
      this.latestFrame = null;
    }

    return this.snapshot();
  }

  getGeometryTransform(name: string): { position: Vec3; matrix: number[] } | null {
    if (!this.module || !this.model || !this.data) {
      return null;
    }

    const geomId = this.module.mj_name2id(this.model, this.module.mjtObj.mjOBJ_GEOM.value, name);
    if (geomId < 0) {
      return null;
    }

    return {
      position: arrayVec3(this.data.geom_xpos, geomId * 3),
      matrix: Array.from(this.data.geom_xmat.slice(geomId * 9, geomId * 9 + 9))
    };
  }

  getBodyPosition(name: string): Vec3 | null {
    if (!this.module || !this.model || !this.data) {
      return null;
    }

    const bodyId = this.module.mj_name2id(this.model, this.module.mjtObj.mjOBJ_BODY.value, name);
    if (bodyId < 0) {
      return null;
    }

    return arrayVec3(this.data.xpos, bodyId * 3);
  }

  private connectLiveBackend(): void {
    this.socket?.close();
    this.socket = new WebSocket(liveWebSocketUrl());

    this.socket.addEventListener("open", () => {
      this.liveConnected = true;
      this.policyMessage = "Python live backend connected";
      this.sendCommand({ type: "playback", playback: this.playback });
    });

    this.socket.addEventListener("message", (event) => {
      const message = parseLiveMessage(event.data);
      if (!message) {
        return;
      }

      if (message.type === "ready") {
        this.liveReady = true;
        this.policyMessage = "Python live backend ready";
        return;
      }

      this.latestFrame = message;
      this.liveReady = true;
      this.policyMessage = message.policyMessage;
    });

    this.socket.addEventListener("close", () => {
      this.liveConnected = false;
      this.liveReady = false;
      this.policyMessage = "Python live backend disconnected";
    });

    this.socket.addEventListener("error", () => {
      this.liveConnected = false;
      this.policyMessage = "Python live backend connection failed";
    });
  }

  private sendCommand(command: Record<string, unknown>): void {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      return;
    }
    this.socket.send(JSON.stringify(command));
  }

  private applyLiveFrame(frame: LiveFrame): void {
    if (!this.module || !this.model || !this.data || !this.ids) {
      return;
    }

    if (
      frame.state.qpos.length !== this.model.nq ||
      frame.state.qvel.length !== this.model.nv ||
      frame.state.ctrl.length !== this.model.nu
    ) {
      this.policyMessage = "Python live frame does not match the loaded MuJoCo model.";
      return;
    }

    copyBuffer(this.data.qpos, frame.state.qpos, this.model.nq);
    copyBuffer(this.data.qvel, frame.state.qvel, this.model.nv);
    copyBuffer(this.data.ctrl, frame.state.ctrl, this.model.nu);
    this.data.time = Number.isFinite(frame.state.time) ? frame.state.time : frame.time;
    this.module.mj_forward(this.model, this.data);

    const eventKey = `${frame.episode}:${frame.step}`;
    const episodeChanged = this.currentEpisode !== frame.episode;
    const floorContact = frame.failureReason === "floor_contact" || frame.ball.position[2] <= 0.025;
    const shouldResetVisuals = frame.reset || episodeChanged || frame.contact.event || floorContact;
    if (shouldResetVisuals && this.lastVisualResetKey !== eventKey) {
      this.resetSerial += 1;
      this.lastVisualResetKey = eventKey;
    }

    this.currentEpisode = frame.episode;
    this.contactCount = frame.contact.count;
    this.contactEvent = frame.contact.event;
    this.lastContact = frame.contact.last;
    this.failureReason = frame.failureReason;
    this.terminated = frame.terminated;
    this.truncated = frame.truncated;
    this.racketAnchor = arrayVec3(this.data.site_xpos, this.ids.racketSite * 3);
  }

  private resetLocalState(): void {
    if (!this.module || !this.model || !this.data || !this.ids) {
      this.fallbackTime = 0;
      this.fallbackBallPosition = [...ZERO_SNAPSHOT.ball.position] as Vec3;
      this.fallbackBallVelocity = [0, 0, 0];
      this.contactCount = 0;
      this.contactEvent = false;
      this.lastContact = null;
      this.failureReason = null;
      this.terminated = false;
      this.truncated = false;
      this.resetSerial += 1;
      return;
    }

    this.module.mj_resetData(this.model, this.data);
    this.module.mj_forward(this.model, this.data);
    this.racketAnchor = arrayVec3(this.data.site_xpos, this.ids.racketSite * 3);
    this.contactCount = 0;
    this.contactEvent = false;
    this.lastContact = null;
    this.failureReason = null;
    this.terminated = false;
    this.truncated = false;
    this.resetSerial += 1;
  }

  private loadModel(module: MainModule, scene: string, sceneFormat?: "xml" | "mjb"): MjModel {
    if (!this.vfs) {
      throw new Error("MuJoCo virtual file system is not initialized.");
    }

    const format = sceneFormat ?? (scene.endsWith(".mjb") ? "mjb" : "xml");
    return format === "mjb" ? module.MjModel.from_binary_path(scene, this.vfs) : module.MjModel.from_xml_path(scene, this.vfs);
  }

  private writeVirtualFile(filePath: string, bytes: Uint8Array): void {
    if (!this.module) {
      return;
    }

    const directory = filePath.slice(0, filePath.lastIndexOf("/"));
    this.ensureVirtualDirectory(directory);
    this.module.FS.writeFile(filePath, bytes);
  }

  private ensureVirtualDirectory(directory: string): void {
    if (!this.module) {
      return;
    }

    let current = "";
    for (const part of directory.split("/").filter(Boolean)) {
      current += `/${part}`;
      try {
        this.module.FS.mkdir(current);
      } catch {
        // Directory already exists in the Emscripten filesystem.
      }
    }
  }

  private resolveIds(module: MainModule, model: MjModel): MujocoIds {
    const ballBody = module.mj_name2id(model, module.mjtObj.mjOBJ_BODY.value, BALL_BODY_NAME);
    const ballJoint = module.mj_name2id(model, module.mjtObj.mjOBJ_JOINT.value, BALL_JOINT_NAME);
    const racketSite = module.mj_name2id(model, module.mjtObj.mjOBJ_SITE.value, RACKET_SITE_NAME);

    if ([ballBody, ballJoint, racketSite].some((id) => id < 0)) {
      throw new Error("MuJoCo model is missing ball or racket identifiers.");
    }

    return {
      ballBody,
      ballDofAdr: Number(model.jnt_dofadr[ballJoint]),
      racketSite
    };
  }

  private ballPosition(): Vec3 {
    if (!this.data || !this.ids) {
      return [...this.fallbackBallPosition] as Vec3;
    }

    return arrayVec3(this.data.xpos, this.ids.ballBody * 3);
  }

  private ballVelocity(): Vec3 {
    if (!this.data || !this.ids) {
      return [...this.fallbackBallVelocity] as Vec3;
    }

    return arrayVec3(this.data.qvel, this.ids.ballDofAdr);
  }

  private snapshot(): SimulationSnapshot {
    if (!this.data || !this.ids) {
      return this.fallbackSnapshot();
    }

    return {
      episode: this.currentEpisode < 0 ? 0 : this.currentEpisode,
      resetSerial: this.resetSerial,
      time: this.data.time,
      ball: {
        position: this.ballPosition(),
        velocity: this.ballVelocity()
      },
      racketPosition: arrayVec3(this.data.site_xpos, this.ids.racketSite * 3),
      contactCount: this.contactCount,
      contactEvent: this.contactEvent,
      lastContactTime: this.lastContact?.time ?? null,
      lastContact: this.lastContact,
      failureReason: this.failureReason,
      terminated: this.terminated,
      truncated: this.truncated,
      mujocoLoaded: true,
      policyLoaded: this.liveConnected && this.liveReady,
      policyMessage: this.policyMessage
    };
  }

  private stepFallback(elapsedSeconds: number): SimulationSnapshot {
    this.fallbackTime += elapsedSeconds;
    this.fallbackBallVelocity[2] -= 9.81 * elapsedSeconds;
    this.fallbackBallPosition = [
      this.fallbackBallPosition[0],
      this.fallbackBallPosition[1],
      this.fallbackBallPosition[2] + this.fallbackBallVelocity[2] * elapsedSeconds
    ];

    if (this.fallbackBallPosition[2] < 0.05) {
      this.fallbackBallPosition = [...ZERO_SNAPSHOT.ball.position] as Vec3;
      this.fallbackBallVelocity = [0, 0, 0];
      this.contactCount = 0;
      this.contactEvent = false;
      this.lastContact = null;
      this.failureReason = "floor_contact";
      this.terminated = true;
      this.resetSerial += 1;
    }

    return this.fallbackSnapshot();
  }

  private fallbackSnapshot(): SimulationSnapshot {
    return {
      episode: 0,
      resetSerial: this.resetSerial,
      time: this.fallbackTime,
      ball: {
        position: this.fallbackBallPosition,
        velocity: this.fallbackBallVelocity
      },
      racketPosition: this.racketAnchor,
      contactCount: this.contactCount,
      contactEvent: this.contactEvent,
      lastContactTime: this.lastContact?.time ?? null,
      lastContact: this.lastContact,
      failureReason: this.failureReason,
      terminated: this.terminated,
      truncated: this.truncated,
      mujocoLoaded: false,
      policyLoaded: false,
      policyMessage: this.policyMessage
    };
  }
}

function liveWebSocketUrl(): string {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/api/live`;
}

function parseLiveMessage(rawData: unknown): LiveMessage | null {
  if (typeof rawData !== "string") {
    return null;
  }

  try {
    const message = JSON.parse(rawData) as LiveMessage;
    return message && typeof message === "object" ? message : null;
  } catch {
    return null;
  }
}

function arrayVec3(arrayLike: ArrayLike<number>, offset: number): Vec3 {
  return [
    Number(arrayLike[offset] ?? 0),
    Number(arrayLike[offset + 1] ?? 0),
    Number(arrayLike[offset + 2] ?? 0)
  ];
}

function copyBuffer(target: ArrayLike<number>, source: ArrayLike<number>, length: number): void {
  const writable = target as { [index: number]: number };
  for (let index = 0; index < length; index += 1) {
    writable[index] = Number(source[index] ?? 0);
  }
}
