import type { MainModule, MjData, MjModel, MjVFS } from "@mujoco/mujoco";

import { loadMujocoAssets, loadAssetManifest } from "./assetLoader";
import type { MujocoAssetManifest } from "./assetLoader";
import { loadMujocoModule } from "./mujocoLoader";
import type {
  BallSpawnSettings,
  ContactEvent,
  LoadingProgress,
  PlaybackState,
  PolicyTrace,
  SimulationSnapshot,
  Vec3
} from "./types";
import { ZERO_SNAPSHOT } from "./types";

const MODEL_ROOT = "/assets/mujoco";
const MODEL_FS_ROOT = "/pingpong_model";
const BALL_BODY_NAME = "ball";
const BALL_JOINT_NAME = "ball_joint";
const RACKET_SITE_NAME = "racket_center";
const INITIAL_RECONNECT_DELAY_MS = 650;
const MAX_RECONNECT_DELAY_MS = 5000;
type SceneFormat = "xml" | "mjb";

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
  modelId?: string | null;
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
  action: number[] | null;
  policyTrace?: PolicyTrace | null;
};

type LiveMessage =
  | LiveFrame
  | {
      type: "ready";
      config: Record<string, unknown>;
    }
  | {
      type: "error";
      message: string;
    };

type LoadingProgressListener = (progress: LoadingProgress) => void;
type FallbackScene = {
  modelRoot: string;
  scene: string;
  sceneFormat: SceneFormat;
  files: string[];
};

export class MujocoWorld {
  private module: MainModule | null = null;
  private vfs: MjVFS | null = null;
  private model: MjModel | null = null;
  private data: MjData | null = null;
  private ids: MujocoIds | null = null;
  private socket: WebSocket | null = null;
  private reconnectTimer: number | null = null;
  private reconnectAttempts = 0;
  private socketSerial = 0;
  private disposed = false;
  private latestFrame: LiveFrame | null = null;
  private pendingSpawn: BallSpawnSettings | null = null;
  private liveReady = false;
  private liveConnected = false;
  private playback: PlaybackState = "playing";
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
  private policyMessage = "Connecting to policy model";
  private liveModelId: string | null = null;
  private latestAction: number[] | null = null;
  private latestPolicyTrace: PolicyTrace | null = null;

  async initialize(onProgress?: LoadingProgressListener): Promise<void> {
    this.disposed = false;
    notifyProgress(onProgress, 6, "Loading MuJoCo physics engine");
    const [module, manifest] = await Promise.all([
      loadMujocoModule().then((loadedModule) => {
        notifyProgress(onProgress, 34, "MuJoCo physics engine loaded");
        return loadedModule;
      }),
      loadAssetManifest().then((loadedManifest) => {
        notifyProgress(onProgress, 12, "Loading 3D scene asset list");
        return loadedManifest;
      })
    ]);

    this.module = module;
    this.vfs = new module.MjVFS();

    const modelRoot = manifest.modelRoot || MODEL_ROOT;

    notifyProgress(onProgress, 38, "Loading compiled 3D scene");
    this.ensureVirtualDirectory(MODEL_FS_ROOT);
    await this.loadSceneAssets(manifest.files, modelRoot, onProgress, 38, 36, "Loading compiled 3D scene");

    notifyProgress(onProgress, 78, "Opening 3D physics scene");
    this.model = await this.openModelWithFallback(module, manifest, onProgress);
    this.data = new module.MjData(this.model);
    this.ids = this.resolveIds(module, this.model);
    this.resetLocalState();
    notifyProgress(onProgress, 86, "Connecting policy model");
    this.connectLiveBackend();
  }

  dispose(): void {
    this.disposed = true;
    this.socketSerial += 1;
    this.clearReconnectTimer();
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

  spawnBall(settings: BallSpawnSettings): SimulationSnapshot {
    this.contactCount = 0;
    this.contactEvent = false;
    this.lastContact = null;
    this.failureReason = null;
    this.terminated = false;
    this.truncated = false;
    this.latestAction = null;
    this.latestPolicyTrace = null;
    this.resetSerial += 1;
    this.pendingSpawn = { ...settings };
    this.flushPendingSpawn();
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
    if (this.disposed) {
      return;
    }

    const serial = this.socketSerial + 1;
    this.socketSerial = serial;
    this.clearReconnectTimer();
    this.socket?.close();
    this.socket = new WebSocket(liveWebSocketUrl());

    this.socket.addEventListener("open", () => {
      if (serial !== this.socketSerial) {
        return;
      }
      this.liveConnected = true;
      this.policyMessage = "Connecting to policy model";
      this.reconnectAttempts = 0;
      this.sendCommand({ type: "playback", playback: this.playback });
      this.flushPendingSpawn();
    });

    this.socket.addEventListener("message", (event) => {
      if (serial !== this.socketSerial) {
        return;
      }
      const message = parseLiveMessage(event.data);
      if (!message) {
        return;
      }

      if (message.type === "ready") {
        this.liveReady = true;
        this.policyMessage = "Policy model ready";
        return;
      }

      if (message.type === "error") {
        this.liveReady = false;
        this.policyMessage = message.message;
        return;
      }

      this.latestFrame = message;
      this.liveReady = true;
      this.policyMessage = message.policyMessage;
    });

    this.socket.addEventListener("close", () => {
      if (serial !== this.socketSerial || this.disposed) {
        return;
      }
      this.liveConnected = false;
      this.liveReady = false;
      this.policyMessage = "Live policy stream reconnecting...";
      this.scheduleReconnect();
    });

    this.socket.addEventListener("error", () => {
      if (serial !== this.socketSerial || this.disposed) {
        return;
      }
      this.liveConnected = false;
      this.policyMessage = "Live policy stream reconnecting...";
      this.scheduleReconnect();
    });
  }

  private scheduleReconnect(): void {
    if (this.disposed || this.reconnectTimer !== null) {
      return;
    }

    const delay = Math.min(MAX_RECONNECT_DELAY_MS, INITIAL_RECONNECT_DELAY_MS * 2 ** this.reconnectAttempts);
    this.reconnectAttempts += 1;
    this.reconnectTimer = window.setTimeout(() => {
      this.reconnectTimer = null;
      this.connectLiveBackend();
    }, delay);
  }

  private clearReconnectTimer(): void {
    if (this.reconnectTimer === null) {
      return;
    }
    window.clearTimeout(this.reconnectTimer);
    this.reconnectTimer = null;
  }

  private sendCommand(command: Record<string, unknown>): boolean {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      return false;
    }
    this.socket.send(JSON.stringify(command));
    return true;
  }

  private flushPendingSpawn(): void {
    if (!this.pendingSpawn) {
      return;
    }

    const settings = this.pendingSpawn;
    const sent = this.sendCommand({
      type: "spawnBall",
      xOffset: settings.xOffset,
      yOffset: settings.yOffset,
      zOffset: settings.zOffset,
      velocityX: settings.velocityX,
      velocityY: settings.velocityY,
      velocityZ: settings.velocityZ
    });

    if (sent) {
      this.pendingSpawn = null;
    }
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
      this.policyMessage = "The server simulation and browser 3D scene are out of sync. Rebuild the web MuJoCo scene asset and refresh.";
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
    this.liveModelId = frame.modelId ?? this.liveModelId;
    this.latestAction = Array.isArray(frame.action) ? frame.action : this.latestAction;
    this.latestPolicyTrace = isPolicyTrace(frame.policyTrace) ? frame.policyTrace : this.latestPolicyTrace;
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
      this.latestAction = null;
      this.latestPolicyTrace = null;
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
    this.latestAction = null;
    this.latestPolicyTrace = null;
    this.resetSerial += 1;
  }

  private async openModelWithFallback(
    module: MainModule,
    manifest: MujocoAssetManifest,
    onProgress?: LoadingProgressListener
  ): Promise<MjModel> {
    try {
      return this.loadModel(module, manifest.scene, manifest.sceneFormat);
    } catch (primaryError) {
      const fallback = fallbackScene(manifest);
      if (!fallback) {
        throw new Error(formatMujocoModelLoadError(primaryError, manifest.scene, manifest.sceneFormat));
      }

      notifyProgress(onProgress, 80, `Compiled scene did not open; loading ${fallback.files.length} source assets`);
      await this.loadSceneAssets(fallback.files, fallback.modelRoot, onProgress, 80, 5, "Loading source 3D scene asset");
      try {
        notifyProgress(onProgress, 85, "Opening source 3D scene");
        return this.loadModel(module, fallback.scene, fallback.sceneFormat);
      } catch (fallbackError) {
        throw new Error(
          formatMujocoModelLoadError(primaryError, manifest.scene, manifest.sceneFormat, fallbackError, fallback.scene)
        );
      }
    }
  }

  private async loadSceneAssets(
    files: string[],
    modelRoot: string,
    onProgress: LoadingProgressListener | undefined,
    progressStart: number,
    progressSpan: number,
    label: string
  ): Promise<void> {
    await loadMujocoAssets(
      files,
      modelRoot,
      (file, bytes) => {
        this.registerModelFile(file, bytes);
      },
      ({ loaded, total }) => {
        const assetProgress = total > 0 ? loaded / total : 1;
        notifyProgress(onProgress, progressStart + assetProgress * progressSpan, `${label} ${loaded}/${total}`);
      }
    );
  }

  private loadModel(module: MainModule, scene: string, sceneFormat?: SceneFormat): MjModel {
    if (!this.vfs) {
      throw new Error("3D scene files are not ready yet.");
    }

    const format = sceneFormat ?? (scene.endsWith(".mjb") ? "mjb" : "xml");
    return format === "mjb" ? module.MjModel.from_binary_path(scene, this.vfs) : module.MjModel.from_xml_path(scene, this.vfs);
  }

  private registerModelFile(filePath: string, bytes: Uint8Array): void {
    const normalizedPath = normalizeModelPath(filePath);
    if (this.vfs) {
      try {
        this.vfs.deleteFile(normalizedPath);
      } catch {
        // Missing files are fine; this keeps repeated model loads deterministic.
      }
      this.vfs.addBuffer(normalizedPath, bytes);
    }

    if (!this.module) {
      return;
    }

    const virtualPath = `${MODEL_FS_ROOT}/${normalizedPath}`;
    const directory = virtualPath.slice(0, virtualPath.lastIndexOf("/"));
    this.ensureVirtualDirectory(directory);
    this.module.FS.writeFile(virtualPath, bytes);
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
      throw new Error("The 3D scene is missing required ball or racket markers. Rebuild the MuJoCo scene asset from rl/assets/scene.xml.");
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
      policyMessage: this.policyMessage,
      modelId: this.liveModelId,
      action: this.latestAction,
      policyTrace: this.latestPolicyTrace
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
      policyMessage: this.policyMessage,
      modelId: this.liveModelId,
      action: this.latestAction,
      policyTrace: this.latestPolicyTrace
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

function isPolicyTrace(value: unknown): value is PolicyTrace {
  if (!value || typeof value !== "object") {
    return false;
  }
  const trace = value as PolicyTrace;
  return (
    Array.isArray(trace.observation) &&
    Array.isArray(trace.hiddenLayers) &&
    Array.isArray(trace.action) &&
    trace.hiddenLayers.every((layer) => Array.isArray(layer))
  );
}

function normalizeModelPath(filePath: string): string {
  return filePath.replace(/^\/+/, "").replace(/^pingpong_model\//, "");
}

function fallbackScene(manifest: MujocoAssetManifest): FallbackScene | null {
  const scene = manifest.fallbackScene ?? manifest.sourceScene;
  const files = manifest.fallbackFiles ?? manifest.sourceFiles;
  if (!scene || !files?.length) {
    return null;
  }

  const uniqueFiles = Array.from(new Set([scene, ...files]));
  const sceneFormat = manifest.fallbackSceneFormat ?? (scene.endsWith(".mjb") ? "mjb" : "xml");
  return {
    modelRoot: manifest.fallbackModelRoot ?? manifest.modelRoot ?? MODEL_ROOT,
    scene,
    sceneFormat,
    files: uniqueFiles
  };
}

function formatMujocoModelLoadError(
  primaryError: unknown,
  scene: string,
  sceneFormat?: SceneFormat,
  fallbackError?: unknown,
  fallbackSceneName?: string
): string {
  const format = sceneFormat ?? (scene.endsWith(".mjb") ? "mjb" : "xml");
  const primaryDetail = compactErrorMessage(primaryError);

  if (fallbackError) {
    return (
      "The browser could not open the 3D physics scene. " +
      `The compiled scene (${scene}) failed first, and the source scene fallback (${fallbackSceneName ?? "scene.xml"}) failed too. ` +
      "Refresh once; if it keeps happening, rebuild the web MuJoCo scene asset with npm run compile:mujoco and redeploy. " +
      `Details: compiled scene: ${primaryDetail}; source scene: ${compactErrorMessage(fallbackError)}`
    );
  }

  if (format === "mjb") {
    return (
      `The compiled 3D scene file (${scene}) could not be opened. ` +
      "MJB is MuJoCo's precompiled binary scene format. " +
      "Refresh once; if it keeps happening, rebuild the web scene asset with npm run compile:mujoco and redeploy. " +
      `Detail: ${primaryDetail}`
    );
  }

  return (
    `The source 3D scene file (${scene}) could not be opened. ` +
    "Check that rl/assets/scene.xml and its mesh files are present on the server, then refresh. " +
    `Detail: ${primaryDetail}`
  );
}

function compactErrorMessage(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  return message.replace(/\s+/g, " ").trim() || "unknown error";
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

function notifyProgress(listener: LoadingProgressListener | undefined, percent: number, message: string): void {
  listener?.({ percent: clampProgress(percent), message });
}

function clampProgress(percent: number): number {
  return Math.min(100, Math.max(0, percent));
}
