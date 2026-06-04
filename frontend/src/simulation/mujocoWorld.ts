import type { MainModule, MjData, MjModel, MjVFS } from "@mujoco/mujoco";

import { loadMujocoAssets, loadAssetManifest } from "./assetLoader";
import { loadMujocoModule } from "./mujocoLoader";
import { loadRolloutTrace, type MujocoStateFrame, type RolloutFrame, type RolloutTrace } from "./rolloutTrace";
import type { ContactEvent, DemoConfig, SimulationSnapshot, Vec3 } from "./types";
import { ZERO_SNAPSHOT } from "./types";

const MODEL_ROOT = "/assets/mujoco";
const MODEL_FS_ROOT = "/pingpong_model";
const BALL_BODY_NAME = "ball";
const BALL_JOINT_NAME = "ball_joint";
const RACKET_SITE_NAME = "racket_center";
const MAX_TRACE_STEPS_PER_RENDER = 8;

type MujocoIds = {
  ballBody: number;
  ballDofAdr: number;
  racketSite: number;
};

export class MujocoWorld {
  private module: MainModule | null = null;
  private vfs: MjVFS | null = null;
  private model: MjModel | null = null;
  private data: MjData | null = null;
  private ids: MujocoIds | null = null;
  private trace: RolloutTrace | null = null;
  private homeCtrl: number[] = [];
  private frameIndex = 0;
  private frameAccumulator = 0;
  private traceEnded = false;
  private contactCount = 0;
  private lastContact: ContactEvent | null = null;
  private fallbackTime = 0;
  private fallbackBallPosition: Vec3 = [...ZERO_SNAPSHOT.ball.position] as Vec3;
  private fallbackBallVelocity: Vec3 = [0, 0, 0];
  private racketAnchor: Vec3 = [...ZERO_SNAPSHOT.racketPosition] as Vec3;
  private policyMessage = "Loading Python rollout trace";

  async initialize(_config: DemoConfig): Promise<void> {
    const [module, manifest, trace] = await Promise.all([
      loadMujocoModule(),
      loadAssetManifest(),
      loadRolloutTrace()
    ]);

    this.module = module;
    this.trace = trace;
    this.vfs = new module.MjVFS();

    const modelRoot = manifest.modelRoot || MODEL_ROOT;

    this.ensureVirtualDirectory(MODEL_FS_ROOT);
    await loadMujocoAssets(manifest.files, modelRoot, (file, bytes) => {
      this.writeVirtualFile(`${MODEL_FS_ROOT}/${file}`, bytes);
    });

    this.model = this.loadModel(module, `${MODEL_FS_ROOT}/${manifest.scene}`, manifest.sceneFormat);
    this.data = new module.MjData(this.model);
    this.ids = this.resolveIds(module, this.model);
    this.homeCtrl = Array.from(this.model.key_ctrl ?? [])
      .slice(0, this.model.nu)
      .map((value) => Number(value));
    this.validateTraceModel();
    this.policyMessage = this.trace
      ? `Python rollout replay loaded: ${basename(this.trace.source.model)}`
      : "Python rollout trace missing. Run scripts/export_web_rollout_from_env.py.";
    this.reset();
  }

  dispose(): void {
    this.data?.delete();
    this.model?.delete();
    this.vfs?.delete();
    this.data = null;
    this.model = null;
    this.vfs = null;
    this.ids = null;
    this.trace = null;
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
    if (!this.module || !this.model || !this.data || !this.ids) {
      this.fallbackTime = 0;
      this.fallbackBallPosition = [...ZERO_SNAPSHOT.ball.position] as Vec3;
      this.fallbackBallVelocity = [0, 0, 0];
      this.resetTracePlayback();
      return this.fallbackSnapshot();
    }

    this.module.mj_resetData(this.model, this.data);
    if (this.trace) {
      this.applyMujocoState(this.trace.initialState);
    } else {
      this.applyHomeCtrl();
    }
    this.module.mj_forward(this.model, this.data);
    this.racketAnchor = arrayVec3(this.data.site_xpos, this.ids.racketSite * 3);
    this.resetTracePlayback();
    return this.snapshot();
  }

  step(elapsedSeconds: number): SimulationSnapshot {
    if (!this.module || !this.model || !this.data || !this.ids) {
      return this.stepFallback(elapsedSeconds);
    }

    if (!this.trace) {
      return this.snapshot();
    }

    if (this.traceEnded) {
      this.reset();
    }

    const controlDt = this.controlDt();
    const elapsed = Number.isFinite(elapsedSeconds) ? Math.max(0, elapsedSeconds) : 0;
    this.frameAccumulator += Math.min(elapsed, controlDt * MAX_TRACE_STEPS_PER_RENDER);

    let steps = 0;
    while (this.frameAccumulator >= controlDt && steps < MAX_TRACE_STEPS_PER_RENDER) {
      this.frameAccumulator -= controlDt;
      this.advanceTraceFrame();
      steps += 1;

      if (this.traceEnded) {
        break;
      }
    }

    if (steps >= MAX_TRACE_STEPS_PER_RENDER && this.frameAccumulator >= controlDt) {
      this.frameAccumulator = 0;
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

  private advanceTraceFrame(): void {
    if (!this.module || !this.model || !this.data || !this.trace) {
      return;
    }

    if (this.frameIndex >= this.trace.frames.length) {
      this.traceEnded = true;
      return;
    }

    const frame = this.trace.frames[this.frameIndex];
    this.applyCtrl(frame.ctrl);

    const substeps = this.substepsPerFrame();
    for (let index = 0; index < substeps; index += 1) {
      this.module.mj_step(this.model, this.data);
    }

    this.applyFrameMetadata(frame);
    this.frameIndex += 1;
    this.traceEnded = frame.terminated || frame.truncated || this.frameIndex >= this.trace.frames.length;
  }

  private applyFrameMetadata(frame: RolloutFrame): void {
    this.contactCount = Math.max(this.contactCount, frame.contact.count);

    if (frame.contact.event && frame.contact.position) {
      this.lastContact = {
        position: frame.contact.position,
        time: frame.time
      };
    }
  }

  private applyMujocoState(state: MujocoStateFrame): void {
    if (!this.model || !this.data) {
      return;
    }

    copyBuffer(this.data.qpos, state.qpos, this.model.nq);
    copyBuffer(this.data.qvel, state.qvel, this.model.nv);
    copyBuffer(this.data.ctrl, state.ctrl, this.model.nu);
    this.data.time = Number.isFinite(state.time) ? state.time : 0;
  }

  private applyCtrl(ctrl: ArrayLike<number>): void {
    if (!this.model || !this.data) {
      return;
    }

    copyBuffer(this.data.ctrl, ctrl, this.model.nu);
  }

  private applyHomeCtrl(): void {
    if (!this.model || !this.data) {
      return;
    }

    for (let index = 0; index < this.model.nu; index += 1) {
      this.data.ctrl[index] = this.homeCtrl[index] ?? 0;
    }
  }

  private controlDt(): number {
    return Math.max(1.0e-6, Number(this.trace?.simulation.controlDt ?? 0.02));
  }

  private substepsPerFrame(): number {
    if (!this.model) {
      return 1;
    }

    const exportedSubsteps = Math.trunc(Number(this.trace?.simulation.substeps ?? 0));
    if (exportedSubsteps > 0) {
      return Math.min(exportedSubsteps, 256);
    }

    const timestep = Math.max(1.0e-6, Number(this.model.opt?.timestep ?? 0.002));
    return Math.max(1, Math.min(256, Math.round(this.controlDt() / timestep)));
  }

  private resetTracePlayback(): void {
    this.frameIndex = 0;
    this.frameAccumulator = 0;
    this.traceEnded = false;
    this.contactCount = 0;
    this.lastContact = null;
  }

  private validateTraceModel(): void {
    if (!this.model || !this.trace) {
      return;
    }

    const mismatches = [
      ["nq", this.trace.simulation.nq, this.model.nq],
      ["nv", this.trace.simulation.nv, this.model.nv],
      ["nu", this.trace.simulation.nu, this.model.nu]
    ].filter(([, expected, actual]) => expected !== actual);

    if (mismatches.length > 0) {
      const detail = mismatches
        .map(([name, expected, actual]) => `${name}: rollout=${expected}, model=${actual}`)
        .join(", ");
      throw new Error(`Python rollout does not match the loaded MuJoCo model (${detail}). Re-export the rollout.`);
    }
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
      time: this.data.time,
      ball: {
        position: this.ballPosition(),
        velocity: this.ballVelocity()
      },
      racketPosition: arrayVec3(this.data.site_xpos, this.ids.racketSite * 3),
      contactCount: this.contactCount,
      lastContactTime: this.lastContact?.time ?? null,
      lastContact: this.lastContact,
      mujocoLoaded: true,
      policyLoaded: Boolean(this.trace),
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
      this.resetTracePlayback();
    }

    return this.fallbackSnapshot();
  }

  private fallbackSnapshot(): SimulationSnapshot {
    return {
      time: this.fallbackTime,
      ball: {
        position: this.fallbackBallPosition,
        velocity: this.fallbackBallVelocity
      },
      racketPosition: this.racketAnchor,
      contactCount: this.contactCount,
      lastContactTime: this.lastContact?.time ?? null,
      lastContact: this.lastContact,
      mujocoLoaded: false,
      policyLoaded: false,
      policyMessage: this.policyMessage
    };
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

function basename(path: string): string {
  return path.split("/").filter(Boolean).at(-1) ?? path;
}
