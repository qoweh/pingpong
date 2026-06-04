import type { MainModule, MjData, MjModel, MjVFS } from "@mujoco/mujoco";

import { fetchAssetBytes, loadAssetManifest } from "./assetLoader";
import { loadMujocoModule } from "./mujocoLoader";
import type { ContactEvent, DemoConfig, SimulationSnapshot, Vec3 } from "./types";
import { ZERO_SNAPSHOT } from "./types";
import { createPolicyRunner, type PolicyRunner } from "../policy/policyRunner";
import { loadPolicyManifest } from "../policy/policyManifest";

const MODEL_ROOT = "/assets/mujoco";
const BALL_BODY_NAME = "ball";
const BALL_GEOM_NAME = "ball_geom";
const BALL_JOINT_NAME = "ball_joint";
const RACKET_GEOM_NAME = "racket_head";
const RACKET_SITE_NAME = "racket_center";
const MAX_PHYSICS_STEPS = 32;
const BALL_MIN_Z = -0.2;
const BALL_MAX_Z = 2.6;
const BALL_MAX_XY = 3.0;

type MujocoIds = {
  ballBody: number;
  ballGeom: number;
  ballJoint: number;
  ballQposAdr: number;
  ballDofAdr: number;
  racketGeom: number;
  racketSite: number;
};

export class MujocoWorld {
  private module: MainModule | null = null;
  private vfs: MjVFS | null = null;
  private model: MjModel | null = null;
  private data: MjData | null = null;
  private ids: MujocoIds | null = null;
  private homeCtrl: number[] = [];
  private policy: PolicyRunner | null = null;
  private contactCount = 0;
  private lastContact: ContactEvent | null = null;
  private contactActive = false;
  private fallbackTime = 0;
  private fallbackBallPosition: Vec3 = [...ZERO_SNAPSHOT.ball.position] as Vec3;
  private fallbackBallVelocity: Vec3 = [0, 0, 0];
  private spawnPosition: Vec3 = [...ZERO_SNAPSHOT.ball.position] as Vec3;
  private policyMessage = "Policy not loaded";

  async initialize(config: DemoConfig): Promise<void> {
    const [module, manifest, policyManifest] = await Promise.all([
      loadMujocoModule(),
      loadAssetManifest(),
      loadPolicyManifest()
    ]);

    this.module = module;
    this.vfs = new module.MjVFS();

    await Promise.all(
      manifest.files.map(async (file) => {
        const bytes = await fetchAssetBytes(`${MODEL_ROOT}/${file}`);
        this.vfs?.addBuffer(file, bytes);
      })
    );

    this.model = module.MjModel.from_xml_path(manifest.scene, this.vfs);
    this.data = new module.MjData(this.model);
    this.ids = this.resolveIds(module, this.model);
    this.homeCtrl = Array.from(this.model.key_ctrl ?? [])
      .slice(0, this.model.nu)
      .map((value) => Number(value));
    this.policy = createPolicyRunner(policyManifest);
    this.policyMessage = this.policy.message;
    this.reset(config.ballPosition);
  }

  dispose(): void {
    this.data?.delete();
    this.model?.delete();
    this.vfs?.delete();
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

  reset(ballPosition: Vec3): SimulationSnapshot {
    if (!this.module || !this.model || !this.data || !this.ids) {
      this.fallbackTime = 0;
      this.fallbackBallPosition = [...ballPosition] as Vec3;
      this.fallbackBallVelocity = [0, 0, 0];
      this.contactCount = 0;
      this.lastContact = null;
      return this.fallbackSnapshot();
    }

    this.module.mj_resetDataKeyframe(this.model, this.data, 0);
    this.applyHomeCtrl();
    this.spawnPosition = [...ballPosition] as Vec3;
    this.spawnBall(ballPosition, [0, 0, 0]);
    this.module.mj_forward(this.model, this.data);
    this.contactCount = 0;
    this.lastContact = null;
    this.contactActive = false;
    return this.snapshot();
  }

  setBallPosition(ballPosition: Vec3): SimulationSnapshot {
    if (!this.module || !this.model || !this.data || !this.ids) {
      this.fallbackBallPosition = [...ballPosition] as Vec3;
      this.fallbackBallVelocity = [0, 0, 0];
      return this.fallbackSnapshot();
    }

    this.spawnBall(ballPosition, [0, 0, 0]);
    this.spawnPosition = [...ballPosition] as Vec3;
    this.module.mj_forward(this.model, this.data);
    return this.snapshot();
  }

  step(elapsedSeconds: number): SimulationSnapshot {
    if (!this.module || !this.model || !this.data || !this.ids) {
      return this.stepFallback(elapsedSeconds);
    }

    const timestep = Number(this.model.opt?.timestep ?? 0.002);
    const substeps = Math.max(1, Math.min(MAX_PHYSICS_STEPS, Math.ceil(elapsedSeconds / timestep)));

    for (let index = 0; index < substeps; index += 1) {
      this.applyPolicyAction();
      this.module.mj_step(this.model, this.data);
      this.updateContactState();
      this.respawnBallIfOutOfBounds();
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

  private resolveIds(module: MainModule, model: MjModel): MujocoIds {
    const ballBody = module.mj_name2id(model, module.mjtObj.mjOBJ_BODY.value, BALL_BODY_NAME);
    const ballGeom = module.mj_name2id(model, module.mjtObj.mjOBJ_GEOM.value, BALL_GEOM_NAME);
    const ballJoint = module.mj_name2id(model, module.mjtObj.mjOBJ_JOINT.value, BALL_JOINT_NAME);
    const racketGeom = module.mj_name2id(model, module.mjtObj.mjOBJ_GEOM.value, RACKET_GEOM_NAME);
    const racketSite = module.mj_name2id(model, module.mjtObj.mjOBJ_SITE.value, RACKET_SITE_NAME);

    if ([ballBody, ballGeom, ballJoint, racketGeom, racketSite].some((id) => id < 0)) {
      throw new Error("MuJoCo model is missing ball or racket identifiers.");
    }

    return {
      ballBody,
      ballGeom,
      ballJoint,
      ballQposAdr: Number(model.jnt_qposadr[ballJoint]),
      ballDofAdr: Number(model.jnt_dofadr[ballJoint]),
      racketGeom,
      racketSite
    };
  }

  private applyHomeCtrl(): void {
    if (!this.model || !this.data) {
      return;
    }

    for (let index = 0; index < this.model.nu; index += 1) {
      this.data.ctrl[index] = this.homeCtrl[index] ?? 0;
    }
  }

  private applyPolicyAction(): void {
    if (!this.model || !this.data || !this.policy) {
      return;
    }

    this.applyHomeCtrl();
    const action = this.policy.nextAction(this.snapshot(), this.model.nu);
    for (let index = 0; index < Math.min(action.length, this.model.nu); index += 1) {
      this.data.ctrl[index] += action[index];
    }
  }

  private spawnBall(position: Vec3, velocity: Vec3): void {
    if (!this.data || !this.ids) {
      return;
    }

    const qposAdr = this.ids.ballQposAdr;
    const qvelAdr = this.ids.ballDofAdr;
    this.data.qpos[qposAdr] = position[0];
    this.data.qpos[qposAdr + 1] = position[1];
    this.data.qpos[qposAdr + 2] = position[2];
    this.data.qpos[qposAdr + 3] = 1;
    this.data.qpos[qposAdr + 4] = 0;
    this.data.qpos[qposAdr + 5] = 0;
    this.data.qpos[qposAdr + 6] = 0;
    this.data.qvel[qvelAdr] = velocity[0];
    this.data.qvel[qvelAdr + 1] = velocity[1];
    this.data.qvel[qvelAdr + 2] = velocity[2];
    this.data.qvel[qvelAdr + 3] = 0;
    this.data.qvel[qvelAdr + 4] = 0;
    this.data.qvel[qvelAdr + 5] = 0;
  }

  private updateContactState(): void {
    if (!this.model || !this.data || !this.ids) {
      return;
    }

    let active = false;
    const contacts = this.data.contact;
    try {
      for (let index = 0; index < this.data.ncon; index += 1) {
        const contact = contacts.get(index);
        if (!contact) {
          continue;
        }

        const pair = [contact.geom1, contact.geom2].sort((a, b) => a - b);
        const target = [this.ids.ballGeom, this.ids.racketGeom].sort((a, b) => a - b);
        if (pair[0] === target[0] && pair[1] === target[1]) {
          active = true;
          if (!this.contactActive) {
            this.contactCount += 1;
            this.lastContact = {
              position: arrayVec3(contact.pos, 0),
              time: this.data.time
            };
          }
          break;
        }
      }
    } finally {
      contacts.delete();
    }

    this.contactActive = active;
  }

  private respawnBallIfOutOfBounds(): void {
    if (!this.module || !this.model || !this.data || !this.ids) {
      return;
    }

    const position = arrayVec3(this.data.xpos, this.ids.ballBody * 3);
    const outOfBounds =
      !position.every(Number.isFinite) ||
      Math.abs(position[0]) > BALL_MAX_XY ||
      Math.abs(position[1]) > BALL_MAX_XY ||
      position[2] < BALL_MIN_Z ||
      position[2] > BALL_MAX_Z;

    if (!outOfBounds) {
      return;
    }

    this.spawnBall(this.spawnPosition, [0, 0, 0]);
    this.contactActive = false;
    this.module.mj_forward(this.model, this.data);
  }

  private snapshot(): SimulationSnapshot {
    if (!this.data || !this.ids) {
      return this.fallbackSnapshot();
    }

    return {
      time: this.data.time,
      ball: {
        position: arrayVec3(this.data.xpos, this.ids.ballBody * 3),
        velocity: arrayVec3(this.data.qvel, this.ids.ballDofAdr)
      },
      racketPosition: arrayVec3(this.data.site_xpos, this.ids.racketSite * 3),
      contactCount: this.contactCount,
      lastContactTime: this.lastContact?.time ?? null,
      lastContact: this.lastContact,
      mujocoLoaded: true,
      policyLoaded: Boolean(this.policy?.loaded),
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

    if (this.fallbackBallPosition[2] < 0.74) {
      this.fallbackBallPosition[2] = 0.74;
      this.fallbackBallVelocity[2] = Math.abs(this.fallbackBallVelocity[2]) * 0.78 + 0.2;
      this.contactCount += 1;
      this.lastContact = {
        position: [...this.fallbackBallPosition] as Vec3,
        time: this.fallbackTime
      };
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
      racketPosition: [0.35, 0, 0.72],
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
