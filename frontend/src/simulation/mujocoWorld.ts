import type { MainModule, MjData, MjModel, MjVFS } from "@mujoco/mujoco";

import { loadMujocoAssets, loadAssetManifest } from "./assetLoader";
import { loadMujocoModule } from "./mujocoLoader";
import { RacketCartesianController } from "./racketCartesianController";
import type { ContactEvent, DemoConfig, SimulationSnapshot, Vec3 } from "./types";
import { ZERO_SNAPSHOT } from "./types";
import { createPolicyRunner, type PolicyRunner } from "../policy/policyRunner";
import { loadPolicyManifest } from "../policy/policyManifest";

const MODEL_ROOT = "/assets/mujoco";
const BALL_BODY_NAME = "ball";
const BALL_GEOM_NAME = "ball_geom";
const BALL_JOINT_NAME = "ball_joint";
const FLOOR_GEOM_NAME = "floor";
const RACKET_BODY_NAME = "racket";
const RACKET_GEOM_NAME = "racket_head";
const RACKET_SITE_NAME = "racket_center";
const MAX_PHYSICS_STEPS = 32;
const CONTROL_DT = 0.02;
const BALL_MIN_Z = -0.2;
const BALL_MAX_Z = 2.6;
const BALL_MAX_XY = 3.0;
const GRAVITY_Z = -9.81;
const DESCENDING_BALL_VELOCITY_THRESHOLD = -0.05;
const TRACKING_STRIKE_PLANE_OFFSET = 0.06;
const TARGET_BALL_HEIGHT = 0.3;
const HEIGHT_TOLERANCE = 0.1;
const NEXT_INTERCEPT_MAX_TIME = 1.25;
const NEXT_INTERCEPT_SUCCESS_RADIUS = 0.04;
const EASY_NEXT_BALL_TARGET_TIME = 0.45;
const EASY_NEXT_BALL_TIME_TOLERANCE = 0.3;
const EASY_NEXT_BALL_TARGET_DESCENDING_SPEED = 1.25;
const EASY_NEXT_BALL_MAX_LATERAL_SPEED = 1.0;
const EASY_NEXT_BALL_SOFT_SPEED_LIMIT = 3.0;

type MujocoIds = {
  ballBody: number;
  ballGeom: number;
  ballJoint: number;
  ballQposAdr: number;
  ballDofAdr: number;
  floorGeom: number;
  racketBody: number;
  racketGeom: number;
  racketGeomIds: Set<number>;
  racketSite: number;
};

type ContactState = {
  racket: boolean;
  floor: boolean;
};

type NextInterceptMetrics = {
  time: number;
  relativeXY: [number, number];
  reachable: boolean;
  recoveryDistance: number;
  recoveryReadiness: number;
  easyScore: number;
};

type PolicyCommand = {
  targetPosition: Vec3;
  targetTilt: [number, number];
  targetVelocity: Vec3 | null;
};

export class MujocoWorld {
  private module: MainModule | null = null;
  private vfs: MjVFS | null = null;
  private model: MjModel | null = null;
  private data: MjData | null = null;
  private ids: MujocoIds | null = null;
  private homeCtrl: number[] = [];
  private policy: PolicyRunner | null = null;
  private controller: RacketCartesianController | null = null;
  private contactCount = 0;
  private lastContact: ContactEvent | null = null;
  private contactActive = false;
  private controlAccumulator = CONTROL_DT;
  private fallbackTime = 0;
  private fallbackBallPosition: Vec3 = [...ZERO_SNAPSHOT.ball.position] as Vec3;
  private fallbackBallVelocity: Vec3 = [0, 0, 0];
  private spawnPosition: Vec3 = [...ZERO_SNAPSHOT.ball.position] as Vec3;
  private racketAnchor: Vec3 = [...ZERO_SNAPSHOT.racketPosition] as Vec3;
  private policyMessage = "Policy not loaded";

  async initialize(config: DemoConfig): Promise<void> {
    const [module, manifest, policyManifest] = await Promise.all([
      loadMujocoModule(),
      loadAssetManifest(),
      loadPolicyManifest()
    ]);

    this.module = module;
    this.vfs = new module.MjVFS();

    await loadMujocoAssets(manifest.files, MODEL_ROOT, (file, bytes) => {
      this.vfs?.addBuffer(file, bytes);
    });

    this.model = module.MjModel.from_xml_path(manifest.scene, this.vfs);
    this.data = new module.MjData(this.model);
    this.ids = this.resolveIds(module, this.model);
    this.homeCtrl = Array.from(this.model.key_ctrl ?? [])
      .slice(0, this.model.nu)
      .map((value) => Number(value));
    this.policy = await createPolicyRunner(policyManifest);
    this.policyMessage = this.policy.message;
    this.controller = new RacketCartesianController(
      { module: this.module, model: this.model, data: this.data },
      { racketSite: this.ids.racketSite, racketGeom: this.ids.racketGeom },
      this.homeCtrl
    );
    this.reset(config.ballPosition);
  }

  dispose(): void {
    this.controller?.dispose();
    this.data?.delete();
    this.model?.delete();
    this.vfs?.delete();
    this.controller = null;
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
      this.resetContactMetrics();
      return this.fallbackSnapshot();
    }

    this.module.mj_resetDataKeyframe(this.model, this.data, 0);
    this.applyHomeCtrl();
    this.spawnPosition = [...ballPosition] as Vec3;
    this.spawnBall(ballPosition, [0, 0, 0]);
    this.module.mj_forward(this.model, this.data);
    this.controller?.reset();
    this.racketAnchor = this.controller?.racketPosition() ?? arrayVec3(this.data.site_xpos, this.ids.racketSite * 3);
    this.controlAccumulator = CONTROL_DT;
    this.resetContactMetrics();
    return this.snapshot();
  }

  setBallPosition(ballPosition: Vec3): SimulationSnapshot {
    if (!this.module || !this.model || !this.data || !this.ids) {
      this.fallbackBallPosition = [...ballPosition] as Vec3;
      this.fallbackBallVelocity = [0, 0, 0];
      this.resetContactMetrics();
      return this.fallbackSnapshot();
    }

    this.spawnBall(ballPosition, [0, 0, 0]);
    this.spawnPosition = [...ballPosition] as Vec3;
    this.module.mj_forward(this.model, this.data);
    this.controlAccumulator = CONTROL_DT;
    this.resetContactMetrics();
    return this.snapshot();
  }

  step(elapsedSeconds: number): SimulationSnapshot {
    if (!this.module || !this.model || !this.data || !this.ids) {
      return this.stepFallback(elapsedSeconds);
    }

    const timestep = Number(this.model.opt?.timestep ?? 0.002);
    const substeps = Math.max(1, Math.min(MAX_PHYSICS_STEPS, Math.ceil(elapsedSeconds / timestep)));

    for (let index = 0; index < substeps; index += 1) {
      this.controlAccumulator += timestep;
      if (this.controlAccumulator >= CONTROL_DT) {
        this.controlAccumulator %= CONTROL_DT;
        this.applyPolicyAction();
      }

      this.module.mj_step(this.model, this.data);
      const contactState = this.updateContactState();
      if (contactState.floor) {
        this.reset(this.spawnPosition);
        break;
      }
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
    const floorGeom = module.mj_name2id(model, module.mjtObj.mjOBJ_GEOM.value, FLOOR_GEOM_NAME);
    const racketBody = module.mj_name2id(model, module.mjtObj.mjOBJ_BODY.value, RACKET_BODY_NAME);
    const racketGeom = module.mj_name2id(model, module.mjtObj.mjOBJ_GEOM.value, RACKET_GEOM_NAME);
    const racketSite = module.mj_name2id(model, module.mjtObj.mjOBJ_SITE.value, RACKET_SITE_NAME);

    if ([ballBody, ballGeom, ballJoint, floorGeom, racketBody, racketGeom, racketSite].some((id) => id < 0)) {
      throw new Error("MuJoCo model is missing ball, floor, or racket identifiers.");
    }

    const racketGeomIds = new Set<number>();
    for (let geomId = 0; geomId < model.ngeom; geomId += 1) {
      const geomName = module.mj_id2name(model, module.mjtObj.mjOBJ_GEOM.value, geomId);
      if (Number(model.geom_bodyid[geomId]) === racketBody && geomName.startsWith("racket_")) {
        racketGeomIds.add(geomId);
      }
    }
    racketGeomIds.add(racketGeom);

    return {
      ballBody,
      ballGeom,
      ballJoint,
      ballQposAdr: Number(model.jnt_qposadr[ballJoint]),
      ballDofAdr: Number(model.jnt_dofadr[ballJoint]),
      floorGeom,
      racketBody,
      racketGeom,
      racketGeomIds,
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
    if (!this.model || !this.data || !this.policy || !this.policy.loaded || !this.controller) {
      this.applyHomeCtrl();
      return;
    }

    const observation = this.buildPolicyObservation();
    const action = this.policy.nextAction(observation);
    const command = this.mapPolicyAction(action);
    this.controller.setTargetPosition(command.targetPosition);
    this.controller.setTargetTilt(command.targetTilt);
    this.controller.setTargetVelocity(command.targetVelocity);

    const jointTargets = this.controller.computeJointTargets();
    for (let index = 0; index < Math.min(7, this.model.nu); index += 1) {
      this.data.ctrl[index] = jointTargets[index] ?? this.homeCtrl[index] ?? 0;
    }
    if (this.model.nu > 7) {
      this.data.ctrl[7] = this.homeCtrl[7] ?? 76.5;
    }
  }

  private mapPolicyAction(action: ArrayLike<number>): PolicyCommand {
    const ballPosition = this.ballPosition();
    const ballVelocity = this.ballVelocity();
    const racketPosition = this.controller?.racketPosition() ?? this.racketAnchor;
    const interceptTime = this.predictedInterceptTime(NEXT_INTERCEPT_MAX_TIME);
    const contactPosition = this.predictedContactPosition(interceptTime);
    const descending = ballVelocity[2] < DESCENDING_BALL_VELOCITY_THRESHOLD;
    const urgency = 1 - clamp(interceptTime / 0.18, 0, 1);
    const heightReadiness = this.preContactHeightReadiness(ballPosition, racketPosition, ballVelocity);
    const strikeReadiness = clamp(Math.max(heightReadiness, urgency), 0, 1);

    const targetPosition: Vec3 = [...this.racketAnchor] as Vec3;
    if (descending && interceptTime > 0) {
      targetPosition[0] = contactPosition[0] + safeAction(action, 0);
      targetPosition[1] = contactPosition[1] + safeAction(action, 1);
      targetPosition[2] = contactPosition[2] + 0.01 + 0.028 * strikeReadiness + safeAction(action, 2);
    } else {
      targetPosition[0] = this.racketAnchor[0] + safeAction(action, 0);
      targetPosition[1] = this.racketAnchor[1] + safeAction(action, 1);
      targetPosition[2] = this.racketAnchor[2] - 0.01 + safeAction(action, 2);
    }

    const correctionX = this.racketAnchor[0] - contactPosition[0];
    const correctionY = this.racketAnchor[1] - contactPosition[1];
    const centeringTilt: [number, number] = [
      clamp(0.7 * correctionX, -0.06, 0.06),
      clamp(-0.7 * correctionY, -0.06, 0.06)
    ];
    const targetTilt: [number, number] = [
      clamp(centeringTilt[0] * strikeReadiness + safeAction(action, 3), -0.12, 0.12),
      clamp(centeringTilt[1] * strikeReadiness + safeAction(action, 4), -0.12, 0.12)
    ];

    const desiredVelocity = this.desiredOutgoingVelocity(contactPosition);
    const requiredRacketVz = (desiredVelocity[2] + 0.8 * Math.min(ballVelocity[2], 0)) / 1.8;
    const interceptVelocity: Vec3 =
      descending && interceptTime > 0
        ? [
            clamp((contactPosition[0] - racketPosition[0]) / Math.max(interceptTime, 0.08), -1.2, 1.2),
            clamp((contactPosition[1] - racketPosition[1]) / Math.max(interceptTime, 0.08), -1.2, 1.2),
            0
          ]
        : [0, 0, 0];
    const targetVelocity = descending
      ? clipVec3Norm(
          [
            interceptVelocity[0] + safeAction(action, 5) + safeAction(action, 11),
            interceptVelocity[1] + safeAction(action, 6) + safeAction(action, 12),
            strikeReadiness * requiredRacketVz + safeAction(action, 7) + safeAction(action, 8)
          ],
          3.2
        )
      : null;

    return {
      targetPosition,
      targetTilt,
      targetVelocity
    };
  }

  private buildPolicyObservation(): Float64Array {
    const observationSize = this.policy?.observationSize ?? 55;
    const observation = new Float64Array(observationSize);
    let offset = 0;
    const push = (values: ArrayLike<number>) => {
      for (let index = 0; index < values.length && offset < observation.length; index += 1) {
        observation[offset] = Number.isFinite(Number(values[index])) ? Number(values[index]) : 0;
        offset += 1;
      }
    };

    const ballPosition = this.ballPosition();
    const ballVelocity = this.ballVelocity();
    const racketPosition = this.controller?.racketPosition() ?? this.racketAnchor;
    const racketVelocity = this.controller?.racketVelocity() ?? [0, 0, 0];
    const targetPosition = this.controller?.targetPosition ?? racketPosition;
    const targetTilt = this.controller?.targetTilt ?? [0, 0];
    const faceNormal = this.controller?.racketFaceNormal() ?? [0, 0, 1];
    const interceptTime = this.predictedInterceptTime(0.35);
    const predictedXY = [
      ballPosition[0] + interceptTime * ballVelocity[0] - racketPosition[0],
      ballPosition[1] + interceptTime * ballVelocity[1] - racketPosition[1]
    ];
    const nextIntercept = this.nextInterceptMetrics();

    push(Array.from({ length: 7 }, (_, index) => Number(this.data?.qpos[index] ?? 0)));
    push(Array.from({ length: 7 }, (_, index) => Number(this.data?.qvel[index] ?? 0)));
    push(racketPosition);
    push(racketVelocity);
    push(targetPosition);
    push(ballPosition);
    push(ballVelocity);
    push(subVec3(ballPosition, racketPosition));
    push(predictedXY);
    push([interceptTime]);
    push(this.phaseOneHot(ballPosition, racketPosition, ballVelocity));
    push([this.timeSinceContact(), Math.min(this.contactCount, 3)]);
    push([
      ...nextIntercept.relativeXY,
      nextIntercept.time,
      nextIntercept.reachable ? 1 : 0,
      nextIntercept.recoveryDistance,
      nextIntercept.recoveryReadiness
    ]);
    push(this.desiredOutgoingVelocity(ballPosition));
    push(faceNormal);
    push(targetTilt);

    return observation;
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

  private updateContactState(): ContactState {
    if (!this.data || !this.ids) {
      return { racket: false, floor: false };
    }

    let racket = false;
    let floor = false;
    const contacts = this.data.contact;
    try {
      for (let index = 0; index < this.data.ncon; index += 1) {
        const contact = contacts.get(index);
        if (!contact) {
          continue;
        }

        if (this.isBallFloorContact(contact.geom1, contact.geom2)) {
          floor = true;
          continue;
        }

        if (this.isBallRacketContact(contact.geom1, contact.geom2)) {
          racket = true;
          if (!this.contactActive) {
            this.contactCount += 1;
            this.lastContact = {
              position: arrayVec3(contact.pos, 0),
              time: this.data.time
            };
          }
        }
      }
    } finally {
      contacts.delete();
    }

    this.contactActive = racket;
    return { racket, floor };
  }

  private isBallRacketContact(geom1: number, geom2: number): boolean {
    if (!this.ids) {
      return false;
    }
    return (
      (geom1 === this.ids.ballGeom && this.ids.racketGeomIds.has(geom2)) ||
      (geom2 === this.ids.ballGeom && this.ids.racketGeomIds.has(geom1))
    );
  }

  private isBallFloorContact(geom1: number, geom2: number): boolean {
    if (!this.ids) {
      return false;
    }
    return (
      (geom1 === this.ids.ballGeom && geom2 === this.ids.floorGeom) ||
      (geom2 === this.ids.ballGeom && geom1 === this.ids.floorGeom)
    );
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

    this.reset(this.spawnPosition);
  }

  private resetContactMetrics(): void {
    this.contactCount = 0;
    this.lastContact = null;
    this.contactActive = false;
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

  private predictedInterceptTime(maxInterceptTime: number): number {
    const ballPosition = this.ballPosition();
    const ballVelocity = this.ballVelocity();
    const targetZ = (this.controller?.racketPosition() ?? this.racketAnchor)[2] + TRACKING_STRIKE_PLANE_OFFSET;
    const times = ballisticInterceptTimes(ballPosition[2], ballVelocity[2], targetZ, maxInterceptTime);
    return times.length > 0 ? times[0] : 0;
  }

  private predictedContactPosition(interceptTime: number): Vec3 {
    const ballPosition = this.ballPosition();
    const ballVelocity = this.ballVelocity();
    return [
      ballPosition[0] + interceptTime * ballVelocity[0],
      ballPosition[1] + interceptTime * ballVelocity[1],
      ballPosition[2] + interceptTime * ballVelocity[2] + 0.5 * GRAVITY_Z * interceptTime * interceptTime
    ];
  }

  private nextInterceptMetrics(): NextInterceptMetrics {
    const ballPosition = this.ballPosition();
    const ballVelocity = this.ballVelocity();
    const racketPosition = this.controller?.racketPosition() ?? this.racketAnchor;
    const targetZ = this.racketAnchor[2] + TRACKING_STRIKE_PLANE_OFFSET;
    const descendingTimes = ballisticInterceptTimes(ballPosition[2], ballVelocity[2], targetZ, NEXT_INTERCEPT_MAX_TIME)
      .filter((timeValue) => ballVelocity[2] + GRAVITY_Z * timeValue < 0);
    if (descendingTimes.length === 0) {
      return {
        time: 0,
        relativeXY: [0, 0],
        reachable: false,
        recoveryDistance: 0,
        recoveryReadiness: 0,
        easyScore: 0
      };
    }

    const time = descendingTimes[0];
    const interceptXY: [number, number] = [
      ballPosition[0] + time * ballVelocity[0],
      ballPosition[1] + time * ballVelocity[1]
    ];
    const anchorError = Math.hypot(interceptXY[0] - this.racketAnchor[0], interceptXY[1] - this.racketAnchor[1]);
    const recoveryDistance = Math.hypot(interceptXY[0] - racketPosition[0], interceptXY[1] - racketPosition[1]);
    const reachable = anchorError <= NEXT_INTERCEPT_SUCCESS_RADIUS;
    const recoverySpeedLimit = 0.06 / CONTROL_DT;
    const requiredRecoverySpeed = recoveryDistance / Math.max(time, CONTROL_DT);
    const speedReadiness = 1 - clamp(requiredRecoverySpeed / recoverySpeedLimit, 0, 1);
    const zoneReadiness = 1 - clamp(anchorError / NEXT_INTERCEPT_SUCCESS_RADIUS, 0, 1);
    const verticalSpeed = ballVelocity[2] + GRAVITY_Z * time;
    const speedNorm = Math.hypot(ballVelocity[0], ballVelocity[1], verticalSpeed);
    const lateralSpeed = Math.hypot(ballVelocity[0], ballVelocity[1]);
    const xyScore = Math.max(1 - anchorError / NEXT_INTERCEPT_SUCCESS_RADIUS, 0);
    const timeScore = Math.max(1 - Math.abs(time - EASY_NEXT_BALL_TARGET_TIME) / EASY_NEXT_BALL_TIME_TOLERANCE, 0);
    const descendingScore = Math.max(
      1 - Math.abs(Math.abs(verticalSpeed) - EASY_NEXT_BALL_TARGET_DESCENDING_SPEED) / EASY_NEXT_BALL_TARGET_DESCENDING_SPEED,
      0
    );
    const easyScore =
      xyScore +
      0.75 * timeScore +
      0.5 * descendingScore -
      0.5 * clamp(lateralSpeed / EASY_NEXT_BALL_MAX_LATERAL_SPEED, 0, 1) -
      0.25 * clamp(Math.max(speedNorm - EASY_NEXT_BALL_SOFT_SPEED_LIMIT, 0) / EASY_NEXT_BALL_SOFT_SPEED_LIMIT, 0, 1) -
      0.5 * clamp(recoveryDistance / (1.5 * NEXT_INTERCEPT_SUCCESS_RADIUS), 0, 1);

    return {
      time,
      relativeXY: [interceptXY[0] - racketPosition[0], interceptXY[1] - racketPosition[1]],
      reachable,
      recoveryDistance,
      recoveryReadiness: clamp(0.5 * speedReadiness + 0.5 * zoneReadiness, 0, 1),
      easyScore
    };
  }

  private desiredOutgoingVelocity(contactPosition: Vec3): Vec3 {
    const targetApexZ = this.racketAnchor[2] + TARGET_BALL_HEIGHT;
    const heightDelta = Math.max(targetApexZ - contactPosition[2], 0.01);
    const desiredVelocityZ = Math.sqrt(2 * Math.abs(GRAVITY_Z) * heightDelta);
    const timeToApex = desiredVelocityZ / Math.abs(GRAVITY_Z);
    const descentHeight = Math.max(targetApexZ - (this.racketAnchor[2] + TRACKING_STRIKE_PLANE_OFFSET), 0.01);
    const desiredXYTime = timeToApex + Math.sqrt((2 * descentHeight) / Math.abs(GRAVITY_Z));
    return [
      (this.racketAnchor[0] - contactPosition[0]) / Math.max(desiredXYTime, 1.0e-6),
      (this.racketAnchor[1] - contactPosition[1]) / Math.max(desiredXYTime, 1.0e-6),
      desiredVelocityZ
    ];
  }

  private phaseOneHot(ballPosition: Vec3, racketPosition: Vec3, ballVelocity: Vec3): [number, number, number, number] {
    const timeSinceContact = this.timeSinceContact();
    const recentContact = timeSinceContact > 0 && timeSinceContact <= NEXT_INTERCEPT_MAX_TIME;
    const phase = [0, 0, 0, 0] as [number, number, number, number];
    if (recentContact) {
      phase[ballVelocity[2] > 0 ? 2 : 3] = 1;
      return phase;
    }

    if (
      ballVelocity[2] < DESCENDING_BALL_VELOCITY_THRESHOLD &&
      this.preContactHeightReadiness(ballPosition, racketPosition, ballVelocity) >= 0.95
    ) {
      phase[1] = 1;
      return phase;
    }

    phase[0] = 1;
    return phase;
  }

  private preContactHeightReadiness(ballPosition: Vec3, racketPosition: Vec3, ballVelocity: Vec3): number {
    if (ballVelocity[2] >= DESCENDING_BALL_VELOCITY_THRESHOLD) {
      return 0;
    }

    const ballHeight = ballPosition[2] - racketPosition[2];
    const preparationHeight = Math.max(TRACKING_STRIKE_PLANE_OFFSET - HEIGHT_TOLERANCE * 0.3, 0.04);
    const activationHeight = clamp(preparationHeight + 0.08, 0.16, 0.22);
    if (ballHeight >= activationHeight) {
      return 0;
    }
    return 1 - clamp((ballHeight - preparationHeight) / Math.max(activationHeight - preparationHeight, 1.0e-6), 0, 1);
  }

  private timeSinceContact(): number {
    if (!this.data || !this.lastContact) {
      return 0;
    }
    return Math.max(this.data.time - this.lastContact.time, 0);
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
      racketPosition: this.controller?.racketPosition() ?? arrayVec3(this.data.site_xpos, this.ids.racketSite * 3),
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

    if (this.fallbackBallPosition[2] < 0.05) {
      this.fallbackBallPosition = [...this.spawnPosition] as Vec3;
      this.fallbackBallVelocity = [0, 0, 0];
      this.resetContactMetrics();
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

function ballisticInterceptTimes(ballZ: number, ballVelocityZ: number, targetZ: number, maxInterceptTime: number): number[] {
  const a = 0.5 * GRAVITY_Z;
  const b = ballVelocityZ;
  const c = ballZ - targetZ;
  const times: number[] = [];
  if (Math.abs(a) < 1.0e-9) {
    if (Math.abs(b) > 1.0e-9) {
      times.push(-c / b);
    }
  } else {
    const discriminant = b * b - 4 * a * c;
    if (discriminant >= 0) {
      const sqrtDiscriminant = Math.sqrt(discriminant);
      const denominator = 2 * a;
      times.push((-b - sqrtDiscriminant) / denominator, (-b + sqrtDiscriminant) / denominator);
    }
  }
  return times.filter((time) => time >= 1.0e-6 && time <= maxInterceptTime).sort((left, right) => left - right);
}

function arrayVec3(arrayLike: ArrayLike<number>, offset: number): Vec3 {
  return [
    Number(arrayLike[offset] ?? 0),
    Number(arrayLike[offset + 1] ?? 0),
    Number(arrayLike[offset + 2] ?? 0)
  ];
}

function subVec3(left: Vec3, right: Vec3): Vec3 {
  return [left[0] - right[0], left[1] - right[1], left[2] - right[2]];
}

function clipVec3Norm(vector: Vec3, maxNorm: number): Vec3 {
  const norm = Math.hypot(vector[0], vector[1], vector[2]);
  if (norm <= maxNorm || norm <= 1.0e-9) {
    return vector;
  }
  const scale = maxNorm / norm;
  return [vector[0] * scale, vector[1] * scale, vector[2] * scale];
}

function safeAction(action: ArrayLike<number>, index: number): number {
  const value = Number(action[index] ?? 0);
  return Number.isFinite(value) ? value : 0;
}

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}
