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
const STRIKE_ZONE_XY_RADIUS = 0.1;
const CONTACT_CENTERING_RADIUS = 0.04;
const NEXT_INTERCEPT_MAX_TIME = 1.25;
const NEXT_INTERCEPT_SUCCESS_RADIUS = 0.04;
const EASY_NEXT_BALL_TARGET_TIME = 0.45;
const EASY_NEXT_BALL_TIME_TOLERANCE = 0.3;
const EASY_NEXT_BALL_TARGET_DESCENDING_SPEED = 1.25;
const EASY_NEXT_BALL_MAX_LATERAL_SPEED = 1.0;
const EASY_NEXT_BALL_SOFT_SPEED_LIMIT = 3.0;
const POST_CONTACT_RETURN_Z_OFFSET = -0.01;
const CONTACT_FRAME_BASE_STRIKE_Z_BOOST = 0.028;
const CONTACT_FRAME_BASE_STRIKE_Z_OFFSET = 0.01;
const CONTACT_FRAME_BASE_STRIKE_TIME_HORIZON = 0.18;
const CONTACT_FRAME_APEX_LIFT_GAIN = 0.12;
const CONTACT_FRAME_APEX_LIFT_MAX = 0.11;
const CONTACT_FRAME_APEX_LIFT_RESTITUTION = 0.8;
const CONTACT_FRAME_APEX_LIFT_REFERENCE_VELOCITY_Z = -1.0;
const CONTACT_FRAME_VELOCITY_LEAD_GAIN = 0.04;
const CONTACT_FRAME_VELOCITY_LEAD_MAX = 0.025;
const CONTACT_FRAME_VELOCITY_TARGET_MAX = 3.2;
const CONTACT_FRAME_INTERCEPT_VELOCITY_GAIN = 0.65;
const CONTACT_FRAME_INTERCEPT_VELOCITY_MAX = 1.2;
const CONTACT_FRAME_INTERCEPT_VELOCITY_TIME_FLOOR = 0.08;
const CONTACT_FRAME_PLANNER_MIN_INTERCEPT_TIME = 0.03;
const CONTACT_FRAME_PLANNER_MAX_INTERCEPT_TIME = 0.6;
const CONTACT_FRAME_STRIKE_HOLD_TIME = 0.08;
const CONTACT_FRAME_STRIKE_HOLD_MIN_READINESS = 0.6;
const CONTACT_FRAME_FOLLOWTHROUGH_GAIN = 1.0;
const CONTACT_FRAME_FOLLOWTHROUGH_TIME = 0.06;
const CONTACT_FRAME_FOLLOWTHROUGH_MAX = 0.055;
const CONTACT_FRAME_LATERAL_BRAKE_GAIN = 0.65;
const CONTACT_FRAME_LATERAL_BRAKE_MAX = 0.25;
const CONTACT_FRAME_LATERAL_BRAKE_RADIUS = 0.12;
const CONTACT_FRAME_TRAJECTORY_TILT_GAIN = 0.7;
const CONTACT_FRAME_TRAJECTORY_TILT_LIMIT: [number, number] = [0.06, 0.06];
const CONTACT_FRAME_TILT_RAMP_TIME = 0.45;
const CONTACT_FRAME_CENTERING_TILT_LIMIT: [number, number] = [0.045, 0.045];
const CONTACT_FRAME_CENTERING_TILT_RADIUS = 0.12;
const CONTACT_FRAME_CENTERING_TILT_DEADBAND = 0.008;
const TARGET_OFFSET_LOW_Z = -0.04;
const TARGET_OFFSET_HIGH_Z = 0.12;
const TARGET_TILT_LIMIT: [number, number] = [0.12, 0.12];
const MIN_DESIRED_APEX_HEIGHT_DELTA = 0.01;

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

type ContactFramePlan = {
  interceptTime: number;
  contactPosition: Vec3;
  desiredVelocity: Vec3;
  desiredTimeToApex: number;
  targetXY: [number, number];
  strikeHoldActive: boolean;
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
      this.spawnPosition = [...ballPosition] as Vec3;
      this.fallbackBallPosition = this.resolveFallbackBallPosition(ballPosition);
      this.fallbackBallVelocity = [0, 0, 0];
      this.resetContactMetrics();
      return this.fallbackSnapshot();
    }

    this.module.mj_resetDataKeyframe(this.model, this.data, 0);
    this.applyHomeCtrl();
    this.module.mj_forward(this.model, this.data);
    this.controller?.reset();
    this.racketAnchor = this.controller?.racketPosition() ?? arrayVec3(this.data.site_xpos, this.ids.racketSite * 3);
    this.spawnPosition = [...ballPosition] as Vec3;
    this.spawnBall(this.resolveBallSpawnPosition(ballPosition), [0, 0, 0]);
    this.module.mj_forward(this.model, this.data);
    this.controlAccumulator = CONTROL_DT;
    this.resetContactMetrics();
    return this.snapshot();
  }

  setBallPosition(ballPosition: Vec3): SimulationSnapshot {
    if (!this.module || !this.model || !this.data || !this.ids) {
      this.spawnPosition = [...ballPosition] as Vec3;
      this.fallbackBallPosition = this.resolveFallbackBallPosition(ballPosition);
      this.fallbackBallVelocity = [0, 0, 0];
      this.resetContactMetrics();
      return this.fallbackSnapshot();
    }

    this.spawnPosition = [...ballPosition] as Vec3;
    this.spawnBall(this.resolveBallSpawnPosition(ballPosition), [0, 0, 0]);
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
    const descending = ballVelocity[2] < DESCENDING_BALL_VELOCITY_THRESHOLD;
    const strikePlaneOffset = this.trackingStrikePlaneOffset(action);
    const plan = this.contactFramePlan(action, strikePlaneOffset);
    const interceptTime = plan?.interceptTime ?? this.predictedInterceptTime(0.35, strikePlaneOffset);
    const contactPosition = plan?.contactPosition ?? this.predictedContactPosition(interceptTime);
    const [radial, tangent] = this.contactFrameBasis(plan, strikePlaneOffset);
    const contactOffsetXY: [number, number] = [
      radial[0] * safeAction(action, 0) + tangent[0] * safeAction(action, 1),
      radial[1] * safeAction(action, 0) + tangent[1] * safeAction(action, 1)
    ];

    let targetPosition: Vec3;
    if (descending) {
      const liftTarget =
        this.strikeLiftFeedforward(interceptTime) +
        this.contactFrameBaseStrikeLift(action, plan, contactPosition, strikePlaneOffset);
      targetPosition = [
        contactPosition[0] + contactOffsetXY[0],
        contactPosition[1] + contactOffsetXY[1],
        (plan ? contactPosition[2] : this.racketAnchor[2]) + liftTarget + safeAction(action, 2)
      ];
      targetPosition = addVec3(
        targetPosition,
        this.contactFrameFollowthroughOffset(action, plan, contactPosition, strikePlaneOffset)
      );
    } else {
      targetPosition = [
        this.racketAnchor[0] + contactOffsetXY[0],
        this.racketAnchor[1] + contactOffsetXY[1],
        this.racketAnchor[2] + POST_CONTACT_RETURN_Z_OFFSET + safeAction(action, 2)
      ];
    }

    const desiredVelocity = this.contactFrameControllerDesiredVelocity(action, plan, contactPosition, strikePlaneOffset);
    const targetTilt = this.contactFrameTargetTilt(action, plan, contactPosition, desiredVelocity);
    const targetVelocity = descending
      ? this.contactFrameVelocityTarget(action, plan, targetPosition, targetTilt, desiredVelocity, racketPosition)
      : null;

    return {
      targetPosition,
      targetTilt,
      targetVelocity
    };
  }

  private trackingStrikePlaneOffset(action?: ArrayLike<number>): number {
    return clamp(
      TRACKING_STRIKE_PLANE_OFFSET + (action ? safeAction(action, 14) : 0),
      TARGET_OFFSET_LOW_Z,
      TARGET_OFFSET_HIGH_Z
    );
  }

  private contactFramePlan(action: ArrayLike<number>, strikePlaneOffset: number): ContactFramePlan | null {
    const ballPosition = this.ballPosition();
    const ballVelocity = this.ballVelocity();
    if (ballVelocity[2] >= DESCENDING_BALL_VELOCITY_THRESHOLD) {
      return null;
    }

    const targetZ = this.racketAnchor[2] + strikePlaneOffset;
    const descendingTimes = ballisticInterceptTimes(
      ballPosition[2],
      ballVelocity[2],
      targetZ,
      CONTACT_FRAME_PLANNER_MAX_INTERCEPT_TIME
    ).filter(
      (timeValue) =>
        timeValue >= CONTACT_FRAME_PLANNER_MIN_INTERCEPT_TIME &&
        ballVelocity[2] + GRAVITY_Z * timeValue < 0
    );
    if (descendingTimes.length === 0) {
      return null;
    }

    const interceptTime = descendingTimes[0];
    const contactPosition: Vec3 = [
      ballPosition[0] + interceptTime * ballVelocity[0],
      ballPosition[1] + interceptTime * ballVelocity[1],
      targetZ
    ];
    const targetApexZ = this.racketAnchor[2] + TARGET_BALL_HEIGHT + safeAction(action, 13);
    const [desiredVelocity, desiredTimeToApex, targetXY] = this.desiredOutgoingVelocity(
      contactPosition,
      targetApexZ,
      [this.racketAnchor[0], this.racketAnchor[1]],
      strikePlaneOffset
    );
    const readiness = this.preContactReadiness(ballPosition, this.controller?.racketPosition() ?? this.racketAnchor, ballVelocity);
    const strikeHoldActive =
      interceptTime <= CONTACT_FRAME_STRIKE_HOLD_TIME && readiness >= CONTACT_FRAME_STRIKE_HOLD_MIN_READINESS;

    return {
      interceptTime,
      contactPosition,
      desiredVelocity,
      desiredTimeToApex,
      targetXY,
      strikeHoldActive
    };
  }

  private contactFrameBasis(plan: ContactFramePlan | null, strikePlaneOffset: number): [[number, number], [number, number]] {
    const ballPosition = this.ballPosition();
    const ballVelocity = this.ballVelocity();
    const interceptTime = plan?.interceptTime ?? this.predictedInterceptTime(0.35, strikePlaneOffset);
    const interceptXY: [number, number] = plan
      ? [plan.contactPosition[0], plan.contactPosition[1]]
      : [ballPosition[0] + interceptTime * ballVelocity[0], ballPosition[1] + interceptTime * ballVelocity[1]];
    let radial: [number, number] = [this.racketAnchor[0] - interceptXY[0], this.racketAnchor[1] - interceptXY[1]];
    let radialNorm = Math.hypot(radial[0], radial[1]);
    if (radialNorm <= 1.0e-6) {
      radial = [-ballVelocity[0], -ballVelocity[1]];
      radialNorm = Math.hypot(radial[0], radial[1]);
    }
    if (radialNorm <= 1.0e-6) {
      radial = [this.racketAnchor[0] - ballPosition[0], this.racketAnchor[1] - ballPosition[1]];
      radialNorm = Math.hypot(radial[0], radial[1]);
    }
    if (radialNorm <= 1.0e-6) {
      radial = [1, 0];
      radialNorm = 1;
    }
    radial = [radial[0] / radialNorm, radial[1] / radialNorm];
    return [radial, [-radial[1], radial[0]]];
  }

  private strikeLiftFeedforward(interceptTime: number): number {
    const ballPosition = this.ballPosition();
    const ballVelocity = this.ballVelocity();
    const heightReadiness = this.preContactHeightReadiness(ballPosition, this.controller?.racketPosition() ?? this.racketAnchor, ballVelocity);
    if (heightReadiness <= 0) {
      return 0;
    }
    const urgency = 1 - clamp(interceptTime / 0.12, 0, 1);
    return clamp(0.04 * heightReadiness * urgency, 0, 0.04);
  }

  private contactFrameBaseStrikeLift(
    action: ArrayLike<number>,
    plan: ContactFramePlan | null,
    contactPosition: Vec3,
    strikePlaneOffset: number
  ): number {
    const ballPosition = this.ballPosition();
    const ballVelocity = this.ballVelocity();
    if (ballVelocity[2] >= DESCENDING_BALL_VELOCITY_THRESHOLD) {
      return 0;
    }
    const interceptTime = plan?.interceptTime ?? this.predictedInterceptTime(0.35, strikePlaneOffset);
    const urgency = 1 - clamp(interceptTime / CONTACT_FRAME_BASE_STRIKE_TIME_HORIZON, 0, 1);
    const strikeReadiness = Math.max(
      this.preContactHeightReadiness(ballPosition, this.controller?.racketPosition() ?? this.racketAnchor, ballVelocity),
      urgency
    );
    return (
      CONTACT_FRAME_BASE_STRIKE_Z_OFFSET +
      CONTACT_FRAME_BASE_STRIKE_Z_BOOST * clamp(strikeReadiness, 0, 1) +
      this.contactFrameApexLift(action, plan, contactPosition, strikePlaneOffset) +
      this.contactFrameVelocityLead(action, plan, contactPosition, strikePlaneOffset)
    );
  }

  private contactFrameApexLift(
    action: ArrayLike<number>,
    plan: ContactFramePlan | null,
    contactPosition: Vec3,
    strikePlaneOffset: number
  ): number {
    const ballVelocity = this.ballVelocity();
    if (ballVelocity[2] >= DESCENDING_BALL_VELOCITY_THRESHOLD) {
      return 0;
    }
    const desiredVelocity = this.contactFrameControllerDesiredVelocity(action, plan, contactPosition, strikePlaneOffset);
    const nominalContactPosition: Vec3 = [
      this.racketAnchor[0],
      this.racketAnchor[1],
      this.racketAnchor[2] + strikePlaneOffset
    ];
    const [nominalDesiredVelocity] = this.desiredOutgoingVelocity(nominalContactPosition);
    const requiredRacketVelocityZ =
      (desiredVelocity[2] + CONTACT_FRAME_APEX_LIFT_RESTITUTION * Math.min(ballVelocity[2], 0)) /
      (1 + CONTACT_FRAME_APEX_LIFT_RESTITUTION);
    const nominalRequiredRacketVelocityZ =
      (nominalDesiredVelocity[2] +
        CONTACT_FRAME_APEX_LIFT_RESTITUTION * CONTACT_FRAME_APEX_LIFT_REFERENCE_VELOCITY_Z) /
      (1 + CONTACT_FRAME_APEX_LIFT_RESTITUTION);
    const velocityExcess = Math.max(requiredRacketVelocityZ - nominalRequiredRacketVelocityZ, 0);
    const interceptTime = plan?.interceptTime ?? this.predictedInterceptTime(0.35, strikePlaneOffset);
    const urgency = 1 - clamp(interceptTime / CONTACT_FRAME_BASE_STRIKE_TIME_HORIZON, 0, 1);
    const strikeReadiness = Math.max(
      this.preContactHeightReadiness(this.ballPosition(), this.controller?.racketPosition() ?? this.racketAnchor, ballVelocity),
      urgency
    );
    return clamp(CONTACT_FRAME_APEX_LIFT_GAIN * velocityExcess * strikeReadiness, 0, CONTACT_FRAME_APEX_LIFT_MAX);
  }

  private contactFrameVelocityLead(
    action: ArrayLike<number>,
    plan: ContactFramePlan | null,
    contactPosition: Vec3,
    strikePlaneOffset: number
  ): number {
    const requiredVelocityZ = this.requiredContactFrameRacketVelocity(
      this.contactFrameControllerDesiredVelocity(action, plan, contactPosition, strikePlaneOffset),
      [0, 0]
    )[2];
    const velocityErrorZ = requiredVelocityZ - (this.controller?.racketVelocity() ?? [0, 0, 0])[2];
    const interceptTime = plan?.interceptTime ?? this.predictedInterceptTime(0.35, strikePlaneOffset);
    const urgency = 1 - clamp(interceptTime / CONTACT_FRAME_BASE_STRIKE_TIME_HORIZON, 0, 1);
    const strikeReadiness = Math.max(
      this.preContactHeightReadiness(this.ballPosition(), this.controller?.racketPosition() ?? this.racketAnchor, this.ballVelocity()),
      urgency
    );
    return clamp(CONTACT_FRAME_VELOCITY_LEAD_GAIN * velocityErrorZ * strikeReadiness, -CONTACT_FRAME_VELOCITY_LEAD_MAX, CONTACT_FRAME_VELOCITY_LEAD_MAX);
  }

  private contactFrameControllerDesiredVelocity(
    action: ArrayLike<number>,
    plan: ContactFramePlan | null,
    contactPosition: Vec3,
    strikePlaneOffset: number
  ): Vec3 {
    const targetApexZ = this.racketAnchor[2] + TARGET_BALL_HEIGHT + safeAction(action, 13);
    const [baseVelocity] = plan
      ? [plan.desiredVelocity, plan.desiredTimeToApex, plan.targetXY]
      : this.desiredOutgoingVelocity(contactPosition, targetApexZ, [this.racketAnchor[0], this.racketAnchor[1]], strikePlaneOffset);
    return [
      baseVelocity[0] + safeAction(action, 6),
      baseVelocity[1] + safeAction(action, 7),
      baseVelocity[2] * Math.max(0, 1 + safeAction(action, 5))
    ];
  }

  private contactFrameTargetTilt(
    action: ArrayLike<number>,
    plan: ContactFramePlan | null,
    contactPosition: Vec3,
    desiredVelocity: Vec3
  ): [number, number] {
    const ballVelocity = this.ballVelocity();
    if (ballVelocity[2] >= DESCENDING_BALL_VELOCITY_THRESHOLD) {
      return [
        clamp(safeAction(action, 3), -TARGET_TILT_LIMIT[0], TARGET_TILT_LIMIT[0]),
        clamp(safeAction(action, 4), -TARGET_TILT_LIMIT[1], TARGET_TILT_LIMIT[1])
      ];
    }

    const interceptTime = plan?.interceptTime ?? this.predictedInterceptTime(0.35);
    const urgency = 1 - clamp(interceptTime / CONTACT_FRAME_TILT_RAMP_TIME, 0, 1);
    const ramp = clamp(
      Math.max(
        this.preContactHeightReadiness(this.ballPosition(), this.controller?.racketPosition() ?? this.racketAnchor, ballVelocity),
        urgency
      ),
      0,
      1
    );

    const impulse = subVec3(desiredVelocity, ballVelocity);
    const impulseNorm = Math.hypot(impulse[0], impulse[1], impulse[2]);
    let trajectoryTilt: [number, number] = [0, 0];
    if (impulseNorm > 1.0e-9 && impulse[2] > 0) {
      const topNormal: Vec3 = [impulse[0] / impulseNorm, impulse[1] / impulseNorm, impulse[2] / impulseNorm];
      const trajectoryScale = Math.max(0, 1 + safeAction(action, 9));
      trajectoryTilt = [
        clamp(
          CONTACT_FRAME_TRAJECTORY_TILT_GAIN * trajectoryScale * Math.asin(clamp(topNormal[0], -0.95, 0.95)) * ramp,
          -CONTACT_FRAME_TRAJECTORY_TILT_LIMIT[0],
          CONTACT_FRAME_TRAJECTORY_TILT_LIMIT[0]
        ),
        clamp(
          -CONTACT_FRAME_TRAJECTORY_TILT_GAIN * trajectoryScale * Math.asin(clamp(topNormal[1], -0.95, 0.95)) * ramp,
          -CONTACT_FRAME_TRAJECTORY_TILT_LIMIT[1],
          CONTACT_FRAME_TRAJECTORY_TILT_LIMIT[1]
        )
      ];
    }

    const correctionX = this.racketAnchor[0] - contactPosition[0];
    const correctionY = this.racketAnchor[1] - contactPosition[1];
    const centeringScale = Math.max(0, 1 + safeAction(action, 10));
    const centeringTilt: [number, number] = [
      this.centeringTiltAxis(correctionX, CONTACT_FRAME_CENTERING_TILT_LIMIT[0]) * centeringScale * ramp,
      -this.centeringTiltAxis(correctionY, CONTACT_FRAME_CENTERING_TILT_LIMIT[1]) * centeringScale * ramp
    ];

    return [
      clamp(trajectoryTilt[0] + centeringTilt[0] + safeAction(action, 3), -TARGET_TILT_LIMIT[0], TARGET_TILT_LIMIT[0]),
      clamp(trajectoryTilt[1] + centeringTilt[1] + safeAction(action, 4), -TARGET_TILT_LIMIT[1], TARGET_TILT_LIMIT[1])
    ];
  }

  private contactFrameVelocityTarget(
    action: ArrayLike<number>,
    plan: ContactFramePlan | null,
    targetPosition: Vec3,
    targetTilt: [number, number],
    desiredVelocity: Vec3,
    racketPosition: Vec3
  ): Vec3 {
    const ballVelocity = this.ballVelocity();
    const interceptTime = plan?.interceptTime ?? this.predictedInterceptTime(NEXT_INTERCEPT_MAX_TIME);
    let targetVelocity: Vec3 = [0, 0, 0];
    if (!plan?.strikeHoldActive && interceptTime > 0) {
      targetVelocity = clipVec3Norm(
        [
          (CONTACT_FRAME_INTERCEPT_VELOCITY_GAIN * (targetPosition[0] - racketPosition[0])) /
            Math.max(interceptTime, CONTACT_FRAME_INTERCEPT_VELOCITY_TIME_FLOOR),
          (CONTACT_FRAME_INTERCEPT_VELOCITY_GAIN * (targetPosition[1] - racketPosition[1])) /
            Math.max(interceptTime, CONTACT_FRAME_INTERCEPT_VELOCITY_TIME_FLOOR),
          (CONTACT_FRAME_INTERCEPT_VELOCITY_GAIN * (targetPosition[2] - racketPosition[2])) /
            Math.max(interceptTime, CONTACT_FRAME_INTERCEPT_VELOCITY_TIME_FLOOR)
        ],
        CONTACT_FRAME_INTERCEPT_VELOCITY_MAX
      );
    }

    const urgency = 1 - clamp(interceptTime / CONTACT_FRAME_BASE_STRIKE_TIME_HORIZON, 0, 1);
    const strikeReadiness = clamp(
      Math.max(
        this.preContactHeightReadiness(this.ballPosition(), this.controller?.racketPosition() ?? this.racketAnchor, ballVelocity),
        urgency
      ),
      0,
      1
    );
    const requiredVelocity = this.requiredContactFrameRacketVelocity(desiredVelocity, targetTilt);
    targetVelocity = addVec3(targetVelocity, scaleVec3(requiredVelocity, strikeReadiness));
    targetVelocity[0] += safeAction(action, 11);
    targetVelocity[1] += safeAction(action, 12);
    targetVelocity[2] += safeAction(action, 8);
    targetVelocity = addVec3(targetVelocity, this.contactFrameLateralBrakeVelocity(targetPosition));
    return clipVec3Norm(targetVelocity, CONTACT_FRAME_VELOCITY_TARGET_MAX);
  }

  private contactFrameFollowthroughOffset(
    action: ArrayLike<number>,
    plan: ContactFramePlan | null,
    contactPosition: Vec3,
    strikePlaneOffset: number
  ): Vec3 {
    const ballVelocity = this.ballVelocity();
    if (ballVelocity[2] >= DESCENDING_BALL_VELOCITY_THRESHOLD) {
      return [0, 0, 0];
    }
    const desiredVelocity = this.contactFrameControllerDesiredVelocity(action, plan, contactPosition, strikePlaneOffset);
    const requiredVelocity = this.requiredContactFrameRacketVelocity(desiredVelocity, [safeAction(action, 3), safeAction(action, 4)]);
    const interceptTime = plan?.interceptTime ?? this.predictedInterceptTime(0.35, strikePlaneOffset);
    const urgency = 1 - clamp(interceptTime / CONTACT_FRAME_BASE_STRIKE_TIME_HORIZON, 0, 1);
    const strikeReadiness = clamp(
      Math.max(
        this.preContactHeightReadiness(this.ballPosition(), this.controller?.racketPosition() ?? this.racketAnchor, ballVelocity),
        urgency
      ),
      0,
      1
    );
    return clipVec3Norm(
      scaleVec3(requiredVelocity, CONTACT_FRAME_FOLLOWTHROUGH_GAIN * strikeReadiness * CONTACT_FRAME_FOLLOWTHROUGH_TIME),
      CONTACT_FRAME_FOLLOWTHROUGH_MAX
    );
  }

  private contactFrameLateralBrakeVelocity(targetPosition: Vec3): Vec3 {
    const ballVelocity = this.ballVelocity();
    if (ballVelocity[2] >= DESCENDING_BALL_VELOCITY_THRESHOLD) {
      return [0, 0, 0];
    }

    const outwardXY: [number, number] = [
      targetPosition[0] - this.racketAnchor[0],
      targetPosition[1] - this.racketAnchor[1]
    ];
    const outwardDistance = Math.hypot(outwardXY[0], outwardXY[1]);
    if (outwardDistance <= NEXT_INTERCEPT_SUCCESS_RADIUS) {
      return [0, 0, 0];
    }
    const outwardDirection: [number, number] = [outwardXY[0] / outwardDistance, outwardXY[1] / outwardDistance];
    const racketVelocity = this.controller?.racketVelocity() ?? [0, 0, 0];
    const racketOutwardSpeed = racketVelocity[0] * outwardDirection[0] + racketVelocity[1] * outwardDirection[1];
    if (racketOutwardSpeed <= 0) {
      return [0, 0, 0];
    }
    const distanceScale = clamp(
      (outwardDistance - NEXT_INTERCEPT_SUCCESS_RADIUS) /
        Math.max(CONTACT_FRAME_LATERAL_BRAKE_RADIUS - NEXT_INTERCEPT_SUCCESS_RADIUS, 1.0e-6),
      0,
      1
    );
    const brakeSpeed = Math.min(CONTACT_FRAME_LATERAL_BRAKE_MAX, CONTACT_FRAME_LATERAL_BRAKE_GAIN * racketOutwardSpeed * distanceScale);
    return [-outwardDirection[0] * brakeSpeed, -outwardDirection[1] * brakeSpeed, 0];
  }

  private requiredContactFrameRacketVelocity(desiredVelocity: Vec3, targetTilt: [number, number]): Vec3 {
    const ballVelocity = this.ballVelocity();
    const faceNormal = targetFaceNormalFromTilt(targetTilt);
    let normal = scaleVec3(faceNormal, -1);
    normal = normalizeVec3(normal);
    const incomingNormalVelocity = Math.min(dotVec3(ballVelocity, normal), 0);
    const desiredNormalVelocity = dotVec3(desiredVelocity, normal);
    const requiredNormalVelocity =
      (desiredNormalVelocity + CONTACT_FRAME_APEX_LIFT_RESTITUTION * incomingNormalVelocity) /
      (1 + CONTACT_FRAME_APEX_LIFT_RESTITUTION);
    return scaleVec3(normal, requiredNormalVelocity);
  }

  private centeringTiltAxis(axisError: number, limit: number): number {
    const magnitude = Math.abs(axisError);
    if (magnitude <= CONTACT_FRAME_CENTERING_TILT_DEADBAND) {
      return 0;
    }
    const scale =
      clamp(
        (magnitude - CONTACT_FRAME_CENTERING_TILT_DEADBAND) /
          Math.max(CONTACT_FRAME_CENTERING_TILT_RADIUS - CONTACT_FRAME_CENTERING_TILT_DEADBAND, 1.0e-6),
        0,
        1
      ) * Math.sign(axisError);
    return limit * scale;
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
    push(this.desiredOutgoingVelocity(ballPosition)[0]);
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

  private resolveBallSpawnPosition(offsetAndHeight: Vec3): Vec3 {
    return [
      this.racketAnchor[0] + offsetAndHeight[0],
      this.racketAnchor[1] + offsetAndHeight[1],
      this.racketAnchor[2] + offsetAndHeight[2]
    ];
  }

  private resolveFallbackBallPosition(offsetAndHeight: Vec3): Vec3 {
    return [
      this.racketAnchor[0] + offsetAndHeight[0],
      this.racketAnchor[1] + offsetAndHeight[1],
      this.racketAnchor[2] + offsetAndHeight[2]
    ];
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

  private predictedInterceptTime(maxInterceptTime: number, strikePlaneOffset = TRACKING_STRIKE_PLANE_OFFSET): number {
    const ballPosition = this.ballPosition();
    const ballVelocity = this.ballVelocity();
    const targetZ = (this.controller?.racketPosition() ?? this.racketAnchor)[2] + strikePlaneOffset;
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

  private desiredOutgoingVelocity(
    contactPosition: Vec3,
    targetApexZ = this.racketAnchor[2] + TARGET_BALL_HEIGHT,
    targetXY: [number, number] = [this.racketAnchor[0], this.racketAnchor[1]],
    strikePlaneOffset = TRACKING_STRIKE_PLANE_OFFSET
  ): [Vec3, number, [number, number]] {
    const heightDelta = Math.max(targetApexZ - contactPosition[2], MIN_DESIRED_APEX_HEIGHT_DELTA);
    const desiredVelocityZ = Math.sqrt(2 * Math.abs(GRAVITY_Z) * heightDelta);
    const timeToApex = desiredVelocityZ / Math.abs(GRAVITY_Z);
    const descentHeight = Math.max(targetApexZ - (this.racketAnchor[2] + strikePlaneOffset), MIN_DESIRED_APEX_HEIGHT_DELTA);
    const desiredXYTime = timeToApex + Math.sqrt((2 * descentHeight) / Math.abs(GRAVITY_Z));
    return [
      [
        (targetXY[0] - contactPosition[0]) / Math.max(desiredXYTime, 1.0e-6),
        (targetXY[1] - contactPosition[1]) / Math.max(desiredXYTime, 1.0e-6),
        desiredVelocityZ
      ],
      timeToApex,
      targetXY
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
      this.preContactReadiness(ballPosition, racketPosition, ballVelocity) >= 0.95
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
    const preparationHeight = clamp(Math.min(TARGET_BALL_HEIGHT, 0.18), 0.12, 0.18);
    const activationHeight = clamp(preparationHeight + 0.08, 0.16, 0.22);
    if (ballHeight >= activationHeight) {
      return 0;
    }
    return 1 - clamp((ballHeight - preparationHeight) / Math.max(activationHeight - preparationHeight, 1.0e-6), 0, 1);
  }

  private preContactReadiness(ballPosition: Vec3, racketPosition: Vec3, ballVelocity: Vec3): number {
    const heightScore = this.preContactHeightReadiness(ballPosition, racketPosition, ballVelocity);
    if (heightScore <= 0) {
      return 0;
    }
    const interceptTime = this.predictedInterceptTime(0.35);
    const interceptXY: [number, number] = [
      ballPosition[0] + interceptTime * ballVelocity[0],
      ballPosition[1] + interceptTime * ballVelocity[1]
    ];
    const trackingError = Math.hypot(interceptXY[0] - racketPosition[0], interceptXY[1] - racketPosition[1]);
    const xyScore = Math.max(1 - trackingError / STRIKE_ZONE_XY_RADIUS, 0);
    return clamp(Math.min(heightScore, xyScore), 0, 1);
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

function addVec3(left: Vec3, right: Vec3): Vec3 {
  return [left[0] + right[0], left[1] + right[1], left[2] + right[2]];
}

function scaleVec3(vector: Vec3, scale: number): Vec3 {
  return [vector[0] * scale, vector[1] * scale, vector[2] * scale];
}

function dotVec3(left: Vec3, right: Vec3): number {
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

function normalizeVec3(vector: Vec3): Vec3 {
  const norm = Math.hypot(vector[0], vector[1], vector[2]);
  if (norm <= 1.0e-9) {
    return [0, 0, 1];
  }
  return [vector[0] / norm, vector[1] / norm, vector[2] / norm];
}

function targetFaceNormalFromTilt(tilt: [number, number]): Vec3 {
  const pitch = tilt[0];
  const roll = tilt[1];
  const sinPitch = Math.sin(pitch);
  const cosPitch = Math.cos(pitch);
  const sinRoll = Math.sin(roll);
  const cosRoll = Math.cos(roll);
  return normalizeVec3([-sinPitch * cosRoll, sinRoll, -cosPitch * cosRoll]);
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
