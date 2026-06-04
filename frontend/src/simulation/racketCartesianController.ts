import type { DoubleBuffer, MainModule, MjData, MjModel } from "@mujoco/mujoco";

import type { Vec3 } from "./types";

type ControllerIds = {
  racketSite: number;
  racketGeom: number;
};

type Runtime = {
  module: MainModule;
  model: MjModel;
  data: MjData;
};

const CONTROL_DT = 0.02;
const DAMPING = 1.0e-4;
const POSITION_GAIN = 1.6;
const ORIENTATION_GAIN = 1.1;
const MAX_POSITION_STEP = 0.06;
const MAX_ORIENTATION_STEP = 0.18;
const VELOCITY_GAIN = 1.0;
const VELOCITY_FEEDBACK_GAIN = 0.55;
const MAX_VELOCITY_STEP = 0.085;
const TARGET_OFFSET_LOW: Vec3 = [-0.12, -0.12, -0.04];
const TARGET_OFFSET_HIGH: Vec3 = [0.12, 0.12, 0.12];
const TARGET_TILT_LIMIT: [number, number] = [0.12, 0.12];
const NULLSPACE_POSTURE_GAIN = 0.2;
const NULLSPACE_POSTURE_MAX_STEP = 0.01;

export class RacketCartesianController {
  private readonly jointIds: number[];
  private readonly qposIndices: number[];
  private readonly dofIndices: number[];
  private readonly jointLow: number[];
  private readonly jointHigh: number[];
  private readonly positionJacobian: DoubleBuffer;
  private readonly rotationJacobian: DoubleBuffer;
  private readonly homeTargets: number[];
  private readonly nullspaceTarget: number[];
  private targetPositionValue: Vec3;
  private targetTiltValue: [number, number] = [0, 0];
  private targetVelocityValue: Vec3 = [0, 0, 0];
  private targetVelocityEnabled = false;
  private anchorPosition: Vec3;
  private targets: number[];

  constructor(private readonly runtime: Runtime, ids: ControllerIds, homeCtrl: number[]) {
    const { module, model, data } = runtime;
    this.jointIds = Array.from({ length: 7 }, (_, index) =>
      module.mj_name2id(model, module.mjtObj.mjOBJ_JOINT.value, `joint${index + 1}`)
    );
    if (this.jointIds.some((id) => id < 0)) {
      throw new Error("MuJoCo model is missing one or more Panda arm joints.");
    }

    this.qposIndices = this.jointIds.map((jointId) => Number(model.jnt_qposadr[jointId]));
    this.dofIndices = this.jointIds.map((jointId) => Number(model.jnt_dofadr[jointId]));
    this.jointLow = this.jointIds.map((jointId) => Number(model.jnt_range[jointId * 2]));
    this.jointHigh = this.jointIds.map((jointId) => Number(model.jnt_range[jointId * 2 + 1]));
    this.positionJacobian = module.DoubleBuffer.FromArray(new Array(model.nv * 3).fill(0));
    this.rotationJacobian = module.DoubleBuffer.FromArray(new Array(model.nv * 3).fill(0));
    this.homeTargets = homeCtrl.slice(0, 7);
    this.nullspaceTarget = this.homeTargets.slice();
    this.anchorPosition = arrayVec3(data.site_xpos, ids.racketSite * 3);
    this.targetPositionValue = [...this.anchorPosition] as Vec3;
    this.targets = this.homeTargets.slice();
  }

  get targetPosition(): Vec3 {
    return [...this.targetPositionValue] as Vec3;
  }

  get targetTilt(): [number, number] {
    return [...this.targetTiltValue] as [number, number];
  }

  get targetVelocity(): Vec3 {
    return [...this.targetVelocityValue] as Vec3;
  }

  get targetFaceNormal(): Vec3 {
    return targetFaceNormalFromTilt(this.targetTiltValue);
  }

  reset(): number[] {
    this.anchorPosition = this.racketPosition();
    this.targetPositionValue = [...this.anchorPosition] as Vec3;
    this.targetTiltValue = [0, 0];
    this.targetVelocityValue = [0, 0, 0];
    this.targetVelocityEnabled = false;
    this.targets = this.homeTargets.slice();
    return this.targets.slice();
  }

  dispose(): void {
    this.positionJacobian.delete();
    this.rotationJacobian.delete();
  }

  setTargetPosition(position: Vec3): Vec3 {
    this.targetPositionValue = this.clipTargetPosition(position);
    return this.targetPosition;
  }

  setTargetTilt(tilt: [number, number]): [number, number] {
    this.targetTiltValue = [
      clamp(tilt[0], -TARGET_TILT_LIMIT[0], TARGET_TILT_LIMIT[0]),
      clamp(tilt[1], -TARGET_TILT_LIMIT[1], TARGET_TILT_LIMIT[1])
    ];
    return this.targetTilt;
  }

  setTargetVelocity(velocity: Vec3 | null): Vec3 {
    if (!velocity) {
      this.targetVelocityValue = [0, 0, 0];
      this.targetVelocityEnabled = false;
      return this.targetVelocity;
    }

    this.targetVelocityValue = [...velocity] as Vec3;
    this.targetVelocityEnabled = true;
    return this.targetVelocity;
  }

  computeJointTargets(): number[] {
    const currentPosition = this.racketPosition();
    const positionError = clipVectorNorm(subVec3(this.targetPositionValue, currentPosition), MAX_POSITION_STEP);
    const velocityStep = this.targetVelocityEnabled ? this.computeVelocityStep() : [0, 0, 0] as Vec3;
    const currentFaceNormal = this.racketFaceNormal();
    const targetFaceNormal = this.targetFaceNormal;
    const orientationError = clipVectorNorm(cross(currentFaceNormal, targetFaceNormal), MAX_ORIENTATION_STEP);

    this.runtime.module.mj_jacSite(
      this.runtime.model,
      this.runtime.data,
      this.positionJacobian,
      this.rotationJacobian,
      this.runtime.module.mj_name2id(this.runtime.model, this.runtime.module.mjtObj.mjOBJ_SITE.value, "racket_center")
    );
    const jacp = bufferView(this.positionJacobian);
    const jacr = bufferView(this.rotationJacobian);
    const taskJacobian = buildTaskJacobian(jacp, jacr, this.dofIndices, this.runtime.model.nv);
    const taskError = [
      POSITION_GAIN * positionError[0] + velocityStep[0],
      POSITION_GAIN * positionError[1] + velocityStep[1],
      POSITION_GAIN * positionError[2] + velocityStep[2],
      ORIENTATION_GAIN * orientationError[0],
      ORIENTATION_GAIN * orientationError[1],
      ORIENTATION_GAIN * orientationError[2]
    ];

    const taskMetric = multiplyMatMatT(taskJacobian, DAMPING);
    const solvedTaskError = solveLinearSystem(taskMetric, taskError);
    let deltaQ = multiplyMatTVec(taskJacobian, solvedTaskError);
    deltaQ = addVec(deltaQ, this.postureNullspaceDelta(taskJacobian, taskMetric));

    const nextTargets = this.qposIndices.map((qposIndex, index) =>
      clamp(Number(this.runtime.data.qpos[qposIndex]) + deltaQ[index], this.jointLow[index], this.jointHigh[index])
    );
    this.targets = nextTargets;
    return nextTargets.slice();
  }

  racketPosition(): Vec3 {
    return arrayVec3(this.runtime.data.site_xpos, this.runtime.module.mj_name2id(
      this.runtime.model,
      this.runtime.module.mjtObj.mjOBJ_SITE.value,
      "racket_center"
    ) * 3);
  }

  racketVelocity(): Vec3 {
    this.runtime.module.mj_jacSite(
      this.runtime.model,
      this.runtime.data,
      this.positionJacobian,
      this.rotationJacobian,
      this.runtime.module.mj_name2id(this.runtime.model, this.runtime.module.mjtObj.mjOBJ_SITE.value, "racket_center")
    );
    const jacp = bufferView(this.positionJacobian);
    const velocity: Vec3 = [0, 0, 0];
    for (let row = 0; row < 3; row += 1) {
      let value = 0;
      for (let column = 0; column < this.runtime.model.nv; column += 1) {
        value += Number(jacp[row * this.runtime.model.nv + column]) * Number(this.runtime.data.qvel[column]);
      }
      velocity[row] = value;
    }
    return velocity;
  }

  racketFaceNormal(): Vec3 {
    const geomId = this.runtime.module.mj_name2id(
      this.runtime.model,
      this.runtime.module.mjtObj.mjOBJ_GEOM.value,
      "racket_head"
    );
    const offset = geomId * 9;
    return normalize([
      Number(this.runtime.data.geom_xmat[offset + 2]),
      Number(this.runtime.data.geom_xmat[offset + 5]),
      Number(this.runtime.data.geom_xmat[offset + 8])
    ]);
  }

  private computeVelocityStep(): Vec3 {
    const racketVelocity = this.racketVelocity();
    const command: Vec3 = [
      VELOCITY_GAIN * this.targetVelocityValue[0] + VELOCITY_FEEDBACK_GAIN * (this.targetVelocityValue[0] - racketVelocity[0]),
      VELOCITY_GAIN * this.targetVelocityValue[1] + VELOCITY_FEEDBACK_GAIN * (this.targetVelocityValue[1] - racketVelocity[1]),
      VELOCITY_GAIN * this.targetVelocityValue[2] + VELOCITY_FEEDBACK_GAIN * (this.targetVelocityValue[2] - racketVelocity[2])
    ];
    return clipVectorNorm([command[0] * CONTROL_DT, command[1] * CONTROL_DT, command[2] * CONTROL_DT], MAX_VELOCITY_STEP);
  }

  private clipTargetPosition(position: Vec3): Vec3 {
    return [
      this.anchorPosition[0] + clamp(position[0] - this.anchorPosition[0], TARGET_OFFSET_LOW[0], TARGET_OFFSET_HIGH[0]),
      this.anchorPosition[1] + clamp(position[1] - this.anchorPosition[1], TARGET_OFFSET_LOW[1], TARGET_OFFSET_HIGH[1]),
      this.anchorPosition[2] + clamp(position[2] - this.anchorPosition[2], TARGET_OFFSET_LOW[2], TARGET_OFFSET_HIGH[2])
    ];
  }

  private postureNullspaceDelta(taskJacobian: number[][], taskMetric: number[][]): number[] {
    const currentJointPositions = this.qposIndices.map((qposIndex) => Number(this.runtime.data.qpos[qposIndex]));
    let postureStep = this.nullspaceTarget.map(
      (target, index) => NULLSPACE_POSTURE_GAIN * (target - currentJointPositions[index])
    );
    postureStep = clipVector(postureStep, NULLSPACE_POSTURE_MAX_STEP);
    const taskInverseLeft = solveMat(taskMetric, taskJacobian);
    const projector = subtractMat(identity(7), multiplyMatMat(transpose(taskJacobian), taskInverseLeft));
    return multiplyMatVec(projector, postureStep);
  }
}

function buildTaskJacobian(jacp: ArrayLike<number>, jacr: ArrayLike<number>, dofIndices: number[], nv: number): number[][] {
  return Array.from({ length: 6 }, (_, row) =>
    dofIndices.map((dofIndex) => Number((row < 3 ? jacp : jacr)[(row % 3) * nv + dofIndex]))
  );
}

function targetFaceNormalFromTilt(tilt: [number, number]): Vec3 {
  const pitch = tilt[0];
  const roll = tilt[1];
  const sinPitch = Math.sin(pitch);
  const cosPitch = Math.cos(pitch);
  const sinRoll = Math.sin(roll);
  const cosRoll = Math.cos(roll);
  return normalize([-sinPitch * cosRoll, sinRoll, -cosPitch * cosRoll]);
}

function bufferView(buffer: DoubleBuffer): Float64Array {
  const candidate = buffer as unknown as { GetView?: () => Float64Array; getView?: () => Float64Array };
  if (candidate.GetView) {
    return candidate.GetView();
  }
  if (candidate.getView) {
    return candidate.getView();
  }
  throw new Error("MuJoCo buffer view accessor is unavailable.");
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

function cross(left: Vec3, right: Vec3): Vec3 {
  return [
    left[1] * right[2] - left[2] * right[1],
    left[2] * right[0] - left[0] * right[2],
    left[0] * right[1] - left[1] * right[0]
  ];
}

function normalize(vector: Vec3): Vec3 {
  const length = Math.hypot(vector[0], vector[1], vector[2]);
  if (length <= 1.0e-9) {
    return [0, 0, 1];
  }
  return [vector[0] / length, vector[1] / length, vector[2] / length];
}

function clipVectorNorm(vector: Vec3, maxNorm: number): Vec3 {
  const length = Math.hypot(vector[0], vector[1], vector[2]);
  if (maxNorm <= 0 || length <= maxNorm || length <= 1.0e-9) {
    return vector;
  }
  const scale = maxNorm / length;
  return [vector[0] * scale, vector[1] * scale, vector[2] * scale];
}

function clipVector(vector: number[], maxNorm: number): number[] {
  const length = Math.hypot(...vector);
  if (maxNorm <= 0 || length <= maxNorm || length <= 1.0e-9) {
    return vector;
  }
  const scale = maxNorm / length;
  return vector.map((value) => value * scale);
}

function addVec(left: number[], right: number[]): number[] {
  return left.map((value, index) => value + (right[index] ?? 0));
}

function multiplyMatMatT(matrix: number[][], damping = 0): number[][] {
  return matrix.map((row, rowIndex) =>
    matrix.map((otherRow, columnIndex) => {
      let sum = rowIndex === columnIndex ? damping : 0;
      for (let index = 0; index < row.length; index += 1) {
        sum += row[index] * otherRow[index];
      }
      return sum;
    })
  );
}

function multiplyMatTVec(matrix: number[][], vector: number[]): number[] {
  return Array.from({ length: matrix[0]?.length ?? 0 }, (_, column) => {
    let sum = 0;
    for (let row = 0; row < matrix.length; row += 1) {
      sum += matrix[row][column] * vector[row];
    }
    return sum;
  });
}

function multiplyMatVec(matrix: number[][], vector: number[]): number[] {
  return matrix.map((row) => row.reduce((sum, value, index) => sum + value * (vector[index] ?? 0), 0));
}

function multiplyMatMat(left: number[][], right: number[][]): number[][] {
  return left.map((row) =>
    Array.from({ length: right[0]?.length ?? 0 }, (_, column) => {
      let sum = 0;
      for (let index = 0; index < right.length; index += 1) {
        sum += row[index] * right[index][column];
      }
      return sum;
    })
  );
}

function subtractMat(left: number[][], right: number[][]): number[][] {
  return left.map((row, rowIndex) => row.map((value, columnIndex) => value - right[rowIndex][columnIndex]));
}

function transpose(matrix: number[][]): number[][] {
  return Array.from({ length: matrix[0]?.length ?? 0 }, (_, column) => matrix.map((row) => row[column]));
}

function identity(size: number): number[][] {
  return Array.from({ length: size }, (_, row) =>
    Array.from({ length: size }, (_, column) => (row === column ? 1 : 0))
  );
}

function solveMat(matrix: number[][], rhs: number[][]): number[][] {
  const columns = transpose(rhs);
  return transpose(columns.map((column) => solveLinearSystem(matrix, column)));
}

function solveLinearSystem(matrix: number[][], rhs: number[]): number[] {
  const size = rhs.length;
  const a = matrix.map((row, index) => [...row, rhs[index]]);

  for (let column = 0; column < size; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < size; row += 1) {
      if (Math.abs(a[row][column]) > Math.abs(a[pivot][column])) {
        pivot = row;
      }
    }
    [a[column], a[pivot]] = [a[pivot], a[column]];

    const divisor = Math.abs(a[column][column]) <= 1.0e-12 ? 1.0e-12 : a[column][column];
    for (let entry = column; entry <= size; entry += 1) {
      a[column][entry] /= divisor;
    }

    for (let row = 0; row < size; row += 1) {
      if (row === column) {
        continue;
      }
      const factor = a[row][column];
      for (let entry = column; entry <= size; entry += 1) {
        a[row][entry] -= factor * a[column][entry];
      }
    }
  }

  return a.map((row) => row[size]);
}

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}
