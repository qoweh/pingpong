import type { Vec3 } from "./types";

export interface RolloutTrace {
  format: "pingpong-web-rollout-v1";
  source: {
    rlRoot: string;
    model: string;
    resolvedModel: string;
    scene: string | null;
  };
  policy: {
    deterministic: boolean;
    observationSize: number;
    actionSize: number;
  };
  simulation: {
    seed: number;
    controlDt: number;
    timestep: number;
    substeps: number;
    nq: number;
    nv: number;
    nu: number;
  };
  envConfig: Record<string, unknown>;
  resetInfo: Record<string, unknown>;
  initialState: MujocoStateFrame;
  frames: RolloutFrame[];
  result: {
    steps: number;
    terminated: boolean;
    truncated: boolean;
    failureReason: string | null;
    contactCount: number;
    successfulBounceCount: number;
  };
}

export interface RolloutFrame {
  index: number;
  time: number;
  ctrl: number[];
  action: number[];
  reward: number;
  terminated: boolean;
  truncated: boolean;
  contact: {
    event: boolean;
    observed: boolean;
    count: number;
    successfulBounceCount: number;
    position: Vec3 | null;
  };
  state: {
    ballPosition: Vec3;
    ballVelocity: Vec3;
    racketPosition: Vec3;
    targetTilt: number[];
  };
  mujocoState: MujocoStateFrame;
  info: {
    phaseName: string | null;
    failureReason: string | null;
    successReason: string | null;
    easyNextBallScore: number | null;
    nextInterceptTime: number | null;
  };
}

export interface MujocoStateFrame {
  qpos: number[];
  qvel: number[];
  ctrl: number[];
  time: number;
}

export async function loadRolloutTrace(): Promise<RolloutTrace | null> {
  const response = await fetch("/assets/demo/rollout.json", { cache: "force-cache" });
  if (!response.ok) {
    return null;
  }

  const trace = (await response.json()) as RolloutTrace;
  if (trace.format !== "pingpong-web-rollout-v1" || trace.frames.length === 0) {
    throw new Error("Invalid Python rollout trace.");
  }

  return trace;
}
