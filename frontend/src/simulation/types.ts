export type Vec3 = [number, number, number];

export type CameraMode =
  | "free"
  | "north"
  | "south"
  | "east"
  | "west"
  | "top"
  | "four";

export type PlaybackState = "playing" | "paused";

export interface VisualizationSettings {
  trail: boolean;
  targetBand: boolean;
  contactMarker: boolean;
}

export interface BallSpawnSettings {
  xOffset: number;
  yOffset: number;
  zOffset: number;
  velocityX: number;
  velocityY: number;
  velocityZ: number;
}

export interface BallSpawnRange {
  min: number;
  max: number;
  step: number;
  trainedMin?: number;
  trainedMax?: number;
  testedMin?: number;
  testedMax?: number;
}

export type BallSpawnRanges = Record<keyof BallSpawnSettings, BallSpawnRange>;

export interface BallSpawnXYConstraint {
  sampling?: "disk" | "square" | string;
  trainedRadius?: number | null;
  testedRadius?: number | null;
}

export interface BallSpawnConfig {
  defaults: BallSpawnSettings;
  ranges: BallSpawnRanges;
  xyConstraint?: BallSpawnXYConstraint;
}

export interface PolicyMetadata {
  className?: string;
  netArch?: unknown;
  architecture?: string[];
}

export interface ModelMetadata {
  id: string;
  name: string;
  displayName: string;
  rawRunName?: string | null;
  dimensionGroup?: string | null;
  versionLabel?: string | null;
  detailName?: string | null;
  sortDimension?: number | null;
  sortVersion?: number | null;
  source: string;
  path: string;
  algorithm: string;
  observationDim?: number | null;
  actionDim?: number | null;
  actionMode?: string | null;
  actionLabels?: string[];
  actionLow?: number[] | null;
  actionHigh?: number[] | null;
  ballSpawn?: BallSpawnConfig;
  trainedRanges?: Record<string, { min: number; max: number }>;
  testedRanges?: Record<string, { min: number; max: number }>;
  trainingSummaryPath?: string | null;
  policy?: PolicyMetadata;
  training?: {
    runName?: string | null;
    timesteps?: number | null;
    preset?: string | null;
    seed?: number | null;
  };
}

export interface ModelsPayload {
  activeModel: string;
  models: ModelMetadata[];
}

export interface BallState {
  position: Vec3;
  velocity: Vec3;
}

export interface ContactEvent {
  position: Vec3;
  time: number;
}

export interface SimulationSnapshot {
  episode: number;
  resetSerial: number;
  time: number;
  ball: BallState;
  racketPosition: Vec3;
  contactCount: number;
  contactEvent: boolean;
  lastContactTime: number | null;
  lastContact: ContactEvent | null;
  failureReason: string | null;
  terminated: boolean;
  truncated: boolean;
  mujocoLoaded: boolean;
  policyLoaded: boolean;
  policyMessage: string;
  modelId: string | null;
  action: number[] | null;
}

export interface LoadingProgress {
  percent: number;
  message: string;
}

export interface DemoConfig {
  targetHeight: number;
  heightTolerance: number;
}

export const DEFAULT_VISUALIZATION: VisualizationSettings = {
  trail: false,
  targetBand: false,
  contactMarker: false
};

export const DEFAULT_BALL_SPAWN: BallSpawnSettings = {
  xOffset: 0,
  yOffset: 0,
  zOffset: 0.34,
  velocityX: 0,
  velocityY: 0,
  velocityZ: 0
};

export const DEFAULT_BALL_SPAWN_CONFIG: BallSpawnConfig = {
  defaults: DEFAULT_BALL_SPAWN,
  ranges: {
    xOffset: { min: -0.16, max: 0.16, step: 0.005, trainedMin: -0.13, trainedMax: 0.13 },
    yOffset: { min: -0.16, max: 0.16, step: 0.005, trainedMin: -0.13, trainedMax: 0.13 },
    zOffset: { min: 0.18, max: 0.56, step: 0.005, trainedMin: 0.22, trainedMax: 0.52 },
    velocityX: { min: -0.06, max: 0.06, step: 0.005, trainedMin: -0.045, trainedMax: 0.045 },
    velocityY: { min: -0.06, max: 0.06, step: 0.005, trainedMin: -0.045, trainedMax: 0.045 },
    velocityZ: { min: -0.18, max: 0.04, step: 0.005, trainedMin: -0.14, trainedMax: 0.04 }
  },
  xyConstraint: {
    sampling: "disk",
    trainedRadius: 0.13,
    testedRadius: 0.16
  }
};

export const DEFAULT_DEMO_CONFIG: DemoConfig = {
  targetHeight: 0.3,
  heightTolerance: 0.1
};

export const ZERO_SNAPSHOT: SimulationSnapshot = {
  episode: 0,
  resetSerial: 0,
  time: 0,
  ball: {
    position: [0.456, -0.007, 0.832],
    velocity: [0, 0, 0]
  },
  racketPosition: [0.472, 0, 0.49],
  contactCount: 0,
  contactEvent: false,
  lastContactTime: null,
  lastContact: null,
  failureReason: null,
  terminated: false,
  truncated: false,
  mujocoLoaded: false,
  policyLoaded: false,
  policyMessage: "Connecting to policy model",
  modelId: null,
  action: null
};
