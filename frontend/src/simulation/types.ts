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
  policyMessage: "Connecting to Python live backend"
};
