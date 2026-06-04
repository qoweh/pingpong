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
  heightLabel: boolean;
  contactMarker: boolean;
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
  time: number;
  ball: BallState;
  racketPosition: Vec3;
  contactCount: number;
  lastContactTime: number | null;
  lastContact: ContactEvent | null;
  mujocoLoaded: boolean;
  policyLoaded: boolean;
  policyMessage: string;
}

export interface DemoConfig {
  targetHeight: number;
  heightTolerance: number;
  ballPosition: Vec3;
}

export const DEFAULT_VISUALIZATION: VisualizationSettings = {
  trail: false,
  targetBand: false,
  heightLabel: false,
  contactMarker: false
};

export const DEFAULT_DEMO_CONFIG: DemoConfig = {
  targetHeight: 0.3,
  heightTolerance: 0.1,
  ballPosition: [0, 0, 0.34]
};

export const ZERO_SNAPSHOT: SimulationSnapshot = {
  time: 0,
  ball: {
    position: DEFAULT_DEMO_CONFIG.ballPosition,
    velocity: [0, 0, 0]
  },
  racketPosition: [0.35, 0, 0.72],
  contactCount: 0,
  lastContactTime: null,
  lastContact: null,
  mujocoLoaded: false,
  policyLoaded: false,
  policyMessage: "Loading MuJoCo WASM"
};
