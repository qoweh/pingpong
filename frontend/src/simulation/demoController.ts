import { MujocoWorld } from "./mujocoWorld";
import type {
  CameraMode,
  DemoConfig,
  PlaybackState,
  SimulationSnapshot,
  Vec3,
  VisualizationSettings
} from "./types";
import { DEFAULT_DEMO_CONFIG, DEFAULT_VISUALIZATION, ZERO_SNAPSHOT } from "./types";
import { ThreeScene } from "../visualization/ThreeScene";

type SnapshotListener = (snapshot: SimulationSnapshot) => void;
type StatusListener = (message: string) => void;

export class DemoController {
  private readonly world = new MujocoWorld();
  private readonly renderer: ThreeScene;
  private playback: PlaybackState = "paused";
  private visualization: VisualizationSettings = { ...DEFAULT_VISUALIZATION };
  private cameraMode: CameraMode = "free";
  private config: DemoConfig = { ...DEFAULT_DEMO_CONFIG };
  private animationFrame = 0;
  private previousTimestamp = 0;
  private snapshot: SimulationSnapshot = ZERO_SNAPSHOT;
  private initialized = false;

  constructor(
    host: HTMLElement,
    private readonly onSnapshot: SnapshotListener,
    private readonly onStatus: StatusListener
  ) {
    this.renderer = new ThreeScene(host);
    window.addEventListener("resize", this.resize);
  }

  async initialize(): Promise<void> {
    this.onStatus("Loading MuJoCo WASM");
    this.loop(0);

    try {
      await this.world.initialize(this.config);
      this.initialized = true;
      this.snapshot = this.world.reset(this.config.ballPosition);
      this.onStatus("MuJoCo WASM loaded");
      this.emit();
    } catch (error) {
      this.onStatus(error instanceof Error ? error.message : "MuJoCo failed to load");
      this.initialized = false;
      this.snapshot = ZERO_SNAPSHOT;
      this.emit();
    }
  }

  dispose(): void {
    cancelAnimationFrame(this.animationFrame);
    window.removeEventListener("resize", this.resize);
    this.renderer.dispose();
    this.world.dispose();
  }

  setPlayback(playback: PlaybackState): void {
    this.playback = playback;
  }

  setCameraMode(cameraMode: CameraMode): void {
    this.cameraMode = cameraMode;
  }

  setVisualization(visualization: VisualizationSettings): void {
    this.visualization = { ...visualization };
  }

  setBallPosition(ballPosition: Vec3): void {
    this.config = { ...this.config, ballPosition };
    this.snapshot = this.world.setBallPosition(ballPosition);
    this.emit();
  }

  reset(): void {
    this.snapshot = this.world.reset(this.config.ballPosition);
    this.emit();
  }

  private readonly resize = (): void => {
    this.renderer.resize();
  };

  private readonly loop = (timestamp: number): void => {
    const elapsed = this.previousTimestamp ? Math.min((timestamp - this.previousTimestamp) / 1000, 0.05) : 0;
    this.previousTimestamp = timestamp;

    if (this.playback === "playing" || !this.initialized) {
      this.snapshot = this.world.step(elapsed || 1 / 60);
      this.emit();
    }

    this.renderer.update(this.snapshot, this.world, this.visualization, this.config);
    this.renderer.render(this.cameraMode);
    this.animationFrame = requestAnimationFrame(this.loop);
  };

  private emit(): void {
    this.onSnapshot(this.snapshot);
  }
}
