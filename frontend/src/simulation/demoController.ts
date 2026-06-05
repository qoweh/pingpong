import { MujocoWorld } from "./mujocoWorld";
import type {
  BallSpawnSettings,
  CameraMode,
  DemoConfig,
  LoadingProgress,
  PlaybackState,
  SimulationSnapshot,
  VisualizationSettings
} from "./types";
import { DEFAULT_DEMO_CONFIG, DEFAULT_VISUALIZATION, ZERO_SNAPSHOT } from "./types";
import { ThreeScene } from "../visualization/ThreeScene";

type SnapshotListener = (snapshot: SimulationSnapshot) => void;
type StatusListener = (message: string) => void;
type ProgressListener = (progress: LoadingProgress) => void;

export class DemoController {
  private readonly world = new MujocoWorld();
  private readonly renderer: ThreeScene;
  private playback: PlaybackState = "playing";
  private visualization: VisualizationSettings = { ...DEFAULT_VISUALIZATION };
  private cameraMode: CameraMode = "free";
  private config: DemoConfig = { ...DEFAULT_DEMO_CONFIG };
  private animationFrame = 0;
  private previousTimestamp = 0;
  private snapshot: SimulationSnapshot = ZERO_SNAPSHOT;
  private lastSnapshotEmit = 0;
  private readyProgressSent = false;
  private loadPercent = 0;

  constructor(
    host: HTMLElement,
    private readonly onSnapshot: SnapshotListener,
    private readonly onStatus: StatusListener,
    private readonly onProgress: ProgressListener
  ) {
    this.renderer = new ThreeScene(host);
    window.addEventListener("resize", this.resize);
  }

  async initialize(): Promise<void> {
    this.reportProgress(0, "Starting simulation");

    try {
      await this.world.initialize((progress) => this.reportProgress(progress.percent, progress.message));
      this.snapshot = this.world.reset();
      this.world.setPlayback(this.playback);
      this.reportProgress(90, "Preparing viewer");
      this.renderer.loadWorld(this.world);
      this.reportProgress(94, "Waiting for policy model");
      this.emit(true);
      this.loop(0);
    } catch (error) {
      this.reportProgress(100, error instanceof Error ? error.message : "Simulation failed to load");
      this.snapshot = ZERO_SNAPSHOT;
      this.emit(true);
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
    this.world.setPlayback(playback);
  }

  setCameraMode(cameraMode: CameraMode): void {
    this.cameraMode = cameraMode;
  }

  setVisualization(visualization: VisualizationSettings): void {
    this.visualization = { ...visualization };
  }

  reset(): void {
    this.snapshot = this.world.reset();
    this.emit(true);
  }

  resetCamera(): void {
    this.renderer.resetFreeCamera();
  }

  spawnBall(settings: BallSpawnSettings): void {
    this.snapshot = this.world.spawnBall(settings);
    this.emit(true);
  }

  private readonly resize = (): void => {
    this.renderer.resize();
  };

  private readonly loop = (timestamp: number): void => {
    const elapsed = this.previousTimestamp ? Math.min((timestamp - this.previousTimestamp) / 1000, 0.05) : 0;
    this.previousTimestamp = timestamp;

    this.snapshot = this.world.step(elapsed || 1 / 60);
    this.emit();

    this.renderer.update(this.snapshot, this.world, this.visualization, this.config);
    this.renderer.render(this.cameraMode);
    this.animationFrame = requestAnimationFrame(this.loop);
  };

  private emit(force = false): void {
    if (this.snapshot.mujocoLoaded && this.snapshot.policyLoaded && !this.readyProgressSent) {
      this.readyProgressSent = true;
      this.reportProgress(100, "Simulation ready");
    }

    const now = performance.now();
    if (!force && now - this.lastSnapshotEmit < 100) {
      return;
    }

    this.lastSnapshotEmit = now;
    this.onSnapshot(this.snapshot);
  }

  private reportProgress(percent: number, message: string): void {
    this.loadPercent = percent <= 0 ? 0 : Math.max(this.loadPercent, percent);
    const progress = {
      percent: Math.round(Math.min(100, Math.max(0, this.loadPercent))),
      message
    };
    this.onStatus(message);
    this.onProgress(progress);
  }
}
