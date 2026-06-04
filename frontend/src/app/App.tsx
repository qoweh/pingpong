import { ExternalLink, PanelRightClose, PanelRightOpen } from "lucide-react";
import { lazy, Suspense, useCallback, useState } from "react";

import { BallControls } from "../controls/BallControls";
import { CameraControls } from "../controls/CameraControls";
import { PlaybackControls } from "../controls/PlaybackControls";
import { VisualizationToggles } from "../controls/VisualizationToggles";
import { DocsPage } from "./DocsPage";
import type {
  BallSpawnSettings,
  CameraMode,
  LoadingProgress,
  PlaybackState,
  SimulationSnapshot,
  VisualizationSettings
} from "../simulation/types";
import { DEFAULT_BALL_SPAWN, DEFAULT_VISUALIZATION, ZERO_SNAPSHOT } from "../simulation/types";

const GITHUB_URL = "https://github.com/qoweh/pingpong";
const INITIAL_LOADING_PROGRESS: LoadingProgress = {
  percent: 0,
  message: "Starting simulation"
};
const SimulationCanvas = lazy(() =>
  import("../components/SimulationCanvas").then((module) => ({ default: module.SimulationCanvas }))
);

export function App() {
  const isDocsPage = window.location.pathname === "/docs";
  const [playback, setPlayback] = useState<PlaybackState>("playing");
  const [cameraMode, setCameraMode] = useState<CameraMode>("free");
  const [visualization, setVisualization] = useState<VisualizationSettings>(DEFAULT_VISUALIZATION);
  const [ballSpawn, setBallSpawn] = useState<BallSpawnSettings>(DEFAULT_BALL_SPAWN);
  const [snapshot, setSnapshot] = useState<SimulationSnapshot>(ZERO_SNAPSHOT);
  const [status, setStatus] = useState("Preparing simulation");
  const [loadingProgress, setLoadingProgress] = useState<LoadingProgress>(INITIAL_LOADING_PROGRESS);
  const [resetSignal, setResetSignal] = useState(0);
  const [ballSpawnSignal, setBallSpawnSignal] = useState(0);
  const [controlsOpen, setControlsOpen] = useState(true);
  const ready = snapshot.mujocoLoaded && snapshot.policyLoaded;
  const ballHeightAboveRacket = snapshot.ball.position[2] - snapshot.racketPosition[2];
  const heightText = ready ? `${ballHeightAboveRacket.toFixed(2)}m` : "--";
  const contactText = ready ? String(snapshot.contactCount) : "--";
  const timeText = ready ? `${snapshot.time.toFixed(2)}s` : "--";

  const reset = useCallback(() => {
    setPlayback("paused");
    setResetSignal((value) => value + 1);
  }, []);

  const updateBallSpawn = useCallback((value: BallSpawnSettings) => {
    setPlayback("paused");
    setBallSpawn(value);
    setBallSpawnSignal((signal) => signal + 1);
  }, []);

  const updateStatus = useCallback((message: string) => {
    setStatus(message);
  }, []);

  const updateLoadingProgress = useCallback((progress: LoadingProgress) => {
    setLoadingProgress(progress);
  }, []);

  return (
    <div className="app-shell">
      <header className="app-header">
        <a className="brand" href="/" aria-label="Home">
          <span className="brand-mark" />
          <span>Ping-Pong Keep-Up</span>
        </a>
        <nav className="nav-links" aria-label="Primary navigation">
          {isDocsPage ? null : <a href="/docs">Docs</a>}
          <a href={GITHUB_URL} target="_blank" rel="noreferrer">
            <ExternalLink size={17} />
            <span>GitHub</span>
          </a>
        </nav>
      </header>

      {isDocsPage ? (
        <DocsPage />
      ) : (
        <main>
          <section className="demo-band" id="demo">
            <div className="demo-layout">
              <section className="viewer-pane" aria-label="Simulation viewer">
                <Suspense fallback={<div className="simulation-canvas" />}>
                  <SimulationCanvas
                    playback={playback}
                    cameraMode={cameraMode}
                    visualization={visualization}
                    ballSpawn={ballSpawn}
                    onSnapshot={setSnapshot}
                    onStatus={updateStatus}
                    onProgress={updateLoadingProgress}
                    resetSignal={resetSignal}
                    ballSpawnSignal={ballSpawnSignal}
                  />
                </Suspense>
                <div className="viewer-title">
                  <h1>Ping-Pong Keep-Up</h1>
                  <p>A trained controller keeps the ball in play inside a physics simulation.</p>
                </div>
                <div className="runtime-status">
                  <span className={ready ? "status-dot ready" : "status-dot"} />
                  <span>{ready ? "Simulation Ready" : snapshot.mujocoLoaded ? snapshot.policyMessage : status}</span>
                </div>
                {!ready ? <LoadingOverlay status={status} snapshot={snapshot} progress={loadingProgress} /> : null}
              </section>

              <div className={controlsOpen ? "control-shell open" : "control-shell closed"}>
                <button
                  className="panel-toggle"
                  type="button"
                  title={controlsOpen ? "Hide controls" : "Show controls"}
                  aria-label={controlsOpen ? "Hide controls" : "Show controls"}
                  onClick={() => setControlsOpen((open) => !open)}
                >
                  {controlsOpen ? <PanelRightClose size={18} /> : <PanelRightOpen size={18} />}
                </button>

                {controlsOpen ? (
                  <aside className="control-pane" aria-label="Simulation controls">
                    <PlaybackControls playback={playback} onPlaybackChange={setPlayback} onReset={reset} />

                    <div className="metrics-grid">
                      <div>
                        <span>Height</span>
                        <strong>{heightText}</strong>
                      </div>
                      <div>
                        <span>Contacts</span>
                        <strong>{contactText}</strong>
                      </div>
                      <div>
                        <span>Time</span>
                        <strong>{timeText}</strong>
                      </div>
                      <div>
                        <span>Controller</span>
                        <strong>{snapshot.policyLoaded ? "Ready" : "Starting"}</strong>
                      </div>
                    </div>

                    <BallControls value={ballSpawn} onChange={updateBallSpawn} />
                    <CameraControls value={cameraMode} onChange={setCameraMode} />
                    <VisualizationToggles value={visualization} onChange={setVisualization} />

                    <div className="policy-note">{snapshot.policyMessage}</div>
                  </aside>
                ) : null}
              </div>
            </div>
          </section>
        </main>
      )}
    </div>
  );
}

function LoadingOverlay({
  status,
  snapshot,
  progress
}: {
  status: string;
  snapshot: SimulationSnapshot;
  progress: LoadingProgress;
}) {
  const percent = Math.min(100, Math.max(0, Math.round(progress.percent)));
  const message = progress.message || status;
  return (
    <div className="loading-overlay" role="status" aria-live="polite">
      <div className="loading-panel">
        <span className="loading-kicker">Starting simulation</span>
        <div className="loading-heading">
          <h2>Preparing the scene</h2>
          <strong>{percent}%</strong>
        </div>
        <div
          className="loading-bar"
          role="progressbar"
          aria-label="Simulation loading progress"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={percent}
        >
          <span style={{ width: `${percent}%` }} />
        </div>
        <div className="loading-steps">
          <span className={snapshot.mujocoLoaded ? "done" : ""}>
            {snapshot.mujocoLoaded ? "3D scene ready" : message}
          </span>
          <span className={snapshot.policyLoaded ? "done" : ""}>{snapshot.policyMessage}</span>
          <span>First uncached load can take several seconds on a server.</span>
        </div>
      </div>
    </div>
  );
}
