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
  PlaybackState,
  SimulationSnapshot,
  VisualizationSettings
} from "../simulation/types";
import { DEFAULT_BALL_SPAWN, DEFAULT_VISUALIZATION, ZERO_SNAPSHOT } from "../simulation/types";

const GITHUB_URL = "https://github.com/qoweh/pingpong";
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
  const [status, setStatus] = useState("Loading");
  const [resetSignal, setResetSignal] = useState(0);
  const [ballResetSignal, setBallResetSignal] = useState(0);
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

  const applyBallSpawn = useCallback((value: BallSpawnSettings) => {
    setPlayback("paused");
    setBallSpawn(value);
    setBallResetSignal((signal) => signal + 1);
  }, []);

  return (
    <div className="app-shell">
      <header className="app-header">
        <a className="brand" href="/" aria-label="Home">
          <span className="brand-mark" />
          <span>Ping-Pong Keep-Up RL</span>
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
              <section className="viewer-pane" aria-label="MuJoCo simulation viewer">
                <Suspense fallback={<div className="simulation-canvas" />}>
                  <SimulationCanvas
                    playback={playback}
                    cameraMode={cameraMode}
                    visualization={visualization}
                    ballSpawn={ballSpawn}
                    onSnapshot={setSnapshot}
                    onStatus={setStatus}
                    resetSignal={resetSignal}
                    ballResetSignal={ballResetSignal}
                  />
                </Suspense>
                <div className="viewer-title">
                  <h1>Ping-Pong Keep-Up with Reinforcement Learning</h1>
                  <p>Live Python PPO control streamed into a MuJoCo WebAssembly viewer.</p>
                </div>
                <div className="runtime-status">
                  <span className={ready ? "status-dot ready" : "status-dot"} />
                  <span>{ready ? "Live MuJoCo" : snapshot.mujocoLoaded ? snapshot.policyMessage : status}</span>
                </div>
                {!ready ? <LoadingOverlay status={status} snapshot={snapshot} /> : null}
              </section>

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
                <aside className="control-pane" aria-label="Demo controls">
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
                      <span>Live RL</span>
                      <strong>{snapshot.policyLoaded ? "Loaded" : "Pending"}</strong>
                    </div>
                  </div>

                  <BallControls value={ballSpawn} onChange={setBallSpawn} onApply={applyBallSpawn} />
                  <CameraControls value={cameraMode} onChange={setCameraMode} />
                  <VisualizationToggles value={visualization} onChange={setVisualization} />

                  <div className="policy-note">{snapshot.policyMessage}</div>
                </aside>
              ) : null}
            </div>
          </section>
        </main>
      )}
    </div>
  );
}

function LoadingOverlay({ status, snapshot }: { status: string; snapshot: SimulationSnapshot }) {
  return (
    <div className="loading-overlay" role="status" aria-live="polite">
      <div className="loading-panel">
        <span className="loading-kicker">Starting live simulation</span>
        <h2>Loading MuJoCo and Python PPO</h2>
        <div className="loading-bar">
          <span />
        </div>
        <div className="loading-steps">
          <span className={snapshot.mujocoLoaded ? "done" : ""}>
            {snapshot.mujocoLoaded ? "MuJoCo WASM ready" : status}
          </span>
          <span className={snapshot.policyLoaded ? "done" : ""}>{snapshot.policyMessage}</span>
          <span>First uncached load can take several seconds on a home server.</span>
        </div>
      </div>
    </div>
  );
}
