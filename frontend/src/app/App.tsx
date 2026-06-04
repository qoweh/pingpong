import { ExternalLink } from "lucide-react";
import { useCallback, useState } from "react";

import { BallControls } from "../controls/BallControls";
import { CameraControls } from "../controls/CameraControls";
import { PlaybackControls } from "../controls/PlaybackControls";
import { VisualizationToggles } from "../controls/VisualizationToggles";
import { SimulationCanvas } from "../components/SimulationCanvas";
import { DocsPage } from "./DocsPage";
import type {
  CameraMode,
  PlaybackState,
  SimulationSnapshot,
  Vec3,
  VisualizationSettings
} from "../simulation/types";
import { DEFAULT_DEMO_CONFIG, DEFAULT_VISUALIZATION, ZERO_SNAPSHOT } from "../simulation/types";

const GITHUB_URL = "https://github.com/qoweh/pingpong";

export function App() {
  const isDocsPage = window.location.pathname === "/docs";
  const [playback, setPlayback] = useState<PlaybackState>("playing");
  const [ballPosition, setBallPosition] = useState<Vec3>(DEFAULT_DEMO_CONFIG.ballPosition);
  const [cameraMode, setCameraMode] = useState<CameraMode>("free");
  const [visualization, setVisualization] = useState<VisualizationSettings>(DEFAULT_VISUALIZATION);
  const [snapshot, setSnapshot] = useState<SimulationSnapshot>(ZERO_SNAPSHOT);
  const [status, setStatus] = useState("Loading");
  const [resetSignal, setResetSignal] = useState(0);

  const reset = useCallback(() => {
    setPlayback("paused");
    setResetSignal((value) => value + 1);
  }, []);

  return (
    <div className="app-shell">
      <header className="app-header">
        <a className="brand" href="/" aria-label="Home">
          <span className="brand-mark" />
          <span>Ping-Pong Keep-Up RL</span>
        </a>
        <nav className="nav-links" aria-label="Primary navigation">
          <a href="/docs">Docs</a>
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
              <SimulationCanvas
                playback={playback}
                ballPosition={ballPosition}
                cameraMode={cameraMode}
                visualization={visualization}
                onSnapshot={setSnapshot}
                onStatus={setStatus}
                resetSignal={resetSignal}
              />
              <div className="viewer-title">
                <h1>Ping-Pong Keep-Up with Reinforcement Learning</h1>
                <p>MuJoCo WebAssembly viewer for the Panda racket keep-up scene.</p>
              </div>
              <div className="runtime-status">
                <span className={snapshot.mujocoLoaded ? "status-dot ready" : "status-dot"} />
                <span>{snapshot.mujocoLoaded ? "MuJoCo WASM" : status}</span>
              </div>
              {visualization.heightLabel ? (
                <div className="height-readout">
                  <span>Ball height: {snapshot.ball.position[2].toFixed(2)}m</span>
                  <span>Target: {DEFAULT_DEMO_CONFIG.targetHeight.toFixed(2)}m</span>
                  <span>
                    Error: {(snapshot.ball.position[2] - DEFAULT_DEMO_CONFIG.targetHeight).toFixed(2)}m
                  </span>
                </div>
              ) : null}
            </section>

            <aside className="control-pane" aria-label="Demo controls">
              <PlaybackControls playback={playback} onPlaybackChange={setPlayback} onReset={reset} />

              <div className="metrics-grid">
                <div>
                  <span>Height</span>
                  <strong>{snapshot.ball.position[2].toFixed(2)}m</strong>
                </div>
                <div>
                  <span>Contacts</span>
                  <strong>{snapshot.contactCount}</strong>
                </div>
                <div>
                  <span>Time</span>
                  <strong>{snapshot.time.toFixed(2)}s</strong>
                </div>
                <div>
                  <span>Policy</span>
                  <strong>{snapshot.policyLoaded ? "Loaded" : "Pending"}</strong>
                </div>
              </div>

              <BallControls value={ballPosition} onChange={setBallPosition} />
              <CameraControls value={cameraMode} onChange={setCameraMode} />
              <VisualizationToggles value={visualization} onChange={setVisualization} />

              <div className="policy-note">{snapshot.policyMessage}</div>
            </aside>
          </div>
        </section>
      </main>
      )}
    </div>
  );
}
