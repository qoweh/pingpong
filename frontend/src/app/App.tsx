import { ExternalLink, PanelLeftClose, PanelLeftOpen, PanelRightClose, PanelRightOpen, RefreshCw } from "lucide-react";
import { lazy, Suspense, useCallback, useEffect, useMemo, useState } from "react";

import { ActionVisualizer } from "../components/ActionVisualizer";
import { PolicyNetworkPanel, PolicyNetworkToggle } from "../components/PolicyNetworkVisualizer";
import { BallControls } from "../controls/BallControls";
import { CameraControls } from "../controls/CameraControls";
import { ModelControls } from "../controls/ModelControls";
import { PlaybackControls } from "../controls/PlaybackControls";
import { VisualizationToggles } from "../controls/VisualizationToggles";
import { DocsPage } from "./DocsPage";
import { clampBallSpawnSettings, parseBallSpawnConfig } from "../simulation/ballSpawnConfig";
import { parseModelsPayload } from "../simulation/modelConfig";
import type {
  BallSpawnConfig,
  BallSpawnSettings,
  CameraMode,
  LoadingProgress,
  ModelsPayload,
  ModelMetadata,
  PlaybackState,
  SimulationSnapshot,
  VisualizationSettings
} from "../simulation/types";
import { DEFAULT_BALL_SPAWN, DEFAULT_BALL_SPAWN_CONFIG, DEFAULT_VISUALIZATION, ZERO_SNAPSHOT } from "../simulation/types";

const GITHUB_URL = "https://github.com/qoweh/pingpong";
const INITIAL_LOADING_PROGRESS: LoadingProgress = {
  percent: 0,
  message: "Starting simulation"
};
const SimulationCanvas = lazy(() =>
  import("../components/SimulationCanvas").then((module) => ({ default: module.SimulationCanvas }))
);

export function App() {
  const isDocsPage = false;
  const [playback, setPlayback] = useState<PlaybackState>("playing");
  const [cameraMode, setCameraMode] = useState<CameraMode>("free");
  const [visualization, setVisualization] = useState<VisualizationSettings>(DEFAULT_VISUALIZATION);
  const [ballSpawn, setBallSpawn] = useState<BallSpawnSettings>(DEFAULT_BALL_SPAWN);
  const [ballSpawnConfig, setBallSpawnConfig] = useState<BallSpawnConfig>(DEFAULT_BALL_SPAWN_CONFIG);
  const [models, setModels] = useState<ModelMetadata[]>([]);
  const [activeModelId, setActiveModelId] = useState<string | null>(null);
  const [modelSwitching, setModelSwitching] = useState(false);
  const [modelError, setModelError] = useState<string | null>(null);
  const [snapshot, setSnapshot] = useState<SimulationSnapshot>(ZERO_SNAPSHOT);
  const [status, setStatus] = useState("Preparing simulation");
  const [loadingProgress, setLoadingProgress] = useState<LoadingProgress>(INITIAL_LOADING_PROGRESS);
  const [resetSignal, setResetSignal] = useState(0);
  const [ballSpawnSignal, setBallSpawnSignal] = useState(0);
  const [cameraResetSignal, setCameraResetSignal] = useState(0);
  const [modelPanelOpen, setModelPanelOpen] = useState(true);
  const [controlsOpen, setControlsOpen] = useState(true);
  const [policyNetworkOpen, setPolicyNetworkOpen] = useState(false);
  const selectedModel = useMemo(
    () => models.find((model) => model.id === activeModelId) ?? models[0] ?? null,
    [activeModelId, models]
  );
  const ready = snapshot.mujocoLoaded && snapshot.policyLoaded;
  const ballHeightAboveRacket = snapshot.ball.position[2] - snapshot.racketPosition[2];
  const heightText = ready ? `${ballHeightAboveRacket.toFixed(2)}m` : "--";
  const contactText = ready ? String(snapshot.contactCount) : "--";
  const timeText = ready ? `${snapshot.time.toFixed(2)}s` : "--";
  const modelSwitchHint = "Different action dimensions may take longer.";
  const showLoadingOverlay = !ready || modelSwitching;

  const reset = useCallback(() => {
    setPlayback("paused");
    setResetSignal((value) => value + 1);
  }, []);

  const updateBallSpawn = useCallback((value: BallSpawnSettings) => {
    setPlayback("paused");
    setBallSpawn(value);
    setBallSpawnSignal((signal) => signal + 1);
  }, []);

  const applyModelsPayload = useCallback((payload: ModelsPayload) => {
    setModels(payload.models);
    setActiveModelId(payload.activeModel);
    const activeModel = payload.models.find((model) => model.id === payload.activeModel) ?? payload.models[0];
    if (activeModel?.ballSpawn) {
      setBallSpawnConfig(activeModel.ballSpawn);
      setBallSpawn((current) => clampBallSpawnSettings(current, activeModel.ballSpawn ?? DEFAULT_BALL_SPAWN_CONFIG, "trained"));
    }
  }, []);

  const selectModel = useCallback(
    async (modelId: string) => {
      if (!modelId || modelId === activeModelId || modelSwitching) {
        return;
      }

      setModelError(null);
      setModelSwitching(true);
      const previousPlayback = playback;
      setPlayback("paused");
      const targetModel = models.find((model) => model.id === modelId);
      if (targetModel?.runtimeCompatible === false) {
        const message = targetModel.compatibilityMessage ?? "This model is not compatible with the current runtime.";
        setModelError(message);
        setStatus(message);
        setLoadingProgress({ percent: 100, message });
        await delay(1200);
        setPlayback(previousPlayback);
        setModelSwitching(false);
        return;
      }
      const targetDimension = targetModel?.actionDim ? `${targetModel.actionDim}D` : "selected";
      setStatus(`Switching to ${targetDimension} policy`);
      setLoadingProgress({
        percent: 8,
        message: `Preparing ${targetDimension} policy`
      });

      try {
        setLoadingProgress({ percent: 24, message: "Requesting model switch" });
        const response = await fetch("/api/models/select", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ modelId })
        });
        setLoadingProgress({ percent: 68, message: "Loading policy runtime" });
        if (!response.ok) {
          throw new Error(await modelSwitchErrorMessage(response));
        }
        const parsed = parseModelsPayload(await response.json());
        setLoadingProgress({ percent: 90, message: "Applying model metadata" });
        if (!parsed) {
          throw new Error("Model response was not readable.");
        }
        applyModelsPayload(parsed);
        setLoadingProgress({ percent: 100, message: "Model ready" });
        setPlayback("playing");
        await delay(180);
      } catch (error) {
        const message = error instanceof Error ? error.message : "Model switch failed.";
        setModelError(message);
        setStatus(message);
        setLoadingProgress({ percent: 100, message });
        await delay(1200);
        setPlayback(previousPlayback);
      } finally {
        setModelSwitching(false);
      }
    },
    [activeModelId, applyModelsPayload, modelSwitching, models, playback]
  );

  const updateStatus = useCallback((message: string) => {
    setStatus(message);
  }, []);

  const updateLoadingProgress = useCallback((progress: LoadingProgress) => {
    setLoadingProgress(progress);
  }, []);

  const handleSnapshot = useCallback((nextSnapshot: SimulationSnapshot) => {
    setSnapshot(nextSnapshot);
    if (nextSnapshot.modelId) {
      setActiveModelId((current) => (current === nextSnapshot.modelId ? current : nextSnapshot.modelId));
    }
  }, []);

  useEffect(() => {
    if (selectedModel?.ballSpawn) {
      setBallSpawnConfig(selectedModel.ballSpawn);
      setBallSpawn((current) => clampBallSpawnSettings(current, selectedModel.ballSpawn ?? DEFAULT_BALL_SPAWN_CONFIG, "trained"));
    }
  }, [selectedModel?.id, selectedModel?.ballSpawn]);

  useEffect(() => {
    if (isDocsPage) {
      return;
    }

    let cancelled = false;

    async function loadConfig() {
      try {
        const modelsResponse = await fetch("/api/models");
        if (modelsResponse.ok) {
          const parsed = parseModelsPayload(await modelsResponse.json());
          if (parsed && !cancelled) {
            applyModelsPayload(parsed);
            return;
          }
        }

        const configResponse = await fetch("/api/config");
        if (!configResponse.ok) {
          return;
        }
        const payload = (await configResponse.json()) as { ballSpawn?: unknown };
        const parsedConfig = parseBallSpawnConfig(payload.ballSpawn);
        if (!cancelled) {
          setBallSpawnConfig(parsedConfig);
          setBallSpawn((current) => clampBallSpawnSettings(current, parsedConfig, "trained"));
        }
      } catch {
        return;
      }
    }

    void loadConfig();

    return () => {
      cancelled = true;
    };
  }, [applyModelsPayload, isDocsPage]);

  return (
    <div className="app-shell">
      <header className="app-header">
        <a className="brand" href="/" aria-label="Home">
          <span className="brand-mark" />
          <span>Ping-Pong Keep-Up</span>
        </a>
        <nav className="nav-links" aria-label="Primary navigation">
          {/* Docs is temporarily hidden while the content is being revised. */}
          {/* {isDocsPage ? null : <a href="/docs">Docs</a>} */}
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
                    onSnapshot={handleSnapshot}
                    onStatus={updateStatus}
                    onProgress={updateLoadingProgress}
                    resetSignal={resetSignal}
                    ballSpawnSignal={ballSpawnSignal}
                    cameraResetSignal={cameraResetSignal}
                  />
                </Suspense>
                <div className="viewer-title">
                  <p>A reinforcement learning agent controls a virtual racket to keep a ping-pong ball in play.</p>
                </div>
                <div className="runtime-status">
                  <span className={ready ? "status-dot ready" : "status-dot"} />
                  <span>{ready ? "Simulation Ready" : snapshot.mujocoLoaded ? snapshot.policyMessage : status}</span>
                </div>
                <div className="shared-session-note">Shared server session: changes affect all viewers.</div>
                {showLoadingOverlay ? (
                  <LoadingOverlay
                    status={status}
                    snapshot={snapshot}
                    progress={loadingProgress}
                    title={modelSwitching ? "Switching model" : "Preparing the scene"}
                    kicker={modelSwitching ? "Model selection" : "Starting simulation"}
                    modelSwitching={modelSwitching}
                  />
                ) : null}
              </section>

              <div className={modelPanelOpen ? "model-shell open" : "model-shell closed"}>
                <button
                  className="panel-toggle left"
                  type="button"
                  title={modelPanelOpen ? "Hide model panel" : "Show model panel"}
                  aria-label={modelPanelOpen ? "Hide model panel" : "Show model panel"}
                  onClick={() => setModelPanelOpen((open) => !open)}
                >
                  {modelPanelOpen ? <PanelLeftClose size={18} /> : <PanelLeftOpen size={18} />}
                </button>

                {modelPanelOpen ? (
                  <aside className="model-pane" aria-label="Model and policy output">
                    <ModelControls
                      models={models}
                      activeModelId={activeModelId}
                      selectedModel={selectedModel}
                      switching={modelSwitching}
                      error={modelError}
                      switchHint={modelSwitchHint}
                      onSelect={selectModel}
                    />
                    <ActionVisualizer action={snapshot.action} model={selectedModel} />
                    <div className="policy-network-toggle-slot">
                      <PolicyNetworkToggle
                        visible={policyNetworkOpen}
                        onToggle={() => setPolicyNetworkOpen((open) => !open)}
                      />
                    </div>
                  </aside>
                ) : null}
                {modelPanelOpen ? (
                  <PolicyNetworkPanel
                    model={selectedModel}
                    trace={snapshot.policyTrace}
                    visible={policyNetworkOpen}
                  />
                ) : null}
              </div>

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

                    <BallControls
                      key={activeModelId ?? "ball-controls"}
                      value={ballSpawn}
                      config={ballSpawnConfig}
                      onChange={updateBallSpawn}
                    />
                    <CameraControls
                      value={cameraMode}
                      onChange={setCameraMode}
                      onResetView={() => {
                        setCameraMode("free");
                        setCameraResetSignal((signal) => signal + 1);
                      }}
                    />
                    <VisualizationToggles value={visualization} onChange={setVisualization} />
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
  progress,
  title,
  kicker,
  modelSwitching
}: {
  status: string;
  snapshot: SimulationSnapshot;
  progress: LoadingProgress;
  title: string;
  kicker: string;
  modelSwitching: boolean;
}) {
  const percent = Math.min(100, Math.max(0, Math.round(progress.percent)));
  const message = progress.message || status;
  return (
    <div className="loading-overlay" role="status" aria-live="polite">
      <div className="loading-panel">
        <span className="loading-kicker">{kicker}</span>
        <div className="loading-heading">
          <h2>{title}</h2>
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
          {modelSwitching ? (
            <>
              <span className="done">3D scene stays loaded</span>
              <span>{message}</span>
              <span>Cached models usually switch faster.</span>
            </>
          ) : (
            <>
              <span className={snapshot.mujocoLoaded ? "done" : ""}>
                {snapshot.mujocoLoaded ? "3D scene ready" : message}
              </span>
              <span className={snapshot.policyLoaded ? "done" : ""}>{snapshot.policyMessage}</span>
              <span>First uncached load can take several seconds on a server.</span>
            </>
          )}
        </div>
        <div className="loading-actions">
          <button className="loading-refresh" type="button" onClick={() => window.location.reload()} aria-label="Refresh page">
            <RefreshCw size={15} />
            <span>Refresh</span>
          </button>
        </div>
      </div>
    </div>
  );
}

async function modelSwitchErrorMessage(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string" && body.detail) {
      return body.detail;
    }
  } catch {
    // Fall back to status text below.
  }
  return `Model switch failed (${response.status}).`;
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}
