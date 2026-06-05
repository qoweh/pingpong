import { useEffect, useRef } from "react";

import { DemoController } from "../simulation/demoController";
import type {
  BallSpawnSettings,
  CameraMode,
  LoadingProgress,
  PlaybackState,
  SimulationSnapshot,
  VisualizationSettings
} from "../simulation/types";

interface SimulationCanvasProps {
  playback: PlaybackState;
  cameraMode: CameraMode;
  visualization: VisualizationSettings;
  ballSpawn: BallSpawnSettings;
  onSnapshot: (snapshot: SimulationSnapshot) => void;
  onStatus: (message: string) => void;
  onProgress: (progress: LoadingProgress) => void;
  resetSignal: number;
  ballSpawnSignal: number;
  cameraResetSignal: number;
}

export function SimulationCanvas({
  playback,
  cameraMode,
  visualization,
  ballSpawn,
  onSnapshot,
  onStatus,
  onProgress,
  resetSignal,
  ballSpawnSignal,
  cameraResetSignal
}: SimulationCanvasProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const controllerRef = useRef<DemoController | null>(null);

  useEffect(() => {
    if (!hostRef.current) {
      return;
    }

    const controller = new DemoController(hostRef.current, onSnapshot, onStatus, onProgress);
    controllerRef.current = controller;
    void controller.initialize();

    return () => {
      controller.dispose();
      controllerRef.current = null;
    };
  }, [onSnapshot, onStatus, onProgress]);

  useEffect(() => {
    controllerRef.current?.setPlayback(playback);
  }, [playback]);

  useEffect(() => {
    controllerRef.current?.setCameraMode(cameraMode);
  }, [cameraMode]);

  useEffect(() => {
    controllerRef.current?.setVisualization(visualization);
  }, [visualization]);

  useEffect(() => {
    if (resetSignal > 0) {
      controllerRef.current?.reset();
    }
  }, [resetSignal]);

  useEffect(() => {
    if (ballSpawnSignal > 0) {
      controllerRef.current?.spawnBall(ballSpawn);
    }
  }, [ballSpawnSignal, ballSpawn]);

  useEffect(() => {
    if (cameraResetSignal > 0) {
      controllerRef.current?.resetCamera();
    }
  }, [cameraResetSignal]);

  return <div ref={hostRef} className="simulation-canvas" />;
}
