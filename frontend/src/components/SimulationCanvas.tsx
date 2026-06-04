import { useEffect, useRef } from "react";

import { DemoController } from "../simulation/demoController";
import type {
  BallSpawnSettings,
  CameraMode,
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
  resetSignal: number;
  ballResetSignal: number;
}

export function SimulationCanvas({
  playback,
  cameraMode,
  visualization,
  ballSpawn,
  onSnapshot,
  onStatus,
  resetSignal,
  ballResetSignal
}: SimulationCanvasProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const controllerRef = useRef<DemoController | null>(null);

  useEffect(() => {
    if (!hostRef.current) {
      return;
    }

    const controller = new DemoController(hostRef.current, onSnapshot, onStatus);
    controllerRef.current = controller;
    void controller.initialize();

    return () => {
      controller.dispose();
      controllerRef.current = null;
    };
  }, [onSnapshot, onStatus]);

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
    if (ballResetSignal > 0) {
      controllerRef.current?.resetBall(ballSpawn);
    }
  }, [ballResetSignal, ballSpawn]);

  return <div ref={hostRef} className="simulation-canvas" />;
}
