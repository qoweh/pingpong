import { useEffect, useRef } from "react";

import { DemoController } from "../simulation/demoController";
import type {
  CameraMode,
  PlaybackState,
  SimulationSnapshot,
  Vec3,
  VisualizationSettings
} from "../simulation/types";

interface SimulationCanvasProps {
  playback: PlaybackState;
  ballPosition: Vec3;
  cameraMode: CameraMode;
  visualization: VisualizationSettings;
  onSnapshot: (snapshot: SimulationSnapshot) => void;
  onStatus: (message: string) => void;
  resetSignal: number;
}

export function SimulationCanvas({
  playback,
  ballPosition,
  cameraMode,
  visualization,
  onSnapshot,
  onStatus,
  resetSignal
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
    controllerRef.current?.setBallPosition(ballPosition);
  }, [ballPosition]);

  useEffect(() => {
    if (resetSignal > 0) {
      controllerRef.current?.reset();
    }
  }, [resetSignal]);

  return <div ref={hostRef} className="simulation-canvas" />;
}
