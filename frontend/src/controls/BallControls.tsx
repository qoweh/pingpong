import { RotateCcw } from "lucide-react";
import { useState } from "react";

import type { BallSpawnConfig, BallSpawnSettings } from "../simulation/types";
import { DEFAULT_BALL_SPAWN_CONFIG } from "../simulation/types";
import {
  clampBallSpawnSettings,
  isWithinTrainedBallSpawnRange,
  rangeBounds,
  type BallSpawnClampMode
} from "../simulation/ballSpawnConfig";

interface BallControlsProps {
  value: BallSpawnSettings;
  config: BallSpawnConfig;
  onChange: (value: BallSpawnSettings) => void;
}

const AXES: Array<{
  key: keyof BallSpawnSettings;
  label: string;
  unit: string;
}> = [
  { key: "xOffset", label: "X Pos", unit: "m" },
  { key: "yOffset", label: "Y Pos", unit: "m" },
  { key: "zOffset", label: "Z Pos", unit: "m" },
  { key: "velocityX", label: "X Vel", unit: "m/s" },
  { key: "velocityY", label: "Y Vel", unit: "m/s" },
  { key: "velocityZ", label: "Z Vel", unit: "m/s" }
];

export function BallControls({ value, config, onChange }: BallControlsProps) {
  const activeConfig = config ?? DEFAULT_BALL_SPAWN_CONFIG;
  const [useExtendedRange, setUseExtendedRange] = useState(false);
  const clampMode: BallSpawnClampMode = useExtendedRange ? "extended" : "trained";
  const trained = isWithinTrainedBallSpawnRange(value, activeConfig);

  const updateValue = (key: keyof BallSpawnSettings, rawValue: number) => {
    if (!Number.isFinite(rawValue)) {
      return;
    }
    onChange(clampBallSpawnSettings({ ...value, [key]: rawValue }, activeConfig, clampMode));
  };

  return (
    <div className="control-section">
      <div className="section-heading">
        <h2>Ball Start</h2>
        <span className={trained ? "range-badge trained" : "range-badge extended"}>
          {trained ? "Trained range" : "Extended range"}
        </span>
      </div>
      <label className="toggle-row compact">
        <input
          type="checkbox"
          checked={useExtendedRange}
          onChange={(event) => {
            const nextMode: BallSpawnClampMode = event.target.checked ? "extended" : "trained";
            setUseExtendedRange(event.target.checked);
            onChange(clampBallSpawnSettings(value, activeConfig, nextMode));
          }}
        />
        <span>Stress range</span>
      </label>
      {AXES.map((axis) => {
        const range = activeConfig.ranges[axis.key];
        const bounds = rangeBounds(range, clampMode);
        const trainedMin = range.trainedMin ?? range.min;
        const trainedMax = range.trainedMax ?? range.max;
        const title = `trained ${trainedMin.toFixed(3)}..${trainedMax.toFixed(3)} ${axis.unit}; tested ${range.min.toFixed(
          3
        )}..${range.max.toFixed(3)} ${axis.unit}`;
        return (
          <label className="range-row" key={axis.key} title={title}>
            <span>{axis.label}</span>
            <input
              type="range"
              min={bounds.min}
              max={bounds.max}
              step={range.step}
              value={value[axis.key]}
              onChange={(event) => {
                updateValue(axis.key, Number(event.target.value));
              }}
            />
            <input
              className="range-number"
              type="number"
              min={bounds.min}
              max={bounds.max}
              step={range.step}
              value={value[axis.key].toFixed(3)}
              aria-label={axis.label}
              onChange={(event) => {
                updateValue(axis.key, Number(event.target.value));
              }}
            />
          </label>
        );
      })}
      <div className="button-row">
        <button
          className="action-button muted full"
          type="button"
          onClick={() => onChange(clampBallSpawnSettings(activeConfig.defaults, activeConfig, clampMode))}
        >
          <RotateCcw size={16} />
          <span>Reset Ball</span>
        </button>
      </div>
    </div>
  );
}
