import { RotateCcw } from "lucide-react";

import type { BallSpawnConfig, BallSpawnSettings } from "../simulation/types";
import { DEFAULT_BALL_SPAWN_CONFIG } from "../simulation/types";

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

  const updateValue = (key: keyof BallSpawnSettings, rawValue: number) => {
    if (!Number.isFinite(rawValue)) {
      return;
    }
    const range = activeConfig.ranges[key];
    const nextValue = Math.min(Math.max(rawValue, range.min), range.max);
    onChange({ ...value, [key]: nextValue });
  };

  return (
    <div className="control-section">
      <h2>Ball Start</h2>
      {AXES.map((axis) => {
        const range = activeConfig.ranges[axis.key];
        const trainedMin = range.trainedMin ?? range.min;
        const trainedMax = range.trainedMax ?? range.max;
        const title = `trained ${trainedMin.toFixed(3)}..${trainedMax.toFixed(3)} ${axis.unit}`;
        return (
          <label className="range-row" key={axis.key} title={title}>
            <span>{axis.label}</span>
            <input
              type="range"
              min={range.min}
              max={range.max}
              step={range.step}
              value={value[axis.key]}
              onChange={(event) => {
                updateValue(axis.key, Number(event.target.value));
              }}
            />
            <input
              className="range-number"
              type="number"
              min={range.min}
              max={range.max}
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
        <button className="action-button muted full" type="button" onClick={() => onChange(activeConfig.defaults)}>
          <RotateCcw size={16} />
          <span>Reset Ball</span>
        </button>
      </div>
    </div>
  );
}
