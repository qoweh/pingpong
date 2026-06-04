import { RotateCcw } from "lucide-react";

import type { BallSpawnSettings } from "../simulation/types";
import { DEFAULT_BALL_SPAWN } from "../simulation/types";

interface BallControlsProps {
  value: BallSpawnSettings;
  onChange: (value: BallSpawnSettings) => void;
}

const AXES: Array<{
  key: keyof BallSpawnSettings;
  label: string;
  min: number;
  max: number;
  step: number;
}> = [
  { key: "xOffset", label: "X Offset", min: -0.12, max: 0.12, step: 0.005 },
  { key: "yOffset", label: "Y Offset", min: -0.12, max: 0.12, step: 0.005 },
  { key: "zOffset", label: "Z Offset", min: 0.15, max: 0.7, step: 0.005 },
  { key: "velocityX", label: "X Velocity", min: -0.45, max: 0.45, step: 0.01 },
  { key: "velocityY", label: "Y Velocity", min: -0.45, max: 0.45, step: 0.01 },
  { key: "velocityZ", label: "Z Velocity", min: -0.4, max: 0.4, step: 0.01 }
];

export function BallControls({ value, onChange }: BallControlsProps) {
  return (
    <div className="control-section">
      <h2>Ball Spawn</h2>
      {AXES.map((axis) => (
        <label className="range-row" key={axis.key}>
          <span>{axis.label}</span>
          <input
            type="range"
            min={axis.min}
            max={axis.max}
            step={axis.step}
            value={value[axis.key]}
            onChange={(event) => {
              onChange({ ...value, [axis.key]: Number(event.target.value) });
            }}
          />
          <output>{value[axis.key].toFixed(2)}</output>
        </label>
      ))}
      <div className="button-row">
        <button className="action-button muted full" type="button" onClick={() => onChange(DEFAULT_BALL_SPAWN)}>
          <RotateCcw size={16} />
          <span>Default</span>
        </button>
      </div>
    </div>
  );
}
