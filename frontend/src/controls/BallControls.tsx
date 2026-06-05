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
  { key: "xOffset", label: "X Position", min: -0.15, max: 0.15, step: 0.005 },
  { key: "yOffset", label: "Y Position", min: -0.15, max: 0.15, step: 0.005 },
  { key: "zOffset", label: "Z Position", min: 0.18, max: 0.56, step: 0.005 },
  { key: "velocityX", label: "X Velocity", min: -0.06, max: 0.06, step: 0.005 },
  { key: "velocityY", label: "Y Velocity", min: -0.06, max: 0.06, step: 0.005 },
  { key: "velocityZ", label: "Z Velocity", min: -0.18, max: 0.04, step: 0.005 }
];

export function BallControls({ value, onChange }: BallControlsProps) {
  return (
    <div className="control-section">
      <h2>Ball Start</h2>
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
          <span>Reset Ball</span>
        </button>
      </div>
    </div>
  );
}
