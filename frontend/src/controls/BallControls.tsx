import { Crosshair, RotateCcw } from "lucide-react";

import type { BallSpawnSettings } from "../simulation/types";
import { DEFAULT_BALL_SPAWN } from "../simulation/types";

interface BallControlsProps {
  value: BallSpawnSettings;
  onChange: (value: BallSpawnSettings) => void;
  onApply: (value: BallSpawnSettings) => void;
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
  { key: "velocityZ", label: "Initial Z Vel", min: -0.4, max: 0.4, step: 0.01 }
];

export function BallControls({ value, onChange, onApply }: BallControlsProps) {
  const applyDefault = () => {
    onChange(DEFAULT_BALL_SPAWN);
    onApply(DEFAULT_BALL_SPAWN);
  };

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
            onChange={(event) => onChange({ ...value, [axis.key]: Number(event.target.value) })}
          />
          <output>{value[axis.key].toFixed(2)}</output>
        </label>
      ))}
      <div className="button-row">
        <button className="action-button" type="button" onClick={() => onApply(value)}>
          <Crosshair size={16} />
          <span>Apply</span>
        </button>
        <button className="action-button muted" type="button" onClick={applyDefault}>
          <RotateCcw size={16} />
          <span>Default</span>
        </button>
      </div>
    </div>
  );
}
