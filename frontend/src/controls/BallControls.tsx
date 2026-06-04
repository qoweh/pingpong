import type { Vec3 } from "../simulation/types";

interface BallControlsProps {
  value: Vec3;
  onChange: (value: Vec3) => void;
}

const AXES: Array<{ label: string; index: 0 | 1 | 2; min: number; max: number; step: number }> = [
  { label: "X", index: 0, min: -0.45, max: 0.85, step: 0.01 },
  { label: "Y", index: 1, min: -0.55, max: 0.55, step: 0.01 },
  { label: "Z", index: 2, min: 0.7, max: 1.6, step: 0.01 }
];

export function BallControls({ value, onChange }: BallControlsProps) {
  return (
    <div className="control-section">
      <h2>Ball Position</h2>
      {AXES.map((axis) => (
        <label className="range-row" key={axis.label}>
          <span>{axis.label}</span>
          <input
            type="range"
            min={axis.min}
            max={axis.max}
            step={axis.step}
            value={value[axis.index]}
            onChange={(event) => {
              const next = [...value] as Vec3;
              next[axis.index] = Number(event.target.value);
              onChange(next);
            }}
          />
          <output>{value[axis.index].toFixed(2)}m</output>
        </label>
      ))}
    </div>
  );
}
