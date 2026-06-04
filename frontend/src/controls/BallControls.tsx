import type { Vec3 } from "../simulation/types";

interface BallControlsProps {
  value: Vec3;
  onChange: (value: Vec3) => void;
}

const AXES: Array<{ label: string; index: 0 | 1 | 2; min: number; max: number; step: number }> = [
  { label: "X Offset", index: 0, min: -0.12, max: 0.12, step: 0.005 },
  { label: "Y Offset", index: 1, min: -0.12, max: 0.12, step: 0.005 },
  { label: "Height", index: 2, min: 0.2, max: 0.55, step: 0.005 }
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
