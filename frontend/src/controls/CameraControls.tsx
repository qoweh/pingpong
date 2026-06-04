import type { CameraMode } from "../simulation/types";

interface CameraControlsProps {
  value: CameraMode;
  onChange: (value: CameraMode) => void;
}

const CAMERA_OPTIONS: Array<{ value: CameraMode; label: string }> = [
  { value: "free", label: "Free View" },
  { value: "north", label: "North View" },
  { value: "south", label: "South View" },
  { value: "east", label: "East View" },
  { value: "west", label: "West View" },
  { value: "top", label: "Top View" },
  { value: "four", label: "4-Camera View" }
];

export function CameraControls({ value, onChange }: CameraControlsProps) {
  return (
    <div className="control-section">
      <h2>Camera</h2>
      <select value={value} onChange={(event) => onChange(event.target.value as CameraMode)}>
        {CAMERA_OPTIONS.map((option) => (
          <option value={option.value} key={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </div>
  );
}
