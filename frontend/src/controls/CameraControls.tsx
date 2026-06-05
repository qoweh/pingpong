import { RotateCcw } from "lucide-react";

import type { CameraMode } from "../simulation/types";

interface CameraControlsProps {
  value: CameraMode;
  onChange: (value: CameraMode) => void;
  onResetView: () => void;
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

export function CameraControls({ value, onChange, onResetView }: CameraControlsProps) {
  return (
    <div className="control-section">
      <h2>Camera</h2>
      <div className="select-action-row">
        <select value={value} onChange={(event) => onChange(event.target.value as CameraMode)}>
          {CAMERA_OPTIONS.map((option) => (
            <option value={option.value} key={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        <button
          className="icon-button compact"
          type="button"
          title="Reset free view"
          aria-label="Reset free view"
          onClick={onResetView}
        >
          <RotateCcw size={17} />
        </button>
      </div>
    </div>
  );
}
