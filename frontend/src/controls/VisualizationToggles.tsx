import type { VisualizationSettings } from "../simulation/types";

interface VisualizationTogglesProps {
  value: VisualizationSettings;
  onChange: (value: VisualizationSettings) => void;
}

const TOGGLES: Array<{ key: "trail" | "targetBand" | "contactMarker"; label: string }> = [
  { key: "trail", label: "Ball trail" },
  { key: "targetBand", label: "Target height" },
  { key: "contactMarker", label: "Contact marker" }
];

export function VisualizationToggles({ value, onChange }: VisualizationTogglesProps) {
  return (
    <div className="control-section">
      <h2>Visualization</h2>
      <div className="toggle-list">
        {TOGGLES.map((toggle) => (
          <label className="toggle-row" key={toggle.key}>
            <input
              type="checkbox"
              checked={value[toggle.key]}
              onChange={(event) => onChange({ ...value, [toggle.key]: event.target.checked })}
            />
            <span>{toggle.label}</span>
          </label>
        ))}
      </div>
    </div>
  );
}
