import type { ModelMetadata } from "../simulation/types";

interface ModelControlsProps {
  models: ModelMetadata[];
  activeModelId: string | null;
  selectedModel: ModelMetadata | null;
  switching: boolean;
  error: string | null;
  onSelect: (modelId: string) => void;
}

export function ModelControls({ models, activeModelId, selectedModel, switching, error, onSelect }: ModelControlsProps) {
  return (
    <div className="control-section model-section">
      <h2>Model</h2>
      <select
        value={activeModelId ?? ""}
        disabled={switching || models.length === 0}
        aria-label="Policy model"
        onChange={(event) => onSelect(event.target.value)}
      >
        {models.map((model) => (
          <option key={model.id} value={model.id}>
            {model.displayName}
          </option>
        ))}
      </select>
      {switching ? <span className="inline-status">Loading selected model...</span> : null}
      {error ? <span className="inline-status error">{error}</span> : null}
      {selectedModel ? (
        <div className="model-info-grid" aria-label="Model information">
          <span>Model</span>
          <strong title={selectedModel.path}>{selectedModel.name}</strong>
          <span>Algorithm</span>
          <strong>{selectedModel.algorithm}</strong>
          <span>Observation</span>
          <strong>{formatDim(selectedModel.observationDim)}</strong>
          <span>Action</span>
          <strong>{formatDim(selectedModel.actionDim)}</strong>
          <span>Action Mode</span>
          <strong title={selectedModel.actionMode ?? ""}>{selectedModel.actionMode ?? "unknown"}</strong>
          <span>Train XY</span>
          <strong>{formatXYRange(selectedModel)}</strong>
          <span>Train Z</span>
          <strong>{formatRange(selectedModel.trainedRanges?.zOffset, "m")}</strong>
          <span>Train Vel</span>
          <strong>{formatRange(selectedModel.trainedRanges?.velocityX, "m/s")}</strong>
          <span>Summary</span>
          <strong title={selectedModel.trainingSummaryPath ?? ""}>{selectedModel.trainingSummaryPath ? "available" : "--"}</strong>
        </div>
      ) : null}
    </div>
  );
}

function formatDim(value: number | null | undefined): string {
  return typeof value === "number" ? `${value}D` : "--";
}

function formatXYRange(model: ModelMetadata): string {
  const trainedRadius = model.ballSpawn?.xyConstraint?.trainedRadius;
  if (model.ballSpawn?.xyConstraint?.sampling === "disk" && typeof trainedRadius === "number") {
    return `r <= ${trainedRadius.toFixed(3)}m`;
  }
  return formatRange(model.trainedRanges?.xOffset, "m");
}

function formatRange(range: { min: number; max: number } | undefined, unit: string): string {
  if (!range) {
    return "--";
  }
  return `${range.min.toFixed(3)}..${range.max.toFixed(3)} ${unit}`;
}
