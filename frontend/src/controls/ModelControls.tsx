import { useMemo } from "react";

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
  const groups = useMemo(() => groupModels(models), [models]);

  return (
    <div className="control-section model-section">
      <div className="section-heading">
        <h2>Model</h2>
        {selectedModel?.dimensionGroup ? <span>{selectedModel.dimensionGroup}</span> : null}
      </div>
      <select
        value={activeModelId ?? ""}
        disabled={switching || models.length === 0}
        aria-label="Policy model"
        onChange={(event) => onSelect(event.target.value)}
      >
        {groups.map((group) => (
          <optgroup key={group.label} label={`${group.label} models`}>
            {group.models.map((model) => (
              <option key={model.id} value={model.id}>
                {optionLabel(model)}
              </option>
            ))}
          </optgroup>
        ))}
      </select>
      {switching ? <span className="inline-status">Loading selected model...</span> : null}
      {error ? <span className="inline-status error">{error}</span> : null}
      {selectedModel ? (
        <div className="model-info-grid" aria-label="Model information">
          <span>Series</span>
          <strong title={selectedModel.path}>{selectedModel.name}</strong>
          <span>Run</span>
          <strong title={selectedModel.rawRunName ?? selectedModel.path}>
            {selectedModel.detailName ?? selectedModel.rawRunName ?? "--"}
          </strong>
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

type ModelGroup = {
  label: string;
  sortDimension: number;
  models: ModelMetadata[];
};

function groupModels(models: ModelMetadata[]): ModelGroup[] {
  const grouped = new Map<string, ModelGroup>();
  for (const model of models) {
    const label = model.dimensionGroup ?? formatDim(model.actionDim);
    const current = grouped.get(label);
    const sortDimension = model.sortDimension ?? model.actionDim ?? -1;
    if (current) {
      current.models.push(model);
    } else {
      grouped.set(label, { label, sortDimension, models: [model] });
    }
  }

  return [...grouped.values()]
    .map((group) => ({
      ...group,
      models: group.models.sort((left, right) => modelOrder(left, right))
    }))
    .sort((left, right) => right.sortDimension - left.sortDimension || left.label.localeCompare(right.label));
}

function modelOrder(left: ModelMetadata, right: ModelMetadata): number {
  const versionOrder = (right.sortVersion ?? 0) - (left.sortVersion ?? 0);
  if (versionOrder !== 0) {
    return versionOrder;
  }
  return left.displayName.localeCompare(right.displayName);
}

function optionLabel(model: ModelMetadata): string {
  if (model.versionLabel || model.detailName) {
    return [model.versionLabel, model.detailName].filter(Boolean).join(" · ");
  }
  return model.displayName;
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
