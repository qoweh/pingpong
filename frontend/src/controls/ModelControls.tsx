import { useMemo } from "react";

import type { ModelMetadata } from "../simulation/types";

interface ModelControlsProps {
  models: ModelMetadata[];
  activeModelId: string | null;
  selectedModel: ModelMetadata | null;
  switching: boolean;
  error: string | null;
  switchHint: string;
  onSelect: (modelId: string) => void;
}

export function ModelControls({
  models,
  activeModelId,
  selectedModel,
  switching,
  error,
  switchHint,
  onSelect
}: ModelControlsProps) {
  const dimensionOptions = useMemo(() => orderModels(models), [models]);

  return (
    <div className="control-section model-section">
      <div className="section-heading">
        <h2>Action Dimension</h2>
      </div>
      <span className="inline-status">{switchHint}</span>
      <div className="dimension-picker" aria-label="Policy action dimensions">
        {dimensionOptions.map((model) => (
          <button
            className={model.id === activeModelId ? "dimension-option active" : "dimension-option"}
            type="button"
            key={model.id}
            disabled={switching || model.runtimeCompatible === false}
            aria-pressed={model.id === activeModelId}
            title={model.name}
            onClick={() => onSelect(model.id)}
          >
            <strong>{formatDim(model.actionDim)}</strong>
            <span>{optionLabel(model)}</span>
          </button>
        ))}
      </div>
      {switching ? <span className="inline-status">Loading selected model...</span> : null}
      {error ? <span className="inline-status error">{error}</span> : null}
      {selectedModel?.runtimeCompatible === false && selectedModel.compatibilityMessage ? (
        <span className="inline-status error">{selectedModel.compatibilityMessage}</span>
      ) : null}
      {selectedModel ? (
        <div className="model-info-grid" aria-label="Model information">
          <span>Observation</span>
          <strong>{formatDim(selectedModel.observationDim)}</strong>
          <span>Action</span>
          <strong>{formatDim(selectedModel.actionDim)}</strong>
        </div>
      ) : null}
    </div>
  );
}

function orderModels(models: ModelMetadata[]): ModelMetadata[] {
  return [...models].sort((left, right) => {
    const dimensionOrder = (right.sortDimension ?? right.actionDim ?? -1) - (left.sortDimension ?? left.actionDim ?? -1);
    if (dimensionOrder !== 0) {
      return dimensionOrder;
    }
    return modelOrder(left, right);
  });
}

function modelOrder(left: ModelMetadata, right: ModelMetadata): number {
  const versionOrder = (right.sortVersion ?? 0) - (left.sortVersion ?? 0);
  if (versionOrder !== 0) {
    return versionOrder;
  }
  return left.displayName.localeCompare(right.displayName);
}

function optionLabel(model: ModelMetadata): string {
  return model.versionLabel || model.name || model.displayName;
}

function formatDim(value: number | null | undefined): string {
  return typeof value === "number" ? `${value}D` : "--";
}
