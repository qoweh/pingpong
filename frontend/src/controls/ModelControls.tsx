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
  const groups = useMemo(() => groupModels(models), [models]);

  return (
    <div className="control-section model-section">
      <div className="section-heading">
        <h2>Model</h2>
        {selectedModel?.dimensionGroup ? <span>{selectedModel.dimensionGroup}</span> : null}
      </div>
      <span className="inline-status">{switchHint}</span>
      <div className="model-keypad" aria-label="Policy models">
        {groups.map((group) => (
          <div className="model-keypad-group" key={group.label}>
            <span>{group.label}</span>
            <div className="model-keypad-buttons">
              {group.models.map((model) => (
                <button
                  className={model.id === activeModelId ? "model-key active" : "model-key"}
                  type="button"
                  key={model.id}
                  disabled={switching || model.runtimeCompatible === false}
                  aria-pressed={model.id === activeModelId}
                  title={model.name}
                  onClick={() => onSelect(model.id)}
                >
                  {optionLabel(model)}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
      {switching ? <span className="inline-status">Loading selected model...</span> : null}
      {error ? <span className="inline-status error">{error}</span> : null}
      {selectedModel?.runtimeCompatible === false && selectedModel.compatibilityMessage ? (
        <span className="inline-status error">{selectedModel.compatibilityMessage}</span>
      ) : null}
      {selectedModel ? (
        <div className="model-info-grid" aria-label="Model information">
          <span>Series</span>
          <strong>{selectedModel.name}</strong>
          <span>Observation</span>
          <strong>{formatDim(selectedModel.observationDim)}</strong>
          <span>Action</span>
          <strong>{formatDim(selectedModel.actionDim)}</strong>
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
  return model.versionLabel || model.name || model.displayName;
}

function formatDim(value: number | null | undefined): string {
  return typeof value === "number" ? `${value}D` : "--";
}
