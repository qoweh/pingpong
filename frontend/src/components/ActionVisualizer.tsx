import type { ModelMetadata } from "../simulation/types";

interface ActionVisualizerProps {
  action: number[] | null;
  model: ModelMetadata | null;
}

export function ActionVisualizer({ action, model }: ActionVisualizerProps) {
  const actionDim = model?.actionDim ?? action?.length ?? 0;
  const labels = labelsFor(model, actionDim);
  const values = Array.from({ length: actionDim }, (_, index) => action?.[index] ?? 0);

  if (actionDim <= 0) {
    return null;
  }

  return (
    <div className="control-section action-visualizer">
      <div className="section-heading">
        <h2>Policy Output</h2>
        <span>{actionDim} controls</span>
      </div>
      <div className="action-bar-list">
        {values.map((value, index) => {
          const limit = actionLimit(model, index);
          const normalized = Math.max(-1, Math.min(1, value / limit));
          const fillStyle =
            normalized >= 0
              ? { left: "50%", width: `${Math.abs(normalized) * 50}%` }
              : { right: "50%", width: `${Math.abs(normalized) * 50}%` };
          return (
            <div className="action-row" key={`${labels[index]}-${index}`}>
              <span title={labels[index]}>{labels[index]}</span>
              <div className="action-track" aria-hidden="true">
                <i style={fillStyle} />
              </div>
              <strong>{value.toFixed(3)}</strong>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function labelsFor(model: ModelMetadata | null, actionDim: number): string[] {
  const labels = model?.actionLabels?.length ? [...model.actionLabels] : [];
  while (labels.length < actionDim) {
    labels.push(`Control ${labels.length + 1}`);
  }
  return labels.slice(0, actionDim);
}

function actionLimit(model: ModelMetadata | null, index: number): number {
  const high = Math.abs(model?.actionHigh?.[index] ?? 0);
  const low = Math.abs(model?.actionLow?.[index] ?? 0);
  const limit = Math.max(high, low);
  return limit > 0 ? limit : 1;
}
