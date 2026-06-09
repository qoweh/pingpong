import { Eye, EyeOff, Network } from "lucide-react";

import type { ModelMetadata, PolicyTrace } from "../simulation/types";

interface PolicyNetworkVisualizerProps {
  model: ModelMetadata | null;
  trace: PolicyTrace | null;
  visible: boolean;
}

interface PolicyNetworkToggleProps {
  visible: boolean;
  onToggle: () => void;
}

type NodePoint = {
  id: string;
  x: number;
  y: number;
  value: number;
  label?: string;
  radius?: number;
};

const V39_INPUT_LABELS: Record<number, string> = {
  15: "racket y",
  16: "racket z",
  23: "ball x",
  24: "ball y",
  25: "ball z",
  28: "ball vz",
  41: "next x",
  42: "next y",
  43: "next t",
  47: "desired vx",
  48: "desired vy",
  49: "desired vz"
};
const V39_INPUT_INDICES = [15, 16, 23, 24, 25, 28, 41, 42, 43, 47, 48, 49];

export function PolicyNetworkToggle({ visible, onToggle }: PolicyNetworkToggleProps) {
  return (
    <button
      className="policy-network-toggle"
      type="button"
      title={visible ? "Hide policy network" : "Show policy network"}
      aria-label={visible ? "Hide policy network" : "Show policy network"}
      aria-pressed={visible}
      onClick={onToggle}
    >
      {visible ? <EyeOff size={15} /> : <Eye size={15} />}
      <span>Policy Network</span>
    </button>
  );
}

export function PolicyNetworkPanel({ model, trace, visible }: PolicyNetworkVisualizerProps) {
  const observationDim = model?.observationDim ?? trace?.observation.length ?? 0;
  const actionDim = model?.actionDim ?? trace?.action.length ?? 0;
  const hiddenDims = hiddenLayerDims(model, trace);
  const architectureText = [observationDim || "?", ...hiddenDims, actionDim || "?"].join(" -> ");

  return (
    <div className={visible ? "policy-network-panel-wrap open" : "policy-network-panel-wrap"} aria-hidden={!visible}>
      {visible ? (
        <div className="policy-network-card" aria-label="Policy decision flow visualization">
          <div className="policy-network-heading">
            <span>
              <Network size={13} />
              Policy Decision Flow
            </span>
            <strong>{architectureText}</strong>
          </div>
          <PolicyNetworkSvg model={model} trace={trace} observationDim={observationDim} actionDim={actionDim} hiddenDims={hiddenDims} />
        </div>
      ) : null}
    </div>
  );
}

function PolicyNetworkSvg({
  model,
  trace,
  observationDim,
  actionDim,
  hiddenDims
}: {
  model: ModelMetadata | null;
  trace: PolicyTrace | null;
  observationDim: number;
  actionDim: number;
  hiddenDims: number[];
}) {
  const inputIndices = selectedInputIndices(observationDim);
  const inputs = inputIndices.map((index) => trace?.observation[index] ?? 0);
  const firstHidden = layerValues(trace?.hiddenLayers[0], hiddenDims[0] ?? 0);
  const secondHidden = layerValues(trace?.hiddenLayers[1], hiddenDims[1] ?? 0);
  const actions = Array.from({ length: Math.max(actionDim, trace?.action.length ?? 0) }, (_, index) => trace?.action[index] ?? 0);
  const outputIndices = selectedOutputIndices(actions.length);
  const outputValues = outputIndices.map((index) => actionDisplaySignal(model, actions[index] ?? 0, index));

  const inputNodes = makeVerticalNodes("in", 72, inputs, inputIndices.map((index) => inputLabel(index, observationDim)), 46, 312);
  const hidden1Nodes = makeGridNodes("h1", 224, 58, 112, 214, firstHidden.values);
  const hidden2Nodes = makeGridNodes("h2", 404, 58, 112, 214, secondHidden.values);
  const outputNodes = makeVerticalNodes("out", 580, outputValues, outputIndices.map((index) => outputLabel(model, index)), 46, 312);

  return (
    <svg className="policy-network-svg" viewBox="0 0 660 360" role="img" aria-label="Live policy network">
      <g className="network-edges">
        {edgesBetween(inputNodes, hidden1Nodes, 4).map((edge) => drawEdge(edge.from, edge.to, edge.key))}
        {edgesBetween(hidden1Nodes, hidden2Nodes, 2).map((edge) => drawEdge(edge.from, edge.to, edge.key))}
        {edgesBetween(hidden2Nodes, outputNodes, 1).map((edge) => drawEdge(edge.from, edge.to, edge.key))}
      </g>
      <NetworkColumn nodes={inputNodes} label="observation" align="left" />
      <NetworkGrid nodes={hidden1Nodes} label="hidden 1" count={firstHidden.total} />
      <NetworkGrid nodes={hidden2Nodes} label="hidden 2" count={secondHidden.total} />
      <NetworkColumn nodes={outputNodes} label="policy output" align="right" />
      <text x="330" y="346" textAnchor="middle" className="network-caption">
        red + / blue - / action color boosted near zero
      </text>
    </svg>
  );
}

function NetworkColumn({ nodes, label, align = "center" }: { nodes: NodePoint[]; label: string; align?: "left" | "center" | "right" }) {
  return (
    <g>
      <text x={nodes[0]?.x ?? 0} y="12" textAnchor="middle" className="network-column-label">
        {label}
      </text>
      {nodes.map((node) => (
        <g key={node.id}>
          <circle cx={node.x} cy={node.y} r={node.radius ?? 6.8} className="network-node" fill={nodeColor(node.value)} />
          {node.label ? (
            <text
              x={align === "left" ? node.x - 12 : node.x + 12}
              y={node.y + 3}
              textAnchor={align === "left" ? "end" : "start"}
              className="network-node-label"
            >
              {node.label}
            </text>
          ) : null}
        </g>
      ))}
    </g>
  );
}

function NetworkGrid({ nodes, label, count }: { nodes: NodePoint[]; label: string; count: number }) {
  const centerX = nodes.reduce((sum, node) => sum + node.x, 0) / Math.max(1, nodes.length);
  return (
    <g>
      <text x={centerX} y="22" textAnchor="middle" className="network-column-label">
        {label}
      </text>
      <text x={centerX} y="38" textAnchor="middle" className="network-caption">
        {count} units
      </text>
      {nodes.map((node) => (
        <circle key={node.id} cx={node.x} cy={node.y} r={node.radius ?? 4.4} className="network-node" fill={nodeColor(node.value)} />
      ))}
    </g>
  );
}

function drawEdge(from: NodePoint, to: NodePoint, key: string) {
  const mixed = (from.value + to.value) / 2;
  const opacity = Math.min(0.52, 0.08 + Math.abs(mixed) * 0.34);
  return (
    <line
      key={key}
      x1={from.x}
      y1={from.y}
      x2={to.x}
      y2={to.y}
      stroke={mixed >= 0 ? "#ff6b5f" : "#6f83ff"}
      strokeOpacity={opacity}
      strokeWidth={0.5 + Math.abs(mixed) * 1.3}
    />
  );
}

function makeVerticalNodes(prefix: string, x: number, values: number[], labels?: string[], top = 28, bottom = 166): NodePoint[] {
  const count = Math.max(1, values.length);
  return values.map((value, index) => ({
    id: `${prefix}-${index}`,
    x,
    y: top + (index * (bottom - top)) / Math.max(1, count - 1),
    value: clampSignal(value),
    label: labels?.[index],
    radius: values.length > 12 ? 5.4 : 6.8
  }));
}

function makeGridNodes(prefix: string, centerX: number, top: number, width: number, height: number, values: number[]): NodePoint[] {
  const count = Math.max(1, values.length);
  const columns = Math.ceil(Math.sqrt(count));
  const rows = Math.ceil(count / columns);
  const radius = count <= 64 ? 4.4 : 3.6;
  return values.map((value, index) => {
    const column = index % columns;
    const row = Math.floor(index / columns);
    return {
      id: `${prefix}-${index}`,
      x: centerX - width / 2 + (column * width) / Math.max(1, columns - 1),
      y: top + (row * height) / Math.max(1, rows - 1),
      value: clampSignal(value),
      radius
    };
  });
}

function edgesBetween(from: NodePoint[], to: NodePoint[], fanout: number) {
  const edges: Array<{ from: NodePoint; to: NodePoint; key: string }> = [];
  from.forEach((source, sourceIndex) => {
    for (let offset = 0; offset < fanout; offset += 1) {
      const targetIndex = (sourceIndex * 2 + offset) % to.length;
      edges.push({ from: source, to: to[targetIndex], key: `${source.id}-${to[targetIndex].id}-${offset}` });
    }
  });
  return edges;
}

function hiddenLayerDims(model: ModelMetadata | null, trace: PolicyTrace | null): number[] {
  const fromTrace = trace?.hiddenLayers.map((layer) => layer.length).filter((value) => value > 0) ?? [];
  if (fromTrace.length) {
    return fromTrace.slice(0, 2);
  }
  const architecture = model?.policy?.architecture ?? [];
  const hidden = architecture
    .map((item) => {
      const match = item.match(/(?:hidden|Linear).*?(\d+)/i);
      return match ? Number(match[1]) : null;
    })
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value) && value > 0);
  return hidden.length ? hidden.slice(0, 2) : [64, 64];
}

function selectedInputIndices(observationDim: number): number[] {
  if (observationDim === 55) {
    return V39_INPUT_INDICES;
  }
  if (observationDim <= 12) {
    return Array.from({ length: Math.max(0, observationDim) }, (_, index) => index);
  }
  return Array.from({ length: 12 }, (_, index) => Math.round((index * (observationDim - 1)) / 11));
}

function selectedOutputIndices(actionDim: number): number[] {
  if (actionDim <= 20) {
    return Array.from({ length: Math.max(0, actionDim) }, (_, index) => index);
  }
  return Array.from({ length: 20 }, (_, index) => Math.round((index * (actionDim - 1)) / 19));
}

function layerValues(values: number[] | undefined, dim: number): { values: number[]; total: number } {
  const effectiveDim = Math.max(dim, values?.length ?? 0);
  if (!effectiveDim) {
    return { values: [0], total: 0 };
  }
  const count = Math.min(effectiveDim, 96);
  const indices =
    effectiveDim <= count
      ? Array.from({ length: effectiveDim }, (_, index) => index)
      : Array.from({ length: count }, (_, index) => Math.round((index * (effectiveDim - 1)) / (count - 1)));
  return {
    values: indices.map((index) => values?.[index] ?? 0),
    total: effectiveDim
  };
}

function inputLabel(index: number, observationDim: number): string {
  return observationDim === 55 && V39_INPUT_LABELS[index] ? V39_INPUT_LABELS[index] : `o${index + 1}`;
}

function outputLabel(model: ModelMetadata | null, index: number): string {
  const label = model?.actionLabels?.[index] ?? `a${index + 1}`;
  return label.length > 10 ? `${label.slice(0, 9)}…` : label;
}

function normalizedAction(model: ModelMetadata | null, value: number, index: number): number {
  const high = Math.abs(model?.actionHigh?.[index] ?? 0);
  const low = Math.abs(model?.actionLow?.[index] ?? 0);
  const limit = Math.max(high, low, 1);
  return value / limit;
}

function actionDisplaySignal(model: ModelMetadata | null, value: number, index: number): number {
  const signal = normalizedAction(model, value, index);
  if (Math.abs(signal) < 0.001) {
    return 0;
  }
  return Math.sign(signal) * Math.min(1, Math.max(0.24, Math.abs(signal) * 2.8));
}

function nodeColor(value: number): string {
  const signal = clampSignal(value);
  if (signal > 0.16) {
    return "#e3483f";
  }
  if (signal < -0.16) {
    return "#3f59d9";
  }
  return "#e3e8ef";
}

function clampSignal(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.max(-1, Math.min(1, value));
}
