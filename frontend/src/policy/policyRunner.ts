import type { PolicyManifest } from "./policyManifest";

interface LinearLayer {
  type: "linear";
  weight: number[][];
  bias: number[];
}

interface JsonMlpPolicy {
  format: "json-mlp";
  name: string;
  observationSize: number;
  actionSize: number;
  actionLow: number[];
  actionHigh: number[];
  activation: "tanh";
  squashOutput: boolean;
  layers: LinearLayer[];
}

export interface PolicyRunner {
  readonly loaded: boolean;
  readonly message: string;
  readonly observationSize: number;
  readonly actionSize: number;
  nextAction(observation: ArrayLike<number>): Float64Array;
}

class JsonMlpPolicyRunner implements PolicyRunner {
  readonly loaded = true;
  readonly observationSize: number;
  readonly actionSize: number;
  readonly message: string;

  constructor(private readonly policy: JsonMlpPolicy) {
    this.observationSize = policy.observationSize;
    this.actionSize = policy.actionSize;
    this.message = `${policy.name} JSON policy loaded`;
  }

  nextAction(observation: ArrayLike<number>): Float64Array {
    let values = new Float64Array(this.observationSize);
    for (let index = 0; index < Math.min(observation.length, this.observationSize); index += 1) {
      values[index] = Number.isFinite(Number(observation[index])) ? Number(observation[index]) : 0;
    }

    for (let layerIndex = 0; layerIndex < this.policy.layers.length; layerIndex += 1) {
      const layer = this.policy.layers[layerIndex];
      const output = new Float64Array(layer.bias.length);
      for (let row = 0; row < layer.weight.length; row += 1) {
        let sum = layer.bias[row] ?? 0;
        const weights = layer.weight[row];
        for (let column = 0; column < weights.length; column += 1) {
          sum += weights[column] * values[column];
        }
        output[row] = sum;
      }

      const isLastLayer = layerIndex === this.policy.layers.length - 1;
      if (!isLastLayer || this.policy.squashOutput) {
        for (let index = 0; index < output.length; index += 1) {
          output[index] = Math.tanh(output[index]);
        }
      }

      values = output;
    }

    const action = new Float64Array(this.actionSize);
    for (let index = 0; index < this.actionSize; index += 1) {
      action[index] = clamp(values[index] ?? 0, this.policy.actionLow[index] ?? -Infinity, this.policy.actionHigh[index] ?? Infinity);
    }
    return action;
  }
}

class HoldPolicyRunner implements PolicyRunner {
  readonly loaded = false;
  readonly observationSize: number;
  readonly actionSize: number;
  readonly message: string;

  constructor(manifest: PolicyManifest) {
    this.observationSize = manifest.observationSize ?? 55;
    this.actionSize = manifest.actionSize ?? 15;
    this.message =
      manifest.format === "sb3-zip"
        ? "SB3 PPO zip is bundled as source artifact; browser export is pending."
        : manifest.message;
  }

  nextAction(): Float64Array {
    return new Float64Array(this.actionSize);
  }
}

export async function createPolicyRunner(manifest: PolicyManifest): Promise<PolicyRunner> {
  if (manifest.format !== "json-mlp" || !manifest.file) {
    return new HoldPolicyRunner(manifest);
  }

  const response = await fetch(manifest.file, { cache: "force-cache" });
  if (!response.ok) {
    return new HoldPolicyRunner({
      ...manifest,
      message: `Policy JSON failed to load: ${response.status}`
    });
  }

  const policy = (await response.json()) as JsonMlpPolicy;
  return new JsonMlpPolicyRunner(policy);
}

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}
