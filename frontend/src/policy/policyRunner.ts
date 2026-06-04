import type { SimulationSnapshot } from "../simulation/types";
import type { PolicyManifest } from "./policyManifest";

export interface PolicyRunner {
  readonly loaded: boolean;
  readonly message: string;
  nextAction(snapshot: SimulationSnapshot, actionSize: number): Float64Array;
}

class HoldPolicyRunner implements PolicyRunner {
  readonly loaded = false;
  readonly message: string;

  constructor(manifest: PolicyManifest) {
    this.message =
      manifest.format === "sb3-zip"
        ? "SB3 PPO zip is bundled as source artifact; browser export is pending."
        : manifest.message;
  }

  nextAction(_snapshot: SimulationSnapshot, actionSize: number): Float64Array {
    return new Float64Array(actionSize);
  }
}

export function createPolicyRunner(manifest: PolicyManifest): PolicyRunner {
  return new HoldPolicyRunner(manifest);
}
