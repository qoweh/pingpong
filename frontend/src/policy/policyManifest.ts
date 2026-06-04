export type PolicyFormat = "onnx" | "json-mlp" | "sb3-zip" | "none";

export interface PolicyManifest {
  format: PolicyFormat;
  name: string;
  file: string | null;
  sourceModel: string | null;
  observationSize: number | null;
  actionSize: number | null;
  message: string;
}

export async function loadPolicyManifest(): Promise<PolicyManifest> {
  const response = await fetch("/assets/policy/policy-manifest.json");
  if (!response.ok) {
    return {
      format: "none",
      name: "No browser policy",
      file: null,
      sourceModel: null,
      observationSize: null,
      actionSize: null,
      message: "Policy manifest is not available."
    };
  }

  return (await response.json()) as PolicyManifest;
}
