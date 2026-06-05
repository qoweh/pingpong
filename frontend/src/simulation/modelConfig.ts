import { parseBallSpawnConfig } from "./ballSpawnConfig";
import type { ModelMetadata, ModelsPayload } from "./types";

export function parseModelsPayload(value: unknown): ModelsPayload | null {
  if (!isRecord(value) || typeof value.activeModel !== "string" || !Array.isArray(value.models)) {
    return null;
  }

  const models = value.models.map(parseModelMetadata).filter((model): model is ModelMetadata => model !== null);
  return {
    activeModel: value.activeModel,
    models
  };
}

function parseModelMetadata(value: unknown): ModelMetadata | null {
  if (!isRecord(value) || typeof value.id !== "string") {
    return null;
  }

  const actionLabels = Array.isArray(value.actionLabels)
    ? value.actionLabels.filter((label): label is string => typeof label === "string")
    : [];

  return {
    id: value.id,
    name: readString(value.name, value.id),
    displayName: readString(value.displayName, readString(value.name, value.id)),
    rawRunName: typeof value.rawRunName === "string" ? value.rawRunName : null,
    dimensionGroup: typeof value.dimensionGroup === "string" ? value.dimensionGroup : null,
    versionLabel: typeof value.versionLabel === "string" ? value.versionLabel : null,
    detailName: typeof value.detailName === "string" ? value.detailName : null,
    sortDimension: readNullableNumber(value.sortDimension),
    sortVersion: readNullableNumber(value.sortVersion),
    source: readString(value.source, "artifacts"),
    path: readString(value.path, ""),
    algorithm: readString(value.algorithm, "PPO"),
    observationDim: readNullableNumber(value.observationDim),
    actionDim: readNullableNumber(value.actionDim),
    actionMode: typeof value.actionMode === "string" ? value.actionMode : null,
    actionLabels,
    actionLow: readNumberArray(value.actionLow),
    actionHigh: readNumberArray(value.actionHigh),
    ballSpawn: parseBallSpawnConfig(value.ballSpawn),
    trainedRanges: parseRangeMap(value.trainedRanges),
    testedRanges: parseRangeMap(value.testedRanges),
    trainingSummaryPath: typeof value.trainingSummaryPath === "string" ? value.trainingSummaryPath : null,
    policy: parsePolicyMetadata(value.policy),
    training: parseTrainingMetadata(value.training)
  };
}

function parsePolicyMetadata(value: unknown): ModelMetadata["policy"] {
  if (!isRecord(value)) {
    return undefined;
  }

  return {
    className: typeof value.className === "string" ? value.className : undefined,
    netArch: value.netArch,
    architecture: Array.isArray(value.architecture)
      ? value.architecture.filter((item): item is string => typeof item === "string")
      : undefined
  };
}

function parseTrainingMetadata(value: unknown): ModelMetadata["training"] {
  if (!isRecord(value)) {
    return undefined;
  }

  return {
    runName: typeof value.runName === "string" ? value.runName : null,
    timesteps: readNullableNumber(value.timesteps),
    preset: typeof value.preset === "string" ? value.preset : null,
    seed: readNullableNumber(value.seed)
  };
}

function readString(value: unknown, fallback: string): string {
  return typeof value === "string" && value ? value : fallback;
}

function readNullableNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function readNumberArray(value: unknown): number[] | null {
  if (!Array.isArray(value)) {
    return null;
  }
  const values = value.filter((item): item is number => typeof item === "number" && Number.isFinite(item));
  return values.length ? values : null;
}

function parseRangeMap(value: unknown): Record<string, { min: number; max: number }> | undefined {
  if (!isRecord(value)) {
    return undefined;
  }

  const parsed: Record<string, { min: number; max: number }> = {};
  for (const [key, rawRange] of Object.entries(value)) {
    if (!isRecord(rawRange)) {
      continue;
    }
    const min = readNullableNumber(rawRange.min);
    const max = readNullableNumber(rawRange.max);
    if (min !== null && max !== null) {
      parsed[key] = { min, max };
    }
  }
  return Object.keys(parsed).length ? parsed : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
