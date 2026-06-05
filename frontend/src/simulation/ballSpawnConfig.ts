import type { BallSpawnConfig, BallSpawnRange, BallSpawnSettings } from "./types";
import { DEFAULT_BALL_SPAWN_CONFIG } from "./types";

export function parseBallSpawnConfig(value: unknown): BallSpawnConfig {
  if (!isRecord(value)) {
    return DEFAULT_BALL_SPAWN_CONFIG;
  }

  const defaultRanges = DEFAULT_BALL_SPAWN_CONFIG.ranges;
  const rangesValue = isRecord(value.ranges) ? value.ranges : {};
  const defaultsValue = isRecord(value.defaults) ? value.defaults : {};
  const ranges = {
    xOffset: parseRange(rangesValue.xOffset, defaultRanges.xOffset),
    yOffset: parseRange(rangesValue.yOffset, defaultRanges.yOffset),
    zOffset: parseRange(rangesValue.zOffset, defaultRanges.zOffset),
    velocityX: parseRange(rangesValue.velocityX, defaultRanges.velocityX),
    velocityY: parseRange(rangesValue.velocityY, defaultRanges.velocityY),
    velocityZ: parseRange(rangesValue.velocityZ, defaultRanges.velocityZ)
  };
  const defaults = buildBallSpawnSettings((key) =>
    clampNumber(readNumber(defaultsValue[key], DEFAULT_BALL_SPAWN_CONFIG.defaults[key]), ranges[key].min, ranges[key].max)
  );

  return { defaults, ranges };
}

export function clampBallSpawnSettings(value: BallSpawnSettings, config: BallSpawnConfig): BallSpawnSettings {
  return buildBallSpawnSettings((key) => clampNumber(value[key], config.ranges[key].min, config.ranges[key].max));
}

function parseRange(value: unknown, fallback: BallSpawnRange): BallSpawnRange {
  if (!isRecord(value)) {
    return fallback;
  }

  const min = readNumber(value.min, fallback.min);
  const max = readNumber(value.max, fallback.max);
  const low = Math.min(min, max);
  const high = Math.max(min, max);
  return {
    min: low,
    max: high,
    step: readPositiveNumber(value.step, fallback.step),
    trainedMin: readNumber(value.trainedMin, fallback.trainedMin ?? low),
    trainedMax: readNumber(value.trainedMax, fallback.trainedMax ?? high)
  };
}

function readNumber(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function readPositiveNumber(value: unknown, fallback: number): number {
  const parsed = readNumber(value, fallback);
  return parsed > 0 ? parsed : fallback;
}

function clampNumber(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function buildBallSpawnSettings(readValue: (key: keyof BallSpawnSettings) => number): BallSpawnSettings {
  return {
    xOffset: readValue("xOffset"),
    yOffset: readValue("yOffset"),
    zOffset: readValue("zOffset"),
    velocityX: readValue("velocityX"),
    velocityY: readValue("velocityY"),
    velocityZ: readValue("velocityZ")
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
