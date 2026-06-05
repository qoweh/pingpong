import type { BallSpawnConfig, BallSpawnRange, BallSpawnSettings, BallSpawnXYConstraint } from "./types";
import { DEFAULT_BALL_SPAWN_CONFIG } from "./types";

export type BallSpawnClampMode = "trained" | "extended";

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
  const xyConstraint = parseXYConstraint(value.xyConstraint, DEFAULT_BALL_SPAWN_CONFIG.xyConstraint);

  return { defaults: clampBallSpawnSettings(defaults, { defaults, ranges, xyConstraint }, "extended"), ranges, xyConstraint };
}

export function clampBallSpawnSettings(
  value: BallSpawnSettings,
  config: BallSpawnConfig,
  mode: BallSpawnClampMode = "extended"
): BallSpawnSettings {
  const clamped = buildBallSpawnSettings((key) => {
    const bounds = rangeBounds(config.ranges[key], mode);
    return clampNumber(value[key], bounds.min, bounds.max);
  });
  return clampXYRadius(clamped, config, mode);
}

export function rangeBounds(range: BallSpawnRange, mode: BallSpawnClampMode): { min: number; max: number } {
  if (mode === "trained") {
    return {
      min: range.trainedMin ?? range.min,
      max: range.trainedMax ?? range.max
    };
  }
  return { min: range.min, max: range.max };
}

export function isWithinTrainedBallSpawnRange(value: BallSpawnSettings, config: BallSpawnConfig): boolean {
  const clamped = clampBallSpawnSettings(value, config, "trained");
  return (Object.keys(value) as Array<keyof BallSpawnSettings>).every((key) => nearlyEqual(value[key], clamped[key]));
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
    trainedMax: readNumber(value.trainedMax, fallback.trainedMax ?? high),
    testedMin: readNumber(value.testedMin, fallback.testedMin ?? low),
    testedMax: readNumber(value.testedMax, fallback.testedMax ?? high)
  };
}

function parseXYConstraint(value: unknown, fallback: BallSpawnXYConstraint | undefined): BallSpawnXYConstraint | undefined {
  if (!isRecord(value)) {
    return fallback;
  }
  return {
    sampling: typeof value.sampling === "string" ? value.sampling : fallback?.sampling,
    trainedRadius: readNullableNumber(value.trainedRadius, fallback?.trainedRadius),
    testedRadius: readNullableNumber(value.testedRadius, fallback?.testedRadius)
  };
}

function readNumber(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function readPositiveNumber(value: unknown, fallback: number): number {
  const parsed = readNumber(value, fallback);
  return parsed > 0 ? parsed : fallback;
}

function readNullableNumber(value: unknown, fallback: number | null | undefined): number | null | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function clampNumber(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function clampXYRadius(value: BallSpawnSettings, config: BallSpawnConfig, mode: BallSpawnClampMode): BallSpawnSettings {
  const xyConstraint = config.xyConstraint;
  if (!xyConstraint || xyConstraint.sampling !== "disk") {
    return value;
  }

  const radius = mode === "trained" ? xyConstraint.trainedRadius : xyConstraint.testedRadius ?? xyConstraint.trainedRadius;
  if (typeof radius !== "number" || !Number.isFinite(radius) || radius <= 0) {
    return value;
  }

  const distance = Math.hypot(value.xOffset, value.yOffset);
  if (distance <= radius || distance <= 0) {
    return value;
  }

  const scale = radius / distance;
  return {
    ...value,
    xOffset: value.xOffset * scale,
    yOffset: value.yOffset * scale
  };
}

function nearlyEqual(left: number, right: number): boolean {
  return Math.abs(left - right) < 1e-9;
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
