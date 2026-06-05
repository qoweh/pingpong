from __future__ import annotations

from math import hypot, isfinite
from pathlib import Path
from typing import Any


BallSpawnConfig = dict[str, Any]

DEFAULT_BALL_SPAWN = {
    "xOffset": 0.0,
    "yOffset": 0.0,
    "zOffset": 0.34,
    "velocityX": 0.0,
    "velocityY": 0.0,
    "velocityZ": 0.0,
}

FALLBACK_SPAWN_RANGES = {
    "xOffset": {"min": -0.16, "max": 0.16, "step": 0.005},
    "yOffset": {"min": -0.16, "max": 0.16, "step": 0.005},
    "zOffset": {"min": 0.18, "max": 0.56, "step": 0.005},
    "velocityX": {"min": -0.06, "max": 0.06, "step": 0.005},
    "velocityY": {"min": -0.06, "max": 0.06, "step": 0.005},
    "velocityZ": {"min": -0.18, "max": 0.04, "step": 0.005},
}

FALLBACK_BALL_SPAWN_CONFIG: BallSpawnConfig = {
    "defaults": DEFAULT_BALL_SPAWN,
    "ranges": {
        key: {**value, "trainedMin": value["min"], "trainedMax": value["max"]}
        for key, value in FALLBACK_SPAWN_RANGES.items()
    },
    "xyConstraint": {"sampling": "square"},
}

V39_TESTED_RANGES = {
    "xOffset": (-0.16, 0.16),
    "yOffset": (-0.16, 0.16),
    "zOffset": (0.18, 0.56),
    "velocityX": (-0.06, 0.06),
    "velocityY": (-0.06, 0.06),
    "velocityZ": (-0.18, 0.04),
}


def build_ball_spawn_config(env_kwargs: dict[str, Any], model_path: Path) -> BallSpawnConfig:
    default_z = finite_float(env_kwargs.get("ball_height"), DEFAULT_BALL_SPAWN["zOffset"])
    xy_range = abs(finite_float(env_kwargs.get("reset_xy_range"), 0.0))
    xy_sampling = str(env_kwargs.get("reset_xy_sampling") or "square")
    velocity_xy_range = abs(finite_float(env_kwargs.get("reset_velocity_xy_range"), 0.0))
    z_min, z_max = height_bounds(env_kwargs, default_z)
    velocity_z_min, velocity_z_max = ordered_pair(env_kwargs.get("reset_velocity_z_range"), (-0.0, 0.0))

    trained_ranges = {
        "xOffset": (-xy_range, xy_range),
        "yOffset": (-xy_range, xy_range),
        "zOffset": (z_min, z_max),
        "velocityX": (-velocity_xy_range, velocity_xy_range),
        "velocityY": (-velocity_xy_range, velocity_xy_range),
        "velocityZ": (velocity_z_min, velocity_z_max),
    }
    merged_ranges = dict(trained_ranges)
    tested_xy_radius = xy_range

    if is_keep_v39_model(model_path):
        merged_ranges = {
            key: (
                min(trained_ranges[key][0], V39_TESTED_RANGES[key][0]),
                max(trained_ranges[key][1], V39_TESTED_RANGES[key][1]),
            )
            for key in trained_ranges
        }
        tested_xy_radius = max(abs(V39_TESTED_RANGES["xOffset"][0]), abs(V39_TESTED_RANGES["xOffset"][1]))

    ranges = {
        key: {
            "min": float(low),
            "max": float(high),
            "step": 0.005,
            "trainedMin": float(trained_ranges[key][0]),
            "trainedMax": float(trained_ranges[key][1]),
        }
        for key, (low, high) in merged_ranges.items()
    }

    return {
        "defaults": {
            **DEFAULT_BALL_SPAWN,
            "zOffset": clamp_float(default_z, ranges["zOffset"]["min"], ranges["zOffset"]["max"], default_z),
        },
        "ranges": ranges,
        "xyConstraint": {
            "sampling": xy_sampling,
            "trainedRadius": float(xy_range) if xy_sampling == "disk" else None,
            "testedRadius": float(tested_xy_radius) if xy_sampling == "disk" else None,
        },
    }


def parse_ball_spawn_options(message: dict[str, Any], config: BallSpawnConfig) -> dict[str, Any]:
    ranges = config.get("ranges", FALLBACK_BALL_SPAWN_CONFIG["ranges"])
    defaults = config.get("defaults", FALLBACK_BALL_SPAWN_CONFIG["defaults"])

    x_offset = clamp_axis(message, ranges, defaults, "xOffset")
    y_offset = clamp_axis(message, ranges, defaults, "yOffset")
    z_offset = clamp_axis(message, ranges, defaults, "zOffset")
    velocity_x = clamp_axis(message, ranges, defaults, "velocityX")
    velocity_y = clamp_axis(message, ranges, defaults, "velocityY")
    velocity_z = clamp_axis(message, ranges, defaults, "velocityZ")
    x_offset, y_offset = clamp_xy_radius(x_offset, y_offset, config)
    return {
        "ball_height": z_offset,
        "ball_xy_offset": [x_offset, y_offset],
        "ball_velocity": [velocity_x, velocity_y, velocity_z],
    }


def clamp_axis(
    message: dict[str, Any],
    ranges: dict[str, dict[str, float]],
    defaults: dict[str, float],
    key: str,
) -> float:
    axis_range = ranges.get(key, FALLBACK_BALL_SPAWN_CONFIG["ranges"][key])
    return clamp_float(
        message.get(key),
        finite_float(axis_range.get("min"), FALLBACK_BALL_SPAWN_CONFIG["ranges"][key]["min"]),
        finite_float(axis_range.get("max"), FALLBACK_BALL_SPAWN_CONFIG["ranges"][key]["max"]),
        finite_float(defaults.get(key), DEFAULT_BALL_SPAWN[key]),
    )


def clamp_xy_radius(x_offset: float, y_offset: float, config: BallSpawnConfig) -> tuple[float, float]:
    xy_constraint = config.get("xyConstraint")
    if not isinstance(xy_constraint, dict) or xy_constraint.get("sampling") != "disk":
        return x_offset, y_offset

    radius = finite_float(xy_constraint.get("testedRadius"), 0.0)
    if radius <= 0.0:
        radius = max(abs(x_offset), abs(y_offset))

    distance = hypot(x_offset, y_offset)
    if distance <= radius or distance <= 0.0:
        return x_offset, y_offset

    scale = radius / distance
    return x_offset * scale, y_offset * scale


def height_bounds(env_kwargs: dict[str, Any], default_z: float) -> tuple[float, float]:
    explicit_bounds = env_kwargs.get("reset_ball_height_bounds")
    if explicit_bounds is not None:
        return ordered_pair(explicit_bounds, (default_z, default_z))

    height_range = abs(finite_float(env_kwargs.get("reset_ball_height_range"), 0.0))
    return default_z - height_range, default_z + height_range


def ordered_pair(value: Any, fallback: tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        low = finite_float(value[0], fallback[0])
        high = finite_float(value[1], fallback[1])
    else:
        low, high = fallback

    return (low, high) if low <= high else (high, low)


def finite_float(value: Any, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if isfinite(parsed) else fallback


def clamp_float(value: Any, low: float, high: float, fallback: float) -> float:
    parsed = finite_float(value, fallback)
    bounded_low, bounded_high = (low, high) if low <= high else (high, low)
    return min(max(parsed, bounded_low), bounded_high)


def is_keep_v39_model(model_path: Path) -> bool:
    normalized = str(model_path).lower()
    return "keep_v39_17d" in normalized or "keep1_v39_17d" in normalized
