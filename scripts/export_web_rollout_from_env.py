from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
DEFAULT_RL_ROOT = Path("/Users/pilt/project-collection/ros2/graduation-prj/pingpong_rl2")
DEFAULT_OUTPUT_PATH = ROOT / "frontend/public/assets/demo/rollout.json"


def main() -> None:
    env_values = read_env(ENV_PATH)
    rl_root = Path(
        env_values.get("PINGPONG_RL_SOURCE_ROOT")
        or os.environ.get("PINGPONG_RL_SOURCE_ROOT")
        or DEFAULT_RL_ROOT
    ).expanduser()
    source_model = (
        env_values.get("PINGPONG_POLICY_MODEL_PATH")
        or os.environ.get("PINGPONG_POLICY_MODEL_PATH")
    )
    if not source_model:
        raise SystemExit("PINGPONG_POLICY_MODEL_PATH is missing from .env")

    sys.path.insert(0, str(rl_root / "src"))

    from pingpong_rl2.envs import PingPongKeepUpGymEnv
    from pingpong_rl2.utils import resolve_env_kwargs_for_model

    model_path = resolve_project_path(source_model)
    output_path = resolve_project_path(
        env_values.get("PINGPONG_WEB_ROLLOUT_PATH")
        or os.environ.get("PINGPONG_WEB_ROLLOUT_PATH")
        or str(DEFAULT_OUTPUT_PATH)
    )
    max_steps = int(
        env_values.get("PINGPONG_WEB_ROLLOUT_STEPS")
        or os.environ.get("PINGPONG_WEB_ROLLOUT_STEPS")
        or 1800
    )
    seed = int(
        env_values.get("PINGPONG_WEB_ROLLOUT_SEED")
        or os.environ.get("PINGPONG_WEB_ROLLOUT_SEED")
        or 251
    )
    deterministic = parse_bool(
        env_values.get("PINGPONG_WEB_ROLLOUT_DETERMINISTIC")
        or os.environ.get("PINGPONG_WEB_ROLLOUT_DETERMINISTIC"),
        default=True,
    )

    env_kwargs = resolve_env_kwargs_for_model(model_path)
    env = PingPongKeepUpGymEnv(**env_kwargs)
    model = PPO.load(model_path, device="cpu")
    observation, reset_info = env.reset(seed=seed)
    sim = env.base_env.sim

    initial_state = {
        "qpos": numeric_list(sim.data.qpos),
        "qvel": numeric_list(sim.data.qvel),
        "ctrl": numeric_list(sim.data.ctrl),
        "time": float(sim.data.time),
    }

    frames: list[dict[str, Any]] = []
    terminated = False
    truncated = False
    final_info: dict[str, Any] = {}
    for frame_index in range(max_steps):
        action, _ = model.predict(observation, deterministic=deterministic)
        observation, reward, terminated, truncated, info = env.step(action)
        final_info = dict(info)
        frames.append(frame_from_env(frame_index, env, action, reward, terminated, truncated, info))
        if terminated or truncated:
            break

    payload = {
        "format": "pingpong-web-rollout-v1",
        "source": {
            "rlRoot": str(rl_root),
            "model": source_model,
            "resolvedModel": str(model_path),
            "scene": str(env.training_config().get("scene_path")),
        },
        "policy": {
            "deterministic": deterministic,
            "observationSize": int(model.observation_space.shape[0]),
            "actionSize": int(model.action_space.shape[0]),
        },
        "simulation": {
            "seed": seed,
            "controlDt": float(sim.control_dt),
            "timestep": float(sim.model.opt.timestep),
            "substeps": int(sim.n_substeps),
            "nq": int(sim.model.nq),
            "nv": int(sim.model.nv),
            "nu": int(sim.model.nu),
        },
        "envConfig": to_jsonable(env.training_config()),
        "resetInfo": to_jsonable(reset_info),
        "initialState": initial_state,
        "frames": frames,
        "result": {
            "steps": len(frames),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "failureReason": final_info.get("failure_reason"),
            "contactCount": int(final_info.get("contact_count", 0) or 0),
            "successfulBounceCount": int(final_info.get("successful_bounce_count", 0) or 0),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"exported rollout to {output_path}")
    print(f"model={source_model}")
    print(f"frames={len(frames)} terminated={terminated} truncated={truncated}")


def frame_from_env(
    frame_index: int,
    env: Any,
    action: np.ndarray,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
) -> dict[str, Any]:
    sim = env.base_env.sim
    contact_position = vector_from_info(info, "contact_mujoco_position")
    return {
        "index": frame_index,
        "time": float(sim.data.time),
        "ctrl": numeric_list(sim.data.ctrl),
        "action": numeric_list(np.asarray(action, dtype=float)),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "contact": {
            "event": bool(info.get("contact_event_during_step", False)),
            "observed": bool(info.get("contact_observed_during_step", False)),
            "count": int(info.get("contact_count", 0) or 0),
            "successfulBounceCount": int(info.get("successful_bounce_count", 0) or 0),
            "position": contact_position,
        },
        "state": {
            "ballPosition": numeric_list(sim.ball_position),
            "ballVelocity": numeric_list(sim.ball_velocity),
            "racketPosition": numeric_list(sim.racket_position),
            "targetTilt": numeric_list(info.get("target_tilt", [0.0, 0.0])),
        },
        "mujocoState": {
            "qpos": numeric_list(sim.data.qpos),
            "qvel": numeric_list(sim.data.qvel),
            "ctrl": numeric_list(sim.data.ctrl),
            "time": float(sim.data.time),
        },
        "info": {
            "phaseName": info.get("phase_name"),
            "failureReason": info.get("failure_reason"),
            "successReason": info.get("success_reason"),
            "easyNextBallScore": optional_float(info.get("easy_next_ball_score")),
            "nextInterceptTime": optional_float(info.get("next_intercept_time")),
        },
    }


def vector_from_info(info: dict[str, Any], prefix: str) -> list[float] | None:
    keys = (f"{prefix}_x", f"{prefix}_y", f"{prefix}_z")
    values = [info.get(key) for key in keys]
    if any(value is None for value in values):
        return None
    return [float(value) for value in values]


def optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def numeric_list(values: Any) -> list[float]:
    array = np.asarray(values, dtype=float).reshape(-1)
    return [float(value) for value in array]


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return numeric_list(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def resolve_project_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else ROOT / path


def parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


if __name__ == "__main__":
    main()
