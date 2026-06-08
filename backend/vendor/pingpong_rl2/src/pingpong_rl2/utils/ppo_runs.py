from __future__ import annotations

import json
from pathlib import Path

from pingpong_rl2.defaults import (
    DEFAULT_BALL_HEIGHT,
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_PPO_POSITION_STRIKE_RUN_NAME,
    DEFAULT_PPO_POSITION_STRIKE_TILT_RUN_NAME,
    DEFAULT_PPO_POSITION_STRIKE_TILT_LIFT_RUN_NAME,
    DEFAULT_PPO_POSITION_CONTACT_FRAME_RUN_NAME,
    DEFAULT_PPO_POSITION_CONTACT_FRAME_VELOCITY_TILT_LATERAL_APEX_RESIDUAL_RUN_NAME,
    DEFAULT_PPO_POSITION_CONTACT_FRAME_VELOCITY_TILT_LATERAL_APEX_TRACKING_RESIDUAL_RUN_NAME,
    DEFAULT_PPO_POSITION_CONTACT_FRAME_VELOCITY_RESIDUAL_RUN_NAME,
    DEFAULT_PPO_POSITION_CONTACT_FRAME_VELOCITY_TILT_LATERAL_RESIDUAL_RUN_NAME,
    DEFAULT_PPO_POSITION_CONTACT_FRAME_VELOCITY_TILT_RESIDUAL_RUN_NAME,
    DEFAULT_PPO_POSITION_TILT_RUN_NAME,
    DEFAULT_PPO_RUN_NAME,
    DEFAULT_RESET_BALL_HEIGHT_RANGE,
    DEFAULT_RESET_VELOCITY_XY_RANGE,
    DEFAULT_RESET_VELOCITY_Z_RANGE,
    DEFAULT_RESET_XY_RANGE,
    DEFAULT_SUCCESS_VELOCITY_THRESHOLD,
    SMOKE_PPO_POSITION_STRIKE_RUN_NAME,
    SMOKE_PPO_POSITION_STRIKE_TILT_RUN_NAME,
    SMOKE_PPO_POSITION_STRIKE_TILT_LIFT_RUN_NAME,
    SMOKE_PPO_POSITION_CONTACT_FRAME_RUN_NAME,
    SMOKE_PPO_POSITION_CONTACT_FRAME_VELOCITY_TILT_LATERAL_APEX_RESIDUAL_RUN_NAME,
    SMOKE_PPO_POSITION_CONTACT_FRAME_VELOCITY_TILT_LATERAL_APEX_TRACKING_RESIDUAL_RUN_NAME,
    SMOKE_PPO_POSITION_CONTACT_FRAME_VELOCITY_RESIDUAL_RUN_NAME,
    SMOKE_PPO_POSITION_CONTACT_FRAME_VELOCITY_TILT_LATERAL_RESIDUAL_RUN_NAME,
    SMOKE_PPO_POSITION_CONTACT_FRAME_VELOCITY_TILT_RESIDUAL_RUN_NAME,
    SMOKE_PPO_POSITION_TILT_RUN_NAME,
    SMOKE_PPO_RUN_NAME,
    default_ppo_model_candidates,
)
from pingpong_rl2.utils.paths import PPO_RUNS_ROOT, resolve_input_path


def default_run_name_for_action_mode(action_mode: str, smoke: bool = False) -> str:
    # action_mode와 smoke 여부를 표준 run 이름으로 변환해 학습/평가 경로 규칙을 통일한다.
    # LINK: backend/vendor/pingpong_rl2/src/pingpong_rl2/defaults.py:66
    smoke_run_names = {
        "position": SMOKE_PPO_RUN_NAME,
        "position_strike": SMOKE_PPO_POSITION_STRIKE_RUN_NAME,
        "position_tilt": SMOKE_PPO_POSITION_TILT_RUN_NAME,
        "position_strike_tilt": SMOKE_PPO_POSITION_STRIKE_TILT_RUN_NAME,
        "position_strike_tilt_lift": SMOKE_PPO_POSITION_STRIKE_TILT_LIFT_RUN_NAME,
        "position_contact_frame": SMOKE_PPO_POSITION_CONTACT_FRAME_RUN_NAME,
        "position_contact_frame_velocity_residual": SMOKE_PPO_POSITION_CONTACT_FRAME_VELOCITY_RESIDUAL_RUN_NAME,
        "position_contact_frame_velocity_tilt_residual": (
            SMOKE_PPO_POSITION_CONTACT_FRAME_VELOCITY_TILT_RESIDUAL_RUN_NAME
        ),
        "position_contact_frame_velocity_tilt_lateral_residual": (
            SMOKE_PPO_POSITION_CONTACT_FRAME_VELOCITY_TILT_LATERAL_RESIDUAL_RUN_NAME
        ),
        "position_contact_frame_velocity_tilt_lateral_apex_residual": (
            SMOKE_PPO_POSITION_CONTACT_FRAME_VELOCITY_TILT_LATERAL_APEX_RESIDUAL_RUN_NAME
        ),
        "position_contact_frame_velocity_tilt_lateral_apex_tracking_residual": (
            SMOKE_PPO_POSITION_CONTACT_FRAME_VELOCITY_TILT_LATERAL_APEX_TRACKING_RESIDUAL_RUN_NAME
        ),
    }
    standard_run_names = {
        "position": DEFAULT_PPO_RUN_NAME,
        "position_strike": DEFAULT_PPO_POSITION_STRIKE_RUN_NAME,
        "position_tilt": DEFAULT_PPO_POSITION_TILT_RUN_NAME,
        "position_strike_tilt": DEFAULT_PPO_POSITION_STRIKE_TILT_RUN_NAME,
        "position_strike_tilt_lift": DEFAULT_PPO_POSITION_STRIKE_TILT_LIFT_RUN_NAME,
        "position_contact_frame": DEFAULT_PPO_POSITION_CONTACT_FRAME_RUN_NAME,
        "position_contact_frame_velocity_residual": DEFAULT_PPO_POSITION_CONTACT_FRAME_VELOCITY_RESIDUAL_RUN_NAME,
        "position_contact_frame_velocity_tilt_residual": (
            DEFAULT_PPO_POSITION_CONTACT_FRAME_VELOCITY_TILT_RESIDUAL_RUN_NAME
        ),
        "position_contact_frame_velocity_tilt_lateral_residual": (
            DEFAULT_PPO_POSITION_CONTACT_FRAME_VELOCITY_TILT_LATERAL_RESIDUAL_RUN_NAME
        ),
        "position_contact_frame_velocity_tilt_lateral_apex_residual": (
            DEFAULT_PPO_POSITION_CONTACT_FRAME_VELOCITY_TILT_LATERAL_APEX_RESIDUAL_RUN_NAME
        ),
        "position_contact_frame_velocity_tilt_lateral_apex_tracking_residual": (
            DEFAULT_PPO_POSITION_CONTACT_FRAME_VELOCITY_TILT_LATERAL_APEX_TRACKING_RESIDUAL_RUN_NAME
        ),
    }
    if smoke:
        if action_mode not in smoke_run_names:
            raise ValueError(f"Unsupported action_mode for smoke run naming: {action_mode!r}.")
        return smoke_run_names[action_mode]
    if action_mode not in standard_run_names:
        raise ValueError(f"Unsupported action_mode for run naming: {action_mode!r}.")
    return standard_run_names[action_mode]


def compose_run_name(base_run_name: str, run_version: str | None = None) -> str:
    version = None if run_version is None else run_version.strip()
    return base_run_name if not version else f"{base_run_name}_{version}"


def resolve_requested_run_name(
    run_name: str | None,
    run_version: str | None = None,
    *,
    action_mode: str = "position",
    smoke: bool = False,
) -> str:
    base_run_name = default_run_name_for_action_mode(action_mode, smoke=smoke) if run_name is None else run_name
    return compose_run_name(base_run_name, run_version)


def model_path_for_run_name(run_name: str, ppo_runs_root: Path = PPO_RUNS_ROOT) -> Path:
    return ppo_runs_root / run_name / f"{run_name}_model.zip"


def best_model_path_for_run_name(run_name: str, ppo_runs_root: Path = PPO_RUNS_ROOT) -> Path:
    return ppo_runs_root / run_name / f"{run_name}_best_model.zip"


def training_summary_path_for_run_name(run_name: str, ppo_runs_root: Path = PPO_RUNS_ROOT) -> Path:
    return ppo_runs_root / run_name / f"{run_name}_training_summary.json"


def infer_run_name_from_model_path(model_path: Path) -> str:
    model_stem = model_path.stem
    return model_stem[:-6] if model_stem.endswith("_model") else model_stem


def infer_training_run_name_from_model_path(model_path: Path) -> str:
    model_stem = model_path.stem
    if model_stem.endswith("_best_model"):
        return model_stem[:-11]
    return infer_run_name_from_model_path(model_path)


def training_summary_candidates_for_model(model_path: Path) -> list[Path]:
    run_name = infer_training_run_name_from_model_path(model_path)
    return [model_path.parent / f"{run_name}_training_summary.json"]


def resolve_saved_model_path(
    model_path: Path | None = None,
    run_name: str | None = None,
    *,
    prefer_best_model: bool = False,
) -> Path:
    # 명시 경로, run 이름, 기본 후보 순서로 실제 사용할 PPO zip 경로를 결정한다.
    if model_path is not None:
        return resolve_input_path(model_path)
    if run_name is not None:
        if prefer_best_model:
            fallback_best_model_path = best_model_path_for_run_name(run_name)
            if fallback_best_model_path.is_file():
                return fallback_best_model_path
        return model_path_for_run_name(run_name)
    candidates = default_ppo_model_candidates(PPO_RUNS_ROOT)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def load_training_summary(summary_path: Path) -> dict[str, object] | None:
    if not summary_path.is_file():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def load_env_config_for_model(model_path: Path) -> dict[str, object] | None:
    # 모델 옆 training summary에서 훈련 당시 env_config를 찾아 runtime 재현에 사용한다.
    # LINK: backend/app/live_simulation.py:190
    for summary_path in training_summary_candidates_for_model(model_path):
        summary = load_training_summary(summary_path)
        if summary is None:
            continue
        env_config = summary.get("env_config")
        if isinstance(env_config, dict):
            return dict(env_config)
    return None


def resolve_env_kwargs_for_model(
    model_path: Path | None = None,
    *,
    scene_path: Path | str | None = None,
    ball_height: float | None = None,
    target_ball_height: float | None = None,
    max_episode_steps: int | None = None,
    reset_ball_height_range: float | None = None,
    reset_ball_height_bounds: tuple[float, float] | list[float] | None = None,
    reset_xy_range: float | None = None,
    reset_xy_sampling: str | None = None,
    reset_velocity_xy_range: float | None = None,
    reset_velocity_z_range: tuple[float, float] | list[float] | None = None,
    reset_ball_angular_velocity_range: float | None = None,
    success_velocity_threshold: float | None = None,
) -> dict[str, object]:
    # 기본 env kwargs에 training summary와 호출자가 준 override를 차례로 합쳐 최종 환경 설정을 만든다.
    # LINK: backend/vendor/pingpong_rl2/src/pingpong_rl2/envs/keepup_env.py:176
    env_kwargs: dict[str, object] = {
        "action_mode": "position",
        "ball_height": DEFAULT_BALL_HEIGHT,
        "target_ball_height": DEFAULT_BALL_HEIGHT,
        "max_episode_steps": DEFAULT_MAX_EPISODE_STEPS,
        "reset_ball_height_range": DEFAULT_RESET_BALL_HEIGHT_RANGE,
        "reset_ball_height_bounds": None,
        "reset_xy_range": DEFAULT_RESET_XY_RANGE,
        "reset_xy_sampling": "square",
        "reset_velocity_xy_range": DEFAULT_RESET_VELOCITY_XY_RANGE,
        "reset_velocity_z_range": tuple(DEFAULT_RESET_VELOCITY_Z_RANGE),
        "reset_ball_angular_velocity_range": 0.0,
        "success_velocity_threshold": DEFAULT_SUCCESS_VELOCITY_THRESHOLD,
    }
    if model_path is not None:
        summary_env_config = load_env_config_for_model(model_path)
        if summary_env_config is not None:
            env_kwargs.update(summary_env_config)

    if scene_path is not None:
        env_kwargs["scene_path"] = str(resolve_input_path(Path(scene_path)))
    if ball_height is not None:
        env_kwargs["ball_height"] = float(ball_height)
        if target_ball_height is None:
            env_kwargs["target_ball_height"] = float(ball_height)
    if target_ball_height is not None:
        env_kwargs["target_ball_height"] = float(target_ball_height)
    if max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = int(max_episode_steps)
    if reset_ball_height_range is not None:
        env_kwargs["reset_ball_height_range"] = float(reset_ball_height_range)
    if reset_ball_height_bounds is not None:
        env_kwargs["reset_ball_height_bounds"] = tuple(float(value) for value in reset_ball_height_bounds)
    if reset_xy_range is not None:
        env_kwargs["reset_xy_range"] = float(reset_xy_range)
    if reset_xy_sampling is not None:
        env_kwargs["reset_xy_sampling"] = str(reset_xy_sampling)
    if reset_velocity_xy_range is not None:
        env_kwargs["reset_velocity_xy_range"] = float(reset_velocity_xy_range)
    if reset_velocity_z_range is not None:
        env_kwargs["reset_velocity_z_range"] = tuple(float(value) for value in reset_velocity_z_range)
    if reset_ball_angular_velocity_range is not None:
        env_kwargs["reset_ball_angular_velocity_range"] = float(reset_ball_angular_velocity_range)
    if success_velocity_threshold is not None:
        env_kwargs["success_velocity_threshold"] = float(success_velocity_threshold)
    return env_kwargs
