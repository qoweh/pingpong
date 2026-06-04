from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from .settings import AppSettings


@dataclass
class LiveCommandState:
    playback: str = "playing"
    reset_requested: bool = False
    reset_options: dict[str, Any] | None = None
    ball_spawn_requested: bool = False
    ball_spawn_options: dict[str, Any] | None = None


class LiveSimulationService:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._load_rl_package(settings.rl_source_root)

        from pingpong_rl2.envs import PingPongKeepUpGymEnv
        from pingpong_rl2.utils import resolve_env_kwargs_for_model

        self.env_class = PingPongKeepUpGymEnv
        self.env_kwargs = resolve_env_kwargs_for_model(settings.model_path)
        self.env_kwargs["scene_path"] = str(settings.scene_path)
        self.policy = PPO.load(settings.model_path, device="cpu")
        self.policy_lock = threading.Lock()
        self.policy_message = f"Live Python PPO: {settings.model_path.name}"
        self.control_dt = 0.02

    def create_session(self) -> "LiveSimulationSession":
        return LiveSimulationSession(self)

    def predict(self, observation: Any) -> np.ndarray:
        with self.policy_lock:
            action, _ = self.policy.predict(
                observation,
                deterministic=self.settings.deterministic_policy,
            )
        return np.asarray(action, dtype=float)

    def config_payload(self) -> dict[str, Any]:
        return {
            "model": portable_path(self.settings.model_path, self.settings.project_root),
            "scene": portable_path(self.settings.scene_path, self.settings.project_root),
            "deterministic": self.settings.deterministic_policy,
            "seed": self.settings.seed,
            "controlDt": self.control_dt,
        }

    @staticmethod
    def _load_rl_package(rl_source_root: Path) -> None:
        source_path = rl_source_root / "src"
        if not (source_path / "pingpong_rl2").is_dir():
            raise RuntimeError(
                f"pingpong_rl2 source is missing at {source_path}. "
                "Copy pingpong_rl2/src into backend/vendor/pingpong_rl2/src "
                "or set PINGPONG_RL_SOURCE_ROOT."
            )
        sys.path.insert(0, str(source_path))


class LiveSimulationSession:
    def __init__(self, service: LiveSimulationService) -> None:
        self.service = service
        self.env = service.env_class(**service.env_kwargs)
        self.episode_index = 0
        self.step_index = 0
        self.reset_pending = False
        self.observation: Any = None
        self.last_info: dict[str, Any] = {}
        self.last_reward: float | None = None
        self.last_contact: dict[str, Any] | None = None
        self.custom_reset_options: dict[str, Any] | None = None
        self.observation, self.last_info = self.env.reset(seed=service.settings.seed)
        self.service.control_dt = self.control_dt

    @property
    def sim(self) -> Any:
        return self.env.base_env.sim

    @property
    def control_dt(self) -> float:
        return float(self.sim.control_dt)

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        if options is not None:
            self.custom_reset_options = dict(options)

        self.episode_index += 1
        self.step_index = 0
        self.reset_pending = False
        self.last_reward = None
        self.last_contact = None
        self.observation, self.last_info = self.env.reset(
            seed=self.service.settings.seed + self.episode_index,
            options=self.custom_reset_options,
        )
        return self.frame(reset=True)

    def spawn_ball(self, options: dict[str, Any]) -> dict[str, Any]:
        self.custom_reset_options = dict(options)
        base_env = self.env.base_env
        z_offset = float(options.get("ball_height", base_env.ball_height))
        ball_xy_offset = np.asarray(options.get("ball_xy_offset", (0.0, 0.0)), dtype=float)
        ball_velocity = np.asarray(options.get("ball_velocity", (0.0, 0.0, 0.0)), dtype=float)

        self.sim.reset_ball_above_racket(
            height=z_offset,
            xy_offset=ball_xy_offset,
            velocity=ball_velocity,
        )
        self.sim.data.time = 0.0
        self._clear_episode_counters()

        self.episode_index += 1
        self.step_index = 0
        self.reset_pending = False
        self.last_reward = None
        self.last_contact = None
        self.observation = base_env.observation().astype(np.float32, copy=False)
        self.last_info = self._spawn_info(ball_xy_offset)
        return self.frame(reset=True)

    def step(self) -> dict[str, Any]:
        if self.reset_pending:
            return self.reset()

        action = self.service.predict(self.observation)
        self.observation, reward, terminated, truncated, info = self.env.step(action)
        self.last_reward = float(reward)
        self.last_info = dict(info)
        self.step_index += 1
        frame = self.frame(
            action=action,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            include_events=True,
        )
        self.reset_pending = bool(terminated or truncated)
        return frame

    def _clear_episode_counters(self) -> None:
        base_env = self.env.base_env
        base_env.step_count = 0
        base_env.contact_count = 0
        base_env.successful_bounce_count = 0
        base_env.stable_cycle_count = 0
        base_env._consecutive_stable_cycle_count = 0
        base_env._consecutive_low_apex_contact_count = 0
        base_env._last_projected_contact_apex_height = None
        base_env._last_contact_apex_shortfall = 0.0
        base_env._last_contact_step = None
        base_env._contact_active_previous_step = False
        base_env._previous_action[:] = 0.0
        base_env._contact_frame_velocity_residual_action[:] = 0.0
        base_env._contact_frame_racket_vz_residual_action = 0.0
        base_env._contact_frame_tilt_scale_residual_action[:] = 0.0
        base_env._contact_frame_racket_xy_residual_action[:] = 0.0
        base_env._contact_frame_target_apex_z_residual_action = 0.0
        base_env._contact_frame_strike_plane_z_residual_action = 0.0
        base_env._contact_frame_tracking_xy_residual_action[:] = 0.0
        base_env._reset_contact_frame_plan()
        base_env._spawn_ball_height_above_racket = float(self.sim.ball_position[2] - self.sim.racket_position[2])

    def _spawn_info(self, ball_xy_offset: np.ndarray) -> dict[str, Any]:
        base_env = self.env.base_env
        return {
            "failure_reason": None,
            "success_reason": None,
            "contact_count": 0,
            "successful_bounce_count": 0,
            "stable_cycle_count": 0,
            "consecutive_stable_cycle_count": 0,
            "last_projected_contact_apex_height_above_racket": None,
            "last_contact_apex_shortfall": 0.0,
            "step_count": 0,
            "target_position": base_env.controller.target_position,
            "target_tilt": base_env.controller.target_tilt,
            "target_velocity": base_env.controller.target_velocity,
            "ball_height_above_racket": float(self.sim.ball_position[2] - self.sim.racket_position[2]),
            "spawn_ball_height_above_racket": base_env._spawn_ball_height_above_racket,
            "spawn_ball_position": self.sim.ball_position.copy(),
            "spawn_ball_velocity": self.sim.ball_velocity.copy(),
            "spawn_ball_angular_velocity": self.sim.ball_angular_velocity.copy(),
            "spawn_ball_xy_offset": ball_xy_offset.copy(),
        }

    def frame(
        self,
        *,
        action: np.ndarray | None = None,
        reward: float | None = None,
        terminated: bool = False,
        truncated: bool = False,
        reset: bool = False,
        include_events: bool = False,
    ) -> dict[str, Any]:
        info = self.last_info
        contact_position = vector_from_info(info, "contact_mujoco_position")
        contact_event = bool(include_events and info.get("contact_event_during_step", False))
        contact_observed = bool(include_events and info.get("contact_observed_during_step", False))
        if contact_event and contact_position is not None:
            self.last_contact = {
                "position": contact_position,
                "time": float(self.sim.data.time),
            }

        return {
            "type": "frame",
            "episode": self.episode_index,
            "step": self.step_index,
            "time": float(self.sim.data.time),
            "reset": reset,
            "terminated": terminated,
            "truncated": truncated,
            "reward": reward,
            "failureReason": info.get("failure_reason"),
            "successReason": info.get("success_reason"),
            "policyLoaded": True,
            "policyMessage": self.service.policy_message,
            "state": {
                "qpos": numeric_list(self.sim.data.qpos),
                "qvel": numeric_list(self.sim.data.qvel),
                "ctrl": numeric_list(self.sim.data.ctrl),
                "time": float(self.sim.data.time),
            },
            "ball": {
                "position": numeric_vec3(self.sim.ball_position),
                "velocity": numeric_vec3(self.sim.ball_velocity),
            },
            "racketPosition": numeric_vec3(self.sim.racket_position),
            "contact": {
                "event": contact_event,
                "observed": contact_observed,
                "count": int(info.get("contact_count", 0) or 0),
                "successfulBounceCount": int(info.get("successful_bounce_count", 0) or 0),
                "position": contact_position,
                "last": self.last_contact,
            },
            "action": numeric_list(action) if action is not None else None,
        }


def vector_from_info(info: dict[str, Any], prefix: str) -> list[float] | None:
    keys = (f"{prefix}_x", f"{prefix}_y", f"{prefix}_z")
    values = [info.get(key) for key in keys]
    if any(value is None for value in values):
        return None
    return [float(value) for value in values]


def numeric_list(values: Any) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=float).reshape(-1)]


def numeric_vec3(values: Any) -> list[float]:
    vector = numeric_list(values)
    return [float(vector[0]), float(vector[1]), float(vector[2])]


def portable_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
