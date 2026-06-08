from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pingpong_rl2.envs.keepup_env import PingPongKeepUpEnv

_CONTACT_FRAME_VELOCITY_RESIDUAL_ACTION_MODES = (
    "position_contact_frame_velocity_residual",
    "position_contact_frame_velocity_tilt_residual",
    "position_contact_frame_velocity_tilt_lateral_residual",
    "position_contact_frame_velocity_tilt_lateral_apex_residual",
    "position_contact_frame_velocity_tilt_lateral_apex_tracking_residual",
)
_CONTACT_FRAME_TILT_SCALE_ACTION_MODES = (
    "position_contact_frame_velocity_tilt_residual",
    "position_contact_frame_velocity_tilt_lateral_residual",
    "position_contact_frame_velocity_tilt_lateral_apex_residual",
    "position_contact_frame_velocity_tilt_lateral_apex_tracking_residual",
)
_CONTACT_FRAME_LATERAL_VELOCITY_RESIDUAL_ACTION_MODES = (
    "position_contact_frame_velocity_tilt_lateral_residual",
    "position_contact_frame_velocity_tilt_lateral_apex_residual",
    "position_contact_frame_velocity_tilt_lateral_apex_tracking_residual",
)
_CONTACT_FRAME_APEX_TIMING_RESIDUAL_ACTION_MODES = (
    "position_contact_frame_velocity_tilt_lateral_apex_residual",
    "position_contact_frame_velocity_tilt_lateral_apex_tracking_residual",
)
_CONTACT_FRAME_TRACKING_RESIDUAL_ACTION_MODES = (
    "position_contact_frame_velocity_tilt_lateral_apex_tracking_residual",
)
_CONTACT_FRAME_ACTION_MODES = (
    "position_contact_frame",
    *_CONTACT_FRAME_VELOCITY_RESIDUAL_ACTION_MODES,
)
_STRIKE_CONTRACT_ACTION_MODES = (
    "position_strike",
    "position_strike_tilt",
    "position_strike_tilt_lift",
    *_CONTACT_FRAME_ACTION_MODES,
)


@dataclass(slots=True)
class HeuristicKeepUpPolicy:
    # PPO가 아닌 규칙 기반 baseline으로, phase별로 strike/recovery 목표를 보정해 action을 만든다.
    # LINK: backend/vendor/pingpong_rl2/src/pingpong_rl2/envs/keepup_env.py:2704
    return_blend: float = 0.72
    recovery_blend: float = 0.52
    strike_z_boost: float = 0.018
    strike_time_horizon: float = 0.14
    strike_xy_correction_gain: float = 0.0
    strike_xy_correction_max: float = 0.02
    tracking_velocity_residual_gain: float = 0.35
    tracking_velocity_residual_max: float = 0.12
    fixed_position_residual: tuple[float, float, float] = (0.0, 0.0, 0.0)
    strike_position_residual: tuple[float, float, float] | None = None
    recovery_position_residual: tuple[float, float, float] | None = None
    strike_phase_only_position_residual: tuple[float, float, float] | None = None
    fixed_tilt_residual_pitch: float = 0.0
    fixed_tilt_residual_roll: float = 0.0
    strike_tilt_residual_pitch: float | None = None
    strike_tilt_residual_roll: float | None = None
    recovery_tilt_residual_pitch: float | None = None
    recovery_tilt_residual_roll: float | None = None
    fixed_followup_lift_residual: float = 0.0
    strike_followup_lift_residual: float | None = None
    recovery_followup_lift_residual: float | None = None

    def reset(self) -> None:
        return None

    def _position_residual_for_phase(self, phase_name: str) -> np.ndarray:
        # prepare/strike/recovery phase에 따라 고정 또는 phase-specific 위치 residual을 선택한다.
        if phase_name in {"prepare", "strike"}:
            position_residual = self.strike_position_residual
        elif phase_name in {"return_shaping", "recovery"}:
            position_residual = self.recovery_position_residual
        else:
            position_residual = None

        if position_residual is None:
            position_residual = self.fixed_position_residual
        resolved_residual = np.asarray(position_residual, dtype=float)
        if phase_name == "strike" and self.strike_phase_only_position_residual is not None:
            resolved_residual = resolved_residual + np.asarray(self.strike_phase_only_position_residual, dtype=float)
        return resolved_residual

    def _state_dependent_strike_xy_residual(self, env: PingPongKeepUpEnv, phase_name: str) -> np.ndarray:
        if phase_name not in {"prepare", "strike"}:
            return np.zeros(3, dtype=float)
        if self.strike_xy_correction_gain <= 0.0 or self.strike_xy_correction_max <= 0.0:
            return np.zeros(3, dtype=float)

        correction_xy = env._controller_anchor_position()[:2] - env._predicted_intercept_xy()
        intercept_time = env._predicted_intercept_time()
        urgency = 1.0 - np.clip(intercept_time / max(self.strike_time_horizon, 1.0e-6), 0.0, 1.0)
        strike_readiness = max(env._pre_contact_height_readiness(), urgency)
        residual_xy = self.strike_xy_correction_gain * strike_readiness * np.asarray(correction_xy, dtype=float)
        residual_xy = np.clip(residual_xy, -self.strike_xy_correction_max, self.strike_xy_correction_max)
        return np.array([residual_xy[0], residual_xy[1], 0.0], dtype=float)

    def _tracking_velocity_residual(self, env: PingPongKeepUpEnv, phase_name: str) -> np.ndarray:
        if phase_name not in {"prepare", "strike"}:
            return np.zeros(2, dtype=float)
        if self.tracking_velocity_residual_gain <= 0.0 or self.tracking_velocity_residual_max <= 0.0:
            return np.zeros(2, dtype=float)

        intercept_time = max(env._predicted_intercept_time(), 0.08)
        intercept_error_xy = env._predicted_intercept_xy() - env.sim.racket_position[:2]
        residual_xy = self.tracking_velocity_residual_gain * np.asarray(intercept_error_xy, dtype=float) / intercept_time
        return np.clip(
            residual_xy,
            -self.tracking_velocity_residual_max,
            self.tracking_velocity_residual_max,
        )

    def _tilt_residual_for_phase(self, phase_name: str) -> np.ndarray:
        if phase_name in {"prepare", "strike"}:
            pitch = self.strike_tilt_residual_pitch
            roll = self.strike_tilt_residual_roll
        elif phase_name in {"return_shaping", "recovery"}:
            pitch = self.recovery_tilt_residual_pitch
            roll = self.recovery_tilt_residual_roll
        else:
            pitch = None
            roll = None

        if pitch is None:
            pitch = self.fixed_tilt_residual_pitch
        if roll is None:
            roll = self.fixed_tilt_residual_roll
        return np.array([pitch, roll], dtype=float)

    def _followup_lift_residual_for_phase(self, phase_name: str) -> float:
        if phase_name in {"prepare", "strike"}:
            followup_lift_residual = self.strike_followup_lift_residual
        elif phase_name in {"return_shaping", "recovery"}:
            followup_lift_residual = self.recovery_followup_lift_residual
        else:
            followup_lift_residual = None

        if followup_lift_residual is None:
            followup_lift_residual = self.fixed_followup_lift_residual
        return float(followup_lift_residual)

    def predict(self, env: PingPongKeepUpEnv) -> np.ndarray:
        # 현재 환경 phase와 intercept 예측을 보고 strike-contract action vector를 구성한다.
        # LINK: backend/vendor/pingpong_rl2/src/pingpong_rl2/envs/keepup_env.py:1625
        if env.action_mode not in _STRIKE_CONTRACT_ACTION_MODES:
            raise ValueError(
                "HeuristicKeepUpPolicy requires a strike-contract action mode so it can steer the strike controller."
            )

        if env.action_mode in _CONTACT_FRAME_ACTION_MODES:
            base_target = env._contact_frame_action_target_position(np.zeros(3, dtype=float))
        else:
            base_target = env._strike_action_target_position(np.zeros(3, dtype=float))
        anchor_position = env._controller_anchor_position()
        desired_target = base_target.copy()
        phase_name = env._phase_name()

        if phase_name in {"prepare", "strike"}:
            intercept_time = env._predicted_intercept_time()
            urgency = 1.0 - np.clip(intercept_time / max(self.strike_time_horizon, 1.0e-6), 0.0, 1.0)
            strike_readiness = max(env._pre_contact_height_readiness(), urgency)
            desired_target[2] = base_target[2] + self.strike_z_boost * strike_readiness
        else:
            next_intercept_metrics = env._next_intercept_metrics()
            if next_intercept_metrics["time"] > 0.0:
                next_intercept_xy = env.sim.racket_position[:2] + np.asarray(next_intercept_metrics["relative_xy"], dtype=float)
                blend = self.return_blend if phase_name == "return_shaping" else self.recovery_blend
                desired_target[:2] = (1.0 - blend) * base_target[:2] + blend * next_intercept_xy
            desired_target[2] = base_target[2]

        desired_target = desired_target + self._position_residual_for_phase(phase_name)
        desired_target = desired_target + self._state_dependent_strike_xy_residual(env, phase_name)
        world_delta = desired_target - base_target
        if env.action_mode in _CONTACT_FRAME_ACTION_MODES:
            radial, tangent, _ = env._contact_frame_basis_xy()
            action = np.array(
                [
                    float(np.dot(world_delta[:2], radial)),
                    float(np.dot(world_delta[:2], tangent)),
                    float(world_delta[2]),
                ],
                dtype=float,
            )
        else:
            action = world_delta
        if env.action_mode == "position_strike_tilt":
            tilt_residual = self._tilt_residual_for_phase(phase_name)
            action = np.concatenate([action, tilt_residual])
        elif env.action_mode == "position_strike_tilt_lift":
            tilt_residual = self._tilt_residual_for_phase(phase_name)
            followup_lift_residual = np.array([self._followup_lift_residual_for_phase(phase_name)], dtype=float)
            action = np.concatenate([action, tilt_residual, followup_lift_residual])
        elif env.action_mode in _CONTACT_FRAME_ACTION_MODES:
            tilt_residual = self._tilt_residual_for_phase(phase_name)
            action = np.concatenate([action, tilt_residual])
            if env.action_mode in _CONTACT_FRAME_VELOCITY_RESIDUAL_ACTION_MODES:
                action = np.concatenate([action, np.zeros(3, dtype=float)])
            if env.action_mode in _CONTACT_FRAME_TILT_SCALE_ACTION_MODES:
                action = np.concatenate([action, np.zeros(3, dtype=float)])
            if env.action_mode in _CONTACT_FRAME_LATERAL_VELOCITY_RESIDUAL_ACTION_MODES:
                action = np.concatenate([action, np.zeros(2, dtype=float)])
            if env.action_mode in _CONTACT_FRAME_APEX_TIMING_RESIDUAL_ACTION_MODES:
                action = np.concatenate([action, np.zeros(2, dtype=float)])
            if env.action_mode in _CONTACT_FRAME_TRACKING_RESIDUAL_ACTION_MODES:
                action = np.concatenate([action, self._tracking_velocity_residual(env, phase_name)])
        return np.clip(action, env.action_low, env.action_high)
