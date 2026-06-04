from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from pingpong_rl2.controllers import RacketCartesianController
from pingpong_rl2.defaults import (
    DEFAULT_ACTION_LIMIT,
    DEFAULT_APEX_MATCH_REWARD_WEIGHT,
    DEFAULT_BALL_HEIGHT,
    DEFAULT_BODY_CONTACT_PENALTY,
    DEFAULT_CONTACT_BONUS,
    DEFAULT_FAILURE_PENALTY,
    DEFAULT_FLOOR_PENALTY,
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_RESET_BALL_HEIGHT_RANGE,
    DEFAULT_RESET_VELOCITY_XY_RANGE,
    DEFAULT_RESET_VELOCITY_Z_RANGE,
    DEFAULT_RESET_XY_RANGE,
    DEFAULT_SUCCESS_VELOCITY_THRESHOLD,
    DEFAULT_TRACKING_REWARD_WEIGHT,
)
from pingpong_rl2.envs.pingpong_sim import PingPongSim

_ACTION_MODES = (
    "position",
    "position_strike",
    "position_tilt",
    "position_strike_tilt",
    "position_strike_tilt_lift",
    "position_contact_frame",
    "position_contact_frame_velocity_residual",
    "position_contact_frame_velocity_tilt_residual",
    "position_contact_frame_velocity_tilt_lateral_residual",
    "position_contact_frame_velocity_tilt_lateral_apex_residual",
    "position_contact_frame_velocity_tilt_lateral_apex_tracking_residual",
)
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
_TILT_ACTION_MODES = (
    "position_tilt",
    "position_strike_tilt",
    "position_strike_tilt_lift",
    *_CONTACT_FRAME_ACTION_MODES,
)
_STRIKE_CONTRACT_ACTION_MODES = (
    "position_strike",
    "position_strike_tilt",
    "position_strike_tilt_lift",
    *_CONTACT_FRAME_ACTION_MODES,
)
_TILT_SLICE_3_TO_5_ACTION_MODES = ("position_strike_tilt_lift", *_CONTACT_FRAME_ACTION_MODES)
_CONTACT_ORACLE_MODES = ("none", "desired_outgoing_velocity")
_RETURN_TARGET_XY_SOURCES = ("controller_anchor", "racket_home", "racket_position", "target_position")
_DESIRED_OUTGOING_XY_MODES = ("next_intercept", "apex")
_RESET_XY_SAMPLING_MODES = ("square", "disk")

_EASY_NEXT_BALL_TARGET_TIME = 0.45
_EASY_NEXT_BALL_TIME_TOLERANCE = 0.30
_EASY_NEXT_BALL_TARGET_DESCENDING_SPEED = 1.25
_EASY_NEXT_BALL_MAX_LATERAL_SPEED = 1.0
_EASY_NEXT_BALL_SOFT_SPEED_LIMIT = 3.0
_MIN_DESIRED_APEX_HEIGHT_DELTA = 0.01
_TRAJECTORY_MATCH_ERROR_SCALE = 1.0

_POSITION_OBSERVATION_COMPONENTS: tuple[tuple[str, int], ...] = (
    ("joint_positions", 7),
    ("joint_velocities", 7),
    ("racket_position", 3),
    ("racket_velocity", 3),
    ("target_position", 3),
    ("ball_position", 3),
    ("ball_velocity", 3),
    ("ball_relative_position", 3),
    ("predicted_intercept_relative_xy", 2),
    ("predicted_intercept_time", 1),
)

_TASK_PHASE_OBSERVATION_COMPONENTS: tuple[tuple[str, int], ...] = (
    ("phase_one_hot", 4),
)

_CONTACT_CONTEXT_OBSERVATION_COMPONENTS: tuple[tuple[str, int], ...] = (
    ("time_since_contact", 1),
    ("successful_bounce_count_clipped", 1),
)

_NEXT_INTERCEPT_OBSERVATION_COMPONENTS: tuple[tuple[str, int], ...] = (
    ("next_intercept_relative_xy", 2),
    ("next_intercept_time", 1),
    ("next_intercept_reachable", 1),
    ("next_intercept_recovery_distance", 1),
    ("next_intercept_recovery_readiness", 1),
)

_DESIRED_OUTGOING_OBSERVATION_COMPONENTS: tuple[tuple[str, int], ...] = (
    ("desired_outgoing_velocity", 3),
)

_VELOCITY_DOMAIN_OBSERVATION_COMPONENTS: tuple[tuple[str, int], ...] = (
    ("relative_velocity", 3),
    ("racket_face_normal", 3),
)

_POSITION_TILT_OBSERVATION_COMPONENTS: tuple[tuple[str, int], ...] = (
    ("target_tilt", 2),
)


def _build_observation_layout(
    action_mode: str,
    include_velocity_domain_observation: bool,
    include_task_phase_observation: bool,
    include_contact_context_observation: bool,
    include_next_intercept_observation: bool,
    include_desired_outgoing_velocity_observation: bool,
) -> tuple[tuple[tuple[str, int], ...], dict[str, slice], int]:
    components = _POSITION_OBSERVATION_COMPONENTS
    if include_task_phase_observation:
        components = components + _TASK_PHASE_OBSERVATION_COMPONENTS
    if include_contact_context_observation:
        components = components + _CONTACT_CONTEXT_OBSERVATION_COMPONENTS
    if include_next_intercept_observation:
        components = components + _NEXT_INTERCEPT_OBSERVATION_COMPONENTS
    if include_desired_outgoing_velocity_observation:
        components = components + _DESIRED_OUTGOING_OBSERVATION_COMPONENTS
    if include_velocity_domain_observation:
        components = components + _VELOCITY_DOMAIN_OBSERVATION_COMPONENTS
    if action_mode in _TILT_ACTION_MODES:
        if not include_velocity_domain_observation:
            components = components + (("racket_face_normal", 3),)
        components = components + _POSITION_TILT_OBSERVATION_COMPONENTS

    observation_slices: dict[str, slice] = {}
    observation_offset = 0
    for component_name, component_size in components:
        observation_slices[component_name] = slice(observation_offset, observation_offset + component_size)
        observation_offset += component_size
    return components, observation_slices, observation_offset


class PingPongKeepUpEnv:
    def __init__(
        self,
        sim: PingPongSim | None = None,
        scene_path: Path | str | None = None,
        action_mode: str = "position",
        action_limit: float = DEFAULT_ACTION_LIMIT,
        lateral_action_limit: float | None = None,
        vertical_action_limit: float | None = None,
        tilt_action_limit: float = 0.05,
        followup_lift_action_limit: float = 0.02,
        max_episode_steps: int | None = DEFAULT_MAX_EPISODE_STEPS,
        success_velocity_threshold: float = DEFAULT_SUCCESS_VELOCITY_THRESHOLD,
        ball_height: float = DEFAULT_BALL_HEIGHT,
        target_ball_height: float = DEFAULT_BALL_HEIGHT,
        height_tolerance: float = 0.10,
        tracking_reward_weight: float = DEFAULT_TRACKING_REWARD_WEIGHT,
        tracking_during_contact_scale: float = 0.0,
        contact_bonus: float = DEFAULT_CONTACT_BONUS,
        apex_match_reward_weight: float = DEFAULT_APEX_MATCH_REWARD_WEIGHT,
        useful_contact_outgoing_x_penalty_weight: float = 0.0,
        desired_outgoing_ball_velocity_x: float = 0.0,
        useful_contact_return_target_xy_reward_weight: float = 0.0,
        return_target_xy_source: str = "controller_anchor",
        return_target_xy_tolerance: float | None = None,
        tilt_angle_penalty_weight: float | None = None,
        tilt_action_delta_penalty_weight: float | None = None,
        descending_ball_velocity_threshold: float = -0.05,
        strike_zone_xy_radius: float = 0.10,
        strike_zone_height_tolerance: float = 0.16,
        contact_centering_radius: float = 0.04,
        min_upward_racket_velocity_z: float = 0.05,
        floor_penalty: float = DEFAULT_FLOOR_PENALTY,
        robot_body_contact_penalty: float = DEFAULT_BODY_CONTACT_PENALTY,
        failure_penalty: float = DEFAULT_FAILURE_PENALTY,
        reset_ball_height_range: float = DEFAULT_RESET_BALL_HEIGHT_RANGE,
        reset_ball_height_bounds: Sequence[float] | None = None,
        reset_xy_range: float = DEFAULT_RESET_XY_RANGE,
        reset_xy_sampling: str = "square",
        reset_velocity_xy_range: float = DEFAULT_RESET_VELOCITY_XY_RANGE,
        reset_velocity_z_range: tuple[float, float] = DEFAULT_RESET_VELOCITY_Z_RANGE,
        reset_ball_angular_velocity_range: float = 0.0,
        target_offset_low: Sequence[float] = (-0.12, -0.12, -0.04),
        target_offset_high: Sequence[float] = (0.12, 0.12, 0.12),
        target_tilt_limit: Sequence[float] = (0.18, 0.18),
        target_pitch_range: Sequence[float] | None = None,
        initial_target_tilt: Sequence[float] | None = None,
        strike_tilt_assist_limit: Sequence[float] | None = None,
        strike_tilt_assist_deadband: float = 0.015,
        strike_tilt_ramp_pitch: float | None = None,
        strike_tilt_ramp_xy_tolerance: float | None = None,
        followup_strike_target_tilt: Sequence[float] | None = None,
        followup_strike_contact_offset_ratio: float = 0.0,
        followup_strike_contact_offset_max: float = 0.0,
        followup_strike_lift_boost: float = 0.0,
        post_contact_return_assist_weight: float = 0.0,
        post_contact_return_max_intercept_time: float = 0.6,
        post_contact_return_z_offset: float = 0.0,
        post_contact_return_predict_during_rise: bool = True,
        next_intercept_reachable_bonus_weight: float = 0.0,
        easy_next_ball_reward_weight: float = 0.0,
        require_reachable_next_intercept_for_success: bool = False,
        require_apex_height_window_for_success: bool = False,
        min_easy_next_ball_score_for_success: float | None = None,
        gate_nonuseful_easy_next_ball_by_apex: bool = False,
        terminate_on_nonuseful_contact: bool = False,
        terminate_on_low_apex_contact: bool = False,
        low_apex_contact_height_threshold: float | None = None,
        low_apex_contact_grace_count: int = 0,
        include_velocity_domain_observation: bool = False,
        include_task_phase_observation: bool = False,
        include_contact_context_observation: bool = False,
        include_next_intercept_observation: bool = False,
        include_desired_outgoing_velocity_observation: bool = False,
        desired_outgoing_xy_mode: str = "next_intercept",
        keepup_target_xy_offset: Sequence[float] = (0.0, 0.0),
        trajectory_match_reward_weight: float = 0.0,
        trajectory_error_penalty_weight: float = 0.0,
        reward_contact_quality_on_any_upward_contact: bool = False,
        contact_oracle_mode: str = "none",
        contact_oracle_blend: float = 1.0,
        next_intercept_max_time: float = 1.25,
        next_intercept_success_radius: float | None = None,
        easy_next_ball_xy_radius: float | None = None,
        controller_position_gain: float = 1.6,
        controller_orientation_gain: float = 0.45,
        controller_max_position_step: float = 0.06,
        controller_max_orientation_step: float = 0.12,
        controller_velocity_gain: float = 1.0,
        controller_velocity_feedback_gain: float = 0.0,
        controller_max_velocity_step: float = 0.02,
        controller_nullspace_posture_gain: float = 0.0,
        controller_nullspace_posture_max_step: float = 0.0,
        controller_nullspace_posture_target: Sequence[float] | None = None,
        controller_body_clearance_gain: float = 0.0,
        controller_body_clearance_margin: float = 0.0,
        controller_body_clearance_vertical_margin: float = 0.30,
        controller_body_clearance_max_step: float = 0.0,
        controller_body_clearance_body_names: Sequence[str] = ("link5",),
        tracking_strike_plane_offset: float = 0.02,
        contact_frame_base_strike_z_boost: float = 0.0,
        contact_frame_base_strike_z_offset: float = 0.0,
        contact_frame_base_strike_time_horizon: float = 0.14,
        contact_frame_base_tilt_residual: Sequence[float] | None = None,
        contact_frame_apex_lift_gain: float = 0.0,
        contact_frame_apex_lift_max: float = 0.0,
        contact_frame_apex_lift_reference_velocity_z: float = -1.0,
        contact_frame_apex_lift_restitution: float = 0.8,
        contact_frame_velocity_lead_gain: float = 0.0,
        contact_frame_velocity_lead_max: float = 0.0,
        contact_frame_velocity_target_gain: float = 0.0,
        contact_frame_velocity_target_max: float = 0.0,
        contact_frame_velocity_scale_action_limit: float = 0.35,
        contact_frame_outgoing_xy_action_limit: float = 0.35,
        contact_frame_racket_vz_action_limit: float = 0.45,
        contact_frame_racket_xy_action_limit: float = 0.35,
        contact_frame_tilt_scale_action_limit: float = 0.75,
        contact_frame_target_apex_z_action_limit: float = 0.08,
        contact_frame_strike_plane_z_action_limit: float = 0.025,
        contact_frame_tracking_xy_action_limit: float = 0.60,
        contact_frame_intercept_velocity_gain: float = 0.0,
        contact_frame_intercept_velocity_max: float = 0.0,
        contact_frame_intercept_velocity_time_floor: float = 0.08,
        contact_frame_planner_enabled: bool = False,
        contact_frame_planner_hold_during_descent: bool = True,
        contact_frame_planner_min_intercept_time: float = 0.03,
        contact_frame_planner_max_intercept_time: float = 0.60,
        contact_frame_planner_target_apex_z_offset: float = 0.0,
        contact_frame_planner_contact_offset_ratio: float = 0.0,
        contact_frame_planner_contact_offset_max: float = 0.0,
        contact_frame_strike_hold_time: float = 0.0,
        contact_frame_strike_hold_min_readiness: float = 0.65,
        contact_frame_followthrough_gain: float = 0.0,
        contact_frame_followthrough_time: float = 0.06,
        contact_frame_followthrough_max: float = 0.0,
        contact_frame_lateral_brake_gain: float = 0.0,
        contact_frame_lateral_brake_max: float = 0.0,
        contact_frame_lateral_brake_radius: float = 0.12,
        contact_frame_trajectory_tilt_gain: float = 0.0,
        contact_frame_trajectory_tilt_limit: Sequence[float] | None = None,
        contact_frame_trajectory_tilt_deadband: float = 0.0,
        contact_frame_tilt_ramp_time: float = 0.16,
        contact_frame_centering_tilt_limit: Sequence[float] | None = None,
        contact_frame_centering_tilt_radius: float | None = None,
        contact_frame_centering_tilt_deadband: float = 0.015,
        contact_frame_action_penalty_weight: float = 0.0,
        next_intercept_xy_error_penalty_weight: float = 0.0,
        post_contact_lateral_velocity_penalty_weight: float = 0.0,
        contact_xy_error_penalty_weight: float = 0.0,
        contact_racket_lateral_velocity_penalty_weight: float = 0.0,
        contact_racket_lateral_velocity_tolerance: float = 0.20,
        contact_racket_outward_velocity_penalty_weight: float = 0.0,
        contact_racket_outward_velocity_tolerance: float = 0.04,
        max_contact_racket_lateral_speed_for_success: float | None = None,
        nonuseful_contact_penalty_weight: float = 0.0,
        contact_apex_under_target_penalty_weight: float = 0.0,
        contact_apex_progress_reward_weight: float = 0.0,
        contact_apex_recovery_progress_reward_weight: float = 0.0,
        gate_contact_apex_progress_by_easy_next_ball: bool = False,
        contact_apex_progress_min_easy_next_ball_score: float | None = None,
        contact_apex_potential_reward_weight: float = 0.0,
        contact_apex_potential_gamma: float = 0.99,
        contact_apex_potential_cap: float = 2.0,
        contact_lateral_stability_reward_weight: float = 0.0,
        contact_lateral_stability_speed_tolerance: float = 0.25,
        contact_lateral_stability_xy_tolerance: float | None = None,
        contact_lateral_stability_min_apex_ratio: float | None = None,
        stable_contact_reward_weight: float = 0.0,
        stable_contact_min_apex_ratio: float | None = None,
        stable_cycle_reward_weight: float = 0.0,
        stable_cycle_reward_cap: int = 4,
        stable_cycle_min_easy_next_ball_score: float | None = None,
        contact_frame_low_apex_recovery_lift_gain: float = 0.0,
        contact_frame_low_apex_recovery_lift_max: float = 0.0,
        contact_frame_low_apex_recovery_velocity_gain: float = 0.0,
        contact_frame_low_apex_recovery_velocity_max: float = 0.0,
    ) -> None:
        if sim is not None and scene_path is not None:
            raise ValueError("scene_path can only be provided when sim is None.")
        self.sim = PingPongSim(scene_path=scene_path) if sim is None else sim
        self.scene_path = str(self.sim.scene_path)
        self.action_mode = str(action_mode)
        self.action_limit = float(action_limit)
        self.lateral_action_limit = (
            0.75 * self.action_limit if lateral_action_limit is None else float(lateral_action_limit)
        )
        self.vertical_action_limit = self.action_limit if vertical_action_limit is None else float(vertical_action_limit)
        self.tilt_action_limit = float(tilt_action_limit)
        self.followup_lift_action_limit = float(followup_lift_action_limit)
        if max_episode_steps is None:
            self.max_episode_steps = None
        else:
            parsed_max_episode_steps = int(max_episode_steps)
            self.max_episode_steps = None if parsed_max_episode_steps <= 0 else parsed_max_episode_steps
        self.success_velocity_threshold = float(success_velocity_threshold)
        self.ball_height = float(ball_height)
        self.target_ball_height = float(target_ball_height)
        self.height_tolerance = float(height_tolerance)
        self.tracking_reward_weight = float(tracking_reward_weight)
        self.tracking_during_contact_scale = float(tracking_during_contact_scale)
        self.contact_bonus = float(contact_bonus)
        self.apex_match_reward_weight = float(apex_match_reward_weight)
        self.useful_contact_outgoing_x_penalty_weight = float(useful_contact_outgoing_x_penalty_weight)
        self.desired_outgoing_ball_velocity_x = float(desired_outgoing_ball_velocity_x)
        self.useful_contact_return_target_xy_reward_weight = float(useful_contact_return_target_xy_reward_weight)
        self.return_target_xy_source = str(return_target_xy_source)
        default_tilt_angle_penalty_weight = 0.04 if self.action_mode in _TILT_ACTION_MODES else 0.0
        default_tilt_action_delta_penalty_weight = 0.10 if self.action_mode in _TILT_ACTION_MODES else 0.0
        self.tilt_angle_penalty_weight = float(
            default_tilt_angle_penalty_weight
            if tilt_angle_penalty_weight is None
            else tilt_angle_penalty_weight
        )
        self.tilt_action_delta_penalty_weight = float(
            default_tilt_action_delta_penalty_weight
            if tilt_action_delta_penalty_weight is None
            else tilt_action_delta_penalty_weight
        )
        self.descending_ball_velocity_threshold = float(descending_ball_velocity_threshold)
        self.strike_zone_xy_radius = float(strike_zone_xy_radius)
        self.strike_zone_height_tolerance = float(strike_zone_height_tolerance)
        self.return_target_xy_tolerance = (
            self.strike_zone_xy_radius if return_target_xy_tolerance is None else float(return_target_xy_tolerance)
        )
        self.contact_centering_radius = float(contact_centering_radius)
        self.min_upward_racket_velocity_z = float(min_upward_racket_velocity_z)
        self.floor_penalty = float(floor_penalty)
        self.robot_body_contact_penalty = float(robot_body_contact_penalty)
        self.failure_penalty = float(failure_penalty)
        self.reset_ball_height_range = float(reset_ball_height_range)
        self.reset_ball_height_bounds = (
            None
            if reset_ball_height_bounds is None
            else (float(reset_ball_height_bounds[0]), float(reset_ball_height_bounds[1]))
        )
        self.reset_xy_range = float(reset_xy_range)
        self.reset_xy_sampling = str(reset_xy_sampling)
        self.reset_velocity_xy_range = float(reset_velocity_xy_range)
        self.reset_velocity_z_range = (float(reset_velocity_z_range[0]), float(reset_velocity_z_range[1]))
        self.reset_ball_angular_velocity_range = float(reset_ball_angular_velocity_range)
        self.target_offset_low = np.asarray(target_offset_low, dtype=float)
        self.target_offset_high = np.asarray(target_offset_high, dtype=float)
        self.target_tilt_limit = np.asarray(target_tilt_limit, dtype=float)
        self.target_pitch_range = None if target_pitch_range is None else np.asarray(target_pitch_range, dtype=float)
        self.initial_target_tilt = (
            None if initial_target_tilt is None else np.asarray(initial_target_tilt, dtype=float)
        )
        self.strike_tilt_assist_limit = (
            None if strike_tilt_assist_limit is None else np.asarray(strike_tilt_assist_limit, dtype=float)
        )
        self.strike_tilt_assist_deadband = float(strike_tilt_assist_deadband)
        self.strike_tilt_ramp_pitch = None if strike_tilt_ramp_pitch is None else float(strike_tilt_ramp_pitch)
        self.strike_tilt_ramp_xy_tolerance = (
            self.contact_centering_radius
            if strike_tilt_ramp_xy_tolerance is None
            else float(strike_tilt_ramp_xy_tolerance)
        )
        self.followup_strike_target_tilt = (
            None if followup_strike_target_tilt is None else np.asarray(followup_strike_target_tilt, dtype=float)
        )
        self.followup_strike_contact_offset_ratio = float(followup_strike_contact_offset_ratio)
        self.followup_strike_contact_offset_max = float(followup_strike_contact_offset_max)
        self.followup_strike_lift_boost = float(followup_strike_lift_boost)
        self.post_contact_return_assist_weight = float(post_contact_return_assist_weight)
        self.post_contact_return_max_intercept_time = float(post_contact_return_max_intercept_time)
        self.post_contact_return_z_offset = float(post_contact_return_z_offset)
        self.post_contact_return_predict_during_rise = bool(post_contact_return_predict_during_rise)
        self.next_intercept_reachable_bonus_weight = float(next_intercept_reachable_bonus_weight)
        self.easy_next_ball_reward_weight = float(easy_next_ball_reward_weight)
        self.require_reachable_next_intercept_for_success = bool(require_reachable_next_intercept_for_success)
        self.require_apex_height_window_for_success = bool(require_apex_height_window_for_success)
        self.min_easy_next_ball_score_for_success = (
            None if min_easy_next_ball_score_for_success is None else float(min_easy_next_ball_score_for_success)
        )
        self.gate_nonuseful_easy_next_ball_by_apex = bool(gate_nonuseful_easy_next_ball_by_apex)
        self.terminate_on_nonuseful_contact = bool(terminate_on_nonuseful_contact)
        self.terminate_on_low_apex_contact = bool(terminate_on_low_apex_contact)
        self.low_apex_contact_height_threshold = (
            None if low_apex_contact_height_threshold is None else float(low_apex_contact_height_threshold)
        )
        self.low_apex_contact_grace_count = int(low_apex_contact_grace_count)
        self.include_velocity_domain_observation = bool(include_velocity_domain_observation)
        self.include_task_phase_observation = bool(include_task_phase_observation)
        self.include_contact_context_observation = bool(include_contact_context_observation)
        self.include_next_intercept_observation = bool(include_next_intercept_observation)
        self.include_desired_outgoing_velocity_observation = bool(include_desired_outgoing_velocity_observation)
        self.desired_outgoing_xy_mode = str(desired_outgoing_xy_mode)
        self.keepup_target_xy_offset = np.asarray(keepup_target_xy_offset, dtype=float)
        self.trajectory_match_reward_weight = float(trajectory_match_reward_weight)
        self.trajectory_error_penalty_weight = float(trajectory_error_penalty_weight)
        self.reward_contact_quality_on_any_upward_contact = bool(reward_contact_quality_on_any_upward_contact)
        self.contact_oracle_mode = str(contact_oracle_mode)
        self.contact_oracle_blend = float(contact_oracle_blend)
        self.next_intercept_max_time = float(next_intercept_max_time)
        self.next_intercept_success_radius = (
            self.strike_zone_xy_radius
            if next_intercept_success_radius is None
            else float(next_intercept_success_radius)
        )
        self.easy_next_ball_xy_radius = (
            self.next_intercept_success_radius
            if easy_next_ball_xy_radius is None
            else float(easy_next_ball_xy_radius)
        )
        self.controller_position_gain = float(controller_position_gain)
        self.controller_orientation_gain = float(controller_orientation_gain)
        self.controller_max_position_step = float(controller_max_position_step)
        self.controller_max_orientation_step = float(controller_max_orientation_step)
        self.controller_velocity_gain = float(controller_velocity_gain)
        self.controller_velocity_feedback_gain = float(controller_velocity_feedback_gain)
        self.controller_max_velocity_step = float(controller_max_velocity_step)
        self.controller_nullspace_posture_gain = float(controller_nullspace_posture_gain)
        self.controller_nullspace_posture_max_step = float(controller_nullspace_posture_max_step)
        self.controller_nullspace_posture_target = (
            None
            if controller_nullspace_posture_target is None
            else np.asarray(controller_nullspace_posture_target, dtype=float)
        )
        self.controller_body_clearance_gain = float(controller_body_clearance_gain)
        self.controller_body_clearance_margin = float(controller_body_clearance_margin)
        self.controller_body_clearance_vertical_margin = float(controller_body_clearance_vertical_margin)
        self.controller_body_clearance_max_step = float(controller_body_clearance_max_step)
        self.controller_body_clearance_body_names = tuple(str(name) for name in controller_body_clearance_body_names)
        self.tracking_strike_plane_offset = float(tracking_strike_plane_offset)
        self.contact_frame_base_strike_z_boost = float(contact_frame_base_strike_z_boost)
        self.contact_frame_base_strike_z_offset = float(contact_frame_base_strike_z_offset)
        self.contact_frame_base_strike_time_horizon = float(contact_frame_base_strike_time_horizon)
        self.contact_frame_base_tilt_residual = (
            None if contact_frame_base_tilt_residual is None else np.asarray(contact_frame_base_tilt_residual, dtype=float)
        )
        self.contact_frame_apex_lift_gain = float(contact_frame_apex_lift_gain)
        self.contact_frame_apex_lift_max = float(contact_frame_apex_lift_max)
        self.contact_frame_apex_lift_reference_velocity_z = float(contact_frame_apex_lift_reference_velocity_z)
        self.contact_frame_apex_lift_restitution = float(contact_frame_apex_lift_restitution)
        self.contact_frame_velocity_lead_gain = float(contact_frame_velocity_lead_gain)
        self.contact_frame_velocity_lead_max = float(contact_frame_velocity_lead_max)
        self.contact_frame_velocity_target_gain = float(contact_frame_velocity_target_gain)
        self.contact_frame_velocity_target_max = float(contact_frame_velocity_target_max)
        self.contact_frame_velocity_scale_action_limit = float(contact_frame_velocity_scale_action_limit)
        self.contact_frame_outgoing_xy_action_limit = float(contact_frame_outgoing_xy_action_limit)
        self.contact_frame_racket_vz_action_limit = float(contact_frame_racket_vz_action_limit)
        self.contact_frame_racket_xy_action_limit = float(contact_frame_racket_xy_action_limit)
        self.contact_frame_tilt_scale_action_limit = float(contact_frame_tilt_scale_action_limit)
        self.contact_frame_target_apex_z_action_limit = float(contact_frame_target_apex_z_action_limit)
        self.contact_frame_strike_plane_z_action_limit = float(contact_frame_strike_plane_z_action_limit)
        self.contact_frame_tracking_xy_action_limit = float(contact_frame_tracking_xy_action_limit)
        self.contact_frame_intercept_velocity_gain = float(contact_frame_intercept_velocity_gain)
        self.contact_frame_intercept_velocity_max = float(contact_frame_intercept_velocity_max)
        self.contact_frame_intercept_velocity_time_floor = float(contact_frame_intercept_velocity_time_floor)
        self.contact_frame_planner_enabled = bool(contact_frame_planner_enabled)
        self.contact_frame_planner_hold_during_descent = bool(contact_frame_planner_hold_during_descent)
        self.contact_frame_planner_min_intercept_time = float(contact_frame_planner_min_intercept_time)
        self.contact_frame_planner_max_intercept_time = float(contact_frame_planner_max_intercept_time)
        self.contact_frame_planner_target_apex_z_offset = float(contact_frame_planner_target_apex_z_offset)
        self.contact_frame_planner_contact_offset_ratio = float(contact_frame_planner_contact_offset_ratio)
        self.contact_frame_planner_contact_offset_max = float(contact_frame_planner_contact_offset_max)
        self.contact_frame_strike_hold_time = float(contact_frame_strike_hold_time)
        self.contact_frame_strike_hold_min_readiness = float(contact_frame_strike_hold_min_readiness)
        self.contact_frame_followthrough_gain = float(contact_frame_followthrough_gain)
        self.contact_frame_followthrough_time = float(contact_frame_followthrough_time)
        self.contact_frame_followthrough_max = float(contact_frame_followthrough_max)
        self.contact_frame_lateral_brake_gain = float(contact_frame_lateral_brake_gain)
        self.contact_frame_lateral_brake_max = float(contact_frame_lateral_brake_max)
        self.contact_frame_lateral_brake_radius = float(contact_frame_lateral_brake_radius)
        self.contact_frame_trajectory_tilt_gain = float(contact_frame_trajectory_tilt_gain)
        self.contact_frame_trajectory_tilt_limit = (
            None
            if contact_frame_trajectory_tilt_limit is None
            else np.asarray(contact_frame_trajectory_tilt_limit, dtype=float)
        )
        self.contact_frame_trajectory_tilt_deadband = float(contact_frame_trajectory_tilt_deadband)
        self.contact_frame_tilt_ramp_time = float(contact_frame_tilt_ramp_time)
        self.contact_frame_centering_tilt_limit = (
            None
            if contact_frame_centering_tilt_limit is None
            else np.asarray(contact_frame_centering_tilt_limit, dtype=float)
        )
        self.contact_frame_centering_tilt_radius = (
            None if contact_frame_centering_tilt_radius is None else float(contact_frame_centering_tilt_radius)
        )
        self.contact_frame_centering_tilt_deadband = float(contact_frame_centering_tilt_deadband)
        self.contact_frame_action_penalty_weight = float(contact_frame_action_penalty_weight)
        self.next_intercept_xy_error_penalty_weight = float(next_intercept_xy_error_penalty_weight)
        self.post_contact_lateral_velocity_penalty_weight = float(post_contact_lateral_velocity_penalty_weight)
        self.contact_xy_error_penalty_weight = float(contact_xy_error_penalty_weight)
        self.contact_racket_lateral_velocity_penalty_weight = float(contact_racket_lateral_velocity_penalty_weight)
        self.contact_racket_lateral_velocity_tolerance = float(contact_racket_lateral_velocity_tolerance)
        self.contact_racket_outward_velocity_penalty_weight = float(
            contact_racket_outward_velocity_penalty_weight
        )
        self.contact_racket_outward_velocity_tolerance = float(contact_racket_outward_velocity_tolerance)
        self.max_contact_racket_lateral_speed_for_success = (
            None
            if max_contact_racket_lateral_speed_for_success is None
            else float(max_contact_racket_lateral_speed_for_success)
        )
        self.nonuseful_contact_penalty_weight = float(nonuseful_contact_penalty_weight)
        self.contact_apex_under_target_penalty_weight = float(contact_apex_under_target_penalty_weight)
        self.contact_apex_progress_reward_weight = float(contact_apex_progress_reward_weight)
        self.contact_apex_recovery_progress_reward_weight = float(contact_apex_recovery_progress_reward_weight)
        self.gate_contact_apex_progress_by_easy_next_ball = bool(gate_contact_apex_progress_by_easy_next_ball)
        self.contact_apex_progress_min_easy_next_ball_score = (
            None
            if contact_apex_progress_min_easy_next_ball_score is None
            else float(contact_apex_progress_min_easy_next_ball_score)
        )
        self.contact_apex_potential_reward_weight = float(contact_apex_potential_reward_weight)
        self.contact_apex_potential_gamma = float(contact_apex_potential_gamma)
        self.contact_apex_potential_cap = float(contact_apex_potential_cap)
        self.contact_lateral_stability_reward_weight = float(contact_lateral_stability_reward_weight)
        self.contact_lateral_stability_speed_tolerance = float(contact_lateral_stability_speed_tolerance)
        self.contact_lateral_stability_xy_tolerance = (
            None
            if contact_lateral_stability_xy_tolerance is None
            else float(contact_lateral_stability_xy_tolerance)
        )
        self.contact_lateral_stability_min_apex_ratio = (
            None
            if contact_lateral_stability_min_apex_ratio is None
            else float(contact_lateral_stability_min_apex_ratio)
        )
        self.stable_contact_reward_weight = float(stable_contact_reward_weight)
        self.stable_contact_min_apex_ratio = (
            None if stable_contact_min_apex_ratio is None else float(stable_contact_min_apex_ratio)
        )
        self.stable_cycle_reward_weight = float(stable_cycle_reward_weight)
        self.stable_cycle_reward_cap = int(stable_cycle_reward_cap)
        self.stable_cycle_min_easy_next_ball_score = (
            None
            if stable_cycle_min_easy_next_ball_score is None
            else float(stable_cycle_min_easy_next_ball_score)
        )
        self.contact_frame_low_apex_recovery_lift_gain = float(contact_frame_low_apex_recovery_lift_gain)
        self.contact_frame_low_apex_recovery_lift_max = float(contact_frame_low_apex_recovery_lift_max)
        self.contact_frame_low_apex_recovery_velocity_gain = float(contact_frame_low_apex_recovery_velocity_gain)
        self.contact_frame_low_apex_recovery_velocity_max = float(contact_frame_low_apex_recovery_velocity_max)
        if self.action_mode not in _ACTION_MODES:
            raise ValueError(f"action_mode must be one of {_ACTION_MODES}, got {self.action_mode!r}.")
        if self.contact_oracle_mode not in _CONTACT_ORACLE_MODES:
            raise ValueError(
                f"contact_oracle_mode must be one of {_CONTACT_ORACLE_MODES}, got {self.contact_oracle_mode!r}."
            )
        if self.desired_outgoing_xy_mode not in _DESIRED_OUTGOING_XY_MODES:
            raise ValueError(
                "desired_outgoing_xy_mode must be one of "
                f"{_DESIRED_OUTGOING_XY_MODES}, got {self.desired_outgoing_xy_mode!r}."
            )
        if self.keepup_target_xy_offset.shape != (2,):
            raise ValueError(
                f"keepup_target_xy_offset must have shape (2,), got {self.keepup_target_xy_offset.shape}."
            )
        if not np.isfinite(self.keepup_target_xy_offset).all():
            raise ValueError(f"keepup_target_xy_offset must be finite, got {self.keepup_target_xy_offset}.")
        if not 0.0 <= self.contact_oracle_blend <= 1.0:
            raise ValueError(
                f"contact_oracle_blend must be within [0, 1], got {self.contact_oracle_blend}."
            )
        if self.action_limit <= 0.0:
            raise ValueError(f"action_limit must be positive, got {self.action_limit}.")
        if self.lateral_action_limit <= 0.0:
            raise ValueError(
                f"lateral_action_limit must be positive, got {self.lateral_action_limit}."
            )
        if self.vertical_action_limit <= 0.0:
            raise ValueError(
                f"vertical_action_limit must be positive, got {self.vertical_action_limit}."
            )
        if self.tilt_action_limit <= 0.0:
            raise ValueError(f"tilt_action_limit must be positive, got {self.tilt_action_limit}.")
        if self.followup_lift_action_limit <= 0.0:
            raise ValueError(
                f"followup_lift_action_limit must be positive, got {self.followup_lift_action_limit}."
            )
        if not 0.0 <= self.tracking_during_contact_scale <= 1.0:
            raise ValueError(
                "tracking_during_contact_scale must be within [0, 1], got "
                f"{self.tracking_during_contact_scale}."
            )
        if self.useful_contact_outgoing_x_penalty_weight < 0.0:
            raise ValueError(
                "useful_contact_outgoing_x_penalty_weight must be non-negative, got "
                f"{self.useful_contact_outgoing_x_penalty_weight}."
            )
        if self.useful_contact_return_target_xy_reward_weight < 0.0:
            raise ValueError(
                "useful_contact_return_target_xy_reward_weight must be non-negative, got "
                f"{self.useful_contact_return_target_xy_reward_weight}."
            )
        if self.return_target_xy_source not in _RETURN_TARGET_XY_SOURCES:
            raise ValueError(
                f"return_target_xy_source must be one of {_RETURN_TARGET_XY_SOURCES}, got {self.return_target_xy_source!r}."
            )
        if self.return_target_xy_tolerance <= 0.0:
            raise ValueError(
                f"return_target_xy_tolerance must be positive, got {self.return_target_xy_tolerance}."
            )
        if self.tilt_angle_penalty_weight < 0.0:
            raise ValueError(
                f"tilt_angle_penalty_weight must be non-negative, got {self.tilt_angle_penalty_weight}."
            )
        if self.tilt_action_delta_penalty_weight < 0.0:
            raise ValueError(
                "tilt_action_delta_penalty_weight must be non-negative, got "
                f"{self.tilt_action_delta_penalty_weight}."
            )
        if self.max_episode_steps is not None and self.max_episode_steps < 1:
            raise ValueError(f"max_episode_steps must be positive or <= 0 for unlimited, got {self.max_episode_steps}.")
        if self.height_tolerance <= 0.0:
            raise ValueError(f"height_tolerance must be positive, got {self.height_tolerance}.")
        if self.target_ball_height <= 0.0:
            raise ValueError(f"target_ball_height must be positive, got {self.target_ball_height}.")
        if self.reset_ball_height_range < 0.0:
            raise ValueError(f"reset_ball_height_range must be non-negative, got {self.reset_ball_height_range}.")
        if self.reset_ball_height_bounds is not None:
            if self.reset_ball_height_bounds[0] > self.reset_ball_height_bounds[1]:
                raise ValueError(
                    "reset_ball_height_bounds must be ordered as (low, high), got "
                    f"{self.reset_ball_height_bounds}."
                )
            if self.reset_ball_height_bounds[0] <= 0.0:
                raise ValueError(
                    "reset_ball_height_bounds must stay above the racket plane, got "
                    f"{self.reset_ball_height_bounds}."
                )
        if self.reset_xy_range < 0.0:
            raise ValueError(f"reset_xy_range must be non-negative, got {self.reset_xy_range}.")
        if self.reset_xy_sampling not in _RESET_XY_SAMPLING_MODES:
            raise ValueError(
                f"reset_xy_sampling must be one of {_RESET_XY_SAMPLING_MODES}, got {self.reset_xy_sampling!r}."
            )
        if self.reset_velocity_z_range[0] > self.reset_velocity_z_range[1]:
            raise ValueError(
                "reset_velocity_z_range must be ordered as (low, high), got "
                f"{self.reset_velocity_z_range}."
            )
        if self.reset_ball_angular_velocity_range < 0.0:
            raise ValueError(
                "reset_ball_angular_velocity_range must be non-negative, got "
                f"{self.reset_ball_angular_velocity_range}."
            )
        if self.target_tilt_limit.shape != (2,):
            raise ValueError(f"target_tilt_limit must have shape (2,), got {self.target_tilt_limit.shape}.")
        if np.any(self.target_tilt_limit < 0.0):
            raise ValueError(f"target_tilt_limit must be non-negative, got {self.target_tilt_limit}.")
        if self.strike_tilt_assist_limit is not None:
            if self.strike_tilt_assist_limit.shape != (2,):
                raise ValueError(
                    "strike_tilt_assist_limit must have shape (2,), got "
                    f"{self.strike_tilt_assist_limit.shape}."
                )
            if np.any(self.strike_tilt_assist_limit < 0.0):
                raise ValueError(
                    f"strike_tilt_assist_limit must be non-negative, got {self.strike_tilt_assist_limit}."
                )
            if np.any(self.strike_tilt_assist_limit > self.target_tilt_limit + 1.0e-9):
                raise ValueError(
                    "strike_tilt_assist_limit must stay within target_tilt_limit, got "
                    f"{self.strike_tilt_assist_limit} with target_tilt_limit={self.target_tilt_limit}."
                )
        if self.strike_tilt_assist_deadband < 0.0:
            raise ValueError(
                f"strike_tilt_assist_deadband must be non-negative, got {self.strike_tilt_assist_deadband}."
            )
        if self.followup_strike_target_tilt is not None:
            if self.followup_strike_target_tilt.shape != (2,):
                raise ValueError(
                    "followup_strike_target_tilt must have shape (2,), got "
                    f"{self.followup_strike_target_tilt.shape}."
                )
            if np.any(np.abs(self.followup_strike_target_tilt) > self.target_tilt_limit + 1.0e-9):
                raise ValueError(
                    "followup_strike_target_tilt must stay within target_tilt_limit, got "
                    f"{self.followup_strike_target_tilt} with target_tilt_limit={self.target_tilt_limit}."
                )
        if self.followup_strike_contact_offset_ratio < 0.0:
            raise ValueError(
                "followup_strike_contact_offset_ratio must be non-negative, got "
                f"{self.followup_strike_contact_offset_ratio}."
            )
        if self.followup_strike_contact_offset_max < 0.0:
            raise ValueError(
                "followup_strike_contact_offset_max must be non-negative, got "
                f"{self.followup_strike_contact_offset_max}."
            )
        if self.followup_strike_lift_boost < 0.0:
            raise ValueError(
                f"followup_strike_lift_boost must be non-negative, got {self.followup_strike_lift_boost}."
            )
        if self.strike_tilt_ramp_pitch is not None:
            if abs(self.strike_tilt_ramp_pitch) > self.target_tilt_limit[0] + 1.0e-9:
                raise ValueError(
                    "strike_tilt_ramp_pitch must stay within +/- target_tilt_limit[0], got "
                    f"{self.strike_tilt_ramp_pitch} with target_tilt_limit={self.target_tilt_limit}."
                )
            if self.strike_tilt_assist_limit is not None:
                raise ValueError(
                    "strike_tilt_ramp_pitch cannot be combined with strike_tilt_assist_limit. "
                    "Choose one strike tilt experiment at a time."
                )
        if not 0.0 <= self.post_contact_return_assist_weight <= 1.0:
            raise ValueError(
                "post_contact_return_assist_weight must be within [0, 1], got "
                f"{self.post_contact_return_assist_weight}."
            )
        if self.post_contact_return_max_intercept_time <= 0.0:
            raise ValueError(
                "post_contact_return_max_intercept_time must be positive, got "
                f"{self.post_contact_return_max_intercept_time}."
            )
        if not np.isfinite(self.post_contact_return_z_offset):
            raise ValueError(
                "post_contact_return_z_offset must be finite, got "
                f"{self.post_contact_return_z_offset}."
            )
        if self.next_intercept_reachable_bonus_weight < 0.0:
            raise ValueError(
                "next_intercept_reachable_bonus_weight must be non-negative, got "
                f"{self.next_intercept_reachable_bonus_weight}."
            )
        if self.easy_next_ball_reward_weight < 0.0:
            raise ValueError(
                f"easy_next_ball_reward_weight must be non-negative, got {self.easy_next_ball_reward_weight}."
            )
        if self.min_easy_next_ball_score_for_success is not None and not np.isfinite(
            self.min_easy_next_ball_score_for_success
        ):
            raise ValueError(
                "min_easy_next_ball_score_for_success must be finite when provided, got "
                f"{self.min_easy_next_ball_score_for_success}."
            )
        if self.low_apex_contact_height_threshold is not None:
            if not np.isfinite(self.low_apex_contact_height_threshold):
                raise ValueError(
                    "low_apex_contact_height_threshold must be finite when provided, got "
                    f"{self.low_apex_contact_height_threshold}."
                )
            if self.low_apex_contact_height_threshold <= 0.0:
                raise ValueError(
                    "low_apex_contact_height_threshold must be positive when provided, got "
                    f"{self.low_apex_contact_height_threshold}."
                )
        if self.low_apex_contact_grace_count < 0:
            raise ValueError(
                f"low_apex_contact_grace_count must be non-negative, got {self.low_apex_contact_grace_count}."
            )
        if self.next_intercept_max_time <= 0.0:
            raise ValueError(
                f"next_intercept_max_time must be positive, got {self.next_intercept_max_time}."
            )
        if self.next_intercept_success_radius <= 0.0:
            raise ValueError(
                "next_intercept_success_radius must be positive, got "
                f"{self.next_intercept_success_radius}."
            )
        if self.easy_next_ball_xy_radius <= 0.0:
            raise ValueError(
                f"easy_next_ball_xy_radius must be positive, got {self.easy_next_ball_xy_radius}."
            )
        if self.contact_frame_base_strike_z_boost < 0.0:
            raise ValueError(
                "contact_frame_base_strike_z_boost must be non-negative, got "
                f"{self.contact_frame_base_strike_z_boost}."
            )
        if self.contact_frame_base_strike_time_horizon <= 0.0:
            raise ValueError(
                "contact_frame_base_strike_time_horizon must be positive, got "
                f"{self.contact_frame_base_strike_time_horizon}."
            )
        if self.trajectory_error_penalty_weight < 0.0:
            raise ValueError(
                "trajectory_error_penalty_weight must be non-negative, got "
                f"{self.trajectory_error_penalty_weight}."
            )
        if self.contact_frame_apex_lift_gain < 0.0:
            raise ValueError(
                f"contact_frame_apex_lift_gain must be non-negative, got {self.contact_frame_apex_lift_gain}."
            )
        if self.contact_frame_apex_lift_max < 0.0:
            raise ValueError(
                f"contact_frame_apex_lift_max must be non-negative, got {self.contact_frame_apex_lift_max}."
            )
        if self.contact_frame_apex_lift_restitution < 0.0:
            raise ValueError(
                "contact_frame_apex_lift_restitution must be non-negative, got "
                f"{self.contact_frame_apex_lift_restitution}."
            )
        if self.contact_frame_velocity_lead_gain < 0.0:
            raise ValueError(
                "contact_frame_velocity_lead_gain must be non-negative, got "
                f"{self.contact_frame_velocity_lead_gain}."
            )
        if self.contact_frame_velocity_lead_max < 0.0:
            raise ValueError(
                "contact_frame_velocity_lead_max must be non-negative, got "
                f"{self.contact_frame_velocity_lead_max}."
            )
        if self.controller_velocity_gain < 0.0:
            raise ValueError(f"controller_velocity_gain must be non-negative, got {self.controller_velocity_gain}.")
        if self.controller_velocity_feedback_gain < 0.0:
            raise ValueError(
                "controller_velocity_feedback_gain must be non-negative, got "
                f"{self.controller_velocity_feedback_gain}."
            )
        if self.controller_max_velocity_step < 0.0:
            raise ValueError(
                f"controller_max_velocity_step must be non-negative, got {self.controller_max_velocity_step}."
            )
        if self.controller_nullspace_posture_gain < 0.0:
            raise ValueError(
                "controller_nullspace_posture_gain must be non-negative, got "
                f"{self.controller_nullspace_posture_gain}."
            )
        if self.controller_nullspace_posture_max_step < 0.0:
            raise ValueError(
                "controller_nullspace_posture_max_step must be non-negative, got "
                f"{self.controller_nullspace_posture_max_step}."
            )
        if self.controller_nullspace_posture_target is not None:
            if self.controller_nullspace_posture_target.shape != (7,):
                raise ValueError(
                    "controller_nullspace_posture_target must have shape (7,), got "
                    f"{self.controller_nullspace_posture_target.shape}."
                )
            if not np.isfinite(self.controller_nullspace_posture_target).all():
                raise ValueError(
                    "controller_nullspace_posture_target must be finite, got "
                    f"{self.controller_nullspace_posture_target}."
                )
        if self.controller_body_clearance_gain < 0.0:
            raise ValueError(
                "controller_body_clearance_gain must be non-negative, got "
                f"{self.controller_body_clearance_gain}."
            )
        if self.controller_body_clearance_margin < 0.0:
            raise ValueError(
                "controller_body_clearance_margin must be non-negative, got "
                f"{self.controller_body_clearance_margin}."
            )
        if self.controller_body_clearance_vertical_margin < 0.0:
            raise ValueError(
                "controller_body_clearance_vertical_margin must be non-negative, got "
                f"{self.controller_body_clearance_vertical_margin}."
            )
        if self.controller_body_clearance_max_step < 0.0:
            raise ValueError(
                "controller_body_clearance_max_step must be non-negative, got "
                f"{self.controller_body_clearance_max_step}."
            )
        if not np.isfinite(self.tracking_strike_plane_offset):
            raise ValueError(f"tracking_strike_plane_offset must be finite, got {self.tracking_strike_plane_offset}.")
        if not (self.target_offset_low[2] <= self.tracking_strike_plane_offset <= self.target_offset_high[2]):
            raise ValueError(
                "tracking_strike_plane_offset must stay within target_offset_low/high z bounds, got "
                f"{self.tracking_strike_plane_offset} with z bounds "
                f"[{self.target_offset_low[2]}, {self.target_offset_high[2]}]."
            )
        if self.contact_frame_velocity_target_gain < 0.0:
            raise ValueError(
                "contact_frame_velocity_target_gain must be non-negative, got "
                f"{self.contact_frame_velocity_target_gain}."
            )
        if self.contact_frame_velocity_target_max < 0.0:
            raise ValueError(
                "contact_frame_velocity_target_max must be non-negative, got "
                f"{self.contact_frame_velocity_target_max}."
            )
        if self.contact_frame_velocity_scale_action_limit < 0.0:
            raise ValueError(
                "contact_frame_velocity_scale_action_limit must be non-negative, got "
                f"{self.contact_frame_velocity_scale_action_limit}."
            )
        if self.contact_frame_outgoing_xy_action_limit < 0.0:
            raise ValueError(
                "contact_frame_outgoing_xy_action_limit must be non-negative, got "
                f"{self.contact_frame_outgoing_xy_action_limit}."
            )
        if self.contact_frame_racket_vz_action_limit < 0.0:
            raise ValueError(
                "contact_frame_racket_vz_action_limit must be non-negative, got "
                f"{self.contact_frame_racket_vz_action_limit}."
            )
        if self.contact_frame_racket_xy_action_limit < 0.0:
            raise ValueError(
                "contact_frame_racket_xy_action_limit must be non-negative, got "
                f"{self.contact_frame_racket_xy_action_limit}."
            )
        if self.contact_frame_tilt_scale_action_limit < 0.0:
            raise ValueError(
                "contact_frame_tilt_scale_action_limit must be non-negative, got "
                f"{self.contact_frame_tilt_scale_action_limit}."
            )
        if self.contact_frame_target_apex_z_action_limit < 0.0:
            raise ValueError(
                "contact_frame_target_apex_z_action_limit must be non-negative, got "
                f"{self.contact_frame_target_apex_z_action_limit}."
            )
        if self.contact_frame_strike_plane_z_action_limit < 0.0:
            raise ValueError(
                "contact_frame_strike_plane_z_action_limit must be non-negative, got "
                f"{self.contact_frame_strike_plane_z_action_limit}."
            )
        if self.contact_frame_tracking_xy_action_limit < 0.0:
            raise ValueError(
                "contact_frame_tracking_xy_action_limit must be non-negative, got "
                f"{self.contact_frame_tracking_xy_action_limit}."
            )
        if self.action_mode in _CONTACT_FRAME_VELOCITY_RESIDUAL_ACTION_MODES:
            if self.contact_frame_velocity_scale_action_limit <= 0.0:
                raise ValueError(
                    "contact_frame_velocity_scale_action_limit must be positive in velocity-residual contact-frame mode."
                )
            if self.contact_frame_outgoing_xy_action_limit <= 0.0:
                raise ValueError(
                    "contact_frame_outgoing_xy_action_limit must be positive in velocity-residual contact-frame mode."
                )
        if self.action_mode in _CONTACT_FRAME_TILT_SCALE_ACTION_MODES:
            if self.contact_frame_racket_vz_action_limit <= 0.0:
                raise ValueError("contact_frame_racket_vz_action_limit must be positive in tilt-scale contact-frame mode.")
            if self.contact_frame_tilt_scale_action_limit <= 0.0:
                raise ValueError("contact_frame_tilt_scale_action_limit must be positive in tilt-scale contact-frame mode.")
        if self.action_mode in _CONTACT_FRAME_LATERAL_VELOCITY_RESIDUAL_ACTION_MODES:
            if self.contact_frame_racket_xy_action_limit <= 0.0:
                raise ValueError("contact_frame_racket_xy_action_limit must be positive in lateral residual mode.")
        if self.action_mode in _CONTACT_FRAME_APEX_TIMING_RESIDUAL_ACTION_MODES:
            if self.contact_frame_target_apex_z_action_limit <= 0.0:
                raise ValueError(
                    "contact_frame_target_apex_z_action_limit must be positive in apex/timing residual mode."
                )
            if self.contact_frame_strike_plane_z_action_limit <= 0.0:
                raise ValueError(
                    "contact_frame_strike_plane_z_action_limit must be positive in apex/timing residual mode."
                )
        if self.action_mode in _CONTACT_FRAME_TRACKING_RESIDUAL_ACTION_MODES:
            if self.contact_frame_tracking_xy_action_limit <= 0.0:
                raise ValueError(
                    "contact_frame_tracking_xy_action_limit must be positive in tracking residual mode."
                )
        if self.contact_frame_intercept_velocity_gain < 0.0:
            raise ValueError(
                "contact_frame_intercept_velocity_gain must be non-negative, got "
                f"{self.contact_frame_intercept_velocity_gain}."
            )
        if self.contact_frame_intercept_velocity_max < 0.0:
            raise ValueError(
                "contact_frame_intercept_velocity_max must be non-negative, got "
                f"{self.contact_frame_intercept_velocity_max}."
            )
        if self.contact_frame_intercept_velocity_time_floor <= 0.0:
            raise ValueError(
                "contact_frame_intercept_velocity_time_floor must be positive, got "
                f"{self.contact_frame_intercept_velocity_time_floor}."
            )
        if self.contact_frame_planner_min_intercept_time < 0.0:
            raise ValueError(
                "contact_frame_planner_min_intercept_time must be non-negative, got "
                f"{self.contact_frame_planner_min_intercept_time}."
            )
        if self.contact_frame_planner_max_intercept_time <= 0.0:
            raise ValueError(
                "contact_frame_planner_max_intercept_time must be positive, got "
                f"{self.contact_frame_planner_max_intercept_time}."
            )
        if self.contact_frame_planner_min_intercept_time > self.contact_frame_planner_max_intercept_time:
            raise ValueError(
                "contact_frame_planner_min_intercept_time must be <= contact_frame_planner_max_intercept_time, got "
                f"{self.contact_frame_planner_min_intercept_time} > {self.contact_frame_planner_max_intercept_time}."
            )
        if not np.isfinite(self.contact_frame_planner_target_apex_z_offset):
            raise ValueError(
                "contact_frame_planner_target_apex_z_offset must be finite, got "
                f"{self.contact_frame_planner_target_apex_z_offset}."
            )
        if self.contact_frame_planner_contact_offset_ratio < 0.0:
            raise ValueError(
                "contact_frame_planner_contact_offset_ratio must be non-negative, got "
                f"{self.contact_frame_planner_contact_offset_ratio}."
            )
        if self.contact_frame_planner_contact_offset_max < 0.0:
            raise ValueError(
                "contact_frame_planner_contact_offset_max must be non-negative, got "
                f"{self.contact_frame_planner_contact_offset_max}."
            )
        if self.contact_frame_strike_hold_time < 0.0:
            raise ValueError(
                f"contact_frame_strike_hold_time must be non-negative, got {self.contact_frame_strike_hold_time}."
            )
        if not 0.0 <= self.contact_frame_strike_hold_min_readiness <= 1.0:
            raise ValueError(
                "contact_frame_strike_hold_min_readiness must be within [0, 1], got "
                f"{self.contact_frame_strike_hold_min_readiness}."
            )
        if self.contact_frame_followthrough_gain < 0.0:
            raise ValueError(
                "contact_frame_followthrough_gain must be non-negative, got "
                f"{self.contact_frame_followthrough_gain}."
            )
        if self.contact_frame_followthrough_time < 0.0:
            raise ValueError(
                "contact_frame_followthrough_time must be non-negative, got "
                f"{self.contact_frame_followthrough_time}."
            )
        if self.contact_frame_followthrough_max < 0.0:
            raise ValueError(
                "contact_frame_followthrough_max must be non-negative, got "
                f"{self.contact_frame_followthrough_max}."
            )
        if self.contact_frame_lateral_brake_gain < 0.0:
            raise ValueError(
                "contact_frame_lateral_brake_gain must be non-negative, got "
                f"{self.contact_frame_lateral_brake_gain}."
            )
        if self.contact_frame_lateral_brake_max < 0.0:
            raise ValueError(
                "contact_frame_lateral_brake_max must be non-negative, got "
                f"{self.contact_frame_lateral_brake_max}."
            )
        if self.contact_frame_lateral_brake_radius <= 0.0:
            raise ValueError(
                "contact_frame_lateral_brake_radius must be positive, got "
                f"{self.contact_frame_lateral_brake_radius}."
            )
        if self.contact_racket_outward_velocity_penalty_weight < 0.0:
            raise ValueError(
                "contact_racket_outward_velocity_penalty_weight must be non-negative, got "
                f"{self.contact_racket_outward_velocity_penalty_weight}."
            )
        if self.contact_racket_outward_velocity_tolerance < 0.0:
            raise ValueError(
                "contact_racket_outward_velocity_tolerance must be non-negative, got "
                f"{self.contact_racket_outward_velocity_tolerance}."
            )
        if self.contact_frame_trajectory_tilt_gain < 0.0:
            raise ValueError(
                "contact_frame_trajectory_tilt_gain must be non-negative, got "
                f"{self.contact_frame_trajectory_tilt_gain}."
            )
        if self.contact_frame_trajectory_tilt_limit is not None:
            if self.contact_frame_trajectory_tilt_limit.shape != (2,):
                raise ValueError(
                    "contact_frame_trajectory_tilt_limit must have shape (2,), got "
                    f"{self.contact_frame_trajectory_tilt_limit.shape}."
                )
            if np.any(self.contact_frame_trajectory_tilt_limit < 0.0):
                raise ValueError(
                    "contact_frame_trajectory_tilt_limit must be non-negative, got "
                    f"{self.contact_frame_trajectory_tilt_limit}."
                )
        if self.contact_frame_trajectory_tilt_deadband < 0.0:
            raise ValueError(
                "contact_frame_trajectory_tilt_deadband must be non-negative, got "
                f"{self.contact_frame_trajectory_tilt_deadband}."
            )
        if self.contact_frame_tilt_ramp_time <= 0.0:
            raise ValueError(
                "contact_frame_tilt_ramp_time must be positive, got "
                f"{self.contact_frame_tilt_ramp_time}."
            )
        if self.contact_frame_base_tilt_residual is not None:
            if self.contact_frame_base_tilt_residual.shape != (2,):
                raise ValueError(
                    "contact_frame_base_tilt_residual must have shape (2,), got "
                    f"{self.contact_frame_base_tilt_residual.shape}."
                )
            if np.any(np.abs(self.contact_frame_base_tilt_residual) > self.target_tilt_limit + 1.0e-9):
                raise ValueError(
                    "contact_frame_base_tilt_residual must stay within target_tilt_limit, got "
                    f"{self.contact_frame_base_tilt_residual} with target_tilt_limit={self.target_tilt_limit}."
                )
        if self.contact_frame_centering_tilt_limit is not None:
            if self.contact_frame_centering_tilt_limit.shape != (2,):
                raise ValueError(
                    "contact_frame_centering_tilt_limit must have shape (2,), got "
                    f"{self.contact_frame_centering_tilt_limit.shape}."
                )
            if np.any(self.contact_frame_centering_tilt_limit < 0.0):
                raise ValueError(
                    "contact_frame_centering_tilt_limit must be non-negative, got "
                    f"{self.contact_frame_centering_tilt_limit}."
                )
            if np.any(self.contact_frame_centering_tilt_limit > self.target_tilt_limit + 1.0e-9):
                raise ValueError(
                    "contact_frame_centering_tilt_limit must stay within target_tilt_limit, got "
                    f"{self.contact_frame_centering_tilt_limit} with target_tilt_limit={self.target_tilt_limit}."
                )
        if self.contact_frame_centering_tilt_radius is not None and self.contact_frame_centering_tilt_radius <= 0.0:
            raise ValueError(
                "contact_frame_centering_tilt_radius must be positive when provided, got "
                f"{self.contact_frame_centering_tilt_radius}."
            )
        if self.contact_frame_centering_tilt_deadband < 0.0:
            raise ValueError(
                "contact_frame_centering_tilt_deadband must be non-negative, got "
                f"{self.contact_frame_centering_tilt_deadband}."
            )
        if self.contact_frame_action_penalty_weight < 0.0:
            raise ValueError(
                "contact_frame_action_penalty_weight must be non-negative, got "
                f"{self.contact_frame_action_penalty_weight}."
            )
        if self.next_intercept_xy_error_penalty_weight < 0.0:
            raise ValueError(
                "next_intercept_xy_error_penalty_weight must be non-negative, got "
                f"{self.next_intercept_xy_error_penalty_weight}."
            )
        if self.post_contact_lateral_velocity_penalty_weight < 0.0:
            raise ValueError(
                "post_contact_lateral_velocity_penalty_weight must be non-negative, got "
                f"{self.post_contact_lateral_velocity_penalty_weight}."
            )
        if self.contact_xy_error_penalty_weight < 0.0:
            raise ValueError(
                f"contact_xy_error_penalty_weight must be non-negative, got {self.contact_xy_error_penalty_weight}."
            )
        if self.contact_racket_lateral_velocity_penalty_weight < 0.0:
            raise ValueError(
                "contact_racket_lateral_velocity_penalty_weight must be non-negative, got "
                f"{self.contact_racket_lateral_velocity_penalty_weight}."
            )
        if self.contact_racket_lateral_velocity_tolerance <= 0.0:
            raise ValueError(
                "contact_racket_lateral_velocity_tolerance must be positive, got "
                f"{self.contact_racket_lateral_velocity_tolerance}."
            )
        if (
            self.max_contact_racket_lateral_speed_for_success is not None
            and self.max_contact_racket_lateral_speed_for_success <= 0.0
        ):
            raise ValueError(
                "max_contact_racket_lateral_speed_for_success must be positive when provided, got "
                f"{self.max_contact_racket_lateral_speed_for_success}."
            )
        if self.nonuseful_contact_penalty_weight < 0.0:
            raise ValueError(
                f"nonuseful_contact_penalty_weight must be non-negative, got {self.nonuseful_contact_penalty_weight}."
            )
        if self.contact_apex_under_target_penalty_weight < 0.0:
            raise ValueError(
                "contact_apex_under_target_penalty_weight must be non-negative, got "
                f"{self.contact_apex_under_target_penalty_weight}."
            )
        if self.contact_apex_progress_reward_weight < 0.0:
            raise ValueError(
                "contact_apex_progress_reward_weight must be non-negative, got "
                f"{self.contact_apex_progress_reward_weight}."
            )
        if self.contact_apex_recovery_progress_reward_weight < 0.0:
            raise ValueError(
                "contact_apex_recovery_progress_reward_weight must be non-negative, got "
                f"{self.contact_apex_recovery_progress_reward_weight}."
            )
        if self.contact_apex_potential_reward_weight < 0.0:
            raise ValueError(
                "contact_apex_potential_reward_weight must be non-negative, got "
                f"{self.contact_apex_potential_reward_weight}."
            )
        if not np.isfinite(self.contact_apex_potential_gamma):
            raise ValueError(f"contact_apex_potential_gamma must be finite, got {self.contact_apex_potential_gamma}.")
        if self.contact_apex_potential_cap <= 0.0:
            raise ValueError(f"contact_apex_potential_cap must be positive, got {self.contact_apex_potential_cap}.")
        if self.contact_apex_progress_min_easy_next_ball_score is not None:
            if not np.isfinite(self.contact_apex_progress_min_easy_next_ball_score):
                raise ValueError(
                    "contact_apex_progress_min_easy_next_ball_score must be finite when provided, got "
                    f"{self.contact_apex_progress_min_easy_next_ball_score}."
                )
            if self.contact_apex_progress_min_easy_next_ball_score < 0.0:
                raise ValueError(
                    "contact_apex_progress_min_easy_next_ball_score must be non-negative, got "
                    f"{self.contact_apex_progress_min_easy_next_ball_score}."
                )
        if self.contact_lateral_stability_reward_weight < 0.0:
            raise ValueError(
                "contact_lateral_stability_reward_weight must be non-negative, got "
                f"{self.contact_lateral_stability_reward_weight}."
            )
        if self.contact_lateral_stability_speed_tolerance <= 0.0:
            raise ValueError(
                "contact_lateral_stability_speed_tolerance must be positive, got "
                f"{self.contact_lateral_stability_speed_tolerance}."
            )
        if self.contact_lateral_stability_xy_tolerance is not None:
            if not np.isfinite(self.contact_lateral_stability_xy_tolerance):
                raise ValueError(
                    "contact_lateral_stability_xy_tolerance must be finite when provided, got "
                    f"{self.contact_lateral_stability_xy_tolerance}."
                )
            if self.contact_lateral_stability_xy_tolerance <= 0.0:
                raise ValueError(
                    "contact_lateral_stability_xy_tolerance must be positive when provided, got "
                    f"{self.contact_lateral_stability_xy_tolerance}."
                )
        if self.contact_lateral_stability_min_apex_ratio is not None:
            if not np.isfinite(self.contact_lateral_stability_min_apex_ratio):
                raise ValueError(
                    "contact_lateral_stability_min_apex_ratio must be finite when provided, got "
                    f"{self.contact_lateral_stability_min_apex_ratio}."
                )
            if self.contact_lateral_stability_min_apex_ratio < 0.0:
                raise ValueError(
                    "contact_lateral_stability_min_apex_ratio must be non-negative, got "
                    f"{self.contact_lateral_stability_min_apex_ratio}."
                )
        if self.stable_contact_reward_weight < 0.0:
            raise ValueError(
                f"stable_contact_reward_weight must be non-negative, got {self.stable_contact_reward_weight}."
            )
        if self.stable_contact_min_apex_ratio is not None:
            if not np.isfinite(self.stable_contact_min_apex_ratio):
                raise ValueError(
                    f"stable_contact_min_apex_ratio must be finite when provided, got {self.stable_contact_min_apex_ratio}."
                )
            if self.stable_contact_min_apex_ratio < 0.0:
                raise ValueError(
                    f"stable_contact_min_apex_ratio must be non-negative, got {self.stable_contact_min_apex_ratio}."
                )
        if self.stable_cycle_reward_weight < 0.0:
            raise ValueError(
                f"stable_cycle_reward_weight must be non-negative, got {self.stable_cycle_reward_weight}."
            )
        if self.stable_cycle_reward_cap < 1:
            raise ValueError(f"stable_cycle_reward_cap must be positive, got {self.stable_cycle_reward_cap}.")
        if self.stable_cycle_min_easy_next_ball_score is not None and not np.isfinite(
            self.stable_cycle_min_easy_next_ball_score
        ):
            raise ValueError(
                "stable_cycle_min_easy_next_ball_score must be finite when provided, got "
                f"{self.stable_cycle_min_easy_next_ball_score}."
            )
        if self.contact_frame_low_apex_recovery_lift_gain < 0.0:
            raise ValueError(
                "contact_frame_low_apex_recovery_lift_gain must be non-negative, got "
                f"{self.contact_frame_low_apex_recovery_lift_gain}."
            )
        if self.contact_frame_low_apex_recovery_lift_max < 0.0:
            raise ValueError(
                "contact_frame_low_apex_recovery_lift_max must be non-negative, got "
                f"{self.contact_frame_low_apex_recovery_lift_max}."
            )
        if self.contact_frame_low_apex_recovery_velocity_gain < 0.0:
            raise ValueError(
                "contact_frame_low_apex_recovery_velocity_gain must be non-negative, got "
                f"{self.contact_frame_low_apex_recovery_velocity_gain}."
            )
        if self.contact_frame_low_apex_recovery_velocity_max < 0.0:
            raise ValueError(
                "contact_frame_low_apex_recovery_velocity_max must be non-negative, got "
                f"{self.contact_frame_low_apex_recovery_velocity_max}."
            )
        if self.strike_tilt_ramp_xy_tolerance < 0.0:
            raise ValueError(
                "strike_tilt_ramp_xy_tolerance must be non-negative, got "
                f"{self.strike_tilt_ramp_xy_tolerance}."
            )
        if self.target_pitch_range is not None:
            if self.target_pitch_range.shape != (2,):
                raise ValueError(
                    f"target_pitch_range must have shape (2,), got {self.target_pitch_range.shape}."
                )
            if self.target_pitch_range[0] > self.target_pitch_range[1]:
                raise ValueError(
                    f"target_pitch_range must be ordered as (low, high), got {self.target_pitch_range}."
                )
            if self.target_pitch_range[0] < -self.target_tilt_limit[0] or self.target_pitch_range[1] > self.target_tilt_limit[0]:
                raise ValueError(
                    "target_pitch_range must stay within +/- target_tilt_limit[0], got "
                    f"{self.target_pitch_range} with target_tilt_limit={self.target_tilt_limit}."
                )
            if self.strike_tilt_ramp_pitch is not None and not (
                self.target_pitch_range[0] <= self.strike_tilt_ramp_pitch <= self.target_pitch_range[1]
            ):
                raise ValueError(
                    "strike_tilt_ramp_pitch must stay within target_pitch_range, got "
                    f"pitch={self.strike_tilt_ramp_pitch} with target_pitch_range={self.target_pitch_range}."
                )
        if self.initial_target_tilt is not None:
            if self.initial_target_tilt.shape != (2,):
                raise ValueError(
                    f"initial_target_tilt must have shape (2,), got {self.initial_target_tilt.shape}."
                )
            if np.any(np.abs(self.initial_target_tilt) > self.target_tilt_limit + 1.0e-9):
                raise ValueError(
                    "initial_target_tilt must stay within target_tilt_limit, got "
                    f"{self.initial_target_tilt} with target_tilt_limit={self.target_tilt_limit}."
                )
            if self.target_pitch_range is not None and not (
                self.target_pitch_range[0] <= self.initial_target_tilt[0] <= self.target_pitch_range[1]
            ):
                raise ValueError(
                    "initial_target_tilt pitch must stay within target_pitch_range, got "
                    f"pitch={self.initial_target_tilt[0]} with target_pitch_range={self.target_pitch_range}."
                )
            if self.strike_tilt_ramp_pitch is not None:
                raise ValueError(
                    "initial_target_tilt cannot be combined with strike_tilt_ramp_pitch because "
                    "the ramped strike experiment assumes neutral tilt outside the strike window."
                )
        self.controller = RacketCartesianController(
            self.sim,
            position_gain=self.controller_position_gain,
            orientation_gain=self.controller_orientation_gain,
            max_position_step=self.controller_max_position_step,
            max_orientation_step=self.controller_max_orientation_step,
            velocity_gain=self.controller_velocity_gain,
            velocity_feedback_gain=self.controller_velocity_feedback_gain,
            max_velocity_step=self.controller_max_velocity_step,
            target_offset_low=self.target_offset_low,
            target_offset_high=self.target_offset_high,
            target_tilt_limit=self.target_tilt_limit,
            nullspace_posture_gain=self.controller_nullspace_posture_gain,
            nullspace_posture_max_step=self.controller_nullspace_posture_max_step,
            nullspace_posture_target=self.controller_nullspace_posture_target,
            body_clearance_gain=self.controller_body_clearance_gain,
            body_clearance_margin=self.controller_body_clearance_margin,
            body_clearance_vertical_margin=self.controller_body_clearance_vertical_margin,
            body_clearance_max_step=self.controller_body_clearance_max_step,
            body_clearance_body_names=self.controller_body_clearance_body_names,
        )
        position_action_limit = np.array(
            [self.lateral_action_limit, self.lateral_action_limit, self.vertical_action_limit],
            dtype=float,
        )
        if self.action_mode in _TILT_ACTION_MODES:
            tilt_action_limit = np.full(2, self.tilt_action_limit, dtype=float)
            self.action_high = np.concatenate([position_action_limit, tilt_action_limit])
            if self.action_mode == "position_strike_tilt_lift":
                self.action_high = np.concatenate(
                    [self.action_high, np.array([self.followup_lift_action_limit], dtype=float)]
                )
            elif self.action_mode in _CONTACT_FRAME_VELOCITY_RESIDUAL_ACTION_MODES:
                velocity_residual_limit = np.array(
                    [
                        self.contact_frame_velocity_scale_action_limit,
                        self.contact_frame_outgoing_xy_action_limit,
                        self.contact_frame_outgoing_xy_action_limit,
                    ],
                    dtype=float,
                )
                self.action_high = np.concatenate([self.action_high, velocity_residual_limit])
                if self.action_mode in _CONTACT_FRAME_TILT_SCALE_ACTION_MODES:
                    velocity_tilt_residual_limit = np.array(
                        [
                            self.contact_frame_racket_vz_action_limit,
                            self.contact_frame_tilt_scale_action_limit,
                            self.contact_frame_tilt_scale_action_limit,
                        ],
                        dtype=float,
                    )
                    self.action_high = np.concatenate([self.action_high, velocity_tilt_residual_limit])
                    if self.action_mode in _CONTACT_FRAME_LATERAL_VELOCITY_RESIDUAL_ACTION_MODES:
                        lateral_velocity_residual_limit = np.array(
                            [
                                self.contact_frame_racket_xy_action_limit,
                                self.contact_frame_racket_xy_action_limit,
                            ],
                            dtype=float,
                        )
                        self.action_high = np.concatenate([self.action_high, lateral_velocity_residual_limit])
                        if self.action_mode in _CONTACT_FRAME_APEX_TIMING_RESIDUAL_ACTION_MODES:
                            apex_timing_residual_limit = np.array(
                                [
                                    self.contact_frame_target_apex_z_action_limit,
                                    self.contact_frame_strike_plane_z_action_limit,
                                ],
                                dtype=float,
                            )
                            self.action_high = np.concatenate([self.action_high, apex_timing_residual_limit])
                            if self.action_mode in _CONTACT_FRAME_TRACKING_RESIDUAL_ACTION_MODES:
                                tracking_residual_limit = np.array(
                                    [
                                        self.contact_frame_tracking_xy_action_limit,
                                        self.contact_frame_tracking_xy_action_limit,
                                    ],
                                    dtype=float,
                                )
                                self.action_high = np.concatenate([self.action_high, tracking_residual_limit])
        else:
            self.action_high = position_action_limit
        self.action_low = -self.action_high.copy()
        self.action_size = int(self.action_high.shape[0])
        self._observation_components, self._observation_slices, self.observation_size = _build_observation_layout(
            self.action_mode,
            self.include_velocity_domain_observation,
            self.include_task_phase_observation,
            self.include_contact_context_observation,
            self.include_next_intercept_observation,
            self.include_desired_outgoing_velocity_observation,
        )
        self._rng = np.random.default_rng()
        self._spawn_ball_height_above_racket = self.ball_height
        self.step_count = 0
        self.contact_count = 0
        self.successful_bounce_count = 0
        self.stable_cycle_count = 0
        self._consecutive_stable_cycle_count = 0
        self._consecutive_low_apex_contact_count = 0
        self._last_projected_contact_apex_height: float | None = None
        self._last_contact_apex_shortfall = 0.0
        self._last_contact_step: int | None = None
        self._contact_active_previous_step = False
        self._previous_action = np.zeros(self.action_size, dtype=float)
        self._contact_frame_velocity_residual_action = np.zeros(3, dtype=float)
        self._contact_frame_racket_vz_residual_action = 0.0
        self._contact_frame_tilt_scale_residual_action = np.zeros(2, dtype=float)
        self._contact_frame_racket_xy_residual_action = np.zeros(2, dtype=float)
        self._contact_frame_target_apex_z_residual_action = 0.0
        self._contact_frame_strike_plane_z_residual_action = 0.0
        self._contact_frame_tracking_xy_residual_action = np.zeros(2, dtype=float)
        self._reset_contact_frame_plan()

    @property
    def observation_slices(self) -> dict[str, slice]:
        return dict(self._observation_slices)

    def seed(self, seed: int | None = None) -> int | None:
        self._rng = np.random.default_rng(seed)
        return seed

    def observation(self) -> np.ndarray:
        ball_relative_position = self.sim.ball_position - self.sim.racket_position
        predicted_intercept_time = self._predicted_intercept_time()
        predicted_intercept_relative_xy = (
            self.sim.ball_position[:2] + predicted_intercept_time * self.sim.ball_velocity[:2] - self.sim.racket_position[:2]
        )
        next_intercept_metrics = self._next_intercept_metrics()
        desired_outgoing_velocity, _, _ = self._desired_outgoing_velocity()
        observation_parts: list[np.ndarray] = [
            self.sim.joint_positions,
            self.sim.joint_velocities,
            self.sim.racket_position,
            self.sim.racket_velocity,
            self.controller.target_position,
            self.sim.ball_position,
            self.sim.ball_velocity,
            ball_relative_position,
            predicted_intercept_relative_xy,
            np.array([predicted_intercept_time], dtype=float),
        ]
        if self.include_task_phase_observation:
            observation_parts.append(self._phase_one_hot())
        if self.include_contact_context_observation:
            time_since_contact = self._time_since_contact()
            observation_parts.extend(
                [
                    np.array([0.0 if time_since_contact is None else time_since_contact], dtype=float),
                    np.array([min(self.successful_bounce_count, 3)], dtype=float),
                ]
            )
        if self.include_next_intercept_observation:
            relative_xy = next_intercept_metrics["relative_xy"]
            observation_parts.extend(
                [
                    relative_xy,
                    np.array([next_intercept_metrics["time"]], dtype=float),
                    np.array([float(next_intercept_metrics["reachable"])] , dtype=float),
                    np.array([next_intercept_metrics["recovery_distance"]], dtype=float),
                    np.array([next_intercept_metrics["recovery_readiness"]], dtype=float),
                ]
            )
        if self.include_desired_outgoing_velocity_observation:
            observation_parts.append(desired_outgoing_velocity)
        if self.include_velocity_domain_observation:
            observation_parts.extend(
                [
                    self.sim.ball_velocity - self.sim.racket_velocity,
                    self.sim.racket_face_normal,
                ]
            )
        if self.action_mode in _TILT_ACTION_MODES:
            if not self.include_velocity_domain_observation:
                observation_parts.append(self.sim.racket_face_normal)
            observation_parts.append(self.controller.target_tilt)
        return np.concatenate(observation_parts)

    def reset(
        self,
        ball_height: float | None = None,
        ball_velocity: Sequence[float] | None = None,
        ball_angular_velocity: Sequence[float] | None = None,
        ball_xy_offset: Sequence[float] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        spawn_height = self._sample_reset_ball_height() if ball_height is None else float(ball_height)
        spawn_velocity = self._sample_reset_velocity() if ball_velocity is None else np.asarray(ball_velocity, dtype=float)
        spawn_angular_velocity = (
            self._sample_reset_ball_angular_velocity()
            if ball_angular_velocity is None
            else np.asarray(ball_angular_velocity, dtype=float)
        )
        spawn_xy_offset = self._sample_reset_xy_offset() if ball_xy_offset is None else np.asarray(ball_xy_offset, dtype=float)
        self.sim.reset(
            ball_height=spawn_height,
            ball_velocity=spawn_velocity,
            ball_angular_velocity=spawn_angular_velocity,
            ball_xy_offset=spawn_xy_offset,
        )
        self.controller.reset()
        if self.action_mode in ("position_tilt", *_STRIKE_CONTRACT_ACTION_MODES) and self.initial_target_tilt is not None:
            self.controller.set_target_tilt(self.initial_target_tilt)
        self.step_count = 0
        self.contact_count = 0
        self.successful_bounce_count = 0
        self.stable_cycle_count = 0
        self._consecutive_stable_cycle_count = 0
        self._consecutive_low_apex_contact_count = 0
        self._last_projected_contact_apex_height = None
        self._last_contact_apex_shortfall = 0.0
        self._last_contact_step = None
        self._contact_active_previous_step = False
        self._previous_action[:] = 0.0
        self._contact_frame_velocity_residual_action[:] = 0.0
        self._contact_frame_racket_vz_residual_action = 0.0
        self._contact_frame_tilt_scale_residual_action[:] = 0.0
        self._contact_frame_racket_xy_residual_action[:] = 0.0
        self._contact_frame_target_apex_z_residual_action = 0.0
        self._contact_frame_strike_plane_z_residual_action = 0.0
        self._contact_frame_tracking_xy_residual_action[:] = 0.0
        self._reset_contact_frame_plan()
        self._spawn_ball_height_above_racket = float(self.sim.ball_position[2] - self.sim.racket_position[2])
        info: dict[str, object] = {
            "contact_count": self.contact_count,
            "successful_bounce_count": self.successful_bounce_count,
            "stable_cycle_count": self.stable_cycle_count,
            "consecutive_stable_cycle_count": self._consecutive_stable_cycle_count,
            "last_projected_contact_apex_height_above_racket": self._last_projected_contact_apex_height,
            "last_contact_apex_shortfall": self._last_contact_apex_shortfall,
            "step_count": self.step_count,
            "target_position": self.controller.target_position,
            "target_tilt": self.controller.target_tilt,
            "target_velocity": self.controller.target_velocity,
            "ball_height_above_racket": self._ball_height_above_racket(),
            "spawn_ball_height_above_racket": self._spawn_ball_height_above_racket,
            "spawn_ball_position": self.sim.ball_position.copy(),
            "spawn_ball_velocity": self.sim.ball_velocity.copy(),
            "spawn_ball_angular_velocity": self.sim.ball_angular_velocity.copy(),
            "spawn_ball_xy_offset": spawn_xy_offset.copy(),
        }
        return self.observation(), info

    def step(self, action: Sequence[float]) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        action_array = np.asarray(action, dtype=float)
        if action_array.shape != (self.action_size,):
            raise ValueError(f"EE delta action must have shape ({self.action_size},), got {action_array.shape}.")
        applied_action = np.clip(action_array, self.action_low, self.action_high)
        followup_lift_residual = 0.0
        if self.action_mode == "position_strike_tilt_lift":
            followup_lift_residual = float(applied_action[5])
        self._contact_frame_velocity_residual_action[:] = 0.0
        self._contact_frame_racket_vz_residual_action = 0.0
        self._contact_frame_tilt_scale_residual_action[:] = 0.0
        self._contact_frame_racket_xy_residual_action[:] = 0.0
        self._contact_frame_target_apex_z_residual_action = 0.0
        self._contact_frame_strike_plane_z_residual_action = 0.0
        self._contact_frame_tracking_xy_residual_action[:] = 0.0
        if self.action_mode in _CONTACT_FRAME_VELOCITY_RESIDUAL_ACTION_MODES:
            self._contact_frame_velocity_residual_action = applied_action[5:8].copy()
        if self.action_mode in _CONTACT_FRAME_TILT_SCALE_ACTION_MODES:
            self._contact_frame_racket_vz_residual_action = float(applied_action[8])
            self._contact_frame_tilt_scale_residual_action = applied_action[9:11].copy()
        if self.action_mode in _CONTACT_FRAME_LATERAL_VELOCITY_RESIDUAL_ACTION_MODES:
            self._contact_frame_racket_xy_residual_action = applied_action[11:13].copy()
        if self.action_mode in _CONTACT_FRAME_APEX_TIMING_RESIDUAL_ACTION_MODES:
            self._contact_frame_target_apex_z_residual_action = float(applied_action[13])
            self._contact_frame_strike_plane_z_residual_action = float(applied_action[14])
        if self.action_mode in _CONTACT_FRAME_TRACKING_RESIDUAL_ACTION_MODES:
            self._contact_frame_tracking_xy_residual_action = applied_action[15:17].copy()
        if self._contact_frame_action_mode():
            self._update_contact_frame_plan()
            requested_target_position = self._contact_frame_action_target_position(applied_action[:3])
        elif self.action_mode in ("position_strike", "position_strike_tilt", "position_strike_tilt_lift"):
            requested_target_position = self._strike_action_target_position(
                applied_action[:3],
                followup_lift_residual=followup_lift_residual,
            )
        else:
            self.controller.add_target_offset(applied_action[:3])
            requested_target_position = self.controller.target_position
        if self.action_mode == "position_tilt":
            next_target_tilt = self._constrained_target_tilt(self.controller.target_tilt + applied_action[3:])
            self.controller.set_target_tilt(next_target_tilt)
        elif self.action_mode == "position_strike":
            self.controller.set_target_tilt(self._position_strike_target_tilt())
        elif self.action_mode == "position_strike_tilt":
            next_target_tilt = self._constrained_target_tilt(self._position_strike_target_tilt() + applied_action[3:])
            self.controller.set_target_tilt(next_target_tilt)
        elif self.action_mode == "position_strike_tilt_lift":
            next_target_tilt = self._constrained_target_tilt(self._position_strike_target_tilt() + applied_action[3:5])
            self.controller.set_target_tilt(next_target_tilt)
        elif self._contact_frame_action_mode():
            next_target_tilt = self._constrained_target_tilt(
                self._position_strike_target_tilt()
                + self._contact_frame_base_strike_tilt()
                + applied_action[3:5]
            )
            self.controller.set_target_tilt(next_target_tilt)
        safe_target_position = self._guarded_target_position(requested_target_position)
        contact_frame_intercept_velocity_target = self._contact_frame_intercept_velocity_target(safe_target_position)
        contact_frame_lateral_brake_velocity = self._contact_frame_lateral_brake_velocity(safe_target_position)
        target_velocity = self._contact_frame_velocity_target(
            safe_target_position,
            lateral_brake_velocity=contact_frame_lateral_brake_velocity,
        )
        self.controller.set_target_velocity(target_velocity if np.any(target_velocity) else None)
        self.controller.set_target_position(safe_target_position)
        self.controller.set_body_clearance_reference(
            self.sim.ball_position,
            active=self._controller_body_clearance_active(),
        )
        joint_targets = self.controller.compute_joint_targets()
        contact_trace = self.sim.step_with_contact_trace(joint_targets=joint_targets, n_substeps=self.sim.n_substeps)
        oracle_info = self._apply_contact_oracle(contact_trace)
        self.step_count += 1

        failure_reason = self._failure_reason()
        robot_body_contact_name = self.sim.ball_robot_body_contact()
        contact_active = bool(contact_trace["contact_observed"] or self.sim.has_contact("ball_geom", "racket_head"))
        contact_event = contact_active and not self._contact_active_previous_step
        if contact_event:
            self.contact_count += 1
            self._last_contact_step = self.step_count
        success_reason = self._success_reason(failure_reason, contact_trace, contact_event)
        if success_reason is not None:
            self.successful_bounce_count += 1

        outgoing_trajectory_metrics = self._contact_outgoing_trajectory_metrics(contact_trace)
        low_apex_contact_observed = self._is_low_apex_contact(
            contact_trace,
            outgoing_trajectory_metrics,
            contact_event=contact_event,
            success_reason=success_reason,
        )
        if success_reason is not None:
            self._consecutive_low_apex_contact_count = 0
        elif low_apex_contact_observed:
            self._consecutive_low_apex_contact_count += 1
        elif contact_event:
            self._consecutive_low_apex_contact_count = 0
        low_apex_contact_failure = (
            low_apex_contact_observed
            and self._consecutive_low_apex_contact_count > self.low_apex_contact_grace_count
        )
        if failure_reason is None and low_apex_contact_failure:
            failure_reason = "low_apex_contact"
        if failure_reason is None and self.terminate_on_nonuseful_contact and contact_event and success_reason is None:
            failure_reason = "nonuseful_contact"
        next_intercept_metrics = self._next_intercept_metrics()
        stable_cycle_observed = self._stable_cycle_observed(
            success_reason=success_reason,
            contact_event=contact_event,
            contact_trace=contact_trace,
            next_intercept_metrics=next_intercept_metrics,
        )
        self._update_stable_cycle_state(
            contact_event=contact_event,
            stable_cycle_observed=stable_cycle_observed,
        )
        phase_name = self._phase_name()

        reward_terms = self._reward_terms(
            failure_reason,
            success_reason,
            contact_event,
            contact_active,
            applied_action,
            contact_trace,
            outgoing_trajectory_metrics,
            stable_cycle_observed=stable_cycle_observed,
            consecutive_stable_cycle_count=self._consecutive_stable_cycle_count,
        )
        reward = float(sum(reward_terms.values()))
        terminated = failure_reason is not None
        truncated = (
            (not terminated)
            and self.max_episode_steps is not None
            and self.step_count >= self.max_episode_steps
        )
        episode_success_reason = None
        if truncated and self.successful_bounce_count > 0:
            episode_success_reason = "keepup_time_limit"
        info: dict[str, object] = {
            "failure_reason": failure_reason,
            "robot_body_contact_name": robot_body_contact_name,
            "success_reason": success_reason,
            "low_apex_contact_observed": low_apex_contact_observed,
            "low_apex_contact_failure": low_apex_contact_failure,
            "low_apex_contact_height_threshold": self._low_apex_contact_height_threshold(),
            "consecutive_low_apex_contact_count": self._consecutive_low_apex_contact_count,
            "stable_cycle_observed": stable_cycle_observed,
            "stable_cycle_count": self.stable_cycle_count,
            "consecutive_stable_cycle_count": self._consecutive_stable_cycle_count,
            "stable_cycle_min_easy_next_ball_score": self.stable_cycle_min_easy_next_ball_score,
            "contact_apex_progress_easy_next_ball_gate": self._contact_apex_progress_easy_next_ball_gate(
                next_intercept_metrics
            ),
            "last_projected_contact_apex_height_above_racket": self._last_projected_contact_apex_height,
            "last_contact_apex_shortfall": self._last_contact_apex_shortfall,
            "contact_frame_low_apex_recovery_lift": self._contact_frame_low_apex_recovery_lift(),
            "contact_frame_low_apex_recovery_velocity": self._contact_frame_low_apex_recovery_velocity(),
            "episode_success_reason": episode_success_reason,
            "reward_terms": reward_terms,
            "contact_event_during_step": contact_event,
            "contact_observed_during_step": bool(contact_trace["contact_observed"]),
            "contact_count": self.contact_count,
            "successful_bounce_count": self.successful_bounce_count,
            "step_count": self.step_count,
            "phase_name": phase_name,
            "phase_one_hot": self._phase_one_hot(),
            "time_since_contact": self._time_since_contact(),
            "applied_action": applied_action.copy(),
            "applied_action_norm": float(np.linalg.norm(applied_action)),
            "applied_action_normalized_norm": self._normalized_action_norm(applied_action),
            "applied_position_action_norm": float(np.linalg.norm(applied_action[:3])),
            "applied_tilt_action_norm": (
                float(np.linalg.norm(applied_action[3:5]))
                if self.action_mode in _TILT_SLICE_3_TO_5_ACTION_MODES
                else (
                    float(np.linalg.norm(applied_action[3:]))
                    if self.action_mode in ("position_tilt", "position_strike_tilt")
                    else 0.0
                )
            ),
            "controller_anchor_position": self._controller_anchor_position(),
            "keepup_target_xy": self._keepup_target_xy(),
            "target_position": self.controller.target_position,
            "ball_height_above_racket": self._ball_height_above_racket(),
            "target_ball_height_above_racket": self._target_ball_height_above_racket(),
            "xy_alignment_error": self._xy_alignment_error(),
            "predicted_intercept_xy_error": self._tracking_alignment_error(),
            "predicted_intercept_time": self._predicted_intercept_time(),
            "tracking_strike_plane_offset": self._tracking_strike_plane_offset(),
            "strike_contact_target_x": float(self._strike_contact_target_xy()[0]),
            "strike_contact_target_y": float(self._strike_contact_target_xy()[1]),
            "followup_strike_contract_active": self._followup_strike_contract_active(),
            "next_intercept_time": next_intercept_metrics["info_time"],
            "next_intercept_x": next_intercept_metrics["info_x"],
            "next_intercept_y": next_intercept_metrics["info_y"],
            "next_intercept_xy_error": next_intercept_metrics["info_xy_error"],
            "next_intercept_reachable": next_intercept_metrics["info_reachable"],
            "next_intercept_vertical_speed": next_intercept_metrics["info_vertical_speed"],
            "next_intercept_speed_norm": next_intercept_metrics["info_speed_norm"],
            "next_intercept_recovery_distance": next_intercept_metrics["info_recovery_distance"],
            "next_intercept_recovery_readiness": next_intercept_metrics["info_recovery_readiness"],
            "easy_next_ball_score": next_intercept_metrics["info_easy_next_ball_score"],
            "next_intercept_success_radius": self.next_intercept_success_radius,
            "easy_next_ball_xy_radius": self.easy_next_ball_xy_radius,
            "strike_lift_feedforward": self._strike_lift_feedforward(),
            "contact_frame_apex_lift": self._contact_frame_apex_lift(),
            "contact_frame_velocity_lead": self._contact_frame_velocity_lead(),
            "contact_frame_velocity_residual_action": self._contact_frame_velocity_residual_action.copy(),
            "contact_frame_vz_scale_action": float(self._contact_frame_velocity_residual_action[0]),
            "contact_frame_vz_scale": self._contact_frame_velocity_residual_scale(),
            "contact_frame_outgoing_x_residual_action": float(self._contact_frame_velocity_residual_action[1]),
            "contact_frame_outgoing_y_residual_action": float(self._contact_frame_velocity_residual_action[2]),
            "contact_frame_racket_vz_residual_action": self._contact_frame_racket_vz_residual(),
            "contact_frame_racket_xy_residual_action": self._contact_frame_racket_xy_residual().copy(),
            "contact_frame_tracking_xy_residual_action": self._contact_frame_tracking_xy_residual().copy(),
            "contact_frame_tilt_scale_residual_action": self._contact_frame_tilt_scale_residual_action.copy(),
            "contact_frame_racket_x_residual_action": float(self._contact_frame_racket_xy_residual()[0]),
            "contact_frame_racket_y_residual_action": float(self._contact_frame_racket_xy_residual()[1]),
            "contact_frame_tracking_x_residual_action": float(self._contact_frame_tracking_xy_residual()[0]),
            "contact_frame_tracking_y_residual_action": float(self._contact_frame_tracking_xy_residual()[1]),
            "contact_frame_target_apex_z_residual_action": self._contact_frame_target_apex_z_residual(),
            "contact_frame_strike_plane_z_residual_action": self._contact_frame_strike_plane_z_residual(),
            "contact_frame_trajectory_tilt_scale": self._contact_frame_trajectory_tilt_scale(),
            "contact_frame_centering_tilt_scale": self._contact_frame_centering_tilt_scale(),
            "contact_frame_controller_desired_velocity": self._contact_frame_controller_desired_velocity()[0],
            "contact_frame_velocity_target": self.controller.target_velocity,
            "contact_frame_intercept_velocity_target": contact_frame_intercept_velocity_target,
            "contact_frame_lateral_brake_velocity": contact_frame_lateral_brake_velocity,
            "contact_frame_planner_active": self._contact_frame_plan_active,
            "contact_frame_strike_hold_active": self._contact_frame_strike_hold_active,
            "controller_body_clearance_active": self._controller_body_clearance_active(),
            "contact_frame_planner_contact_position": (
                self._contact_frame_plan_contact_position.copy() if self._contact_frame_plan_active else None
            ),
            "contact_frame_planner_contact_target_xy": (
                self._contact_frame_planner_contact_target_xy(self._contact_frame_plan_contact_position[:2])
                if self._contact_frame_plan_active
                else None
            ),
            "contact_frame_planner_intercept_time": (
                self._contact_frame_plan_intercept_time if self._contact_frame_plan_active else None
            ),
            "contact_frame_planner_target_xy": (
                self._contact_frame_plan_target_xy.copy() if self._contact_frame_plan_active else None
            ),
            "contact_frame_planner_target_apex_z": (
                self._contact_frame_plan_target_apex_z if self._contact_frame_plan_active else None
            ),
            "contact_frame_planner_resolved_target_apex_z": (
                self._contact_frame_resolved_target_apex_z(self._contact_frame_plan_target_apex_z)
                if self._contact_frame_plan_active
                else None
            ),
            "contact_frame_planner_desired_velocity": (
                self._contact_frame_plan_desired_velocity.copy() if self._contact_frame_plan_active else None
            ),
            "contact_frame_followthrough_offset": self._contact_frame_followthrough_offset(),
            "contact_frame_trajectory_tilt": self._contact_frame_trajectory_tilt(),
            "contact_frame_centering_tilt": self._contact_frame_centering_tilt(),
            "racket_face_normal": self.sim.racket_face_normal,
            "target_tilt": self.controller.target_tilt,
            "target_velocity": self.controller.target_velocity,
            "projected_contact_apex_height_above_racket": (
                self._projected_contact_apex_height_above_racket(contact_trace) if contact_event else None
            ),
            "desired_outgoing_velocity_x": outgoing_trajectory_metrics["desired_outgoing_velocity_x"],
            "desired_outgoing_velocity_y": outgoing_trajectory_metrics["desired_outgoing_velocity_y"],
            "desired_outgoing_velocity_z": outgoing_trajectory_metrics["desired_outgoing_velocity_z"],
            "actual_outgoing_velocity_x": outgoing_trajectory_metrics["actual_outgoing_velocity_x"],
            "actual_outgoing_velocity_y": outgoing_trajectory_metrics["actual_outgoing_velocity_y"],
            "actual_outgoing_velocity_z": outgoing_trajectory_metrics["actual_outgoing_velocity_z"],
            "actual_outgoing_velocity_source": outgoing_trajectory_metrics["actual_outgoing_velocity_source"],
            "oracle_contact_applied": oracle_info["oracle_contact_applied"],
            "oracle_contact_mode": oracle_info["oracle_contact_mode"],
            "oracle_contact_blend": oracle_info["oracle_contact_blend"],
            "oracle_contact_base_source": oracle_info["oracle_contact_base_source"],
            "outgoing_velocity_error_norm": outgoing_trajectory_metrics["outgoing_velocity_error_norm"],
            "outgoing_velocity_xy_error": outgoing_trajectory_metrics["outgoing_velocity_xy_error"],
            "outgoing_velocity_z_error": outgoing_trajectory_metrics["outgoing_velocity_z_error"],
            "contact_substep_outgoing_velocity_error_norm": outgoing_trajectory_metrics[
                "contact_substep_outgoing_velocity_error_norm"
            ],
            "contact_substep_outgoing_velocity_xy_error": outgoing_trajectory_metrics[
                "contact_substep_outgoing_velocity_xy_error"
            ],
            "contact_substep_outgoing_velocity_z_error": outgoing_trajectory_metrics[
                "contact_substep_outgoing_velocity_z_error"
            ],
            "contact_substep_predicted_apex_xy_error": outgoing_trajectory_metrics[
                "contact_substep_predicted_apex_xy_error"
            ],
            "contact_substep_predicted_next_intercept_xy_error": outgoing_trajectory_metrics[
                "contact_substep_predicted_next_intercept_xy_error"
            ],
            "desired_time_to_apex": outgoing_trajectory_metrics["desired_time_to_apex"],
            "desired_outgoing_target_x": outgoing_trajectory_metrics["desired_outgoing_target_x"],
            "desired_outgoing_target_y": outgoing_trajectory_metrics["desired_outgoing_target_y"],
            "desired_outgoing_target_z": outgoing_trajectory_metrics["desired_outgoing_target_z"],
            "desired_outgoing_xy_mode": outgoing_trajectory_metrics["desired_outgoing_xy_mode"],
            "desired_outgoing_apex_x": outgoing_trajectory_metrics["desired_outgoing_apex_x"],
            "desired_outgoing_apex_y": outgoing_trajectory_metrics["desired_outgoing_apex_y"],
            "predicted_next_intercept_x_from_actual_velocity": outgoing_trajectory_metrics[
                "predicted_next_intercept_x_from_actual_velocity"
            ],
            "predicted_next_intercept_y_from_actual_velocity": outgoing_trajectory_metrics[
                "predicted_next_intercept_y_from_actual_velocity"
            ],
            "predicted_next_intercept_time_from_actual_velocity": outgoing_trajectory_metrics[
                "predicted_next_intercept_time_from_actual_velocity"
            ],
            "predicted_next_intercept_xy_error": outgoing_trajectory_metrics[
                "predicted_next_intercept_xy_error"
            ],
            "predicted_apex_x_from_actual_velocity": outgoing_trajectory_metrics[
                "predicted_apex_x_from_actual_velocity"
            ],
            "predicted_apex_y_from_actual_velocity": outgoing_trajectory_metrics[
                "predicted_apex_y_from_actual_velocity"
            ],
            "predicted_apex_xy_error": outgoing_trajectory_metrics["predicted_apex_xy_error"],
            "contact_ball_position_x": contact_trace.get("contact_ball_position_x"),
            "contact_ball_position_y": contact_trace.get("contact_ball_position_y"),
            "contact_ball_position_z": contact_trace.get("contact_ball_position_z"),
            "contact_ball_velocity_x": contact_trace.get("contact_ball_velocity_x"),
            "contact_ball_velocity_y": contact_trace.get("contact_ball_velocity_y"),
            "contact_ball_height_above_racket": contact_trace.get("contact_ball_height_above_racket"),
            "contact_xy_alignment_error": contact_trace.get("contact_xy_alignment_error"),
            "contact_ball_speed_norm": contact_trace.get("contact_ball_speed_norm"),
            "contact_racket_velocity_x": contact_trace.get("contact_racket_velocity_x"),
            "contact_racket_velocity_y": contact_trace.get("contact_racket_velocity_y"),
            "contact_racket_lateral_speed": self._contact_racket_lateral_speed(contact_trace),
            "contact_racket_outward_speed": self._contact_racket_outward_speed(contact_trace),
            "racket_velocity_z": float(self.sim.racket_velocity[2]),
            "contact_ball_velocity_z": contact_trace.get("contact_ball_velocity_z"),
            "contact_racket_velocity_z": contact_trace.get("contact_racket_velocity_z"),
            "contact_racket_speed_norm": contact_trace.get("contact_racket_speed_norm"),
            "contact_racket_face_normal_x": contact_trace.get("contact_racket_face_normal_x"),
            "contact_racket_face_normal_y": contact_trace.get("contact_racket_face_normal_y"),
            "contact_racket_face_normal_z": contact_trace.get("contact_racket_face_normal_z"),
            "tilt_magnitude_norm": self._normalized_tilt_magnitude(),
            "tilt_action_delta_norm": self._normalized_tilt_action_delta(applied_action),
            "terminated": terminated,
            "truncated": truncated,
        }
        for key, value in contact_trace.items():
            info.setdefault(key, value)
        self._update_contact_apex_memory(contact_event=contact_event, contact_trace=contact_trace)
        self._contact_active_previous_step = contact_active
        self._previous_action = applied_action.copy()
        return self.observation(), reward, terminated, truncated, info

    def training_config(self) -> dict[str, object]:
        return {
            "scene_path": self.scene_path,
            "action_mode": self.action_mode,
            "action_limit": self.action_limit,
            "lateral_action_limit": self.lateral_action_limit,
            "vertical_action_limit": self.vertical_action_limit,
            "tilt_action_limit": self.tilt_action_limit,
            "followup_lift_action_limit": self.followup_lift_action_limit,
            "max_episode_steps": self.max_episode_steps,
            "success_velocity_threshold": self.success_velocity_threshold,
            "ball_height": self.ball_height,
            "target_ball_height": self.target_ball_height,
            "height_tolerance": self.height_tolerance,
            "tracking_reward_weight": self.tracking_reward_weight,
            "tracking_during_contact_scale": self.tracking_during_contact_scale,
            "contact_bonus": self.contact_bonus,
            "apex_match_reward_weight": self.apex_match_reward_weight,
            "useful_contact_outgoing_x_penalty_weight": self.useful_contact_outgoing_x_penalty_weight,
            "desired_outgoing_ball_velocity_x": self.desired_outgoing_ball_velocity_x,
            "useful_contact_return_target_xy_reward_weight": self.useful_contact_return_target_xy_reward_weight,
            "return_target_xy_source": self.return_target_xy_source,
            "return_target_xy_tolerance": self.return_target_xy_tolerance,
            "tilt_angle_penalty_weight": self.tilt_angle_penalty_weight,
            "tilt_action_delta_penalty_weight": self.tilt_action_delta_penalty_weight,
            "descending_ball_velocity_threshold": self.descending_ball_velocity_threshold,
            "strike_zone_xy_radius": self.strike_zone_xy_radius,
            "strike_zone_height_tolerance": self.strike_zone_height_tolerance,
            "contact_centering_radius": self.contact_centering_radius,
            "min_upward_racket_velocity_z": self.min_upward_racket_velocity_z,
            "reset_ball_height_range": self.reset_ball_height_range,
            "reset_ball_height_bounds": (
                None if self.reset_ball_height_bounds is None else list(self.reset_ball_height_bounds)
            ),
            "reset_xy_range": self.reset_xy_range,
            "reset_xy_sampling": self.reset_xy_sampling,
            "reset_velocity_xy_range": self.reset_velocity_xy_range,
            "reset_velocity_z_range": list(self.reset_velocity_z_range),
            "reset_ball_angular_velocity_range": self.reset_ball_angular_velocity_range,
            "target_offset_low": self.target_offset_low.tolist(),
            "target_offset_high": self.target_offset_high.tolist(),
            "target_tilt_limit": self.target_tilt_limit.tolist(),
            "target_pitch_range": None if self.target_pitch_range is None else self.target_pitch_range.tolist(),
            "initial_target_tilt": None if self.initial_target_tilt is None else self.initial_target_tilt.tolist(),
            "strike_tilt_assist_limit": (
                None if self.strike_tilt_assist_limit is None else self.strike_tilt_assist_limit.tolist()
            ),
            "strike_tilt_assist_deadband": self.strike_tilt_assist_deadband,
            "strike_tilt_ramp_pitch": self.strike_tilt_ramp_pitch,
            "strike_tilt_ramp_xy_tolerance": self.strike_tilt_ramp_xy_tolerance,
            "followup_strike_target_tilt": (
                None if self.followup_strike_target_tilt is None else self.followup_strike_target_tilt.tolist()
            ),
            "followup_strike_contact_offset_ratio": self.followup_strike_contact_offset_ratio,
            "followup_strike_contact_offset_max": self.followup_strike_contact_offset_max,
            "followup_strike_lift_boost": self.followup_strike_lift_boost,
            "post_contact_return_assist_weight": self.post_contact_return_assist_weight,
            "post_contact_return_max_intercept_time": self.post_contact_return_max_intercept_time,
            "post_contact_return_z_offset": self.post_contact_return_z_offset,
            "post_contact_return_predict_during_rise": self.post_contact_return_predict_during_rise,
            "next_intercept_reachable_bonus_weight": self.next_intercept_reachable_bonus_weight,
            "easy_next_ball_reward_weight": self.easy_next_ball_reward_weight,
            "require_reachable_next_intercept_for_success": self.require_reachable_next_intercept_for_success,
            "require_apex_height_window_for_success": self.require_apex_height_window_for_success,
            "min_easy_next_ball_score_for_success": self.min_easy_next_ball_score_for_success,
            "gate_nonuseful_easy_next_ball_by_apex": self.gate_nonuseful_easy_next_ball_by_apex,
            "terminate_on_nonuseful_contact": self.terminate_on_nonuseful_contact,
            "terminate_on_low_apex_contact": self.terminate_on_low_apex_contact,
            "low_apex_contact_height_threshold": self.low_apex_contact_height_threshold,
            "low_apex_contact_grace_count": self.low_apex_contact_grace_count,
            "include_velocity_domain_observation": self.include_velocity_domain_observation,
            "include_task_phase_observation": self.include_task_phase_observation,
            "include_contact_context_observation": self.include_contact_context_observation,
            "include_next_intercept_observation": self.include_next_intercept_observation,
            "include_desired_outgoing_velocity_observation": self.include_desired_outgoing_velocity_observation,
            "desired_outgoing_xy_mode": self.desired_outgoing_xy_mode,
            "keepup_target_xy_offset": self.keepup_target_xy_offset.tolist(),
            "trajectory_match_reward_weight": self.trajectory_match_reward_weight,
            "trajectory_error_penalty_weight": self.trajectory_error_penalty_weight,
            "reward_contact_quality_on_any_upward_contact": self.reward_contact_quality_on_any_upward_contact,
            "contact_oracle_mode": self.contact_oracle_mode,
            "contact_oracle_blend": self.contact_oracle_blend,
            "next_intercept_max_time": self.next_intercept_max_time,
            "next_intercept_success_radius": self.next_intercept_success_radius,
            "easy_next_ball_xy_radius": self.easy_next_ball_xy_radius,
            "controller_nullspace_posture_gain": self.controller_nullspace_posture_gain,
            "controller_nullspace_posture_max_step": self.controller_nullspace_posture_max_step,
            "controller_nullspace_posture_target": (
                None
                if self.controller_nullspace_posture_target is None
                else self.controller_nullspace_posture_target.tolist()
            ),
            "controller_body_clearance_gain": self.controller_body_clearance_gain,
            "controller_body_clearance_margin": self.controller_body_clearance_margin,
            "controller_body_clearance_vertical_margin": self.controller_body_clearance_vertical_margin,
            "controller_body_clearance_max_step": self.controller_body_clearance_max_step,
            "controller_body_clearance_body_names": list(self.controller_body_clearance_body_names),
            "tracking_strike_plane_offset": self.tracking_strike_plane_offset,
            "controller_position_gain": self.controller_position_gain,
            "controller_orientation_gain": self.controller_orientation_gain,
            "controller_max_position_step": self.controller_max_position_step,
            "controller_max_orientation_step": self.controller_max_orientation_step,
            "controller_velocity_gain": self.controller_velocity_gain,
            "controller_velocity_feedback_gain": self.controller_velocity_feedback_gain,
            "controller_max_velocity_step": self.controller_max_velocity_step,
            "contact_frame_base_strike_z_boost": self.contact_frame_base_strike_z_boost,
            "contact_frame_base_strike_z_offset": self.contact_frame_base_strike_z_offset,
            "contact_frame_base_strike_time_horizon": self.contact_frame_base_strike_time_horizon,
            "contact_frame_base_tilt_residual": (
                None if self.contact_frame_base_tilt_residual is None else self.contact_frame_base_tilt_residual.tolist()
            ),
            "contact_frame_apex_lift_gain": self.contact_frame_apex_lift_gain,
            "contact_frame_apex_lift_max": self.contact_frame_apex_lift_max,
            "contact_frame_apex_lift_reference_velocity_z": self.contact_frame_apex_lift_reference_velocity_z,
            "contact_frame_apex_lift_restitution": self.contact_frame_apex_lift_restitution,
            "contact_frame_velocity_lead_gain": self.contact_frame_velocity_lead_gain,
            "contact_frame_velocity_lead_max": self.contact_frame_velocity_lead_max,
            "contact_frame_velocity_target_gain": self.contact_frame_velocity_target_gain,
            "contact_frame_velocity_target_max": self.contact_frame_velocity_target_max,
            "contact_frame_velocity_scale_action_limit": self.contact_frame_velocity_scale_action_limit,
            "contact_frame_outgoing_xy_action_limit": self.contact_frame_outgoing_xy_action_limit,
            "contact_frame_racket_vz_action_limit": self.contact_frame_racket_vz_action_limit,
            "contact_frame_racket_xy_action_limit": self.contact_frame_racket_xy_action_limit,
            "contact_frame_tilt_scale_action_limit": self.contact_frame_tilt_scale_action_limit,
            "contact_frame_target_apex_z_action_limit": self.contact_frame_target_apex_z_action_limit,
            "contact_frame_strike_plane_z_action_limit": self.contact_frame_strike_plane_z_action_limit,
            "contact_frame_tracking_xy_action_limit": self.contact_frame_tracking_xy_action_limit,
            "contact_frame_intercept_velocity_gain": self.contact_frame_intercept_velocity_gain,
            "contact_frame_intercept_velocity_max": self.contact_frame_intercept_velocity_max,
            "contact_frame_intercept_velocity_time_floor": self.contact_frame_intercept_velocity_time_floor,
            "contact_frame_planner_enabled": self.contact_frame_planner_enabled,
            "contact_frame_planner_hold_during_descent": self.contact_frame_planner_hold_during_descent,
            "contact_frame_planner_min_intercept_time": self.contact_frame_planner_min_intercept_time,
            "contact_frame_planner_max_intercept_time": self.contact_frame_planner_max_intercept_time,
            "contact_frame_planner_target_apex_z_offset": self.contact_frame_planner_target_apex_z_offset,
            "contact_frame_planner_contact_offset_ratio": self.contact_frame_planner_contact_offset_ratio,
            "contact_frame_planner_contact_offset_max": self.contact_frame_planner_contact_offset_max,
            "contact_frame_strike_hold_time": self.contact_frame_strike_hold_time,
            "contact_frame_strike_hold_min_readiness": self.contact_frame_strike_hold_min_readiness,
            "contact_frame_followthrough_gain": self.contact_frame_followthrough_gain,
            "contact_frame_followthrough_time": self.contact_frame_followthrough_time,
            "contact_frame_followthrough_max": self.contact_frame_followthrough_max,
            "contact_frame_lateral_brake_gain": self.contact_frame_lateral_brake_gain,
            "contact_frame_lateral_brake_max": self.contact_frame_lateral_brake_max,
            "contact_frame_lateral_brake_radius": self.contact_frame_lateral_brake_radius,
            "contact_frame_trajectory_tilt_gain": self.contact_frame_trajectory_tilt_gain,
            "contact_frame_trajectory_tilt_limit": (
                None
                if self.contact_frame_trajectory_tilt_limit is None
                else self.contact_frame_trajectory_tilt_limit.tolist()
            ),
            "contact_frame_trajectory_tilt_deadband": self.contact_frame_trajectory_tilt_deadband,
            "contact_frame_tilt_ramp_time": self.contact_frame_tilt_ramp_time,
            "contact_frame_centering_tilt_limit": (
                None
                if self.contact_frame_centering_tilt_limit is None
                else self.contact_frame_centering_tilt_limit.tolist()
            ),
            "contact_frame_centering_tilt_radius": self.contact_frame_centering_tilt_radius,
            "contact_frame_centering_tilt_deadband": self.contact_frame_centering_tilt_deadband,
            "contact_frame_action_penalty_weight": self.contact_frame_action_penalty_weight,
            "next_intercept_xy_error_penalty_weight": self.next_intercept_xy_error_penalty_weight,
            "post_contact_lateral_velocity_penalty_weight": self.post_contact_lateral_velocity_penalty_weight,
            "contact_xy_error_penalty_weight": self.contact_xy_error_penalty_weight,
            "contact_racket_lateral_velocity_penalty_weight": self.contact_racket_lateral_velocity_penalty_weight,
            "contact_racket_lateral_velocity_tolerance": self.contact_racket_lateral_velocity_tolerance,
            "contact_racket_outward_velocity_penalty_weight": (
                self.contact_racket_outward_velocity_penalty_weight
            ),
            "contact_racket_outward_velocity_tolerance": self.contact_racket_outward_velocity_tolerance,
            "max_contact_racket_lateral_speed_for_success": self.max_contact_racket_lateral_speed_for_success,
            "nonuseful_contact_penalty_weight": self.nonuseful_contact_penalty_weight,
            "contact_apex_under_target_penalty_weight": self.contact_apex_under_target_penalty_weight,
            "contact_apex_progress_reward_weight": self.contact_apex_progress_reward_weight,
            "contact_apex_recovery_progress_reward_weight": self.contact_apex_recovery_progress_reward_weight,
            "gate_contact_apex_progress_by_easy_next_ball": self.gate_contact_apex_progress_by_easy_next_ball,
            "contact_apex_progress_min_easy_next_ball_score": self.contact_apex_progress_min_easy_next_ball_score,
            "contact_apex_potential_reward_weight": self.contact_apex_potential_reward_weight,
            "contact_apex_potential_gamma": self.contact_apex_potential_gamma,
            "contact_apex_potential_cap": self.contact_apex_potential_cap,
            "contact_lateral_stability_reward_weight": self.contact_lateral_stability_reward_weight,
            "contact_lateral_stability_speed_tolerance": self.contact_lateral_stability_speed_tolerance,
            "contact_lateral_stability_xy_tolerance": self.contact_lateral_stability_xy_tolerance,
            "contact_lateral_stability_min_apex_ratio": self.contact_lateral_stability_min_apex_ratio,
            "stable_contact_reward_weight": self.stable_contact_reward_weight,
            "stable_contact_min_apex_ratio": self.stable_contact_min_apex_ratio,
            "stable_cycle_reward_weight": self.stable_cycle_reward_weight,
            "stable_cycle_reward_cap": self.stable_cycle_reward_cap,
            "stable_cycle_min_easy_next_ball_score": self.stable_cycle_min_easy_next_ball_score,
            "contact_frame_low_apex_recovery_lift_gain": self.contact_frame_low_apex_recovery_lift_gain,
            "contact_frame_low_apex_recovery_lift_max": self.contact_frame_low_apex_recovery_lift_max,
            "contact_frame_low_apex_recovery_velocity_gain": self.contact_frame_low_apex_recovery_velocity_gain,
            "contact_frame_low_apex_recovery_velocity_max": self.contact_frame_low_apex_recovery_velocity_max,
        }

    def set_reset_distribution(
        self,
        *,
        reset_xy_range: float | None = None,
        reset_xy_sampling: str | None = None,
        reset_ball_height_range: float | None = None,
        reset_ball_height_bounds: Sequence[float] | None = None,
        reset_velocity_xy_range: float | None = None,
        reset_velocity_z_range: Sequence[float] | None = None,
        reset_ball_angular_velocity_range: float | None = None,
    ) -> dict[str, object]:
        if reset_xy_range is not None:
            parsed_xy_range = float(reset_xy_range)
            if parsed_xy_range < 0.0:
                raise ValueError(f"reset_xy_range must be non-negative, got {parsed_xy_range}.")
            self.reset_xy_range = parsed_xy_range
        if reset_xy_sampling is not None:
            parsed_xy_sampling = str(reset_xy_sampling)
            if parsed_xy_sampling not in _RESET_XY_SAMPLING_MODES:
                raise ValueError(
                    f"reset_xy_sampling must be one of {_RESET_XY_SAMPLING_MODES}, got {parsed_xy_sampling!r}."
                )
            self.reset_xy_sampling = parsed_xy_sampling
        if reset_ball_height_range is not None:
            parsed_height_range = float(reset_ball_height_range)
            if parsed_height_range < 0.0:
                raise ValueError(f"reset_ball_height_range must be non-negative, got {parsed_height_range}.")
            self.reset_ball_height_range = parsed_height_range
        if reset_ball_height_bounds is not None:
            parsed_height_bounds = (float(reset_ball_height_bounds[0]), float(reset_ball_height_bounds[1]))
            if parsed_height_bounds[0] > parsed_height_bounds[1]:
                raise ValueError(
                    "reset_ball_height_bounds must be ordered as (low, high), got "
                    f"{parsed_height_bounds}."
                )
            if parsed_height_bounds[0] <= 0.0:
                raise ValueError(
                    "reset_ball_height_bounds must stay above the racket plane, got "
                    f"{parsed_height_bounds}."
                )
            self.reset_ball_height_bounds = parsed_height_bounds
        if reset_velocity_xy_range is not None:
            parsed_velocity_xy_range = float(reset_velocity_xy_range)
            if parsed_velocity_xy_range < 0.0:
                raise ValueError(f"reset_velocity_xy_range must be non-negative, got {parsed_velocity_xy_range}.")
            self.reset_velocity_xy_range = parsed_velocity_xy_range
        if reset_velocity_z_range is not None:
            parsed_velocity_z_range = (float(reset_velocity_z_range[0]), float(reset_velocity_z_range[1]))
            if parsed_velocity_z_range[0] > parsed_velocity_z_range[1]:
                raise ValueError(
                    "reset_velocity_z_range must be ordered as (low, high), got "
                    f"{parsed_velocity_z_range}."
                )
            self.reset_velocity_z_range = parsed_velocity_z_range
        if reset_ball_angular_velocity_range is not None:
            parsed_angular_velocity_range = float(reset_ball_angular_velocity_range)
            if parsed_angular_velocity_range < 0.0:
                raise ValueError(
                    "reset_ball_angular_velocity_range must be non-negative, got "
                    f"{parsed_angular_velocity_range}."
                )
            self.reset_ball_angular_velocity_range = parsed_angular_velocity_range
        return self.training_config()

    def close(self) -> None:
        return None

    def _contact_frame_action_mode(self) -> bool:
        return self.action_mode in _CONTACT_FRAME_ACTION_MODES

    def _contact_frame_velocity_residual_scale(self) -> float:
        if self.action_mode not in _CONTACT_FRAME_VELOCITY_RESIDUAL_ACTION_MODES:
            return 1.0
        return float(max(0.0, 1.0 + self._contact_frame_velocity_residual_action[0]))

    def _apply_contact_frame_velocity_residual(self, desired_velocity: Sequence[float]) -> np.ndarray:
        resolved_desired_velocity = np.asarray(desired_velocity, dtype=float).copy()
        if self.action_mode not in _CONTACT_FRAME_VELOCITY_RESIDUAL_ACTION_MODES:
            return resolved_desired_velocity
        resolved_desired_velocity[2] *= self._contact_frame_velocity_residual_scale()
        resolved_desired_velocity[:2] += self._contact_frame_velocity_residual_action[1:3]
        return resolved_desired_velocity

    def _contact_frame_racket_vz_residual(self) -> float:
        if self.action_mode not in _CONTACT_FRAME_TILT_SCALE_ACTION_MODES:
            return 0.0
        return float(self._contact_frame_racket_vz_residual_action)

    def _contact_frame_racket_xy_residual(self) -> np.ndarray:
        if self.action_mode not in _CONTACT_FRAME_LATERAL_VELOCITY_RESIDUAL_ACTION_MODES:
            return np.zeros(2, dtype=float)
        return np.asarray(self._contact_frame_racket_xy_residual_action, dtype=float)

    def _contact_frame_tracking_xy_residual(self) -> np.ndarray:
        if self.action_mode not in _CONTACT_FRAME_TRACKING_RESIDUAL_ACTION_MODES:
            return np.zeros(2, dtype=float)
        if float(self.sim.ball_velocity[2]) >= self.descending_ball_velocity_threshold:
            return np.zeros(2, dtype=float)
        return np.asarray(self._contact_frame_tracking_xy_residual_action, dtype=float)

    def _contact_frame_target_apex_z_residual(self) -> float:
        if self.action_mode not in _CONTACT_FRAME_APEX_TIMING_RESIDUAL_ACTION_MODES:
            return 0.0
        return float(getattr(self, "_contact_frame_target_apex_z_residual_action", 0.0))

    def _contact_frame_strike_plane_z_residual(self) -> float:
        if self.action_mode not in _CONTACT_FRAME_APEX_TIMING_RESIDUAL_ACTION_MODES:
            return 0.0
        return float(getattr(self, "_contact_frame_strike_plane_z_residual_action", 0.0))

    def _contact_frame_resolved_target_apex_z(self, base_target_apex_z: float) -> float:
        return float(base_target_apex_z + self._contact_frame_target_apex_z_residual())

    def _contact_frame_trajectory_tilt_scale(self) -> float:
        if self.action_mode not in _CONTACT_FRAME_TILT_SCALE_ACTION_MODES:
            return 1.0
        return float(max(0.0, 1.0 + self._contact_frame_tilt_scale_residual_action[0]))

    def _contact_frame_centering_tilt_scale(self) -> float:
        if self.action_mode not in _CONTACT_FRAME_TILT_SCALE_ACTION_MODES:
            return 1.0
        return float(max(0.0, 1.0 + self._contact_frame_tilt_scale_residual_action[1]))

    def _reset_contact_frame_plan(self) -> None:
        self._contact_frame_plan_active = False
        self._contact_frame_strike_hold_active = False
        self._contact_frame_plan_intercept_time = 0.0
        self._contact_frame_plan_contact_position = np.zeros(3, dtype=float)
        self._contact_frame_plan_target_xy = np.zeros(2, dtype=float)
        self._contact_frame_plan_target_apex_z = 0.0
        self._contact_frame_plan_desired_velocity = np.zeros(3, dtype=float)
        self._contact_frame_plan_time_to_apex = 0.0

    def _contact_frame_planner_contact_time_and_position(self) -> tuple[float, np.ndarray | None]:
        target_z = float(self._controller_anchor_position()[2] + self._tracking_strike_plane_offset())
        candidate_times = self._ballistic_intercept_times(
            target_z,
            max_intercept_time=self.contact_frame_planner_max_intercept_time,
        )
        descending_times = [
            time_value
            for time_value in candidate_times
            if (
                time_value >= self.contact_frame_planner_min_intercept_time
                and float(self.sim.ball_velocity[2] + self._gravity_z() * time_value) < 0.0
            )
        ]
        if not descending_times:
            return 0.0, None
        intercept_time = min(descending_times)
        contact_position = np.asarray(self.sim.ball_position, dtype=float).copy()
        contact_position[:2] = self.sim.ball_position[:2] + intercept_time * self.sim.ball_velocity[:2]
        contact_position[2] = target_z
        return float(intercept_time), contact_position

    def _update_contact_frame_plan(self) -> None:
        if not self.contact_frame_planner_enabled or not self._contact_frame_action_mode():
            self._reset_contact_frame_plan()
            return
        if float(self.sim.ball_velocity[2]) >= self.descending_ball_velocity_threshold:
            self._reset_contact_frame_plan()
            return

        intercept_time, contact_position = self._contact_frame_planner_contact_time_and_position()
        if contact_position is None:
            self._reset_contact_frame_plan()
            return

        was_strike_hold_active = self._contact_frame_strike_hold_active
        if not self._contact_frame_plan_active or not self.contact_frame_planner_hold_during_descent:
            self._contact_frame_plan_target_xy = self._keepup_target_xy()
            self._contact_frame_plan_target_apex_z = float(
                self._controller_anchor_position()[2]
                + self._target_ball_height_above_racket()
                + self.contact_frame_planner_target_apex_z_offset
            )
            self._contact_frame_strike_hold_active = False
        if (
            not self._contact_frame_strike_hold_active
            and self.contact_frame_strike_hold_time > 0.0
            and intercept_time <= self.contact_frame_strike_hold_time
            and self._pre_contact_readiness() >= self.contact_frame_strike_hold_min_readiness
        ):
            self._contact_frame_strike_hold_active = True

        if was_strike_hold_active:
            contact_position = self._contact_frame_plan_contact_position.copy()

        desired_velocity, desired_time_to_apex, _ = self._desired_outgoing_velocity(
            contact_position,
            target_xy=self._contact_frame_plan_target_xy,
            target_apex_z=self._contact_frame_resolved_target_apex_z(self._contact_frame_plan_target_apex_z),
        )
        self._contact_frame_plan_active = True
        self._contact_frame_plan_intercept_time = float(intercept_time)
        self._contact_frame_plan_contact_position = np.asarray(contact_position, dtype=float)
        self._contact_frame_plan_desired_velocity = np.asarray(desired_velocity, dtype=float)
        self._contact_frame_plan_time_to_apex = float(desired_time_to_apex)

    def _contact_frame_planned_contact_position(self) -> np.ndarray:
        if self._contact_frame_plan_active:
            return self._contact_frame_plan_contact_position.copy()
        return self._predicted_contact_position(max_intercept_time=self.next_intercept_max_time)

    def _contact_frame_planned_desired_velocity(
        self,
        contact_position: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        if self._contact_frame_plan_active and contact_position is None:
            if self.action_mode not in _CONTACT_FRAME_APEX_TIMING_RESIDUAL_ACTION_MODES:
                return (
                    self._contact_frame_plan_desired_velocity.copy(),
                    float(self._contact_frame_plan_time_to_apex),
                    self._contact_frame_plan_target_xy.copy(),
                )
            return self._desired_outgoing_velocity(
                self._contact_frame_planned_contact_position(),
                target_xy=self._contact_frame_plan_target_xy,
                target_apex_z=self._contact_frame_resolved_target_apex_z(self._contact_frame_plan_target_apex_z),
            )
        resolved_contact_position = (
            self._contact_frame_planned_contact_position()
            if contact_position is None
            else np.asarray(contact_position, dtype=float)
        )
        if self._contact_frame_plan_active:
            return self._desired_outgoing_velocity(
                resolved_contact_position,
                target_xy=self._contact_frame_plan_target_xy,
                target_apex_z=self._contact_frame_resolved_target_apex_z(self._contact_frame_plan_target_apex_z),
            )
        return self._desired_outgoing_velocity(resolved_contact_position)

    def _contact_frame_controller_desired_velocity(
        self,
        contact_position: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        desired_velocity, desired_time_to_apex, target_xy = self._contact_frame_planned_desired_velocity(
            contact_position
        )
        return (
            self._apply_contact_frame_velocity_residual(desired_velocity),
            desired_time_to_apex,
            target_xy,
        )

    def _contact_float(self, contact_trace: dict[str, object] | None, key: str, default: float) -> float:
        if contact_trace is None:
            return float(default)
        value = contact_trace.get(key)
        if value is None:
            return float(default)
        return float(value)

    def _contact_vector(self, contact_trace: dict[str, object] | None, prefix: str) -> np.ndarray | None:
        if contact_trace is None:
            return None
        x_value = contact_trace.get(f"{prefix}_x")
        y_value = contact_trace.get(f"{prefix}_y")
        z_value = contact_trace.get(f"{prefix}_z")
        if x_value is None or y_value is None or z_value is None:
            return None
        return np.array([float(x_value), float(y_value), float(z_value)], dtype=float)

    def _contact_racket_lateral_speed(self, contact_trace: dict[str, object] | None) -> float:
        if contact_trace is None:
            return float(np.linalg.norm(self.sim.racket_velocity[:2]))
        velocity_x = contact_trace.get("contact_racket_velocity_x")
        velocity_y = contact_trace.get("contact_racket_velocity_y")
        if velocity_x is None or velocity_y is None:
            return float(np.linalg.norm(self.sim.racket_velocity[:2]))
        return float(np.linalg.norm(np.array([float(velocity_x), float(velocity_y)], dtype=float)))

    def _contact_racket_outward_speed(self, contact_trace: dict[str, object] | None) -> float:
        if contact_trace is None:
            return 0.0
        velocity = self._contact_vector(contact_trace, "contact_racket_velocity")
        contact_position = self._contact_vector(contact_trace, "contact_ball_position")
        if velocity is None or contact_position is None:
            return 0.0
        outward_xy = contact_position[:2] - self._keepup_target_xy()
        outward_distance = float(np.linalg.norm(outward_xy))
        if outward_distance <= max(self.next_intercept_success_radius, 1.0e-6):
            return 0.0
        outward_direction = outward_xy / outward_distance
        return max(float(np.dot(velocity[:2], outward_direction)), 0.0)

    def _contact_racket_outward_velocity_penalty_term(self, contact_trace: dict[str, object] | None) -> float:
        if self.contact_racket_outward_velocity_penalty_weight <= 0.0:
            return 0.0
        outward_speed = self._contact_racket_outward_speed(contact_trace)
        excess_outward_speed = max(outward_speed - self.contact_racket_outward_velocity_tolerance, 0.0)
        normalized_outward_speed = min(
            excess_outward_speed / max(self.contact_racket_outward_velocity_tolerance, 1.0e-6),
            4.0,
        )
        return -self.contact_racket_outward_velocity_penalty_weight * normalized_outward_speed

    def _resolved_outgoing_ball_velocity(self, contact_trace: dict[str, object] | None) -> tuple[np.ndarray | None, str | None]:
        if contact_trace is None:
            return None, None

        oracle_velocity = self._contact_vector(contact_trace, "oracle_post_contact_ball_velocity")
        if oracle_velocity is not None:
            return oracle_velocity, "oracle_post_contact_ball_velocity"

        contact_end_velocity = self._contact_vector(contact_trace, "contact_end_ball_velocity")
        if contact_end_velocity is not None:
            return contact_end_velocity, "contact_end_ball_velocity"

        for offset in range(5, 0, -1):
            post_contact_velocity = self._contact_vector(contact_trace, f"post_contact_{offset}_ball_velocity")
            if post_contact_velocity is None:
                continue
            if contact_trace.get(f"post_contact_{offset}_contact_active") is False:
                return post_contact_velocity, f"post_contact_{offset}_ball_velocity"

        for offset in range(5, 0, -1):
            post_contact_velocity = self._contact_vector(contact_trace, f"post_contact_{offset}_ball_velocity")
            if post_contact_velocity is not None:
                return post_contact_velocity, f"post_contact_{offset}_ball_velocity"

        contact_velocity = self._contact_vector(contact_trace, "contact_ball_velocity")
        if contact_velocity is not None:
            return contact_velocity, "contact_ball_velocity"
        return None, None

    def _apply_contact_oracle(self, contact_trace: dict[str, object]) -> dict[str, object]:
        oracle_info = {
            "oracle_contact_applied": False,
            "oracle_contact_mode": self.contact_oracle_mode,
            "oracle_contact_blend": self.contact_oracle_blend,
            "oracle_contact_base_source": None,
        }
        if self.contact_oracle_mode == "none" or not bool(contact_trace.get("contact_observed", False)):
            return oracle_info

        contact_ball_position = self._contact_vector(contact_trace, "contact_ball_position")
        if contact_ball_position is None:
            return oracle_info

        if self._contact_frame_action_mode() and self._contact_frame_plan_active:
            desired_velocity, _, _ = self._contact_frame_planned_desired_velocity(contact_ball_position)
        else:
            desired_velocity, _, _ = self._desired_outgoing_velocity(contact_ball_position)
        base_velocity, base_source = self._resolved_outgoing_ball_velocity(contact_trace)
        if base_velocity is None:
            base_velocity = self._contact_vector(contact_trace, "contact_ball_velocity")
            if base_velocity is not None:
                base_source = "contact_ball_velocity"
        if base_velocity is None:
            base_velocity = np.asarray(self.sim.ball_velocity, dtype=float)
            base_source = "sim_ball_velocity"

        oracle_velocity = (1.0 - self.contact_oracle_blend) * base_velocity + self.contact_oracle_blend * desired_velocity
        self.sim.set_ball_velocity(oracle_velocity)
        contact_trace.update(self.sim._trace_vector_fields("oracle_desired_outgoing_velocity", desired_velocity))
        contact_trace.update(self.sim._trace_vector_fields("oracle_preoverride_outgoing_ball_velocity", base_velocity))
        contact_trace.update(self.sim._trace_vector_fields("oracle_post_contact_ball_velocity", oracle_velocity))
        contact_trace["oracle_contact_applied"] = True
        contact_trace["oracle_contact_mode"] = self.contact_oracle_mode
        contact_trace["oracle_contact_blend"] = self.contact_oracle_blend
        contact_trace["oracle_contact_base_source"] = base_source
        oracle_info["oracle_contact_applied"] = True
        oracle_info["oracle_contact_base_source"] = base_source
        return oracle_info

    def _predicted_descending_intercept_from_velocity(
        self,
        start_position: np.ndarray,
        velocity: np.ndarray,
        target_z: float,
    ) -> tuple[float | None, np.ndarray | None]:
        quadratic_a = 0.5 * self._gravity_z()
        quadratic_b = float(velocity[2])
        quadratic_c = float(start_position[2] - target_z)
        candidate_times: list[float] = []
        if abs(quadratic_a) < 1.0e-9:
            if abs(quadratic_b) > 1.0e-9:
                candidate_times.append(-quadratic_c / quadratic_b)
        else:
            discriminant = quadratic_b * quadratic_b - 4.0 * quadratic_a * quadratic_c
            if discriminant >= 0.0:
                sqrt_discriminant = float(np.sqrt(discriminant))
                denominator = 2.0 * quadratic_a
                candidate_times.extend(
                    [
                        (-quadratic_b - sqrt_discriminant) / denominator,
                        (-quadratic_b + sqrt_discriminant) / denominator,
                    ]
                )
        descending_times = [
            time_value
            for time_value in candidate_times
            if time_value > 1.0e-6 and float(velocity[2] + self._gravity_z() * time_value) < 0.0
        ]
        if not descending_times:
            return None, None
        intercept_time = min(descending_times)
        intercept_xy = np.asarray(start_position[:2] + velocity[:2] * intercept_time, dtype=float)
        return float(intercept_time), intercept_xy

    def _desired_outgoing_target_z(self) -> float:
        return float(self._controller_anchor_position()[2] + self._tracking_strike_plane_offset())

    def _trajectory_metrics_from_velocity(
        self,
        contact_ball_position: np.ndarray,
        actual_velocity: np.ndarray,
        desired_velocity: np.ndarray,
        desired_time_to_apex: float,
        desired_target_xy: np.ndarray,
        desired_target_z: float,
    ) -> dict[str, float]:
        gravity_magnitude = max(abs(self._gravity_z()), 1.0e-6)
        velocity_error = actual_velocity - desired_velocity
        actual_time_to_apex = max(float(actual_velocity[2]), 0.0) / gravity_magnitude
        predicted_apex_xy = contact_ball_position[:2] + actual_velocity[:2] * actual_time_to_apex
        desired_apex_xy = contact_ball_position[:2] + desired_velocity[:2] * desired_time_to_apex
        predicted_next_time, predicted_next_xy = self._predicted_descending_intercept_from_velocity(
            contact_ball_position,
            actual_velocity,
            desired_target_z,
        )
        if predicted_next_xy is None:
            predicted_next_x = None
            predicted_next_y = None
            predicted_next_error = None
        else:
            predicted_next_x = float(predicted_next_xy[0])
            predicted_next_y = float(predicted_next_xy[1])
            predicted_next_error = float(np.linalg.norm(predicted_next_xy - desired_target_xy))
        return {
            "outgoing_velocity_error_norm": float(np.linalg.norm(velocity_error)),
            "outgoing_velocity_xy_error": float(np.linalg.norm(velocity_error[:2])),
            "outgoing_velocity_z_error": float(abs(velocity_error[2])),
            "predicted_apex_x": float(predicted_apex_xy[0]),
            "predicted_apex_y": float(predicted_apex_xy[1]),
            "predicted_apex_xy_error": float(np.linalg.norm(predicted_apex_xy - desired_apex_xy)),
            "desired_apex_x": float(desired_apex_xy[0]),
            "desired_apex_y": float(desired_apex_xy[1]),
            "predicted_next_intercept_x": predicted_next_x,
            "predicted_next_intercept_y": predicted_next_y,
            "predicted_next_intercept_time": predicted_next_time,
            "predicted_next_intercept_xy_error": predicted_next_error,
        }

    def _ball_height_above_racket(self) -> float:
        return float(self.sim.ball_position[2] - self.sim.racket_position[2])

    def _xy_alignment_error(self) -> float:
        return float(np.linalg.norm(self.sim.ball_position[:2] - self.sim.racket_position[:2]))

    def _tracking_strike_plane_offset(self) -> float:
        offset = self.tracking_strike_plane_offset + self._contact_frame_strike_plane_z_residual()
        return float(np.clip(offset, self.target_offset_low[2], self.target_offset_high[2]))

    def _predicted_intercept_time(self, max_intercept_time: float = 0.35) -> float:
        target_z = float(self.sim.racket_position[2]) + self._tracking_strike_plane_offset()
        valid_times = self._ballistic_intercept_times(target_z, max_intercept_time=max_intercept_time)
        if valid_times:
            return min(valid_times)
        return 0.0

    def _predicted_intercept_xy(self, max_intercept_time: float = 0.35) -> np.ndarray:
        intercept_time = self._predicted_intercept_time(max_intercept_time=max_intercept_time)
        return np.asarray(self.sim.ball_position[:2] + intercept_time * self.sim.ball_velocity[:2], dtype=float)

    def _predicted_contact_position(self, max_intercept_time: float = 0.35) -> np.ndarray:
        intercept_time = self._predicted_intercept_time(max_intercept_time=max_intercept_time)
        contact_position = np.asarray(self.sim.ball_position, dtype=float).copy()
        contact_position[:2] = self.sim.ball_position[:2] + intercept_time * self.sim.ball_velocity[:2]
        contact_position[2] = (
            float(self.sim.ball_position[2])
            + float(self.sim.ball_velocity[2]) * intercept_time
            + 0.5 * self._gravity_z() * intercept_time * intercept_time
        )
        return contact_position

    def _tracking_alignment_error(self) -> float:
        return float(np.linalg.norm(self._predicted_intercept_xy() - self.sim.racket_position[:2]))

    def _gravity_z(self) -> float:
        return float(self.sim.model.opt.gravity[2])

    def _ballistic_intercept_times(self, target_z: float, max_intercept_time: float) -> list[float]:
        quadratic_a = 0.5 * self._gravity_z()
        quadratic_b = float(self.sim.ball_velocity[2])
        quadratic_c = float(self.sim.ball_position[2] - target_z)

        candidate_times: list[float] = []
        if abs(quadratic_a) < 1.0e-9:
            if abs(quadratic_b) > 1.0e-9:
                candidate_times.append(-quadratic_c / quadratic_b)
        else:
            discriminant = quadratic_b * quadratic_b - 4.0 * quadratic_a * quadratic_c
            if discriminant >= 0.0:
                sqrt_discriminant = float(np.sqrt(discriminant))
                denominator = 2.0 * quadratic_a
                candidate_times.extend(
                    [
                        (-quadratic_b - sqrt_discriminant) / denominator,
                        (-quadratic_b + sqrt_discriminant) / denominator,
                    ]
                )
        return sorted(time_value for time_value in candidate_times if 1.0e-6 <= time_value <= max_intercept_time)

    def _time_since_contact(self) -> float | None:
        if self._last_contact_step is None:
            return None
        return float(max(self.step_count - self._last_contact_step, 0) * self.sim.control_dt)

    def _phase_name(self) -> str:
        time_since_contact = self._time_since_contact()
        recent_contact = time_since_contact is not None and time_since_contact <= self.next_intercept_max_time
        ball_velocity_z = float(self.sim.ball_velocity[2])
        if recent_contact:
            if ball_velocity_z > 0.0:
                return "return_shaping"
            return "recovery"
        if ball_velocity_z < self.descending_ball_velocity_threshold:
            if self._pre_contact_upward_ready():
                return "strike"
            return "prepare"
        return "prepare"

    def _phase_one_hot(self) -> np.ndarray:
        phase_name = self._phase_name()
        phase_vector = np.zeros(4, dtype=float)
        phase_index = {
            "prepare": 0,
            "strike": 1,
            "return_shaping": 2,
            "recovery": 3,
        }[phase_name]
        phase_vector[phase_index] = 1.0
        return phase_vector

    def _next_intercept_metrics(self) -> dict[str, object]:
        anchor_position = self._controller_anchor_position()
        target_xy = self._keepup_target_xy()
        target_z = float(anchor_position[2] + self._tracking_strike_plane_offset())
        candidate_times = self._ballistic_intercept_times(target_z, max_intercept_time=self.next_intercept_max_time)
        descending_times = [
            time_value
            for time_value in candidate_times
            if float(self.sim.ball_velocity[2] + self._gravity_z() * time_value) < 0.0
        ]
        default_metrics: dict[str, object] = {
            "time": 0.0,
            "relative_xy": np.zeros(2, dtype=float),
            "reachable": False,
            "recovery_distance": 0.0,
            "recovery_readiness": 0.0,
            "info_time": None,
            "info_x": None,
            "info_y": None,
            "info_xy_error": None,
            "info_reachable": None,
            "info_vertical_speed": None,
            "info_speed_norm": None,
            "info_recovery_distance": None,
            "info_recovery_readiness": None,
            "easy_next_ball_score": 0.0,
            "info_easy_next_ball_score": None,
        }
        if not descending_times:
            return default_metrics

        next_intercept_time = min(descending_times)
        next_intercept_xy = np.asarray(
            self.sim.ball_position[:2] + next_intercept_time * self.sim.ball_velocity[:2],
            dtype=float,
        )
        anchor_xy_error = float(np.linalg.norm(next_intercept_xy - target_xy))
        recovery_distance = float(np.linalg.norm(next_intercept_xy - self.sim.racket_position[:2]))
        reachable = anchor_xy_error <= self.next_intercept_success_radius
        recovery_speed_limit = max(self.controller_max_position_step / max(self.sim.control_dt, 1.0e-6), 1.0e-6)
        required_recovery_speed = recovery_distance / max(next_intercept_time, self.sim.control_dt)
        speed_readiness = 1.0 - np.clip(required_recovery_speed / recovery_speed_limit, 0.0, 1.0)
        zone_readiness = 1.0 - np.clip(
            anchor_xy_error / max(self.next_intercept_success_radius, 1.0e-6),
            0.0,
            1.0,
        )
        recovery_readiness = float(np.clip(0.5 * speed_readiness + 0.5 * zone_readiness, 0.0, 1.0))
        next_intercept_vertical_speed = float(self.sim.ball_velocity[2] + self._gravity_z() * next_intercept_time)
        next_intercept_speed_norm = float(
            np.linalg.norm(
                np.array(
                    [self.sim.ball_velocity[0], self.sim.ball_velocity[1], next_intercept_vertical_speed],
                    dtype=float,
                )
            )
        )
        lateral_speed = float(np.linalg.norm(self.sim.ball_velocity[:2]))
        xy_score = max(1.0 - anchor_xy_error / max(self.easy_next_ball_xy_radius, 1.0e-6), 0.0)
        time_score = max(
            1.0 - abs(next_intercept_time - _EASY_NEXT_BALL_TARGET_TIME) / _EASY_NEXT_BALL_TIME_TOLERANCE,
            0.0,
        )
        descending_score = max(
            1.0
            - abs(abs(next_intercept_vertical_speed) - _EASY_NEXT_BALL_TARGET_DESCENDING_SPEED)
            / _EASY_NEXT_BALL_TARGET_DESCENDING_SPEED,
            0.0,
        )
        lateral_speed_penalty = float(np.clip(lateral_speed / _EASY_NEXT_BALL_MAX_LATERAL_SPEED, 0.0, 1.0))
        excessive_speed_penalty = float(
            np.clip(
                max(next_intercept_speed_norm - _EASY_NEXT_BALL_SOFT_SPEED_LIMIT, 0.0)
                / _EASY_NEXT_BALL_SOFT_SPEED_LIMIT,
                0.0,
                1.0,
            )
        )
        recovery_distance_penalty = float(
            np.clip(recovery_distance / max(1.5 * self.next_intercept_success_radius, 1.0e-6), 0.0, 1.0)
        )
        easy_next_ball_score = float(
            xy_score
            + 0.75 * time_score
            + 0.5 * descending_score
            - 0.5 * lateral_speed_penalty
            - 0.25 * excessive_speed_penalty
            - 0.5 * recovery_distance_penalty
        )
        return {
            "time": float(next_intercept_time),
            "relative_xy": np.asarray(next_intercept_xy - self.sim.racket_position[:2], dtype=float),
            "reachable": bool(reachable),
            "recovery_distance": recovery_distance,
            "recovery_readiness": recovery_readiness,
            "easy_next_ball_score": easy_next_ball_score,
            "info_time": float(next_intercept_time),
            "info_x": float(next_intercept_xy[0]),
            "info_y": float(next_intercept_xy[1]),
            "info_xy_error": anchor_xy_error,
            "info_reachable": bool(reachable),
            "info_vertical_speed": next_intercept_vertical_speed,
            "info_speed_norm": next_intercept_speed_norm,
            "info_recovery_distance": recovery_distance,
            "info_recovery_readiness": recovery_readiness,
            "info_easy_next_ball_score": easy_next_ball_score,
        }

    def _target_ball_height_above_racket(self) -> float:
        return self.target_ball_height

    def _minimum_useful_apex_height(self) -> float:
        if self.require_apex_height_window_for_success:
            return float(max(self._target_ball_height_above_racket() - self.height_tolerance, 0.0))
        return self._target_ball_height_above_racket()

    def _maximum_useful_apex_height(self) -> float | None:
        if not self.require_apex_height_window_for_success:
            return None
        return float(self._target_ball_height_above_racket() + self.height_tolerance)

    def _apex_height_in_success_window(self, projected_apex_height: float) -> bool:
        if projected_apex_height + 1.0e-6 < self._minimum_useful_apex_height():
            return False
        maximum_useful_apex_height = self._maximum_useful_apex_height()
        if maximum_useful_apex_height is not None and projected_apex_height - 1.0e-6 > maximum_useful_apex_height:
            return False
        return True

    def _low_apex_contact_height_threshold(self) -> float:
        if self.low_apex_contact_height_threshold is not None:
            return self.low_apex_contact_height_threshold
        return max(self._target_ball_height_above_racket() - self.height_tolerance, 0.0)

    def _is_low_apex_contact(
        self,
        contact_trace: dict[str, object] | None,
        outgoing_trajectory_metrics: dict[str, object],
        *,
        contact_event: bool,
        success_reason: str | None,
    ) -> bool:
        if not self.terminate_on_low_apex_contact or not contact_event or success_reason is not None:
            return False
        actual_outgoing_velocity_z = outgoing_trajectory_metrics.get("actual_outgoing_velocity_z")
        if actual_outgoing_velocity_z is None:
            actual_outgoing_velocity_z = None if contact_trace is None else contact_trace.get("contact_ball_velocity_z")
        if actual_outgoing_velocity_z is None or float(actual_outgoing_velocity_z) <= self.success_velocity_threshold:
            return False
        projected_apex = self._projected_contact_apex_height_above_racket(contact_trace)
        return bool(projected_apex < self._low_apex_contact_height_threshold())

    def _failure_z_bounds(self) -> tuple[float, float]:
        dynamic_upper_bound = (
            self.sim.racket_position[2]
            + self._target_ball_height_above_racket()
            + self.height_tolerance
            + max(self.height_tolerance, 0.20)
        )
        return (-0.05, max(2.0, float(dynamic_upper_bound)))

    def _failure_reason(self) -> str | None:
        return self.sim.failure_reason(
            z_bounds=self._failure_z_bounds(),
            x_bounds=(0.0, 1.35),
            y_bounds=(-0.6, 0.6),
        )

    def _projected_apex_height_above_racket(self, ball_height_above_racket: float, ball_velocity_z: float) -> float:
        current_height = max(float(ball_height_above_racket), 0.0)
        if ball_velocity_z <= 0.0:
            return current_height
        gravity_magnitude = max(abs(self._gravity_z()), 1.0e-6)
        return current_height + float(ball_velocity_z * ball_velocity_z / (2.0 * gravity_magnitude))

    def _projected_contact_apex_height_above_racket(self, contact_trace: dict[str, object] | None) -> float:
        outgoing_velocity, _ = self._resolved_outgoing_ball_velocity(contact_trace)
        contact_ball_velocity_z = float(self.sim.ball_velocity[2]) if outgoing_velocity is None else float(outgoing_velocity[2])
        contact_ball_position_z = None if contact_trace is None else contact_trace.get("contact_ball_position_z")
        if contact_ball_position_z is not None:
            gravity_magnitude = max(abs(self._gravity_z()), 1.0e-6)
            apex_gain = (
                float(contact_ball_velocity_z * contact_ball_velocity_z / (2.0 * gravity_magnitude))
                if contact_ball_velocity_z > 0.0
                else 0.0
            )
            apex_world_z = float(contact_ball_position_z) + apex_gain
            return float(apex_world_z - self._controller_anchor_position()[2])
        contact_ball_height = self._contact_float(
            contact_trace,
            "contact_ball_height_above_racket",
            default=self._ball_height_above_racket(),
        )
        return self._projected_apex_height_above_racket(contact_ball_height, contact_ball_velocity_z)

    def _projected_contact_apex_xy(self, contact_trace: dict[str, object] | None) -> np.ndarray | None:
        if contact_trace is None:
            return None
        contact_ball_position_x = contact_trace.get("contact_ball_position_x")
        contact_ball_position_y = contact_trace.get("contact_ball_position_y")
        if contact_ball_position_x is None or contact_ball_position_y is None:
            return None
        outgoing_velocity, _ = self._resolved_outgoing_ball_velocity(contact_trace)
        if outgoing_velocity is None:
            outgoing_velocity = np.asarray(self.sim.ball_velocity, dtype=float)
        gravity_magnitude = max(abs(self._gravity_z()), 1.0e-6)
        projected_apex_time = max(float(outgoing_velocity[2]), 0.0) / gravity_magnitude
        return np.array(
            [
                float(contact_ball_position_x) + float(outgoing_velocity[0]) * projected_apex_time,
                float(contact_ball_position_y) + float(outgoing_velocity[1]) * projected_apex_time,
            ],
            dtype=float,
        )

    def _desired_outgoing_velocity(
        self,
        ball_position: np.ndarray | None = None,
        *,
        target_xy: Sequence[float] | None = None,
        target_apex_z: float | None = None,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        contact_ball_position = (
            np.asarray(self.sim.ball_position, dtype=float)
            if ball_position is None
            else np.asarray(ball_position, dtype=float)
        )
        anchor_position = self._controller_anchor_position()
        desired_target_xy = self._keepup_target_xy() if target_xy is None else np.asarray(target_xy, dtype=float)
        resolved_target_apex_z = (
            float(anchor_position[2] + self._target_ball_height_above_racket())
            if target_apex_z is None
            else float(target_apex_z)
        )
        gravity_magnitude = max(abs(self._gravity_z()), 1.0e-6)
        height_delta = max(resolved_target_apex_z - float(contact_ball_position[2]), _MIN_DESIRED_APEX_HEIGHT_DELTA)
        desired_velocity_z = float(np.sqrt(2.0 * gravity_magnitude * height_delta))
        desired_time_to_apex = desired_velocity_z / gravity_magnitude
        desired_xy_time = desired_time_to_apex
        if self.desired_outgoing_xy_mode == "next_intercept":
            target_z = self._desired_outgoing_target_z()
            descent_height = max(resolved_target_apex_z - target_z, _MIN_DESIRED_APEX_HEIGHT_DELTA)
            desired_xy_time += float(np.sqrt(2.0 * descent_height / gravity_magnitude))
        desired_velocity_xy = (desired_target_xy - contact_ball_position[:2]) / max(desired_xy_time, 1.0e-6)
        desired_velocity = np.array(
            [float(desired_velocity_xy[0]), float(desired_velocity_xy[1]), desired_velocity_z],
            dtype=float,
        )
        return desired_velocity, float(desired_time_to_apex), desired_target_xy

    def _contact_outgoing_trajectory_metrics(self, contact_trace: dict[str, object] | None) -> dict[str, object]:
        default_metrics: dict[str, object] = {
            "desired_outgoing_velocity_x": None,
            "desired_outgoing_velocity_y": None,
            "desired_outgoing_velocity_z": None,
            "actual_outgoing_velocity_x": None,
            "actual_outgoing_velocity_y": None,
            "actual_outgoing_velocity_z": None,
            "actual_outgoing_velocity_source": None,
            "outgoing_velocity_error_norm": None,
            "outgoing_velocity_xy_error": None,
            "outgoing_velocity_z_error": None,
            "contact_substep_outgoing_velocity_error_norm": None,
            "contact_substep_outgoing_velocity_xy_error": None,
            "contact_substep_outgoing_velocity_z_error": None,
            "contact_substep_predicted_apex_xy_error": None,
            "contact_substep_predicted_next_intercept_xy_error": None,
            "desired_time_to_apex": None,
            "desired_outgoing_target_x": None,
            "desired_outgoing_target_y": None,
            "desired_outgoing_target_z": None,
            "desired_outgoing_xy_mode": self.desired_outgoing_xy_mode,
            "desired_outgoing_apex_x": None,
            "desired_outgoing_apex_y": None,
            "predicted_next_intercept_x_from_actual_velocity": None,
            "predicted_next_intercept_y_from_actual_velocity": None,
            "predicted_next_intercept_time_from_actual_velocity": None,
            "predicted_next_intercept_xy_error": None,
            "predicted_apex_x_from_actual_velocity": None,
            "predicted_apex_y_from_actual_velocity": None,
            "predicted_apex_xy_error": None,
        }
        if contact_trace is None:
            return default_metrics

        contact_ball_position_x = contact_trace.get("contact_ball_position_x")
        contact_ball_position_y = contact_trace.get("contact_ball_position_y")
        contact_ball_position_z = contact_trace.get("contact_ball_position_z")
        contact_ball_velocity_x = contact_trace.get("contact_ball_velocity_x")
        contact_ball_velocity_y = contact_trace.get("contact_ball_velocity_y")
        contact_ball_velocity_z = contact_trace.get("contact_ball_velocity_z")
        if (
            contact_ball_position_x is None
            or contact_ball_position_y is None
            or contact_ball_position_z is None
            or contact_ball_velocity_x is None
            or contact_ball_velocity_y is None
            or contact_ball_velocity_z is None
        ):
            return default_metrics

        contact_ball_position = np.array(
            [
                float(contact_ball_position_x),
                float(contact_ball_position_y),
                float(contact_ball_position_z),
            ],
            dtype=float,
        )
        contact_substep_velocity = np.array(
            [
                float(contact_ball_velocity_x),
                float(contact_ball_velocity_y),
                float(contact_ball_velocity_z),
            ],
            dtype=float,
        )
        actual_velocity, actual_velocity_source = self._resolved_outgoing_ball_velocity(contact_trace)
        if actual_velocity is None:
            actual_velocity = contact_substep_velocity
            actual_velocity_source = "contact_ball_velocity"
        if self._contact_frame_action_mode() and self._contact_frame_plan_active:
            desired_velocity, desired_time_to_apex, desired_target_xy = self._contact_frame_planned_desired_velocity(
                contact_ball_position
            )
        else:
            desired_velocity, desired_time_to_apex, desired_target_xy = self._desired_outgoing_velocity(
                contact_ball_position
            )
        desired_target_z = self._desired_outgoing_target_z()
        resolved_metrics = self._trajectory_metrics_from_velocity(
            contact_ball_position,
            actual_velocity,
            desired_velocity,
            desired_time_to_apex,
            desired_target_xy,
            desired_target_z,
        )
        contact_substep_metrics = self._trajectory_metrics_from_velocity(
            contact_ball_position,
            contact_substep_velocity,
            desired_velocity,
            desired_time_to_apex,
            desired_target_xy,
            desired_target_z,
        )
        return {
            "desired_outgoing_velocity_x": float(desired_velocity[0]),
            "desired_outgoing_velocity_y": float(desired_velocity[1]),
            "desired_outgoing_velocity_z": float(desired_velocity[2]),
            "actual_outgoing_velocity_x": float(actual_velocity[0]),
            "actual_outgoing_velocity_y": float(actual_velocity[1]),
            "actual_outgoing_velocity_z": float(actual_velocity[2]),
            "actual_outgoing_velocity_source": actual_velocity_source,
            "outgoing_velocity_error_norm": resolved_metrics["outgoing_velocity_error_norm"],
            "outgoing_velocity_xy_error": resolved_metrics["outgoing_velocity_xy_error"],
            "outgoing_velocity_z_error": resolved_metrics["outgoing_velocity_z_error"],
            "contact_substep_outgoing_velocity_error_norm": contact_substep_metrics[
                "outgoing_velocity_error_norm"
            ],
            "contact_substep_outgoing_velocity_xy_error": contact_substep_metrics[
                "outgoing_velocity_xy_error"
            ],
            "contact_substep_outgoing_velocity_z_error": contact_substep_metrics[
                "outgoing_velocity_z_error"
            ],
            "contact_substep_predicted_apex_xy_error": contact_substep_metrics["predicted_apex_xy_error"],
            "contact_substep_predicted_next_intercept_xy_error": contact_substep_metrics[
                "predicted_next_intercept_xy_error"
            ],
            "desired_time_to_apex": float(desired_time_to_apex),
            "desired_outgoing_target_x": float(desired_target_xy[0]),
            "desired_outgoing_target_y": float(desired_target_xy[1]),
            "desired_outgoing_target_z": float(desired_target_z),
            "desired_outgoing_xy_mode": self.desired_outgoing_xy_mode,
            "desired_outgoing_apex_x": resolved_metrics["desired_apex_x"],
            "desired_outgoing_apex_y": resolved_metrics["desired_apex_y"],
            "predicted_next_intercept_x_from_actual_velocity": resolved_metrics["predicted_next_intercept_x"],
            "predicted_next_intercept_y_from_actual_velocity": resolved_metrics["predicted_next_intercept_y"],
            "predicted_next_intercept_time_from_actual_velocity": resolved_metrics["predicted_next_intercept_time"],
            "predicted_next_intercept_xy_error": resolved_metrics["predicted_next_intercept_xy_error"],
            "predicted_apex_x_from_actual_velocity": resolved_metrics["predicted_apex_x"],
            "predicted_apex_y_from_actual_velocity": resolved_metrics["predicted_apex_y"],
            "predicted_apex_xy_error": resolved_metrics["predicted_apex_xy_error"],
        }

    def _return_target_xy(self) -> np.ndarray:
        if self.return_target_xy_source in ("controller_anchor", "racket_home"):
            return self._keepup_target_xy()
        if self.return_target_xy_source == "racket_position":
            return np.asarray(self.sim.racket_position[:2], dtype=float)
        return np.asarray(self.controller.target_position[:2], dtype=float)

    def _return_target_xy_term(self, contact_trace: dict[str, object] | None) -> float:
        if self.useful_contact_return_target_xy_reward_weight <= 0.0:
            return 0.0
        projected_apex_xy = self._projected_contact_apex_xy(contact_trace)
        if projected_apex_xy is None:
            return 0.0
        target_xy = self._return_target_xy()
        xy_error = float(np.linalg.norm(projected_apex_xy - target_xy))
        xy_match = max(1.0 - xy_error / self.return_target_xy_tolerance, 0.0)
        return float(self.useful_contact_return_target_xy_reward_weight * xy_match)

    def _apex_match_term(self, contact_trace: dict[str, object] | None) -> float:
        projected_apex = self._projected_contact_apex_height_above_racket(contact_trace)
        height_error = abs(projected_apex - self._target_ball_height_above_racket())
        height_match = max(1.0 - height_error / self.height_tolerance, 0.0)
        return float(self.apex_match_reward_weight * height_match)

    def _apex_height_match_score(self, contact_trace: dict[str, object] | None) -> float:
        projected_apex = self._projected_contact_apex_height_above_racket(contact_trace)
        height_error = abs(projected_apex - self._target_ball_height_above_racket())
        return float(max(1.0 - height_error / self.height_tolerance, 0.0))

    def _contact_apex_under_target_penalty_term(self, contact_trace: dict[str, object] | None) -> float:
        if self.contact_apex_under_target_penalty_weight <= 0.0:
            return 0.0
        projected_apex = self._projected_contact_apex_height_above_racket(contact_trace)
        under_target_gap = max(self._minimum_useful_apex_height() - projected_apex, 0.0)
        normalized_gap = min(under_target_gap / max(self.height_tolerance, 1.0e-6), 4.0)
        return float(-self.contact_apex_under_target_penalty_weight * normalized_gap)

    def _contact_apex_progress_term(self, contact_trace: dict[str, object] | None) -> float:
        if self.contact_apex_progress_reward_weight <= 0.0:
            return 0.0
        projected_apex = self._projected_contact_apex_height_above_racket(contact_trace)
        target_apex = max(self._target_ball_height_above_racket(), 1.0e-6)
        progress = float(np.clip(projected_apex / target_apex, 0.0, 1.0))
        return float(self.contact_apex_progress_reward_weight * progress)

    def _contact_apex_recovery_progress_term(self, contact_trace: dict[str, object] | None) -> float:
        if self.contact_apex_recovery_progress_reward_weight <= 0.0:
            return 0.0
        previous_apex = self._last_projected_contact_apex_height
        if previous_apex is None:
            return 0.0
        target_apex = self._target_ball_height_above_racket()
        previous_shortfall = target_apex - previous_apex
        if previous_shortfall <= 0.0:
            return 0.0
        projected_apex = self._projected_contact_apex_height_above_racket(contact_trace)
        improvement = projected_apex - previous_apex
        if improvement <= 0.0:
            return 0.0
        normalized_improvement = min(improvement / max(self.height_tolerance, 1.0e-6), 2.0)
        return float(self.contact_apex_recovery_progress_reward_weight * normalized_improvement)

    def _contact_apex_potential_score(self, projected_apex: float) -> float:
        target_apex = self._target_ball_height_above_racket()
        shortfall = max(target_apex - projected_apex, 0.0)
        normalized_shortfall = min(shortfall / max(self.height_tolerance, 1.0e-6), self.contact_apex_potential_cap)
        return float(max(1.0 - normalized_shortfall / self.contact_apex_potential_cap, 0.0))

    def _contact_apex_potential_term(self, contact_trace: dict[str, object] | None) -> float:
        if self.contact_apex_potential_reward_weight <= 0.0:
            return 0.0
        previous_apex = self._last_projected_contact_apex_height
        if previous_apex is None:
            return 0.0
        current_apex = self._projected_contact_apex_height_above_racket(contact_trace)
        previous_score = self._contact_apex_potential_score(previous_apex)
        current_score = self._contact_apex_potential_score(current_apex)
        return float(
            self.contact_apex_potential_reward_weight
            * (self.contact_apex_potential_gamma * current_score - previous_score)
        )

    def _projected_apex_min_ratio_satisfied(
        self,
        contact_trace: dict[str, object] | None,
        min_ratio: float | None,
    ) -> bool:
        if min_ratio is None:
            return True
        target_apex = self._target_ball_height_above_racket()
        projected_apex = self._projected_contact_apex_height_above_racket(contact_trace)
        return bool(projected_apex + 1.0e-6 >= target_apex * min_ratio)

    def _contact_apex_progress_easy_next_ball_gate(
        self,
        next_intercept_metrics: dict[str, object] | None = None,
    ) -> float:
        if not self.gate_contact_apex_progress_by_easy_next_ball:
            return 1.0
        if next_intercept_metrics is None:
            next_intercept_metrics = self._next_intercept_metrics()
        easy_score = max(float(next_intercept_metrics["easy_next_ball_score"]), 0.0)
        min_easy_score = self.contact_apex_progress_min_easy_next_ball_score
        if min_easy_score is not None and easy_score < min_easy_score:
            return 0.0
        return float(np.clip(easy_score, 0.0, 1.0))

    def _contact_lateral_stability_term(self, contact_trace: dict[str, object] | None) -> float:
        if self.contact_lateral_stability_reward_weight <= 0.0:
            return 0.0
        if not self._projected_apex_min_ratio_satisfied(
            contact_trace,
            self.contact_lateral_stability_min_apex_ratio,
        ):
            return 0.0
        outgoing_velocity, _ = self._resolved_outgoing_ball_velocity(contact_trace)
        if outgoing_velocity is None:
            outgoing_velocity = np.asarray(self.sim.ball_velocity, dtype=float)
        lateral_speed = float(np.linalg.norm(np.asarray(outgoing_velocity[:2], dtype=float)))
        lateral_score = max(1.0 - lateral_speed / self.contact_lateral_stability_speed_tolerance, 0.0)
        if lateral_score <= 0.0:
            return 0.0
        projected_apex_xy = self._projected_contact_apex_xy(contact_trace)
        if projected_apex_xy is None:
            return 0.0
        xy_tolerance = (
            self.return_target_xy_tolerance
            if self.contact_lateral_stability_xy_tolerance is None
            else self.contact_lateral_stability_xy_tolerance
        )
        xy_error = float(np.linalg.norm(projected_apex_xy - self._return_target_xy()))
        xy_score = max(1.0 - xy_error / xy_tolerance, 0.0)
        return float(self.contact_lateral_stability_reward_weight * lateral_score * xy_score)

    def _update_contact_apex_memory(
        self,
        *,
        contact_event: bool,
        contact_trace: dict[str, object] | None,
    ) -> None:
        if not contact_event:
            return
        projected_apex = self._projected_contact_apex_height_above_racket(contact_trace)
        self._last_projected_contact_apex_height = projected_apex
        self._last_contact_apex_shortfall = max(self._minimum_useful_apex_height() - projected_apex, 0.0)

    def _normalized_last_contact_apex_shortfall(self) -> float:
        if self._last_contact_apex_shortfall <= 0.0:
            return 0.0
        return float(np.clip(self._last_contact_apex_shortfall / max(self.height_tolerance, 1.0e-6), 0.0, 2.0))

    def _contact_frame_low_apex_recovery_lift(self) -> float:
        if not self._contact_frame_action_mode():
            return 0.0
        if self.contact_frame_low_apex_recovery_lift_gain <= 0.0:
            return 0.0
        if self.contact_frame_low_apex_recovery_lift_max <= 0.0:
            return 0.0
        if float(self.sim.ball_velocity[2]) >= self.descending_ball_velocity_threshold:
            return 0.0
        normalized_shortfall = self._normalized_last_contact_apex_shortfall()
        if normalized_shortfall <= 0.0:
            return 0.0
        strike_readiness = self._contact_frame_strike_readiness()
        lift = self.contact_frame_low_apex_recovery_lift_gain * normalized_shortfall * strike_readiness
        return float(np.clip(lift, 0.0, self.contact_frame_low_apex_recovery_lift_max))

    def _contact_frame_low_apex_recovery_velocity(self) -> float:
        if not self._contact_frame_action_mode():
            return 0.0
        if self.contact_frame_low_apex_recovery_velocity_gain <= 0.0:
            return 0.0
        if self.contact_frame_low_apex_recovery_velocity_max <= 0.0:
            return 0.0
        if float(self.sim.ball_velocity[2]) >= self.descending_ball_velocity_threshold:
            return 0.0
        normalized_shortfall = self._normalized_last_contact_apex_shortfall()
        if normalized_shortfall <= 0.0:
            return 0.0
        strike_readiness = self._contact_frame_strike_readiness()
        velocity = self.contact_frame_low_apex_recovery_velocity_gain * normalized_shortfall * strike_readiness
        return float(np.clip(velocity, 0.0, self.contact_frame_low_apex_recovery_velocity_max))

    def _stable_contact_term(
        self,
        contact_trace: dict[str, object] | None,
        next_intercept_metrics: dict[str, object],
    ) -> float:
        if self.stable_contact_reward_weight <= 0.0:
            return 0.0
        if not self._projected_apex_min_ratio_satisfied(contact_trace, self.stable_contact_min_apex_ratio):
            return 0.0
        height_score = self._apex_height_match_score(contact_trace)
        easy_score = max(float(next_intercept_metrics["easy_next_ball_score"]), 0.0)
        return float(self.stable_contact_reward_weight * height_score * easy_score)

    def _stable_cycle_observed(
        self,
        *,
        success_reason: str | None,
        contact_event: bool,
        contact_trace: dict[str, object] | None,
        next_intercept_metrics: dict[str, object],
    ) -> bool:
        if not contact_event or success_reason is None:
            return False
        projected_apex = self._projected_contact_apex_height_above_racket(contact_trace)
        if not self._apex_height_in_success_window(projected_apex):
            return False
        if not bool(next_intercept_metrics["reachable"]):
            return False
        min_easy_score = self.stable_cycle_min_easy_next_ball_score
        if min_easy_score is not None:
            if float(next_intercept_metrics["easy_next_ball_score"]) < min_easy_score:
                return False
        return True

    def _update_stable_cycle_state(
        self,
        *,
        contact_event: bool,
        stable_cycle_observed: bool,
    ) -> None:
        if not contact_event:
            return
        if stable_cycle_observed:
            self.stable_cycle_count += 1
            self._consecutive_stable_cycle_count += 1
            return
        self._consecutive_stable_cycle_count = 0

    def _stable_cycle_term(
        self,
        *,
        stable_cycle_observed: bool,
        consecutive_stable_cycle_count: int | None = None,
    ) -> float:
        if self.stable_cycle_reward_weight <= 0.0 or not stable_cycle_observed:
            return 0.0
        count = (
            self._consecutive_stable_cycle_count
            if consecutive_stable_cycle_count is None
            else int(consecutive_stable_cycle_count)
        )
        if count <= 0:
            return 0.0
        capped_count = min(count, self.stable_cycle_reward_cap)
        streak_scale = 1.0 + 0.5 * float(capped_count - 1)
        return float(self.stable_cycle_reward_weight * streak_scale)

    def _normalized_tilt_magnitude(self) -> float:
        if self.action_mode not in ("position_tilt", *_STRIKE_CONTRACT_ACTION_MODES):
            return 0.0
        normalized_tilt = self.controller.target_tilt / np.maximum(self.target_tilt_limit, 1.0e-6)
        return float(np.linalg.norm(normalized_tilt) / np.sqrt(2.0))

    def _normalized_tilt_action_delta(self, action: np.ndarray) -> float:
        if self.action_mode not in _TILT_ACTION_MODES:
            return 0.0
        tilt_action = action[3:5] if self.action_mode in _TILT_SLICE_3_TO_5_ACTION_MODES else action[3:]
        previous_tilt_action = self._previous_action[3:5] if self.action_mode in _TILT_SLICE_3_TO_5_ACTION_MODES else self._previous_action[3:]
        tilt_delta = tilt_action - previous_tilt_action
        return float(np.linalg.norm(tilt_delta) / max(np.sqrt(2.0) * self.tilt_action_limit, 1.0e-6))

    def _normalized_action_norm(self, action: np.ndarray) -> float:
        normalized_action = np.asarray(action, dtype=float) / np.maximum(self.action_high, 1.0e-6)
        return float(np.linalg.norm(normalized_action) / np.sqrt(float(self.action_size)))

    def _constrained_target_tilt(self, tilt: np.ndarray) -> np.ndarray:
        if self.action_mode not in ("position_tilt", *_STRIKE_CONTRACT_ACTION_MODES):
            return np.asarray(tilt, dtype=float)
        constrained_tilt = np.asarray(tilt, dtype=float).copy()
        if self.target_pitch_range is not None:
            constrained_tilt[0] = np.clip(constrained_tilt[0], self.target_pitch_range[0], self.target_pitch_range[1])
        return constrained_tilt

    def _preparation_target_height_above_racket(self) -> float:
        return float(np.clip(min(self._target_ball_height_above_racket(), 0.18), 0.12, 0.18))

    def _tracking_term(self) -> float:
        if float(self.sim.ball_velocity[2]) >= self.descending_ball_velocity_threshold:
            return 0.0
        ball_height_above_racket = self._ball_height_above_racket()
        vertical_error = abs(ball_height_above_racket - self._preparation_target_height_above_racket())
        if vertical_error > self.strike_zone_height_tolerance:
            return 0.0
        vertical_score = max(1.0 - vertical_error / self.strike_zone_height_tolerance, 0.0)
        xy_score = max(1.0 - self._tracking_alignment_error() / self.strike_zone_xy_radius, 0.0)
        return float(self.tracking_reward_weight * xy_score * vertical_score)

    def _success_reason(
        self,
        failure_reason: str | None,
        contact_trace: dict[str, object],
        contact_event: bool,
    ) -> str | None:
        if failure_reason is not None or not contact_event:
            return None
        outgoing_velocity, _ = self._resolved_outgoing_ball_velocity(contact_trace)
        contact_ball_velocity_z = float(self.sim.ball_velocity[2]) if outgoing_velocity is None else float(outgoing_velocity[2])
        contact_racket_velocity_z = self._contact_float(
            contact_trace,
            "contact_racket_velocity_z",
            float(self.sim.racket_velocity[2]),
        )
        contact_xy_alignment_error = self._contact_float(
            contact_trace,
            "contact_xy_alignment_error",
            self._xy_alignment_error(),
        )
        if contact_ball_velocity_z <= self.success_velocity_threshold:
            return None
        if contact_racket_velocity_z <= self.min_upward_racket_velocity_z:
            return None
        if contact_xy_alignment_error > self.contact_centering_radius:
            return None
        if self.max_contact_racket_lateral_speed_for_success is not None:
            if self._contact_racket_lateral_speed(contact_trace) > self.max_contact_racket_lateral_speed_for_success:
                return None
        projected_apex_height = self._projected_contact_apex_height_above_racket(contact_trace)
        if not self._apex_height_in_success_window(projected_apex_height):
            return None
        next_intercept_metrics = self._next_intercept_metrics()
        if self.require_reachable_next_intercept_for_success and not bool(next_intercept_metrics["reachable"]):
            return None
        if self.min_easy_next_ball_score_for_success is not None:
            if float(next_intercept_metrics["easy_next_ball_score"]) < self.min_easy_next_ball_score_for_success:
                return None
        return "useful_keepup_bounce"

    def _reward_terms(
        self,
        failure_reason: str | None,
        success_reason: str | None,
        contact_event: bool,
        contact_active: bool,
        applied_action: np.ndarray,
        contact_trace: dict[str, object],
        outgoing_trajectory_metrics: dict[str, object] | None = None,
        *,
        stable_cycle_observed: bool = False,
        consecutive_stable_cycle_count: int | None = None,
    ) -> dict[str, float]:
        if outgoing_trajectory_metrics is None:
            outgoing_trajectory_metrics = self._contact_outgoing_trajectory_metrics(contact_trace)
        tracking_scale = self.tracking_during_contact_scale if contact_active else 1.0
        reward_terms = {
            "tracking_term": tracking_scale * self._tracking_term(),
            "contact_bonus": 0.0,
            "apex_match_term": 0.0,
            "return_target_xy_term": 0.0,
            "next_intercept_reachable_bonus": 0.0,
            "easy_next_ball_term": 0.0,
            "trajectory_match_term": 0.0,
            "trajectory_error_penalty": 0.0,
            "next_intercept_xy_error_penalty": 0.0,
            "post_contact_lateral_velocity_penalty": 0.0,
            "contact_xy_error_penalty": 0.0,
            "contact_racket_lateral_velocity_penalty": 0.0,
            "contact_racket_outward_velocity_penalty": 0.0,
            "nonuseful_contact_penalty": 0.0,
            "contact_apex_under_target_penalty": 0.0,
            "contact_apex_progress_term": 0.0,
            "contact_apex_recovery_progress_term": 0.0,
            "contact_apex_potential_term": 0.0,
            "contact_lateral_stability_term": 0.0,
            "stable_contact_term": 0.0,
            "stable_cycle_term": 0.0,
            "outgoing_x_term": 0.0,
            "failure_penalty": 0.0,
            "contact_frame_action_penalty": 0.0,
            "tilt_angle_penalty": 0.0,
            "tilt_action_delta_penalty": 0.0,
        }
        if self.action_mode in _TILT_ACTION_MODES:
            reward_terms["tilt_angle_penalty"] = -self.tilt_angle_penalty_weight * self._normalized_tilt_magnitude()
            reward_terms["tilt_action_delta_penalty"] = (
                -self.tilt_action_delta_penalty_weight * self._normalized_tilt_action_delta(applied_action)
            )
        if self._contact_frame_action_mode() and self.contact_frame_action_penalty_weight > 0.0:
            reward_terms["contact_frame_action_penalty"] = -self.contact_frame_action_penalty_weight * float(
                self._normalized_action_norm(applied_action)
            )
        if contact_event and success_reason is not None:
            next_intercept_metrics = self._next_intercept_metrics()
            reward_terms["contact_bonus"] = self.contact_bonus
            reward_terms["apex_match_term"] = self._apex_match_term(contact_trace)
            reward_terms["return_target_xy_term"] = self._return_target_xy_term(contact_trace)
            if bool(next_intercept_metrics["reachable"]):
                reward_terms["next_intercept_reachable_bonus"] = self.next_intercept_reachable_bonus_weight
            reward_terms["easy_next_ball_term"] = self.easy_next_ball_reward_weight * max(
                float(next_intercept_metrics["easy_next_ball_score"]),
                0.0,
            )
        if contact_event:
            actual_outgoing_velocity_z = outgoing_trajectory_metrics.get("actual_outgoing_velocity_z")
            if actual_outgoing_velocity_z is None:
                actual_outgoing_velocity_z = contact_trace.get("contact_ball_velocity_z")
            if actual_outgoing_velocity_z is None:
                actual_outgoing_velocity_z = float(self.sim.ball_velocity[2])
            if float(actual_outgoing_velocity_z) > self.success_velocity_threshold:
                next_intercept_metrics: dict[str, object] | None = None
                if self.gate_contact_apex_progress_by_easy_next_ball:
                    next_intercept_metrics = self._next_intercept_metrics()
                apex_progress_gate = self._contact_apex_progress_easy_next_ball_gate(next_intercept_metrics)
                reward_terms["contact_apex_progress_term"] = (
                    apex_progress_gate * self._contact_apex_progress_term(contact_trace)
                )
                reward_terms["contact_apex_recovery_progress_term"] = (
                    apex_progress_gate * self._contact_apex_recovery_progress_term(contact_trace)
                )
                reward_terms["contact_apex_potential_term"] = self._contact_apex_potential_term(contact_trace)
                if success_reason is None and self.reward_contact_quality_on_any_upward_contact:
                    next_intercept_metrics = self._next_intercept_metrics()
                    easy_score_scale = (
                        self._apex_height_match_score(contact_trace)
                        if self.gate_nonuseful_easy_next_ball_by_apex
                        else 1.0
                    )
                    reward_terms["apex_match_term"] = self._apex_match_term(contact_trace)
                    reward_terms["easy_next_ball_term"] = self.easy_next_ball_reward_weight * max(
                        float(next_intercept_metrics["easy_next_ball_score"]),
                        0.0,
                    ) * easy_score_scale
                if self.stable_contact_reward_weight > 0.0:
                    if next_intercept_metrics is None:
                        next_intercept_metrics = self._next_intercept_metrics()
                    reward_terms["stable_contact_term"] = self._stable_contact_term(
                        contact_trace,
                        next_intercept_metrics,
                    )
                reward_terms["contact_lateral_stability_term"] = self._contact_lateral_stability_term(contact_trace)
                reward_terms["stable_cycle_term"] = self._stable_cycle_term(
                    stable_cycle_observed=stable_cycle_observed,
                    consecutive_stable_cycle_count=consecutive_stable_cycle_count,
                )
                if self.next_intercept_xy_error_penalty_weight > 0.0:
                    if next_intercept_metrics is None:
                        next_intercept_metrics = self._next_intercept_metrics()
                    next_intercept_xy_error = next_intercept_metrics["info_xy_error"]
                    if next_intercept_xy_error is not None:
                        normalized_error = min(
                            float(next_intercept_xy_error) / max(self.next_intercept_success_radius, 1.0e-6),
                            4.0,
                        )
                        reward_terms["next_intercept_xy_error_penalty"] = (
                            -self.next_intercept_xy_error_penalty_weight * normalized_error
                        )
                if self.post_contact_lateral_velocity_penalty_weight > 0.0:
                    actual_outgoing_velocity_x = outgoing_trajectory_metrics.get("actual_outgoing_velocity_x")
                    actual_outgoing_velocity_y = outgoing_trajectory_metrics.get("actual_outgoing_velocity_y")
                    if actual_outgoing_velocity_x is None:
                        actual_outgoing_velocity_x = contact_trace.get("contact_ball_velocity_x", 0.0)
                    if actual_outgoing_velocity_y is None:
                        actual_outgoing_velocity_y = contact_trace.get("contact_ball_velocity_y", 0.0)
                    lateral_speed = float(
                        np.linalg.norm(
                            np.array(
                                [float(actual_outgoing_velocity_x), float(actual_outgoing_velocity_y)],
                                dtype=float,
                            )
                        )
                    )
                    normalized_lateral_speed = min(lateral_speed / max(_EASY_NEXT_BALL_MAX_LATERAL_SPEED, 1.0e-6), 4.0)
                    reward_terms["post_contact_lateral_velocity_penalty"] = (
                        -self.post_contact_lateral_velocity_penalty_weight * normalized_lateral_speed
                    )
                if self.contact_xy_error_penalty_weight > 0.0:
                    contact_xy_alignment_error = self._contact_float(
                        contact_trace,
                        "contact_xy_alignment_error",
                        self._xy_alignment_error(),
                    )
                    normalized_contact_error = min(
                        contact_xy_alignment_error / max(self.contact_centering_radius, 1.0e-6),
                        4.0,
                    )
                    reward_terms["contact_xy_error_penalty"] = (
                        -self.contact_xy_error_penalty_weight * normalized_contact_error
                    )
                if self.contact_racket_lateral_velocity_penalty_weight > 0.0:
                    lateral_speed = self._contact_racket_lateral_speed(contact_trace)
                    excess_lateral_speed = max(lateral_speed - self.contact_racket_lateral_velocity_tolerance, 0.0)
                    normalized_lateral_speed = min(
                        excess_lateral_speed / max(self.contact_racket_lateral_velocity_tolerance, 1.0e-6),
                        4.0,
                    )
                    reward_terms["contact_racket_lateral_velocity_penalty"] = (
                        -self.contact_racket_lateral_velocity_penalty_weight * normalized_lateral_speed
                    )
                reward_terms["contact_racket_outward_velocity_penalty"] = (
                    self._contact_racket_outward_velocity_penalty_term(contact_trace)
                )
                reward_terms["contact_apex_under_target_penalty"] = (
                    self._contact_apex_under_target_penalty_term(contact_trace)
                )
        if contact_event and success_reason is None and self.nonuseful_contact_penalty_weight > 0.0:
            reward_terms["nonuseful_contact_penalty"] = -self.nonuseful_contact_penalty_weight
        if contact_event and self.useful_contact_outgoing_x_penalty_weight > 0.0:
            actual_outgoing_velocity_z = outgoing_trajectory_metrics.get("actual_outgoing_velocity_z")
            if actual_outgoing_velocity_z is None:
                actual_outgoing_velocity_z = contact_trace.get("contact_ball_velocity_z")
            if actual_outgoing_velocity_z is not None and float(actual_outgoing_velocity_z) > 0.0:
                actual_outgoing_velocity_x = outgoing_trajectory_metrics.get("actual_outgoing_velocity_x")
                if actual_outgoing_velocity_x is None:
                    actual_outgoing_velocity_x = contact_trace.get("contact_ball_velocity_x", 0.0)
                contact_ball_velocity_x = float(actual_outgoing_velocity_x)
                outward_x_error = max(
                    contact_ball_velocity_x - self.desired_outgoing_ball_velocity_x,
                    0.0,
                )
                reward_terms["outgoing_x_term"] = (
                    -self.useful_contact_outgoing_x_penalty_weight * outward_x_error
                )
        if contact_event and self.trajectory_match_reward_weight > 0.0:
            actual_outgoing_velocity_z = outgoing_trajectory_metrics.get("actual_outgoing_velocity_z")
            outgoing_velocity_error_norm = outgoing_trajectory_metrics.get("outgoing_velocity_error_norm")
            if (
                actual_outgoing_velocity_z is not None
                and float(actual_outgoing_velocity_z) > 0.0
                and outgoing_velocity_error_norm is not None
            ):
                reward_terms["trajectory_match_term"] = self.trajectory_match_reward_weight * float(
                    np.exp(-float(outgoing_velocity_error_norm) / _TRAJECTORY_MATCH_ERROR_SCALE)
                )
        if contact_event and self.trajectory_error_penalty_weight > 0.0:
            actual_outgoing_velocity_z = outgoing_trajectory_metrics.get("actual_outgoing_velocity_z")
            outgoing_velocity_error_norm = outgoing_trajectory_metrics.get("outgoing_velocity_error_norm")
            if (
                actual_outgoing_velocity_z is not None
                and float(actual_outgoing_velocity_z) > 0.0
                and outgoing_velocity_error_norm is not None
            ):
                normalized_error = min(float(outgoing_velocity_error_norm) / _TRAJECTORY_MATCH_ERROR_SCALE, 3.0)
                reward_terms["trajectory_error_penalty"] = -self.trajectory_error_penalty_weight * normalized_error
        if failure_reason == "floor_contact":
            reward_terms["failure_penalty"] = self.floor_penalty
        elif failure_reason == "robot_body_contact":
            reward_terms["failure_penalty"] = self.robot_body_contact_penalty
        elif failure_reason is not None:
            reward_terms["failure_penalty"] = self.failure_penalty
        return reward_terms

    def _sample_reset_xy_offset(self) -> np.ndarray:
        if self.reset_xy_range <= 0.0:
            return np.zeros(2, dtype=float)
        if self.reset_xy_sampling == "disk":
            radius = self.reset_xy_range * float(np.sqrt(self._rng.uniform(0.0, 1.0)))
            angle = float(self._rng.uniform(0.0, 2.0 * np.pi))
            return np.array([radius * np.cos(angle), radius * np.sin(angle)], dtype=float)
        return self._rng.uniform(-self.reset_xy_range, self.reset_xy_range, size=2)

    def _sample_reset_ball_height(self) -> float:
        if self.reset_ball_height_bounds is not None:
            return float(self._rng.uniform(self.reset_ball_height_bounds[0], self.reset_ball_height_bounds[1]))
        if self.reset_ball_height_range <= 0.0:
            return self.ball_height
        return float(self.ball_height + self._rng.uniform(-self.reset_ball_height_range, self.reset_ball_height_range))

    def _sample_reset_velocity(self) -> np.ndarray:
        velocity = np.zeros(3, dtype=float)
        if self.reset_velocity_xy_range > 0.0:
            velocity[:2] = self._rng.uniform(-self.reset_velocity_xy_range, self.reset_velocity_xy_range, size=2)
        velocity[2] = self._rng.uniform(self.reset_velocity_z_range[0], self.reset_velocity_z_range[1])
        return velocity

    def _sample_reset_ball_angular_velocity(self) -> np.ndarray:
        if self.reset_ball_angular_velocity_range <= 0.0:
            return np.zeros(3, dtype=float)
        return self._rng.uniform(
            -self.reset_ball_angular_velocity_range,
            self.reset_ball_angular_velocity_range,
            size=3,
        )

    def _controller_anchor_position(self) -> np.ndarray:
        anchor_position = getattr(self.controller, "_anchor_position", None)
        if anchor_position is None:
            return np.asarray(self.sim.racket_position, dtype=float)
        return np.asarray(anchor_position, dtype=float)

    def _keepup_target_xy(self) -> np.ndarray:
        return np.asarray(self._controller_anchor_position()[:2], dtype=float) + self.keepup_target_xy_offset

    def _pre_contact_readiness(self) -> float:
        if float(self.sim.ball_velocity[2]) >= self.descending_ball_velocity_threshold:
            return 0.0
        preparation_height = self._preparation_target_height_above_racket()
        activation_height = float(np.clip(preparation_height + 0.08, 0.16, 0.22))
        ball_height = self._ball_height_above_racket()
        if ball_height >= activation_height:
            return 0.0
        height_score = 1.0 - np.clip(
            (ball_height - preparation_height) / max(activation_height - preparation_height, 1.0e-6),
            0.0,
            1.0,
        )
        xy_score = max(1.0 - self._tracking_alignment_error() / self.strike_zone_xy_radius, 0.0)
        return float(np.clip(min(height_score, xy_score), 0.0, 1.0))

    def _pre_contact_height_readiness(self) -> float:
        if float(self.sim.ball_velocity[2]) >= self.descending_ball_velocity_threshold:
            return 0.0
        preparation_height = self._preparation_target_height_above_racket()
        activation_height = float(np.clip(preparation_height + 0.08, 0.16, 0.22))
        ball_height = self._ball_height_above_racket()
        if ball_height >= activation_height:
            return 0.0
        return float(
            1.0
            - np.clip(
                (ball_height - preparation_height) / max(activation_height - preparation_height, 1.0e-6),
                0.0,
                1.0,
            )
        )

    def _pre_contact_upward_ready(self) -> bool:
        return self._pre_contact_readiness() >= 0.95

    def _pre_contact_xy_limit(self) -> float:
        full_xy_limit = min(
            float(np.min(self.target_offset_high[:2])),
            float(np.min(-self.target_offset_low[:2])),
            self.strike_zone_xy_radius + self.contact_centering_radius,
        )
        base_xy_limit = min(max(self.contact_centering_radius, 0.4 * full_xy_limit), full_xy_limit)
        height_catchup_weight = 1.0 if self.action_mode in ("position", *_STRIKE_CONTRACT_ACTION_MODES) else 0.5
        readiness = max(self._pre_contact_readiness(), height_catchup_weight * self._pre_contact_height_readiness())
        return float(base_xy_limit + readiness * (full_xy_limit - base_xy_limit))

    def _pre_contact_xy_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        pre_contact_xy_limit = self._pre_contact_xy_limit()
        low = np.full(2, -pre_contact_xy_limit, dtype=float)
        high = np.full(2, pre_contact_xy_limit, dtype=float)
        return low, high

    def _strike_lift_feedforward(self) -> float:
        if float(self.sim.ball_velocity[2]) >= self.descending_ball_velocity_threshold:
            return 0.0
        height_readiness = self._pre_contact_height_readiness()
        if height_readiness <= 0.0:
            return 0.0
        intercept_time = (
            self._contact_frame_plan_intercept_time
            if self._contact_frame_plan_active
            else self._predicted_intercept_time()
        )
        urgency = 1.0 - np.clip(intercept_time / 0.12, 0.0, 1.0)
        return float(np.clip(0.04 * height_readiness * urgency, 0.0, 0.04))

    def _strike_tilt_assist_target(self) -> np.ndarray:
        if self.action_mode not in _STRIKE_CONTRACT_ACTION_MODES or self.strike_tilt_assist_limit is None:
            return np.zeros(2, dtype=float)
        if self._contact_active_previous_step:
            return np.zeros(2, dtype=float)
        if float(self.sim.ball_velocity[2]) >= self.descending_ball_velocity_threshold:
            return np.zeros(2, dtype=float)

        correction_xy = self._keepup_target_xy() - self._predicted_intercept_xy()
        axis_active = np.abs(correction_xy) > self.strike_tilt_assist_deadband
        if not np.any(axis_active):
            return np.zeros(2, dtype=float)

        height_readiness = self._pre_contact_height_readiness()
        intercept_time = (
            self._contact_frame_plan_intercept_time
            if self._contact_frame_plan_active
            else self._predicted_intercept_time()
        )
        urgency = 1.0 - np.clip(intercept_time / self.contact_frame_tilt_ramp_time, 0.0, 1.0)
        ramp = float(np.clip(max(height_readiness, urgency), 0.0, 1.0))
        pitch = 0.0
        roll = 0.0
        if axis_active[0]:
            pitch = self.strike_tilt_assist_limit[0] * ramp * float(np.sign(correction_xy[0]))
        if axis_active[1]:
            roll = -self.strike_tilt_assist_limit[1] * ramp * float(np.sign(correction_xy[1]))
        return np.array([pitch, roll], dtype=float)

    def _strike_tilt_ramp_target(self) -> np.ndarray:
        if self.action_mode not in _STRIKE_CONTRACT_ACTION_MODES or self.strike_tilt_ramp_pitch is None:
            return np.zeros(2, dtype=float)
        if self._contact_active_previous_step:
            return np.zeros(2, dtype=float)
        if float(self.sim.ball_velocity[2]) >= self.descending_ball_velocity_threshold:
            return np.zeros(2, dtype=float)
        if self._xy_alignment_error() > self.strike_tilt_ramp_xy_tolerance:
            return np.zeros(2, dtype=float)

        height_readiness = self._pre_contact_height_readiness()
        if height_readiness <= 0.0:
            return np.zeros(2, dtype=float)
        intercept_time = (
            self._contact_frame_plan_intercept_time
            if self._contact_frame_plan_active
            else self._predicted_intercept_time()
        )
        urgency = 1.0 - np.clip(intercept_time / self.contact_frame_tilt_ramp_time, 0.0, 1.0)
        ramp = float(np.clip(max(height_readiness, urgency), 0.0, 1.0))
        return np.array([self.strike_tilt_ramp_pitch * ramp, 0.0], dtype=float)

    def _followup_strike_contract_active(self) -> bool:
        return self.action_mode in _STRIKE_CONTRACT_ACTION_MODES and self.successful_bounce_count > 0

    def _centered_contact_target_xy(
        self,
        contact_xy: Sequence[float],
        *,
        offset_ratio: float,
        offset_max: float,
    ) -> np.ndarray:
        resolved_contact_xy = np.asarray(contact_xy, dtype=float)
        if offset_ratio <= 0.0 or offset_max <= 0.0:
            return resolved_contact_xy
        anchor_xy = self._keepup_target_xy()
        correction_xy = anchor_xy - resolved_contact_xy
        correction_norm = float(np.linalg.norm(correction_xy))
        if correction_norm <= 1.0e-9:
            return resolved_contact_xy
        offset_distance = min(
            float(offset_max),
            float(offset_ratio) * correction_norm,
            0.75 * self.contact_centering_radius,
        )
        return resolved_contact_xy + offset_distance * correction_xy / correction_norm

    def _followup_strike_target_xy(self, predicted_intercept_xy: np.ndarray) -> np.ndarray:
        if not self._followup_strike_contract_active():
            return np.asarray(predicted_intercept_xy, dtype=float)
        return self._centered_contact_target_xy(
            predicted_intercept_xy,
            offset_ratio=self.followup_strike_contact_offset_ratio,
            offset_max=self.followup_strike_contact_offset_max,
        )

    def _contact_frame_planner_contact_target_xy(self, planned_contact_xy: Sequence[float]) -> np.ndarray:
        if not self._contact_frame_plan_active:
            return np.asarray(planned_contact_xy, dtype=float)
        return self._centered_contact_target_xy(
            planned_contact_xy,
            offset_ratio=self.contact_frame_planner_contact_offset_ratio,
            offset_max=self.contact_frame_planner_contact_offset_max,
        )

    def _strike_contact_target_xy(self) -> np.ndarray:
        predicted_intercept_xy = self._predicted_intercept_xy()
        if float(self.sim.ball_velocity[2]) >= self.descending_ball_velocity_threshold:
            return predicted_intercept_xy
        return self._followup_strike_target_xy(predicted_intercept_xy)

    def _followup_strike_base_tilt(self) -> np.ndarray:
        if not self._followup_strike_contract_active() or self.followup_strike_target_tilt is None:
            return np.zeros(2, dtype=float)
        return np.asarray(self.followup_strike_target_tilt, dtype=float)

    def _position_strike_target_tilt(self) -> np.ndarray:
        target_tilt = self._followup_strike_base_tilt()
        if self.initial_target_tilt is not None and not self._followup_strike_contract_active():
            target_tilt = target_tilt + np.asarray(self.initial_target_tilt, dtype=float)
        if self.strike_tilt_ramp_pitch is not None:
            target_tilt = target_tilt + self._strike_tilt_ramp_target()
        elif self.strike_tilt_assist_limit is not None:
            target_tilt = target_tilt + self._strike_tilt_assist_target()
        return self._constrained_target_tilt(target_tilt)

    def _post_contact_return_target_xy(self) -> np.ndarray:
        anchor_xy = self._keepup_target_xy()
        if self.action_mode not in _STRIKE_CONTRACT_ACTION_MODES:
            return anchor_xy
        if self.post_contact_return_assist_weight <= 0.0:
            return anchor_xy
        if self.successful_bounce_count <= 0:
            return anchor_xy
        if float(self.sim.ball_velocity[2]) <= 0.0:
            return anchor_xy
        if not self.post_contact_return_predict_during_rise:
            return anchor_xy
        predicted_return_xy = self._predicted_intercept_xy(
            max_intercept_time=self.post_contact_return_max_intercept_time
        )
        assist_weight = float(np.clip(self.post_contact_return_assist_weight, 0.0, 1.0))
        return (1.0 - assist_weight) * anchor_xy + assist_weight * predicted_return_xy

    def _effective_followup_strike_lift_boost(self, followup_lift_residual: float = 0.0) -> float:
        if self.action_mode != "position_strike_tilt_lift":
            return self.followup_strike_lift_boost
        max_followup_lift_boost = self.followup_strike_lift_boost + self.followup_lift_action_limit
        return float(np.clip(self.followup_strike_lift_boost + followup_lift_residual, 0.0, max_followup_lift_boost))

    def _strike_action_target_position(
        self,
        action: Sequence[float],
        *,
        followup_lift_residual: float = 0.0,
    ) -> np.ndarray:
        action_array = np.asarray(action, dtype=float)
        if action_array.shape != (3,):
            raise ValueError(f"Strike action must have shape (3,), got {action_array.shape}.")
        anchor_position = self._controller_anchor_position()
        target_position = anchor_position.copy()
        if float(self.sim.ball_velocity[2]) >= self.descending_ball_velocity_threshold:
            target_position[:2] = self._post_contact_return_target_xy()
            return target_position + action_array
        target_position[:2] = self._strike_contact_target_xy() + action_array[:2]
        lift_target = self._strike_lift_feedforward()
        if self._followup_strike_contract_active():
            followup_lift_boost = self._effective_followup_strike_lift_boost(followup_lift_residual)
            lift_target += followup_lift_boost * max(self._pre_contact_height_readiness(), 0.0)
        target_position[2] = anchor_position[2] + lift_target + action_array[2]
        return target_position

    def _contact_frame_basis_xy(self) -> tuple[np.ndarray, np.ndarray, str]:
        predicted_intercept_xy = (
            self._contact_frame_plan_contact_position[:2]
            if self._contact_frame_plan_active
            else self._predicted_intercept_xy()
        )
        anchor_xy = self._keepup_target_xy()
        radial = anchor_xy - predicted_intercept_xy
        frame_source = "anchor_minus_intercept"
        radial_norm = float(np.linalg.norm(radial))
        if radial_norm <= 1.0e-6:
            incoming_xy = -np.asarray(self.sim.ball_velocity[:2], dtype=float)
            incoming_norm = float(np.linalg.norm(incoming_xy))
            if incoming_norm > 1.0e-6:
                radial = incoming_xy
                radial_norm = incoming_norm
                frame_source = "negative_ball_velocity"
        if radial_norm <= 1.0e-6:
            radial = anchor_xy - np.asarray(self.sim.ball_position[:2], dtype=float)
            radial_norm = float(np.linalg.norm(radial))
            frame_source = "anchor_minus_ball"
        if radial_norm <= 1.0e-6:
            radial = np.array([1.0, 0.0], dtype=float)
            radial_norm = 1.0
            frame_source = "world_x_fallback"
        radial = radial / radial_norm
        tangent = np.array([-radial[1], radial[0]], dtype=float)
        return radial, tangent, frame_source

    def _contact_frame_base_strike_lift(self) -> float:
        if not self._contact_frame_action_mode():
            return 0.0
        if float(self.sim.ball_velocity[2]) >= self.descending_ball_velocity_threshold:
            return 0.0
        intercept_time = (
            self._contact_frame_plan_intercept_time
            if self._contact_frame_plan_active
            else self._predicted_intercept_time()
        )
        urgency = 1.0 - np.clip(
            intercept_time / max(self.contact_frame_base_strike_time_horizon, 1.0e-6),
            0.0,
            1.0,
        )
        strike_readiness = max(self._pre_contact_height_readiness(), urgency)
        base_lift = float(
            self.contact_frame_base_strike_z_offset
            + self.contact_frame_base_strike_z_boost * np.clip(strike_readiness, 0.0, 1.0)
        )
        return (
            base_lift
            + self._contact_frame_apex_lift()
            + self._contact_frame_low_apex_recovery_lift()
            + self._contact_frame_velocity_lead()
        )

    def _contact_frame_apex_lift(self) -> float:
        if not self._contact_frame_action_mode():
            return 0.0
        if self.contact_frame_apex_lift_gain <= 0.0 or self.contact_frame_apex_lift_max <= 0.0:
            return 0.0
        if float(self.sim.ball_velocity[2]) >= self.descending_ball_velocity_threshold:
            return 0.0

        contact_position = self._contact_frame_planned_contact_position()
        desired_velocity, _, _ = self._contact_frame_controller_desired_velocity(contact_position)
        anchor_position = self._controller_anchor_position()
        nominal_contact_position = anchor_position.copy()
        nominal_contact_position[2] = float(anchor_position[2] + self._tracking_strike_plane_offset())
        nominal_desired_velocity, _, _ = self._desired_outgoing_velocity(nominal_contact_position)

        restitution = self.contact_frame_apex_lift_restitution
        required_racket_velocity_z = (
            float(desired_velocity[2]) + restitution * min(float(self.sim.ball_velocity[2]), 0.0)
        ) / max(1.0 + restitution, 1.0e-6)
        nominal_required_racket_velocity_z = (
            float(nominal_desired_velocity[2]) + restitution * self.contact_frame_apex_lift_reference_velocity_z
        ) / max(1.0 + restitution, 1.0e-6)
        velocity_excess = max(required_racket_velocity_z - nominal_required_racket_velocity_z, 0.0)

        intercept_time = (
            self._contact_frame_plan_intercept_time
            if self._contact_frame_plan_active
            else self._predicted_intercept_time()
        )
        urgency = 1.0 - np.clip(
            intercept_time / max(self.contact_frame_base_strike_time_horizon, 1.0e-6),
            0.0,
            1.0,
        )
        strike_readiness = max(self._pre_contact_height_readiness(), urgency)
        lift = self.contact_frame_apex_lift_gain * velocity_excess * np.clip(strike_readiness, 0.0, 1.0)
        return float(np.clip(lift, 0.0, self.contact_frame_apex_lift_max))

    def _required_contact_frame_racket_velocity_z(self) -> float:
        contact_position = self._contact_frame_planned_contact_position()
        desired_velocity, _, _ = self._contact_frame_controller_desired_velocity(contact_position)
        restitution = self.contact_frame_apex_lift_restitution
        incoming_velocity_z = min(float(self.sim.ball_velocity[2]), 0.0)
        return float((float(desired_velocity[2]) + restitution * incoming_velocity_z) / max(1.0 + restitution, 1.0e-6))

    def _required_contact_frame_racket_velocity(
        self,
        contact_position: Sequence[float] | None = None,
        desired_velocity: Sequence[float] | None = None,
        face_normal: Sequence[float] | None = None,
    ) -> np.ndarray:
        resolved_contact_position = (
            self._contact_frame_planned_contact_position()
            if contact_position is None
            else np.asarray(contact_position, dtype=float)
        )
        resolved_desired_velocity = (
            self._contact_frame_controller_desired_velocity(resolved_contact_position)[0]
            if desired_velocity is None
            else np.asarray(desired_velocity, dtype=float)
        )
        incoming_velocity = np.asarray(self.sim.ball_velocity, dtype=float)
        normal = -np.asarray(
            self.controller.target_face_normal if face_normal is None else face_normal,
            dtype=float,
        )
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1.0e-9:
            normal = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            normal = normal / normal_norm

        restitution = self.contact_frame_apex_lift_restitution
        incoming_normal_velocity = min(float(np.dot(incoming_velocity, normal)), 0.0)
        desired_normal_velocity = float(np.dot(resolved_desired_velocity, normal))
        required_normal_velocity = (
            desired_normal_velocity + restitution * incoming_normal_velocity
        ) / max(1.0 + restitution, 1.0e-6)
        return normal * required_normal_velocity

    def _contact_frame_intercept_velocity_target(self, target_position: Sequence[float] | None = None) -> np.ndarray:
        if not self._contact_frame_action_mode():
            return np.zeros(3, dtype=float)
        if self.contact_frame_intercept_velocity_gain <= 0.0 or self.contact_frame_intercept_velocity_max <= 0.0:
            return np.zeros(3, dtype=float)
        if float(self.sim.ball_velocity[2]) >= self.descending_ball_velocity_threshold:
            return np.zeros(3, dtype=float)

        intercept_time = (
            self._contact_frame_plan_intercept_time
            if self._contact_frame_plan_active
            else self._predicted_intercept_time(max_intercept_time=self.next_intercept_max_time)
        )
        if intercept_time <= 0.0:
            return np.zeros(3, dtype=float)

        resolved_target_position = (
            np.asarray(target_position, dtype=float)
            if target_position is not None
            else self._contact_frame_planned_contact_position()
        )
        target_delta = resolved_target_position - np.asarray(self.sim.racket_position, dtype=float)
        time_to_target = max(float(intercept_time), self.contact_frame_intercept_velocity_time_floor)
        reference_velocity = self.contact_frame_intercept_velocity_gain * target_delta / time_to_target
        reference_speed = float(np.linalg.norm(reference_velocity))
        if reference_speed > self.contact_frame_intercept_velocity_max:
            reference_velocity = reference_velocity * (self.contact_frame_intercept_velocity_max / reference_speed)
        return reference_velocity

    def _contact_frame_lateral_brake_velocity(self, target_position: Sequence[float] | None = None) -> np.ndarray:
        if not self._contact_frame_action_mode():
            return np.zeros(3, dtype=float)
        if self.contact_frame_lateral_brake_gain <= 0.0 or self.contact_frame_lateral_brake_max <= 0.0:
            return np.zeros(3, dtype=float)
        if float(self.sim.ball_velocity[2]) >= self.descending_ball_velocity_threshold:
            return np.zeros(3, dtype=float)

        contact_xy = (
            self._contact_frame_planned_contact_position()[:2]
            if target_position is None
            else np.asarray(target_position, dtype=float)[:2]
        )
        outward_xy = contact_xy - self._keepup_target_xy()
        outward_distance = float(np.linalg.norm(outward_xy))
        min_distance = float(self.next_intercept_success_radius)
        if outward_distance <= min_distance:
            return np.zeros(3, dtype=float)

        outward_direction = outward_xy / outward_distance
        racket_outward_speed = float(np.dot(self.sim.racket_velocity[:2], outward_direction))
        if racket_outward_speed <= 0.0:
            return np.zeros(3, dtype=float)

        effective_radius = max(self.contact_frame_lateral_brake_radius, min_distance + 1.0e-6)
        distance_scale = float(
            np.clip(
                (outward_distance - min_distance) / max(effective_radius - min_distance, 1.0e-6),
                0.0,
                1.0,
            )
        )
        brake_speed = min(
            self.contact_frame_lateral_brake_max,
            self.contact_frame_lateral_brake_gain * racket_outward_speed * distance_scale,
        )
        brake_velocity = np.zeros(3, dtype=float)
        brake_velocity[:2] = -outward_direction * brake_speed
        return brake_velocity

    def _contact_frame_velocity_target(
        self,
        target_position: Sequence[float] | None = None,
        *,
        lateral_brake_velocity: Sequence[float] | None = None,
    ) -> np.ndarray:
        if not self._contact_frame_action_mode():
            return np.zeros(3, dtype=float)
        intercept_velocity = self._contact_frame_intercept_velocity_target(target_position)
        if self._contact_frame_strike_hold_active:
            intercept_velocity = np.zeros(3, dtype=float)
        if self.contact_frame_velocity_target_max <= 0.0:
            return intercept_velocity
        if float(self.sim.ball_velocity[2]) >= self.descending_ball_velocity_threshold:
            return intercept_velocity

        target_velocity = intercept_velocity.copy()
        if self.contact_frame_velocity_target_gain > 0.0:
            required_velocity = self._required_contact_frame_racket_velocity()
            intercept_time = (
                self._contact_frame_plan_intercept_time
                if self._contact_frame_plan_active
                else self._predicted_intercept_time()
            )
            urgency = 1.0 - np.clip(
                intercept_time / max(self.contact_frame_base_strike_time_horizon, 1.0e-6),
                0.0,
                1.0,
            )
            strike_readiness = float(np.clip(max(self._pre_contact_height_readiness(), urgency), 0.0, 1.0))
            target_velocity = target_velocity + self.contact_frame_velocity_target_gain * strike_readiness * required_velocity
        target_velocity[:2] += self._contact_frame_tracking_xy_residual()
        target_velocity[:2] += self._contact_frame_racket_xy_residual()
        target_velocity[2] += self._contact_frame_racket_vz_residual()
        target_velocity[2] += self._contact_frame_low_apex_recovery_velocity()
        target_velocity += (
            self._contact_frame_lateral_brake_velocity(target_position)
            if lateral_brake_velocity is None
            else np.asarray(lateral_brake_velocity, dtype=float)
        )
        target_speed = float(np.linalg.norm(target_velocity))
        max_speed = max(self.contact_frame_velocity_target_max, self.contact_frame_intercept_velocity_max)
        if target_speed > max_speed:
            target_velocity = target_velocity * (max_speed / target_speed)
        return target_velocity

    def _contact_frame_strike_readiness(self) -> float:
        if float(self.sim.ball_velocity[2]) >= self.descending_ball_velocity_threshold:
            return 0.0
        intercept_time = (
            self._contact_frame_plan_intercept_time
            if self._contact_frame_plan_active
            else self._predicted_intercept_time()
        )
        urgency = 1.0 - np.clip(
            intercept_time / max(self.contact_frame_base_strike_time_horizon, 1.0e-6),
            0.0,
            1.0,
        )
        return float(np.clip(max(self._pre_contact_height_readiness(), urgency), 0.0, 1.0))

    def _contact_frame_followthrough_offset(self) -> np.ndarray:
        if not self._contact_frame_action_mode():
            return np.zeros(3, dtype=float)
        if (
            self.contact_frame_followthrough_gain <= 0.0
            or self.contact_frame_followthrough_time <= 0.0
            or self.contact_frame_followthrough_max <= 0.0
        ):
            return np.zeros(3, dtype=float)
        if float(self.sim.ball_velocity[2]) >= self.descending_ball_velocity_threshold:
            return np.zeros(3, dtype=float)

        followthrough_offset = (
            self.contact_frame_followthrough_gain
            * self._contact_frame_strike_readiness()
            * self._required_contact_frame_racket_velocity()
            * self.contact_frame_followthrough_time
        )
        offset_norm = float(np.linalg.norm(followthrough_offset))
        if offset_norm > self.contact_frame_followthrough_max:
            followthrough_offset = followthrough_offset * (self.contact_frame_followthrough_max / offset_norm)
        return followthrough_offset

    def _contact_frame_velocity_lead(self) -> float:
        if not self._contact_frame_action_mode():
            return 0.0
        if self.contact_frame_velocity_lead_gain <= 0.0 or self.contact_frame_velocity_lead_max <= 0.0:
            return 0.0
        if float(self.sim.ball_velocity[2]) >= self.descending_ball_velocity_threshold:
            return 0.0

        required_velocity_z = self._required_contact_frame_racket_velocity_z()
        current_velocity_z = float(self.sim.racket_velocity[2])
        velocity_error_z = required_velocity_z - current_velocity_z
        intercept_time = (
            self._contact_frame_plan_intercept_time
            if self._contact_frame_plan_active
            else self._predicted_intercept_time()
        )
        urgency = 1.0 - np.clip(
            intercept_time / max(self.contact_frame_base_strike_time_horizon, 1.0e-6),
            0.0,
            1.0,
        )
        strike_readiness = float(np.clip(max(self._pre_contact_height_readiness(), urgency), 0.0, 1.0))
        lead = self.contact_frame_velocity_lead_gain * velocity_error_z * strike_readiness
        return float(np.clip(lead, -self.contact_frame_velocity_lead_max, self.contact_frame_velocity_lead_max))

    def _contact_frame_strike_tilt_active(self) -> bool:
        return (
            self._contact_frame_action_mode()
            and self._phase_name() in {"prepare", "strike", "recovery"}
            and float(self.sim.ball_velocity[2]) < self.descending_ball_velocity_threshold
        )

    def _contact_frame_centering_tilt(self) -> np.ndarray:
        if not self._contact_frame_action_mode() or self.contact_frame_centering_tilt_limit is None:
            return np.zeros(2, dtype=float)
        if not self._contact_frame_strike_tilt_active():
            return np.zeros(2, dtype=float)

        contact_xy = (
            self._contact_frame_plan_contact_position[:2]
            if self._contact_frame_plan_active
            else self._predicted_intercept_xy()
        )
        correction_xy = self._keepup_target_xy() - contact_xy
        radius = (
            self.contact_frame_centering_tilt_radius
            if self.contact_frame_centering_tilt_radius is not None
            else self.contact_centering_radius
        )
        effective_radius = max(float(radius), self.contact_frame_centering_tilt_deadband + 1.0e-6)
        scale_xy = np.zeros(2, dtype=float)
        for axis_index in range(2):
            axis_error = float(correction_xy[axis_index])
            axis_magnitude = abs(axis_error)
            if axis_magnitude <= self.contact_frame_centering_tilt_deadband:
                continue
            scale_xy[axis_index] = np.sign(axis_error) * np.clip(
                (axis_magnitude - self.contact_frame_centering_tilt_deadband)
                / max(effective_radius - self.contact_frame_centering_tilt_deadband, 1.0e-6),
                0.0,
                1.0,
            )

        if not np.any(scale_xy):
            return np.zeros(2, dtype=float)
        intercept_time = (
            self._contact_frame_plan_intercept_time
            if self._contact_frame_plan_active
            else self._predicted_intercept_time()
        )
        urgency = 1.0 - np.clip(intercept_time / self.contact_frame_tilt_ramp_time, 0.0, 1.0)
        ramp = float(np.clip(max(self._pre_contact_height_readiness(), urgency), 0.0, 1.0))
        target_tilt = np.array(
            [
                self.contact_frame_centering_tilt_limit[0] * scale_xy[0] * ramp,
                -self.contact_frame_centering_tilt_limit[1] * scale_xy[1] * ramp,
            ],
            dtype=float,
        )
        target_tilt = target_tilt * self._contact_frame_centering_tilt_scale()
        return np.clip(target_tilt, -self.contact_frame_centering_tilt_limit, self.contact_frame_centering_tilt_limit)

    def _contact_frame_trajectory_tilt(self) -> np.ndarray:
        if not self._contact_frame_action_mode() or self.contact_frame_trajectory_tilt_limit is None:
            return np.zeros(2, dtype=float)
        if self.contact_frame_trajectory_tilt_gain <= 0.0:
            return np.zeros(2, dtype=float)
        if not self._contact_frame_strike_tilt_active():
            return np.zeros(2, dtype=float)

        contact_position = self._contact_frame_planned_contact_position()
        desired_velocity, _, _ = self._contact_frame_controller_desired_velocity(contact_position)
        impulse_direction = desired_velocity - np.asarray(self.sim.ball_velocity, dtype=float)
        impulse_norm = float(np.linalg.norm(impulse_direction))
        if impulse_norm <= 1.0e-9 or float(impulse_direction[2]) <= 0.0:
            return np.zeros(2, dtype=float)

        top_normal = impulse_direction / impulse_norm
        raw_tilt = np.array(
            [
                np.arcsin(float(np.clip(top_normal[0], -0.95, 0.95))),
                -np.arcsin(float(np.clip(top_normal[1], -0.95, 0.95))),
            ],
            dtype=float,
        )
        if self.contact_frame_trajectory_tilt_deadband > 0.0:
            raw_tilt[np.abs(raw_tilt) <= self.contact_frame_trajectory_tilt_deadband] = 0.0

        intercept_time = (
            self._contact_frame_plan_intercept_time
            if self._contact_frame_plan_active
            else self._predicted_intercept_time()
        )
        urgency = 1.0 - np.clip(intercept_time / self.contact_frame_tilt_ramp_time, 0.0, 1.0)
        ramp = float(np.clip(max(self._pre_contact_height_readiness(), urgency), 0.0, 1.0))
        target_tilt = (
            self.contact_frame_trajectory_tilt_gain
            * self._contact_frame_trajectory_tilt_scale()
            * raw_tilt
            * ramp
        )
        return np.clip(target_tilt, -self.contact_frame_trajectory_tilt_limit, self.contact_frame_trajectory_tilt_limit)

    def _contact_frame_base_strike_tilt(self) -> np.ndarray:
        if not self._contact_frame_action_mode():
            return np.zeros(2, dtype=float)
        if not self._contact_frame_strike_tilt_active():
            return np.zeros(2, dtype=float)
        target_tilt = np.zeros(2, dtype=float)
        if self.contact_frame_base_tilt_residual is not None:
            target_tilt = target_tilt + np.asarray(self.contact_frame_base_tilt_residual, dtype=float)
        target_tilt = target_tilt + self._contact_frame_trajectory_tilt()
        target_tilt = target_tilt + self._contact_frame_centering_tilt()
        return target_tilt

    def _contact_frame_action_target_position(self, action: Sequence[float]) -> np.ndarray:
        action_array = np.asarray(action, dtype=float)
        if action_array.shape != (3,):
            raise ValueError(f"Contact-frame action must have shape (3,), got {action_array.shape}.")
        anchor_position = self._controller_anchor_position()
        target_position = anchor_position.copy()
        radial, tangent, _ = self._contact_frame_basis_xy()
        contact_offset_xy = radial * action_array[0] + tangent * action_array[1]
        if float(self.sim.ball_velocity[2]) >= self.descending_ball_velocity_threshold:
            target_position[:2] = self._post_contact_return_target_xy() + contact_offset_xy
            target_position[2] = anchor_position[2] + self.post_contact_return_z_offset + action_array[2]
            return target_position
        contact_position = (
            self._contact_frame_planned_contact_position()
            if self._contact_frame_plan_active
            else self._predicted_contact_position()
        )
        base_contact_xy = (
            self._contact_frame_planner_contact_target_xy(contact_position[:2])
            if self._contact_frame_plan_active
            else self._strike_contact_target_xy()
        )
        target_position[:2] = base_contact_xy + contact_offset_xy
        lift_target = self._strike_lift_feedforward() + self._contact_frame_base_strike_lift()
        if self._contact_frame_plan_active:
            target_position[2] = float(contact_position[2]) + lift_target + action_array[2]
        else:
            target_position[2] = anchor_position[2] + lift_target + action_array[2]
        target_position = target_position + self._contact_frame_followthrough_offset()
        return target_position

    def _controller_body_clearance_active(self) -> bool:
        if (
            self.controller_body_clearance_gain <= 0.0
            or self.controller_body_clearance_margin <= 0.0
            or self.controller_body_clearance_max_step <= 0.0
        ):
            return False
        if self._ball_height_above_racket() < -0.05:
            return False
        if float(self.sim.ball_velocity[2]) < self.descending_ball_velocity_threshold:
            return True
        if self.contact_count <= 0 and self._last_contact_step is None:
            return False
        time_since_contact = self._time_since_contact()
        if time_since_contact is not None and time_since_contact > self.next_intercept_max_time:
            return False
        clearance_height = self._target_ball_height_above_racket() + self.controller_body_clearance_vertical_margin
        return self._ball_height_above_racket() <= clearance_height

    def _body_safe_target_position(self, target_position: Sequence[float]) -> np.ndarray:
        safe_target = np.asarray(target_position, dtype=float).copy()
        keepout_specs = (("link5", 0.12), ("link6", 0.10), ("link7", 0.09), ("hand", 0.08))
        anchor_xy = np.asarray(self.sim.racket_position[:2], dtype=float)
        for body_name, keepout_radius in keepout_specs:
            try:
                body_id = self.sim.model.body(body_name).id
            except Exception:
                continue
            body_position = np.asarray(self.sim.data.xpos[body_id], dtype=float)
            delta_xy = safe_target[:2] - body_position[:2]
            distance_xy = float(np.linalg.norm(delta_xy))
            if distance_xy >= keepout_radius:
                continue
            if distance_xy <= 1.0e-9:
                fallback_direction = safe_target[:2] - anchor_xy
                fallback_norm = float(np.linalg.norm(fallback_direction))
                delta_direction = np.array([1.0, 0.0], dtype=float) if fallback_norm <= 1.0e-9 else fallback_direction / fallback_norm
            else:
                delta_direction = delta_xy / distance_xy
            safe_target[:2] = body_position[:2] + keepout_radius * delta_direction
        return safe_target

    def _guarded_target_position(self, target_position: Sequence[float]) -> np.ndarray:
        safe_target = self._body_safe_target_position(target_position)
        anchor_position = self._controller_anchor_position()
        pre_contact_xy_low, pre_contact_xy_high = self._pre_contact_xy_bounds()
        safe_target[:2] = anchor_position[:2] + np.clip(
            safe_target[:2] - anchor_position[:2],
            pre_contact_xy_low,
            pre_contact_xy_high,
        )
        if not self._pre_contact_upward_ready():
            pre_contact_lift_limit = 0.02 + self._strike_lift_feedforward()
            safe_target[2] = min(float(safe_target[2]), float(anchor_position[2] + pre_contact_lift_limit))
        return safe_target
