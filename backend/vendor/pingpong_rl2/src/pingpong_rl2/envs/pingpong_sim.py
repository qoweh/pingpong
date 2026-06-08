from __future__ import annotations

from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np

from pingpong_rl2.defaults import DEFAULT_BALL_HEIGHT, DEFAULT_CONTROL_DT
from pingpong_rl2.utils.paths import SCENE_XML_PATH, resolve_input_path


class PingPongSim:
    def __init__(self, scene_path: Path | str | None = None, control_dt: float = DEFAULT_CONTROL_DT) -> None:
        # MuJoCo XML scene을 열고 공/라켓/관절 id를 캐시해 이후 step 접근을 빠르게 만든다.
        # LINK: backend/vendor/pingpong_rl2/src/pingpong_rl2/envs/keepup_env.py:176
        scene_file = resolve_input_path(Path(scene_path)) if scene_path is not None else SCENE_XML_PATH
        self.scene_path = scene_file.resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)
        self.control_dt = float(control_dt)
        self.n_substeps = max(1, int(round(self.control_dt / self.model.opt.timestep)))

        self.ball_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
        self.ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.racket_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "racket")
        self.racket_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "racket_center")
        self.racket_head_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "racket_head")

        if (
            self.ball_joint_id < 0
            or self.ball_body_id < 0
            or self.racket_body_id < 0
            or self.racket_site_id < 0
            or self.racket_head_geom_id < 0
        ):
            raise ValueError(
                "Scene is missing one of the required objects: ball_joint, ball, racket, racket_center, racket_head."
            )

        self._ball_qpos_adr = self.model.jnt_qposadr[self.ball_joint_id]
        self._ball_dof_adr = self.model.jnt_dofadr[self.ball_joint_id]
        self._home_ctrl = self.model.key_ctrl[0].copy()
        self._home_joint_targets = self._home_ctrl[:7].copy()
        self._default_ball_height = DEFAULT_BALL_HEIGHT
        self._racket_jacobian = np.zeros((3, self.model.nv), dtype=float)
        self.reset()

    @property
    def home_joint_targets(self) -> np.ndarray:
        return self._home_joint_targets.copy()

    @property
    def joint_positions(self) -> np.ndarray:
        return self.data.qpos[:7].copy()

    @property
    def joint_velocities(self) -> np.ndarray:
        return self.data.qvel[:7].copy()

    @property
    def ball_position(self) -> np.ndarray:
        return self.data.xpos[self.ball_body_id].copy()

    @property
    def racket_position(self) -> np.ndarray:
        return self.data.site_xpos[self.racket_site_id].copy()

    @property
    def racket_velocity(self) -> np.ndarray:
        mujoco.mj_jacSite(
            self.model,
            self.data,
            self._racket_jacobian,
            None,
            self.racket_site_id,
        )
        return self._racket_jacobian @ self.data.qvel

    @property
    def racket_face_normal(self) -> np.ndarray:
        racket_xmat = np.asarray(self.data.geom_xmat[self.racket_head_geom_id], dtype=float).reshape(3, 3)
        return racket_xmat[:, 2].copy()

    @property
    def ball_velocity(self) -> np.ndarray:
        return self.data.qvel[self._ball_dof_adr:self._ball_dof_adr + 3].copy()

    @property
    def ball_angular_velocity(self) -> np.ndarray:
        return self.data.qvel[self._ball_dof_adr + 3:self._ball_dof_adr + 6].copy()

    def set_ball_velocity(
        self,
        velocity: Sequence[float],
        angular_velocity: Sequence[float] | None = None,
    ) -> np.ndarray:
        velocity_array = np.asarray(velocity, dtype=float)
        if velocity_array.shape != (3,):
            raise ValueError(f"Ball velocity must have shape (3,), got {velocity_array.shape}.")

        self.data.qvel[self._ball_dof_adr:self._ball_dof_adr + 3] = velocity_array
        if angular_velocity is not None:
            angular_velocity_array = np.asarray(angular_velocity, dtype=float)
            if angular_velocity_array.shape != (3,):
                raise ValueError(
                    f"Ball angular velocity must have shape (3,), got {angular_velocity_array.shape}."
                )
            self.data.qvel[self._ball_dof_adr + 3:self._ball_dof_adr + 6] = angular_velocity_array
        mujoco.mj_forward(self.model, self.data)
        return self.ball_velocity

    def reset(
        self,
        ball_position: Sequence[float] | None = None,
        ball_velocity: Sequence[float] = (0.0, 0.0, 0.0),
        ball_angular_velocity: Sequence[float] | None = None,
        ball_height: float | None = None,
        ball_xy_offset: Sequence[float] = (0.0, 0.0),
    ) -> mujoco.MjData:
        # keyframe home pose로 되돌린 뒤 공을 racket 기준 위치나 명시 위치에 다시 놓는다.
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.ctrl[:] = self._home_ctrl
        mujoco.mj_forward(self.model, self.data)

        if ball_position is None:
            spawn_height = self._default_ball_height if ball_height is None else float(ball_height)
            self.reset_ball_above_racket(
                height=spawn_height,
                xy_offset=ball_xy_offset,
                velocity=ball_velocity,
                angular_velocity=ball_angular_velocity,
            )
        else:
            self.spawn_ball(ball_position, ball_velocity, angular_velocity=ball_angular_velocity)

        return self.data

    def spawn_ball(
        self,
        position: Sequence[float],
        velocity: Sequence[float] = (0.0, 0.0, 0.0),
        angular_velocity: Sequence[float] | None = None,
    ) -> np.ndarray:
        # free joint qpos/qvel에 공 위치, 자세, 선속도, 각속도를 직접 써서 공을 재배치한다.
        # LINK: backend/app/live_simulation.py:482
        position_array = np.asarray(position, dtype=float)
        velocity_array = np.asarray(velocity, dtype=float)
        angular_velocity_array = (
            np.zeros(3, dtype=float)
            if angular_velocity is None
            else np.asarray(angular_velocity, dtype=float)
        )

        if position_array.shape != (3,):
            raise ValueError(f"Ball position must have shape (3,), got {position_array.shape}.")
        if velocity_array.shape != (3,):
            raise ValueError(f"Ball velocity must have shape (3,), got {velocity_array.shape}.")
        if angular_velocity_array.shape != (3,):
            raise ValueError(
                f"Ball angular velocity must have shape (3,), got {angular_velocity_array.shape}."
            )

        qpos = self.data.qpos
        qvel = self.data.qvel

        qpos[self._ball_qpos_adr:self._ball_qpos_adr + 3] = position_array
        qpos[self._ball_qpos_adr + 3:self._ball_qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        qvel[self._ball_dof_adr:self._ball_dof_adr + 3] = velocity_array
        qvel[self._ball_dof_adr + 3:self._ball_dof_adr + 6] = angular_velocity_array
        mujoco.mj_forward(self.model, self.data)
        return self.ball_position

    def reset_ball_above_racket(
        self,
        height: float = 0.25,
        xy_offset: Sequence[float] = (0.0, 0.0),
        velocity: Sequence[float] = (0.0, 0.0, 0.0),
        angular_velocity: Sequence[float] | None = None,
    ) -> np.ndarray:
        xy_offset_array = np.asarray(xy_offset, dtype=float)
        if xy_offset_array.shape != (2,):
            raise ValueError(f"xy_offset must have shape (2,), got {xy_offset_array.shape}.")

        spawn_position = self.racket_position + np.array([xy_offset_array[0], xy_offset_array[1], height])
        return self.spawn_ball(spawn_position, velocity, angular_velocity=angular_velocity)

    def set_arm_joint_targets(self, joint_targets: Sequence[float], gripper_target: float | None = None) -> np.ndarray:
        joint_targets_array = np.asarray(joint_targets, dtype=float)
        if joint_targets_array.shape != (7,):
            raise ValueError(f"Arm targets must have shape (7,), got {joint_targets_array.shape}.")

        self.data.ctrl[:7] = joint_targets_array
        if gripper_target is not None:
            self.data.ctrl[7] = gripper_target
        return self.data.ctrl[:8].copy()

    @staticmethod
    def _trace_vector_fields(prefix: str, vector: Sequence[float] | np.ndarray | None) -> dict[str, float | None]:
        if vector is None:
            return {
                f"{prefix}_x": None,
                f"{prefix}_y": None,
                f"{prefix}_z": None,
            }

        vector_array = np.asarray(vector, dtype=float)
        if vector_array.shape != (3,):
            raise ValueError(f"Expected a 3D vector for {prefix}, got {vector_array.shape}.")
        return {
            f"{prefix}_x": float(vector_array[0]),
            f"{prefix}_y": float(vector_array[1]),
            f"{prefix}_z": float(vector_array[2]),
        }

    @classmethod
    def _empty_contact_trace(cls) -> dict[str, object]:
        contact_trace: dict[str, object] = {
            "contact_observed": False,
            "contact_active_at_step_start": False,
            "contact_active_at_trace_end": False,
            "contact_started_during_trace": False,
            "contact_substep": None,
            "contact_end_substep": None,
            "contact_geom1_name": None,
            "contact_geom2_name": None,
            "contact_ball_height_above_racket": None,
            "contact_xy_alignment_error": None,
            "contact_ball_speed_norm": None,
            "contact_racket_speed_norm": None,
        }
        for prefix in (
            "pre_contact_ball_position",
            "pre_contact_ball_velocity",
            "pre_contact_racket_velocity",
            "pre_contact_relative_velocity",
            "contact_ball_position",
            "contact_ball_velocity",
            "contact_racket_velocity",
            "contact_racket_center_velocity",
            "contact_relative_velocity",
            "contact_racket_face_normal",
            "contact_racket_acceleration",
            "contact_mujoco_position",
            "contact_mujoco_normal",
            "contact_mujoco_normal_racket_to_ball",
            "contact_end_ball_velocity",
        ):
            contact_trace.update(cls._trace_vector_fields(prefix, None))
        for offset in range(1, 6):
            contact_trace.update(cls._trace_vector_fields(f"post_contact_{offset}_ball_velocity", None))
            contact_trace[f"post_contact_{offset}_contact_active"] = None
        return contact_trace

    def _matching_contact(self, contact_geoms: tuple[str, str]) -> dict[str, object] | None:
        target_pair = tuple(sorted(contact_geoms))
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1_name = self.model.geom(int(contact.geom1)).name or ""
            geom2_name = self.model.geom(int(contact.geom2)).name or ""
            if tuple(sorted((geom1_name, geom2_name))) != target_pair:
                continue

            contact_position = np.asarray(contact.pos[:3], dtype=float).copy()
            contact_normal = np.asarray(contact.frame[:3], dtype=float).copy()
            normal_norm = np.linalg.norm(contact_normal)
            if normal_norm > 1.0e-9:
                contact_normal = contact_normal / normal_norm

            racket_to_ball_normal = contact_normal.copy()
            if geom1_name == "ball_geom" and geom2_name == "racket_head":
                racket_to_ball_normal = -contact_normal
            elif geom1_name == "racket_head" and geom2_name == "ball_geom":
                racket_to_ball_normal = contact_normal

            return {
                "contact_index": index,
                "geom1_name": geom1_name,
                "geom2_name": geom2_name,
                "contact_position": contact_position,
                "contact_normal": contact_normal,
                "contact_normal_racket_to_ball": racket_to_ball_normal,
            }
        return None

    def contact_pairs(self) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1 = self.model.geom(contact.geom1).name
            geom2 = self.model.geom(contact.geom2).name
            pairs.append(tuple(sorted((geom1, geom2))))
        return pairs

    def ball_robot_body_contact(self) -> str | None:
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1 = self.model.geom(int(contact.geom1))
            geom2 = self.model.geom(int(contact.geom2))
            geom1_name = geom1.name or ""
            geom2_name = geom2.name or ""
            if geom1_name != "ball_geom" and geom2_name != "ball_geom":
                continue

            other_geom = geom2 if geom1_name == "ball_geom" else geom1
            other_geom_name = other_geom.name or ""
            other_body_name = self.model.body(int(np.asarray(other_geom.bodyid).item())).name
            if other_geom_name == "floor" or other_body_name in {"ball", "racket"}:
                continue
            return other_body_name
        return None

    def has_contact(self, geom_a: str, geom_b: str) -> bool:
        target = tuple(sorted((geom_a, geom_b)))
        return target in self.contact_pairs()

    def state_is_finite(self) -> bool:
        return bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())

    def failure_reason(
        self,
        x_bounds: tuple[float, float] = (0.0, 1.35),
        y_bounds: tuple[float, float] = (-0.6, 0.6),
        z_bounds: tuple[float, float] = (-0.05, 2.0),
        max_ball_speed: float = 8.0,
    ) -> str | None:
        # 물리 state, 바닥/로봇 접촉, 영역 이탈, 과속을 episode 실패 사유로 판정한다.
        if not self.state_is_finite():
            return "nonfinite_state"
        if self.has_contact("ball_geom", "floor"):
            return "floor_contact"
        if self.ball_robot_body_contact() is not None:
            return "robot_body_contact"

        ball_position = self.ball_position
        within_x = x_bounds[0] <= ball_position[0] <= x_bounds[1]
        within_y = y_bounds[0] <= ball_position[1] <= y_bounds[1]
        within_z = z_bounds[0] <= ball_position[2] <= z_bounds[1]
        if not (within_x and within_y and within_z):
            return "ball_out_of_bounds"
        if np.linalg.norm(self.ball_velocity) > max_ball_speed:
            return "ball_speed_limit"
        return None

    def step(
        self,
        joint_targets: Sequence[float] | None = None,
        gripper_target: float | None = None,
        n_substeps: int | None = None,
    ) -> mujoco.MjData:
        # 제어 주기(control_dt)를 MuJoCo timestep 여러 번으로 나누어 실제 물리를 진행한다.
        if joint_targets is not None:
            self.set_arm_joint_targets(joint_targets, gripper_target)

        step_count = self.n_substeps if n_substeps is None else max(1, int(n_substeps))
        for _ in range(step_count):
            mujoco.mj_step(self.model, self.data)
        return self.data

    def step_with_contact_trace(
        self,
        joint_targets: Sequence[float] | None = None,
        gripper_target: float | None = None,
        n_substeps: int | None = None,
        contact_geoms: tuple[str, str] = ("ball_geom", "racket_head"),
    ) -> dict[str, object]:
        # substep 동안 공-라켓 접촉이 발생한 순간의 위치/속도/법선을 reward 계산용 trace로 남긴다.
        # LINK: backend/vendor/pingpong_rl2/src/pingpong_rl2/envs/keepup_env.py:1625
        if joint_targets is not None:
            self.set_arm_joint_targets(joint_targets, gripper_target)

        step_count = self.n_substeps if n_substeps is None else max(1, int(n_substeps))
        contact_trace = self._empty_contact_trace()
        contact_active_previous_substep = self._matching_contact(contact_geoms) is not None
        contact_trace["contact_active_at_step_start"] = contact_active_previous_substep
        previous_ball_position = self.ball_position
        previous_ball_velocity = self.ball_velocity
        previous_racket_velocity = self.racket_velocity
        for substep_index in range(1, step_count + 1):
            mujoco.mj_step(self.model, self.data)
            ball_position = self.ball_position
            ball_velocity = self.ball_velocity
            racket_velocity = self.racket_velocity
            racket_acceleration = (racket_velocity - previous_racket_velocity) / self.model.opt.timestep
            racket_face_normal = self.racket_face_normal
            previous_racket_velocity = racket_velocity
            matching_contact = self._matching_contact(contact_geoms)
            contact_active = matching_contact is not None

            if not contact_trace["contact_observed"] and matching_contact is not None:
                racket_position = self.racket_position
                pre_contact_relative_velocity = previous_ball_velocity - previous_racket_velocity
                contact_relative_velocity = ball_velocity - racket_velocity
                contact_trace["contact_observed"] = True
                contact_trace["contact_started_during_trace"] = not bool(contact_trace["contact_active_at_step_start"])
                contact_trace["contact_substep"] = substep_index
                contact_trace["contact_geom1_name"] = matching_contact["geom1_name"]
                contact_trace["contact_geom2_name"] = matching_contact["geom2_name"]
                contact_trace["contact_ball_height_above_racket"] = float(ball_position[2] - racket_position[2])
                contact_trace["contact_xy_alignment_error"] = float(np.linalg.norm(ball_position[:2] - racket_position[:2]))
                contact_trace["contact_ball_speed_norm"] = float(np.linalg.norm(ball_velocity))
                contact_trace["contact_racket_speed_norm"] = float(np.linalg.norm(racket_velocity))
                contact_trace["contact_racket_acceleration_norm"] = float(np.linalg.norm(racket_acceleration))
                contact_trace.update(self._trace_vector_fields("pre_contact_ball_position", previous_ball_position))
                contact_trace.update(self._trace_vector_fields("pre_contact_ball_velocity", previous_ball_velocity))
                contact_trace.update(self._trace_vector_fields("pre_contact_racket_velocity", previous_racket_velocity))
                contact_trace.update(self._trace_vector_fields("pre_contact_relative_velocity", pre_contact_relative_velocity))
                contact_trace.update(self._trace_vector_fields("contact_ball_position", ball_position))
                contact_trace.update(self._trace_vector_fields("contact_ball_velocity", ball_velocity))
                contact_trace.update(self._trace_vector_fields("contact_racket_velocity", racket_velocity))
                contact_trace.update(self._trace_vector_fields("contact_racket_center_velocity", racket_velocity))
                contact_trace.update(self._trace_vector_fields("contact_relative_velocity", contact_relative_velocity))
                contact_trace.update(self._trace_vector_fields("contact_racket_face_normal", racket_face_normal))
                contact_trace.update(self._trace_vector_fields("contact_racket_acceleration", racket_acceleration))
                contact_trace.update(self._trace_vector_fields("contact_mujoco_position", matching_contact["contact_position"]))
                contact_trace.update(self._trace_vector_fields("contact_mujoco_normal", matching_contact["contact_normal"]))
                contact_trace.update(
                    self._trace_vector_fields(
                        "contact_mujoco_normal_racket_to_ball",
                        matching_contact["contact_normal_racket_to_ball"],
                    )
                )

            if contact_trace["contact_observed"]:
                contact_substep = int(contact_trace["contact_substep"])
                post_contact_offset = substep_index - contact_substep
                if 1 <= post_contact_offset <= 5:
                    contact_trace.update(
                        self._trace_vector_fields(f"post_contact_{post_contact_offset}_ball_velocity", ball_velocity)
                    )
                    contact_trace[f"post_contact_{post_contact_offset}_contact_active"] = contact_active
                if (
                    contact_trace["contact_end_substep"] is None
                    and contact_active_previous_substep
                    and not contact_active
                ):
                    contact_trace["contact_end_substep"] = substep_index
                    contact_trace.update(self._trace_vector_fields("contact_end_ball_velocity", ball_velocity))

            previous_ball_position = ball_position
            previous_ball_velocity = ball_velocity
            contact_active_previous_substep = contact_active

        contact_trace["contact_active_at_trace_end"] = contact_active_previous_substep
        return contact_trace
