from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from .ball_spawn import build_ball_spawn_config, parse_ball_spawn_options
from .model_catalog import (
    ModelRecord,
    build_model_catalog,
    env_kwargs_with_summary_hints,
    infer_run_name,
    is_catalog_visible,
    read_json,
    training_summary_path,
    with_loaded_policy_metadata,
)
from .settings import AppSettings


LOGGER = logging.getLogger(__name__)


class ModelSelectionError(RuntimeError):
    pass


@dataclass
class LiveCommandState:
    playback: str = "playing"
    reset_requested: bool = False
    reset_options: dict[str, Any] | None = None
    ball_spawn_requested: bool = False
    ball_spawn_options: dict[str, Any] | None = None


@dataclass
class RuntimeModel:
    # PPO policy와 그 policy를 실행하기 위한 환경 설정을 하나의 runtime 단위로 묶는다.
    # LINK: backend/app/model_catalog.py:70
    record: ModelRecord
    env_kwargs: dict[str, Any]
    ball_spawn_config: dict[str, Any]
    policy: PPO
    policy_lock: threading.Lock
    policy_message: str
    control_dt: float = 0.02

    @property
    def model_path(self) -> Path:
        return self.record.path

    @property
    def metadata(self) -> dict[str, Any]:
        return self.record.metadata

    def predict(self, observation: Any, deterministic: bool) -> np.ndarray:
        # 서버 live loop는 매 step마다 observation을 policy 입력 형태로 맞춘 뒤 PPO action을 계산한다.
        # LINK: backend/vendor/pingpong_rl2/src/pingpong_rl2/envs/keepup_env.py:1501
        policy_observation = self.policy_observation(observation)
        with self.policy_lock:
            action, _ = self.policy.predict(
                policy_observation,
                deterministic=deterministic,
            )
        return np.asarray(action, dtype=float)

    def policy_observation(self, observation: Any) -> np.ndarray:
        # 오래된 모델의 observation 차원이 다를 때는 현재 observation에서 호환 가능한 부분만 투영한다.
        raw_observation = np.asarray(observation, dtype=np.float32).reshape(-1)
        expected_dim = int(self.metadata.get("observationDim") or 0)
        if expected_dim and raw_observation.shape[0] != expected_dim:
            projected = legacy_policy_observation(raw_observation, expected_dim)
            if projected is not None:
                return projected
        return raw_observation


class LiveSimulationService:
    def __init__(self, settings: AppSettings) -> None:
        # vendored RL 패키지를 로드한 뒤 모델 카탈로그와 최초 active runtime을 준비한다.
        # LINK: backend/vendor/pingpong_rl2/src/pingpong_rl2/utils/ppo_runs.py:181
        self.settings = settings
        self._load_rl_package(settings.rl_source_root)

        from pingpong_rl2.envs import PingPongKeepUpEnv, PingPongKeepUpGymEnv
        from pingpong_rl2.utils import resolve_env_kwargs_for_model

        self.env_class = PingPongKeepUpGymEnv
        self.env_kwarg_names = set(inspect.signature(PingPongKeepUpEnv.__init__).parameters) - {"self"}
        self.resolve_env_kwargs_for_model = resolve_env_kwargs_for_model
        self.model_catalog = build_model_catalog(settings.project_root, settings.model_path, self._resolve_env_kwargs)
        self._runtime_lock = threading.RLock()
        self._runtime_cache: dict[str, RuntimeModel] = {}
        self.active_runtime = self._load_runtime_for_path(settings.model_path)

    def create_session(self) -> "LiveSimulationSession":
        # active runtime을 기준으로 실제 MuJoCo/Gym session을 새로 만든다.
        # LINK: backend/app/live_simulation.py:434
        with self._runtime_lock:
            runtime = self.active_runtime
        return LiveSimulationSession(self, runtime)

    def current_runtime(self) -> RuntimeModel:
        with self._runtime_lock:
            return self.active_runtime

    def activate_runtime(self, runtime: RuntimeModel) -> None:
        with self._runtime_lock:
            self.active_runtime = runtime

    @property
    def ball_spawn_config(self) -> dict[str, Any]:
        with self._runtime_lock:
            return self.active_runtime.ball_spawn_config

    def config_payload(self, runtime: RuntimeModel | None = None) -> dict[str, Any]:
        # 프론트 초기 로딩과 ready 메시지에서 공유하는 현재 runtime 설정 payload를 만든다.
        active_runtime = runtime
        if active_runtime is None:
            with self._runtime_lock:
                active_runtime = self.active_runtime
        return {
            "model": portable_path(active_runtime.model_path, self.settings.project_root),
            "modelId": active_runtime.record.id,
            "scene": portable_path(self.settings.scene_path, self.settings.project_root),
            "deterministic": self.settings.deterministic_policy,
            "seed": self.settings.seed,
            "controlDt": active_runtime.control_dt,
            "ballSpawn": active_runtime.ball_spawn_config,
            "modelInfo": active_runtime.metadata,
        }

    def models_payload(self) -> dict[str, Any]:
        # UI에는 dimension별 대표/최신 모델만 노출하되 현재 활성 모델은 항상 포함한다.
        # LINK: backend/app/model_catalog.py:197
        with self._runtime_lock:
            active_runtime = self.active_runtime
            active_model_id = active_runtime.record.id
            active_metadata = active_runtime.metadata

        models = []
        for record in self.model_catalog.values():
            if record.id != active_model_id and not is_catalog_visible(record):
                continue
            metadata = active_metadata if record.id == active_model_id else record.metadata
            models.append(metadata)

        models.sort(key=lambda item: model_sort_key(item, active_model_id))
        return {
            "activeModel": active_model_id,
            "models": models,
        }

    def select_model(self, model_id: str) -> dict[str, Any]:
        # 모델 zip을 runtime으로 로드하거나 캐시에서 꺼내 active runtime으로 교체한다.
        # LINK: backend/app/model_catalog.py:130
        record = self.model_catalog.get(model_id)
        if record is None or not is_catalog_visible(record):
            raise KeyError(model_id)

        started_at = time.perf_counter()
        with self._runtime_lock:
            was_cached = record.id in self._runtime_cache
        runtime = self._runtime_for_record(record)
        with self._runtime_lock:
            self.active_runtime = runtime
        LOGGER.info(
            "Selected model %s in %.3fs (%s)",
            model_id,
            time.perf_counter() - started_at,
            "cached runtime" if was_cached else "loaded runtime",
        )
        return {
            **self.models_payload(),
            "config": self.config_payload(runtime),
        }

    def _resolve_env_kwargs(self, model_path: Path) -> dict[str, Any]:
        # 훈련 summary의 환경 힌트와 현재 배포 scene을 합쳐 live runtime용 env kwargs를 만든다.
        # LINK: backend/vendor/pingpong_rl2/src/pingpong_rl2/utils/ppo_runs.py:181
        env_kwargs = dict(self.resolve_env_kwargs_for_model(model_path))
        summary = read_json(training_summary_path(model_path, infer_run_name(model_path)))
        env_kwargs = env_kwargs_with_summary_hints(env_kwargs, summary)
        env_kwargs["scene_path"] = str(self.settings.scene_path)
        env_kwargs["max_episode_steps"] = 0
        return self._supported_env_kwargs(env_kwargs)

    def _supported_env_kwargs(self, env_kwargs: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in env_kwargs.items() if key in self.env_kwarg_names}

    def _load_runtime_for_path(self, model_path: Path) -> RuntimeModel:
        record = next((item for item in self.model_catalog.values() if item.path == model_path.resolve()), None)
        if record is None:
            self.model_catalog = build_model_catalog(self.settings.project_root, model_path, self._resolve_env_kwargs)
            record = next(item for item in self.model_catalog.values() if item.path == model_path.resolve())
        return self._runtime_for_record(record)

    def _runtime_for_record(self, record: ModelRecord) -> RuntimeModel:
        with self._runtime_lock:
            cached = self._runtime_cache.get(record.id)
            if cached is not None:
                return cached

            runtime = self._load_runtime(record)
            self._runtime_cache[record.id] = runtime
            return runtime

    def _load_runtime(self, record: ModelRecord) -> RuntimeModel:
        # SB3 PPO zip을 CPU로 로드하고, 실제 policy 객체에서 읽은 metadata로 카탈로그를 보강한다.
        env_kwargs = self._resolve_env_kwargs(record.path)
        ball_spawn_config = build_ball_spawn_config(env_kwargs, record.path)
        policy = PPO.load(str(record.path), device="cpu")
        metadata = with_loaded_policy_metadata(
            {
                **record.metadata,
                "ballSpawn": ball_spawn_config,
                "trainedRanges": trained_ranges(ball_spawn_config),
                "testedRanges": tested_ranges(ball_spawn_config),
            },
            policy,
        )
        enriched_record = ModelRecord(
            id=record.id,
            name=record.name,
            display_name=record.display_name,
            source=record.source,
            path=record.path,
            metadata=metadata,
        )
        self.model_catalog[record.id] = enriched_record
        return RuntimeModel(
            record=enriched_record,
            env_kwargs=env_kwargs,
            ball_spawn_config=ball_spawn_config,
            policy=policy,
            policy_lock=threading.Lock(),
            policy_message=f"Model: {metadata.get('name') or record.id}",
        )

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


class LiveSimulationHub:
    def __init__(self, service: LiveSimulationService) -> None:
        # 하나의 공유 session을 돌리고 여러 WebSocket 구독자에게 최신 frame을 fan-out한다.
        # LINK: backend/app/main.py:97
        self.service = service
        self.command_state = LiveCommandState()
        self.session: LiveSimulationSession | None = None
        self.subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self.last_frame: dict[str, Any] | None = None
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pingpong-live")

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="pingpong-live-hub")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        # 새 클라이언트가 붙으면 ready 메시지와 마지막 frame을 즉시 보내 초기 화면을 채운다.
        await self.start()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=3)
        async with self._lock:
            session = await self._ensure_session_locked()
            self.subscribers.add(queue)
            offer_queue_message(queue, {"type": "ready", "config": self.service.config_payload(session.runtime)})
            frame = self.last_frame or await self._run_sync(session.frame)
            offer_queue_message(queue, frame)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self.subscribers.discard(queue)

    async def handle_message(self, message: dict[str, Any]) -> None:
        # WebSocket 명령은 즉시 시뮬레이션을 건드리지 않고 다음 loop tick에서 적용될 상태로 저장한다.
        # LINK: backend/app/ball_spawn.py:104
        message_type = message.get("type")
        async with self._lock:
            if message_type == "reset":
                self.command_state.reset_requested = True
            elif message_type == "playback":
                playback = message.get("playback")
                if playback in {"playing", "paused"}:
                    self.command_state.playback = playback
            elif message_type == "spawnBall":
                self.command_state.ball_spawn_options = parse_ball_spawn_options(message, self.service.ball_spawn_config)
                self.command_state.ball_spawn_requested = True

    async def select_model(self, model_id: str) -> dict[str, Any]:
        # 모델 교체는 session 재생성과 frame 발행까지 한 lock 안에서 처리해 중간 상태 노출을 막는다.
        async with self._lock:
            previous_runtime = await self._run_sync(self.service.current_runtime)
            previous_session = self.session
            try:
                payload = await self._run_sync(self.service.select_model, model_id)
                next_session = await self._run_sync(self.service.create_session)
            except Exception as exc:
                await self._run_sync(self.service.activate_runtime, previous_runtime)
                self.session = previous_session
                LOGGER.exception("Model selection failed for %s", model_id)
                raise ModelSelectionError(model_selection_message(model_id, exc)) from exc

            self.session = next_session
            self.command_state.reset_requested = False
            self.command_state.reset_options = None
            self.command_state.ball_spawn_requested = False
            self.command_state.ball_spawn_options = None
            ready = {"type": "ready", "config": self.service.config_payload(self.session.runtime)}
            frame = await self._run_sync(self.session.frame, reset=True)
            self.last_frame = frame

        self._publish(ready)
        self._publish(frame)
        return payload

    async def _run(self) -> None:
        # shared live loop: 명령 적용, PPO action 계산, env step, frame publish를 반복한다.
        # LINK: backend/app/live_simulation.py:509
        while True:
            try:
                if not self.subscribers:
                    await asyncio.sleep(0.25)
                    continue

                async with self._lock:
                    session = await self._ensure_session_locked()
                    state = self.command_state
                    if state.reset_requested:
                        state.reset_requested = False
                        reset_options = state.reset_options
                        state.reset_options = None
                        frame = await self._run_sync(session.reset, reset_options)
                    elif state.ball_spawn_requested:
                        state.ball_spawn_requested = False
                        spawn_options = state.ball_spawn_options or {}
                        state.ball_spawn_options = None
                        frame = await self._run_sync(session.spawn_ball, spawn_options)
                    elif state.playback == "playing":
                        frame = await self._run_sync(session.step)
                    else:
                        frame = await self._run_sync(session.frame)

                    self.last_frame = frame
                    delay = session.control_dt if state.playback == "playing" else 0.1

                self._publish(frame)
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                raise
            except Exception:
                LOGGER.exception("Live simulation loop failed")
                self.command_state.playback = "paused"
                self._publish(
                    {
                        "type": "error",
                        "message": "Live simulation paused because the selected model could not run. Select another model.",
                    }
                )
                await asyncio.sleep(0.5)

    async def _ensure_session_locked(self) -> "LiveSimulationSession":
        # active runtime이 바뀌었거나 아직 session이 없으면 새 Gym/MuJoCo session을 만든다.
        if self.session is None or self.session.runtime is not self.service.active_runtime:
            self.session = await self._run_sync(self.service.create_session)
            self.last_frame = await self._run_sync(self.session.frame, reset=True)
        return self.session

    async def _run_sync(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        # MuJoCo/PPO의 blocking 계산을 전용 스레드에서 실행해 FastAPI event loop를 막지 않는다.
        loop = asyncio.get_running_loop()
        call = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(self._executor, call)

    def _publish(self, message: dict[str, Any]) -> None:
        for queue in list(self.subscribers):
            offer_queue_message(queue, message)


def offer_queue_message(queue: asyncio.Queue[dict[str, Any]], message: dict[str, Any]) -> None:
    # 느린 클라이언트가 오래된 frame을 쌓지 않도록 queue가 꽉 차면 가장 오래된 메시지를 버린다.
    if queue.full():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    try:
        queue.put_nowait(message)
    except asyncio.QueueFull:
        pass


def model_selection_message(model_id: str, error: Exception) -> str:
    detail = str(error)
    if "action_mode must be one of" in detail:
        return f"Model {model_id} is not compatible with the current runtime action mode."
    if "observation shape mismatch" in detail:
        return f"Model {model_id} is not compatible with the current runtime observation format."
    if "action shape mismatch" in detail:
        return f"Model {model_id} is not compatible with the current runtime action format."
    if detail:
        return f"Model {model_id} could not be loaded: {detail}"
    return f"Model {model_id} could not be loaded."


class LiveSimulationSession:
    def __init__(self, service: LiveSimulationService, runtime: RuntimeModel) -> None:
        # 하나의 runtime으로 실제 Gym 환경을 열고 policy/action shape 호환성을 검증한다.
        # LINK: backend/vendor/pingpong_rl2/src/pingpong_rl2/envs/gym_env.py:15
        self.service = service
        self.runtime = runtime
        self.env = service.env_class(**runtime.env_kwargs)
        self.episode_index = 0
        self.step_index = 0
        self.reset_pending = False
        self.observation: Any = None
        self.last_info: dict[str, Any] = {}
        self.last_reward: float | None = None
        self.last_contact: dict[str, Any] | None = None
        self.custom_reset_options: dict[str, Any] | None = None
        self.observation, self.last_info = self.env.reset(seed=service.settings.seed)
        self.runtime.control_dt = self.control_dt
        self._validate_policy_shapes()

    @property
    def ball_spawn_config(self) -> dict[str, Any]:
        return self.runtime.ball_spawn_config

    @property
    def sim(self) -> Any:
        return self.env.base_env.sim

    @property
    def control_dt(self) -> float:
        return float(self.sim.control_dt)

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        # episode를 새로 시작하고 reset option이 있으면 다음 reset에도 이어 쓰도록 저장한다.
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
        self._validate_observation_shape()
        return self.frame(reset=True)

    def spawn_ball(self, options: dict[str, Any]) -> dict[str, Any]:
        # 공 스폰 UI 명령은 환경 전체 reset 대신 현재 racket 기준 공 위치/속도만 재배치한다.
        # LINK: backend/app/ball_spawn.py:104
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
        self._validate_observation_shape()
        self.last_info = self._spawn_info(ball_xy_offset)
        return self.frame(reset=True)

    def step(self) -> dict[str, Any]:
        # observation -> PPO action -> Gym step -> frame 생성이 live inference의 한 tick이다.
        # LINK: backend/vendor/pingpong_rl2/src/pingpong_rl2/envs/keepup_env.py:1625
        if self.reset_pending:
            return self.reset()

        action = self.runtime.predict(self.observation, deterministic=self.service.settings.deterministic_policy)
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

    def _validate_policy_shapes(self) -> None:
        # 모델 metadata와 현재 env의 observation/action 차원이 맞지 않으면 모델 전환을 실패시킨다.
        self._validate_observation_shape()
        expected_action_dim = int(self.runtime.metadata.get("actionDim") or 0)
        actual_action_dim = action_space_dim(getattr(self.env, "action_space", None))
        if expected_action_dim and actual_action_dim and expected_action_dim != actual_action_dim:
            raise ValueError(
                "Model/runtime action shape mismatch: "
                f"model expects {expected_action_dim} controls, runtime produced {actual_action_dim}."
            )

    def _validate_observation_shape(self) -> None:
        expected_observation_dim = int(self.runtime.metadata.get("observationDim") or 0)
        actual_observation_dim = int(self.runtime.policy_observation(self.observation).shape[0])
        if expected_observation_dim and expected_observation_dim != actual_observation_dim:
            raise ValueError(
                "Model/runtime observation shape mismatch: "
                f"model expects {expected_observation_dim} values, runtime produced {actual_observation_dim}."
            )

    def _clear_episode_counters(self) -> None:
        # 수동 공 스폰 뒤에도 reward/contact 통계가 새 episode처럼 보이도록 내부 카운터를 초기화한다.
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
        # 브라우저가 3D scene에 그대로 복사할 MuJoCo state와 UI 지표를 하나의 frame으로 직렬화한다.
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
            "policyMessage": self.runtime.policy_message,
            "modelId": self.runtime.record.id,
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


def action_space_dim(action_space: Any) -> int | None:
    shape = getattr(action_space, "shape", None)
    if isinstance(shape, tuple) and shape:
        return int(shape[0])
    if isinstance(shape, list) and shape:
        return int(shape[0])
    return None


def legacy_policy_observation(observation: np.ndarray, expected_dim: int) -> np.ndarray | None:
    # 예전 정책 zip이 기대하는 26D/29D observation만 현재 observation에서 호환 추출한다.
    if observation.shape[0] < 35:
        return None
    if expected_dim == 29:
        return np.concatenate((observation[:17], observation[20:32])).astype(np.float32, copy=False)
    if expected_dim == 26:
        return np.concatenate((observation[:17], observation[20:29])).astype(np.float32, copy=False)
    return None


def portable_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)


def display_model_name(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_best_model"):
        return stem[:-11]
    if stem.endswith("_model"):
        return stem[:-6]
    return stem


def trained_ranges(ball_spawn_config: dict[str, Any]) -> dict[str, dict[str, float]]:
    ranges = ball_spawn_config.get("ranges", {})
    return {
        key: {
            "min": float(value.get("trainedMin", value.get("min", 0.0))),
            "max": float(value.get("trainedMax", value.get("max", 0.0))),
        }
        for key, value in ranges.items()
        if isinstance(value, dict)
    }


def tested_ranges(ball_spawn_config: dict[str, Any]) -> dict[str, dict[str, float]]:
    ranges = ball_spawn_config.get("ranges", {})
    return {
        key: {
            "min": float(value.get("min", 0.0)),
            "max": float(value.get("max", 0.0)),
        }
        for key, value in ranges.items()
        if isinstance(value, dict)
    }


def model_sort_key(model: dict[str, Any], active_model_id: str) -> tuple[int, int, int, str]:
    model_id = str(model.get("id") or "")
    name = str(model.get("displayName") or model.get("name") or model_id)
    dimension = model.get("sortDimension")
    version = model.get("sortVersion")
    active_rank = 0 if str(model.get("id") or "") == active_model_id else 1
    return (
        -int(dimension) if isinstance(dimension, int) else 1,
        -int(version) if isinstance(version, int) else 0,
        active_rank,
        name,
    )
