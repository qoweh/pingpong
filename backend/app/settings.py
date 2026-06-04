from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppSettings:
    project_root: Path
    frontend_dist: Path
    rl_source_root: Path
    model_path: Path
    scene_path: Path
    server_port: int
    deterministic_policy: bool
    seed: int


def load_settings() -> AppSettings:
    project_root = Path(os.environ.get("PINGPONG_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
    env_values = read_env(project_root / ".env")

    def value(name: str, default: str) -> str:
        return os.environ.get(name) or env_values.get(name) or default

    model_path = resolve_path(
        value(
            "PINGPONG_POLICY_MODEL_PATH",
            "rl/artifacts/pmk_cf_self_rally_v25/pmk_cf_self_rally_v25_model.zip",
        ),
        project_root,
    )
    return AppSettings(
        project_root=project_root,
        frontend_dist=resolve_path(value("PINGPONG_FRONTEND_DIST", "frontend/dist"), project_root),
        rl_source_root=resolve_path(
            value("PINGPONG_RL_SOURCE_ROOT", "backend/vendor/pingpong_rl2"),
            project_root,
        ),
        model_path=model_path,
        scene_path=resolve_path(value("PINGPONG_MUJOCO_SCENE_PATH", "rl/assets/scene.xml"), project_root),
        server_port=int(value("SERVER_PORT", value("PINGPONG_WEB_PORT", "8079"))),
        deterministic_policy=parse_bool(value("PINGPONG_POLICY_DETERMINISTIC", "true")),
        seed=int(value("PINGPONG_LIVE_SEED", "251")),
    )


def read_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        values[key.strip()] = raw_value.strip().strip('"').strip("'")
    return values


def resolve_path(raw_path: str, project_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}
