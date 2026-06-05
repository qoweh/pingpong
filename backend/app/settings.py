from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL_PATH = "rl/artifacts/keep_v39_17d/keep_v39_17d_model.zip"
DEFAULT_SCENE_PATH = "rl/assets/scene.xml"


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

    raw_model_path = value("PINGPONG_POLICY_MODEL_PATH", DEFAULT_MODEL_PATH)
    raw_scene_path = value("PINGPONG_MUJOCO_SCENE_PATH", DEFAULT_SCENE_PATH)
    model_path = resolve_existing_file(
        raw_model_path,
        project_root,
        env_name="PINGPONG_POLICY_MODEL_PATH",
        label="Policy model",
        include_stable_baselines_zip_candidate=True,
    )
    scene_path = resolve_existing_file(
        raw_scene_path,
        project_root,
        env_name="PINGPONG_MUJOCO_SCENE_PATH",
        label="MuJoCo scene",
    )
    return AppSettings(
        project_root=project_root,
        frontend_dist=resolve_path(value("PINGPONG_FRONTEND_DIST", "frontend/dist"), project_root),
        rl_source_root=resolve_path(
            value("PINGPONG_RL_SOURCE_ROOT", "backend/vendor/pingpong_rl2"),
            project_root,
        ),
        model_path=model_path,
        scene_path=scene_path,
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


def resolve_existing_file(
    raw_path: str,
    project_root: Path,
    *,
    env_name: str,
    label: str,
    include_stable_baselines_zip_candidate: bool = False,
) -> Path:
    path = resolve_path(raw_path, project_root)
    candidates = [path]

    if include_stable_baselines_zip_candidate:
        if path.name.endswith(".zip.zip"):
            candidates.append(path.with_name(path.name.removesuffix(".zip")))
        elif path.suffix != ".zip":
            candidates.append(Path(f"{path}.zip"))

    for candidate in dedupe_paths(candidates):
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        missing_runtime_file_message(
            label=label,
            env_name=env_name,
            raw_path=raw_path,
            candidates=dedupe_paths(candidates),
            project_root=project_root,
        )
    )


def dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def missing_runtime_file_message(
    *,
    label: str,
    env_name: str,
    raw_path: str,
    candidates: list[Path],
    project_root: Path,
) -> str:
    candidate_lines = "\n".join(f"  - {display_path(candidate, project_root)}" for candidate in candidates)
    return (
        f"{label} file is missing.\n"
        f"{env_name}={raw_path}\n"
        f"Checked:\n{candidate_lines}\n"
        "For Docker deployment, make sure runtime files exist in the server project "
        "before running docker compose build, because backend/Dockerfile copies ./rl into the image."
    )


def display_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}
