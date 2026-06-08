from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = PACKAGE_ROOT
WORKSPACE_ROOT = PACKAGE_ROOT.parent
DOCS_ROOT = PACKAGE_ROOT / "docs"
ARTIFACT_ROOT = PACKAGE_ROOT / "artifacts"
PPO_RUNS_ROOT = ARTIFACT_ROOT / "ppo_runs"
BENCHMARK_ROOT = ARTIFACT_ROOT / "benchmarks"
ASSET_ROOT = PACKAGE_ROOT / "assets"
SCENE_XML_PATH = ASSET_ROOT / "scene.xml"


def resolve_input_path(path: Path) -> Path:
    # 입력 파일은 절대경로, 현재 작업 디렉터리, 패키지 루트 순서로 해석한다.
    if path.is_absolute():
        return path.resolve()

    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path

    return (PACKAGE_ROOT / path).resolve()


def resolve_output_path(path: Path) -> Path:
    # 출력 파일은 패키지 루트를 기준으로 만들어 학습 artifact 위치를 일정하게 유지한다.
    if path.is_absolute():
        return path.resolve()
    return (PACKAGE_ROOT / path).resolve()
