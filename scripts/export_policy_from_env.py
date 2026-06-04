from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
FRONTEND_POLICY_PATH = ROOT / "frontend/public/assets/policy/final-policy.json"
TRAINING_POLICY_PATH = ROOT / "training-artifacts/final-policy/final-policy.json"
POLICY_MANIFEST_PATH = ROOT / "frontend/public/assets/policy/policy-manifest.json"


def main() -> None:
    env = read_env(ENV_PATH)
    source_model = env.get("PINGPONG_POLICY_MODEL_PATH")
    if not source_model:
        raise SystemExit("PINGPONG_POLICY_MODEL_PATH is missing from .env")

    model_path = resolve_model_path(source_model)
    model = PPO.load(model_path, device="cpu")
    policy = model.policy

    layers = []
    for module in list(policy.mlp_extractor.policy_net) + [policy.action_net]:
        if isinstance(module, nn.Linear):
            layers.append(
                {
                    "type": "linear",
                    "weight": module.weight.detach().cpu().tolist(),
                    "bias": module.bias.detach().cpu().tolist(),
                }
            )

    action_space = model.action_space
    observation_size = int(model.observation_space.shape[0])
    action_size = int(action_space.shape[0])
    run_name = infer_run_name(source_model)
    export: dict[str, Any] = {
        "format": "json-mlp",
        "name": run_name,
        "sourceModel": source_model,
        "observationSize": observation_size,
        "actionSize": action_size,
        "actionLow": action_space.low.astype(float).tolist(),
        "actionHigh": action_space.high.astype(float).tolist(),
        "activation": "tanh",
        "squashOutput": bool(getattr(policy, "squash_output", False)),
        "layers": layers,
    }

    write_json(FRONTEND_POLICY_PATH, export)
    write_json(TRAINING_POLICY_PATH, export)
    write_json(
        POLICY_MANIFEST_PATH,
        {
            "format": "json-mlp",
            "name": run_name,
            "file": "/assets/policy/final-policy.json",
            "sourceModel": source_model,
            "observationSize": observation_size,
            "actionSize": action_size,
            "message": "PPO actor weights are exported from PINGPONG_POLICY_MODEL_PATH.",
        },
    )

    print(f"exported {run_name}")
    print(f"sourceModel={source_model}")
    print(f"observationSize={observation_size} actionSize={action_size}")


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


def resolve_model_path(source_model: str) -> Path:
    path = Path(source_model)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise SystemExit(f"Policy model does not exist: {path}")
    return path


def infer_run_name(source_model: str) -> str:
    path = Path(source_model)
    name = path.stem
    return name.removesuffix("_best_model").removesuffix("_model")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
