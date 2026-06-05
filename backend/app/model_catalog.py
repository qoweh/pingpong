from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .ball_spawn import BallSpawnConfig, build_ball_spawn_config


EnvResolver = Callable[[Path], dict[str, Any]]


@dataclass(frozen=True)
class ModelRecord:
    id: str
    name: str
    display_name: str
    source: str
    path: Path
    metadata: dict[str, Any]


def build_model_catalog(
    project_root: Path,
    active_model_path: Path,
    resolve_env_kwargs: EnvResolver,
) -> dict[str, ModelRecord]:
    paths = representative_model_paths(project_root / "rl" / "artifacts")
    paths[active_model_path.resolve()] = active_model_path.resolve()

    records: dict[str, ModelRecord] = {}
    used_ids: set[str] = set()
    for path in sorted(paths.values(), key=lambda item: sort_key(item, project_root)):
        record = build_model_record(path, project_root, resolve_env_kwargs, used_ids)
        records[record.id] = record
        used_ids.add(record.id)

    return records


def representative_model_paths(artifacts_root: Path) -> dict[Path, Path]:
    if not artifacts_root.is_dir():
        return {}

    by_run_dir: dict[Path, Path] = {}
    for path in artifacts_root.rglob("*.zip"):
        if "checkpoints" in path.parts:
            continue
        if not (path.name.endswith("_model.zip") or path.name.endswith("_best_model.zip")):
            continue

        current = by_run_dir.get(path.parent)
        if current is None or is_preferred_model_path(path, current):
            by_run_dir[path.parent] = path.resolve()

    return {path.resolve(): path.resolve() for path in by_run_dir.values()}


def is_preferred_model_path(candidate: Path, current: Path) -> bool:
    candidate_is_regular = candidate.name.endswith("_model.zip") and not candidate.name.endswith("_best_model.zip")
    current_is_regular = current.name.endswith("_model.zip") and not current.name.endswith("_best_model.zip")
    if candidate_is_regular != current_is_regular:
        return candidate_is_regular
    return candidate.name < current.name


def build_model_record(
    model_path: Path,
    project_root: Path,
    resolve_env_kwargs: EnvResolver,
    used_ids: set[str],
) -> ModelRecord:
    raw_run_name = infer_run_name(model_path)
    zip_metadata = read_sb3_zip_metadata(model_path)
    summary_path = training_summary_path(model_path, raw_run_name)
    summary = read_json(summary_path)
    env_kwargs = resolve_env_kwargs(model_path)
    ball_spawn = build_ball_spawn_config(env_kwargs, model_path)

    observation_dim = read_shape_dim(zip_metadata.get("observation_space"))
    action_dim = read_shape_dim(zip_metadata.get("action_space"))
    action_low = parse_space_bounds(zip_metadata.get("action_space"), "low_repr")
    action_high = parse_space_bounds(zip_metadata.get("action_space"), "high_repr")
    action_mode = str(env_kwargs.get("action_mode") or summary_value(summary, "action_mode") or "position")
    public_name = public_model_name(raw_run_name, action_dim)
    model_id = unique_id(raw_run_name, used_ids)

    labels = normalized_action_labels(summary_action_labels(summary), action_mode, action_dim)

    metadata = {
        "id": model_id,
        "name": public_name,
        "displayName": display_model_name(raw_run_name, action_dim, model_path),
        "source": model_source(model_path),
        "path": portable_path(model_path, project_root),
        "algorithm": "PPO",
        "observationDim": observation_dim,
        "actionDim": action_dim,
        "actionMode": action_mode,
        "actionLabels": labels,
        "actionLow": action_low,
        "actionHigh": action_high,
        "ballSpawn": ball_spawn,
        "trainedRanges": trained_ranges(ball_spawn),
        "testedRanges": tested_ranges(ball_spawn),
        "trainingSummaryPath": portable_path(summary_path, project_root) if summary_path.is_file() else None,
        "policy": policy_metadata(zip_metadata, observation_dim, action_dim),
        "training": training_metadata(summary),
    }

    return ModelRecord(
        id=model_id,
        name=public_name,
        display_name=metadata["displayName"],
        source=metadata["source"],
        path=model_path.resolve(),
        metadata=metadata,
    )


def with_loaded_policy_metadata(metadata: dict[str, Any], policy: Any) -> dict[str, Any]:
    enriched = dict(metadata)
    policy_info = dict(enriched.get("policy") or {})
    architecture = loaded_policy_architecture(
        policy,
        int(enriched.get("observationDim") or 0),
        int(enriched.get("actionDim") or 0),
    )
    if architecture:
        policy_info["architecture"] = architecture
    policy_info["className"] = policy_info.get("className") or policy.policy.__class__.__name__
    enriched["policy"] = policy_info
    return enriched


def read_sb3_zip_metadata(model_path: Path) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(model_path) as archive:
            return json.loads(archive.read("data").decode("utf-8"))
    except (OSError, KeyError, json.JSONDecodeError, zipfile.BadZipFile):
        return {}


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def training_summary_path(model_path: Path, raw_run_name: str) -> Path:
    return model_path.parent / f"{raw_run_name}_training_summary.json"


def infer_run_name(model_path: Path) -> str:
    stem = model_path.stem
    if stem.endswith("_best_model"):
        return stem[:-11]
    if stem.endswith("_model"):
        return stem[:-6]
    return stem


def read_shape_dim(space_value: Any) -> int | None:
    if not isinstance(space_value, dict):
        return None
    shape = space_value.get("_shape")
    if isinstance(shape, list) and shape:
        value = shape[0]
        return int(value) if isinstance(value, int) else None
    return None


def parse_space_bounds(space_value: Any, key: str) -> list[float] | None:
    if not isinstance(space_value, dict):
        return None
    raw = space_value.get(key)
    if not isinstance(raw, str):
        return None
    values = [float(match.group(0)) for match in re.finditer(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:e[-+]?\d+)?", raw)]
    return values or None


def summary_value(summary: dict[str, Any] | None, key: str) -> Any:
    if summary is None:
        return None
    env_config = summary.get("env_config")
    if isinstance(env_config, dict) and key in env_config:
        return env_config[key]
    config = summary.get("config")
    if isinstance(config, dict) and key in config:
        return config[key]
    return summary.get(key)


def policy_metadata(zip_metadata: dict[str, Any], observation_dim: int | None, action_dim: int | None) -> dict[str, Any]:
    policy_class = zip_metadata.get("policy_class")
    policy_kwargs = zip_metadata.get("policy_kwargs")
    class_name = None
    if isinstance(policy_class, dict):
        module = policy_class.get("__module__")
        type_name = policy_class.get(":type:")
        class_name = "ActorCriticPolicy" if isinstance(type_name, str) and "ActorCriticPolicy" in type_name else None
        if class_name is None and isinstance(module, str):
            class_name = module.rsplit(".", 1)[-1]
    net_arch = policy_kwargs.get("net_arch") if isinstance(policy_kwargs, dict) else None
    architecture = default_policy_architecture(observation_dim, action_dim, class_name, net_arch)
    return {
        "className": class_name or "ActorCriticPolicy",
        "netArch": net_arch,
        "architecture": architecture,
    }


def loaded_policy_architecture(policy: Any, observation_dim: int, action_dim: int) -> list[str]:
    policy_object = getattr(policy, "policy", None)
    extractor = getattr(policy_object, "mlp_extractor", None)
    layers = linear_layer_widths(getattr(extractor, "policy_net", None))
    if not layers:
        layers = linear_layer_widths(getattr(extractor, "shared_net", None))
    lines = [f"Input ({observation_dim})"] if observation_dim else ["Input"]
    lines.extend(f"Dense({width})" for width in layers)
    lines.append(f"Actor Head ({action_dim})" if action_dim else "Actor Head")
    lines.append("Critic Head (1)")
    return lines


def linear_layer_widths(module: Any) -> list[int]:
    if module is None or not hasattr(module, "modules"):
        return []
    widths: list[int] = []
    for layer in module.modules():
        if layer.__class__.__name__ == "Linear" and hasattr(layer, "out_features"):
            widths.append(int(layer.out_features))
    return widths


def default_policy_architecture(
    observation_dim: int | None,
    action_dim: int | None,
    class_name: str | None,
    net_arch: Any,
) -> list[str]:
    lines = [f"Input ({observation_dim})" if observation_dim else "Input"]
    if isinstance(net_arch, list):
        lines.extend(f"Dense({value})" for value in net_arch if isinstance(value, int))
    elif isinstance(net_arch, dict):
        pi_layers = net_arch.get("pi")
        vf_layers = net_arch.get("vf")
        if isinstance(pi_layers, list):
            lines.extend(f"Actor Dense({value})" for value in pi_layers if isinstance(value, int))
        if isinstance(vf_layers, list):
            lines.extend(f"Critic Dense({value})" for value in vf_layers if isinstance(value, int))
    elif class_name:
        lines.append(class_name)
    lines.append(f"Actor Head ({action_dim})" if action_dim else "Actor Head")
    lines.append("Critic Head (1)")
    return lines


def action_labels(action_mode: str, action_dim: int | None) -> list[str]:
    if not action_dim or action_dim <= 0:
        return []

    labels = labels_for_action_mode(action_mode)
    if len(labels) < action_dim:
        labels.extend(f"Action[{index}]" for index in range(len(labels), action_dim))
    return labels[:action_dim]


def normalized_action_labels(summary_labels: list[str] | None, action_mode: str, action_dim: int | None) -> list[str]:
    if not action_dim or action_dim <= 0:
        return []
    labels = list(summary_labels or [])
    fallback = action_labels(action_mode, action_dim)
    while len(labels) < action_dim:
        labels.append(fallback[len(labels)] if len(labels) < len(fallback) else f"Action[{len(labels)}]")
    return labels[:action_dim]


def summary_action_labels(summary: dict[str, Any] | None) -> list[str] | None:
    if summary is None:
        return None
    for container in (summary, summary.get("env_config"), summary.get("config")):
        if not isinstance(container, dict):
            continue
        raw_labels = container.get("actionLabels") or container.get("action_labels")
        if isinstance(raw_labels, list) and all(isinstance(label, str) for label in raw_labels):
            return list(raw_labels)
    return None


def labels_for_action_mode(action_mode: str) -> list[str]:
    if action_mode == "position":
        return ["Target X", "Target Y", "Target Z"]
    if action_mode == "position_tilt":
        return ["Target X", "Target Y", "Target Z", "Tilt X", "Tilt Y"]
    if action_mode == "position_strike":
        return ["Strike X", "Strike Y", "Strike Z"]
    if action_mode == "position_strike_tilt":
        return ["Strike X", "Strike Y", "Strike Z", "Tilt X", "Tilt Y"]
    if action_mode == "position_strike_tilt_lift":
        return ["Strike X", "Strike Y", "Strike Z", "Tilt X", "Tilt Y", "Lift"]

    if action_mode.startswith("position_contact_frame"):
        labels = ["Radial X", "Tangent Y", "Strike Z", "Tilt X", "Tilt Y"]
        if "velocity" in action_mode:
            labels.extend(["Velocity Scale", "Outgoing X", "Outgoing Y"])
        if "tilt" in action_mode and "velocity_tilt" in action_mode:
            labels.extend(["Racket VZ", "Tilt Scale X", "Tilt Scale Y"])
        if "lateral" in action_mode:
            labels.extend(["Racket VX", "Racket VY"])
        if "apex" in action_mode:
            labels.extend(["Target Apex Z", "Strike Plane Z"])
        if "tracking" in action_mode:
            labels.extend(["Tracking X", "Tracking Y"])
        return labels

    return []


def trained_ranges(ball_spawn: BallSpawnConfig) -> dict[str, dict[str, float]]:
    ranges = ball_spawn.get("ranges", {})
    return {
        key: {
            "min": float(value.get("trainedMin", value.get("min", 0.0))),
            "max": float(value.get("trainedMax", value.get("max", 0.0))),
        }
        for key, value in ranges.items()
        if isinstance(value, dict)
    }


def tested_ranges(ball_spawn: BallSpawnConfig) -> dict[str, dict[str, float]]:
    ranges = ball_spawn.get("ranges", {})
    return {
        key: {
            "min": float(value.get("min", 0.0)),
            "max": float(value.get("max", 0.0)),
        }
        for key, value in ranges.items()
        if isinstance(value, dict)
    }


def training_metadata(summary: dict[str, Any] | None) -> dict[str, Any]:
    if summary is None:
        return {}
    config = summary.get("config")
    return {
        "runName": summary.get("run_name"),
        "timesteps": summary.get("completed_timesteps") or (config.get("total_timesteps") if isinstance(config, dict) else None),
        "preset": config.get("preset") if isinstance(config, dict) else None,
        "seed": config.get("seed") if isinstance(config, dict) else None,
    }


def public_model_name(raw_run_name: str, action_dim: int | None) -> str:
    dim = f"_{action_dim}d" if action_dim else ""
    if raw_run_name.startswith("keep1_"):
        stripped = raw_run_name.replace("keep1_", "keep_", 1)
        match = re.match(r"(keep_v\d+(?:_\d+d)?)(?:_|$)", stripped)
        return match.group(1) if match else stripped
    if raw_run_name.startswith("pmk_cf_self_rally_"):
        version = re.search(r"v\d+", raw_run_name)
        return f"{version.group(0)}{dim}" if version else f"contact_frame{dim}"
    if raw_run_name.startswith("ppo_keepup_"):
        version = re.search(r"v\d+", raw_run_name)
        return f"rl1_{version.group(0)}{dim}" if version else f"rl1{dim}"
    return raw_run_name


def display_model_name(raw_run_name: str, action_dim: int | None, model_path: Path) -> str:
    dim = f"{action_dim}D" if action_dim else "PPO"
    title_parts: list[str] = []
    version = re.search(r"v\d+", raw_run_name)
    if raw_run_name == "keep_v39_17d":
        title_parts = ["V39", dim, "Current"]
    elif raw_run_name.startswith("keep1_"):
        title_parts = [version.group(0).upper() if version else "Keep", dim]
        title_parts.extend(descriptor_words(raw_run_name))
    elif raw_run_name.startswith("pmk_cf_self_rally_"):
        title_parts = [version.group(0).upper() if version else "CF", dim]
        title_parts.extend(descriptor_words(raw_run_name))
    elif raw_run_name.startswith("ppo_keepup_"):
        title_parts = ["RL1", version.group(0).upper() if version else "KeepUp", dim]
        title_parts.extend(descriptor_words(raw_run_name))
    else:
        title_parts = [raw_run_name, dim]

    if model_path.name.endswith("_best_model.zip"):
        title_parts.append("Best")
    return " ".join(part for part in title_parts if part)


def descriptor_words(raw_run_name: str) -> list[str]:
    descriptors: list[str] = []
    keywords = [
        ("balanced", "Balanced"),
        ("tracking_spin", "Tracking Spin"),
        ("tracking_staged", "Tracking Staged"),
        ("strong_axis", "Strong Axis"),
        ("curriculum", "Curriculum"),
        ("recover", "Recover"),
        ("polish", "Polish"),
        ("long", "Long"),
        ("init", "Init"),
        ("tilt", "Tilt"),
        ("rebound", "Rebound"),
        ("baseline", "Baseline"),
        ("active_hit", "Active Hit"),
        ("robot_base", "Robot Base"),
    ]
    for key, label in keywords:
        if key in raw_run_name and label not in descriptors:
            descriptors.append(label)
    return descriptors[:2]


def model_source(model_path: Path) -> str:
    parts = model_path.parts
    if "legacy_models" not in parts:
        return "current"
    index = parts.index("legacy_models")
    return parts[index + 1] if len(parts) > index + 1 else "legacy"


def unique_id(raw_run_name: str, used_ids: set[str]) -> str:
    base = raw_run_name
    if base not in used_ids:
        return base
    index = 2
    while f"{base}_{index}" in used_ids:
        index += 1
    return f"{base}_{index}"


def sort_key(path: Path, project_root: Path) -> tuple[int, int, str]:
    relative = portable_path(path, project_root)
    source_rank = 0 if "keep_v39_17d" in relative else 1
    version = re.findall(r"v(\d+)", relative)
    version_rank = -int(version[-1]) if version else 0
    return (source_rank, version_rank, relative)


def portable_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)
