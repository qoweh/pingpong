from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .ball_spawn import BallSpawnConfig, build_ball_spawn_config


EnvResolver = Callable[[Path], dict[str, Any]]
SUPPORTED_ACTION_MODES = {
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
}
CATALOG_EXCLUDED_RUN_NAMES = {
    "keep1_v40_17d_v39_polish",
    "pmk_cf_self_rally_v29_first_contact_chase_sector",
}
CATALOG_EXCLUDED_ACTION_DIMS = {19}
CATALOG_REPRESENTATIVE_RUN_NAMES_BY_ACTION_DIM = {
    5: "ppo_keepup_v9",
}
SUMMARY_ENV_HINT_KEYS = {
    "action_mode",
    "action_limit",
    "lateral_action_limit",
    "vertical_action_limit",
    "tilt_action_limit",
    "followup_lift_action_limit",
    "success_velocity_threshold",
    "ball_height",
    "target_ball_height",
    "height_tolerance",
    "reset_ball_height_range",
    "reset_ball_height_bounds",
    "reset_xy_range",
    "reset_xy_sampling",
    "reset_velocity_xy_range",
    "reset_velocity_z_range",
    "reset_ball_angular_velocity_range",
    "target_offset_low",
    "target_offset_high",
    "target_tilt_limit",
    "target_pitch_range",
    "initial_target_tilt",
}
SUMMARY_ENV_KEY_ALIASES = {
    "position_gain": "controller_position_gain",
    "orientation_gain": "controller_orientation_gain",
    "max_position_step": "controller_max_position_step",
    "max_orientation_step": "controller_max_orientation_step",
}


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
        if infer_run_name(path) in CATALOG_EXCLUDED_RUN_NAMES:
            continue
        record = build_model_record(path, project_root, resolve_env_kwargs, used_ids)
        if record.metadata.get("actionDim") in CATALOG_EXCLUDED_ACTION_DIMS:
            continue
        records[record.id] = record
        used_ids.add(record.id)

    return assign_dimension_versions(records)


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
    env_kwargs = env_kwargs_with_summary_hints(resolve_env_kwargs(model_path), summary)
    ball_spawn = build_ball_spawn_config(env_kwargs, model_path)

    observation_dim = read_shape_dim(zip_metadata.get("observation_space"))
    action_dim = read_shape_dim(zip_metadata.get("action_space"))
    action_low = parse_space_bounds(zip_metadata.get("action_space"), "low_repr")
    action_high = parse_space_bounds(zip_metadata.get("action_space"), "high_repr")
    action_mode = str(env_kwargs.get("action_mode") or summary_value(summary, "action_mode") or "position")
    dimension_group = dimension_group_label(action_dim)
    detail_name = compact_model_detail(raw_run_name, model_path)
    public_name = raw_run_name
    model_id = unique_id(raw_run_name, used_ids)

    labels = normalized_action_labels(summary_action_labels(summary), action_mode, action_dim)
    runtime_compatible = action_mode in SUPPORTED_ACTION_MODES

    metadata = {
        "id": model_id,
        "name": public_name,
        "displayName": f"{dimension_group} · {detail_name}",
        "rawRunName": raw_run_name,
        "dimensionGroup": dimension_group,
        "versionLabel": None,
        "detailName": detail_name,
        "source": model_source(model_path),
        "path": portable_path(model_path, project_root),
        "algorithm": "PPO",
        "observationDim": observation_dim,
        "actionDim": action_dim,
        "actionMode": action_mode,
        "runtimeCompatible": runtime_compatible,
        "compatibilityMessage": None
        if runtime_compatible
        else f"This legacy model uses an unsupported action mode: {action_mode}.",
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


def assign_dimension_versions(records: dict[str, ModelRecord]) -> dict[str, ModelRecord]:
    grouped: dict[str, list[ModelRecord]] = {}
    for record in records.values():
        group = str(record.metadata.get("dimensionGroup") or dimension_group_label(record.metadata.get("actionDim")))
        grouped.setdefault(group, []).append(record)

    assigned: list[ModelRecord] = []
    for group, group_records in grouped.items():
        chronological = sorted(group_records, key=model_chronology_key)
        for version_number, record in enumerate(chronological, start=1):
            version_label = f"V{version_number}"
            detail_name = str(record.metadata.get("detailName") or record.metadata.get("rawRunName") or record.id)
            public_name = f"{group} {version_label}"
            display_name = f"{public_name} · {detail_name}"
            metadata = {
                **record.metadata,
                "name": public_name,
                "displayName": display_name,
                "dimensionGroup": group,
                "versionLabel": version_label,
                "sortDimension": sort_dimension_value(record.metadata.get("actionDim")),
                "sortVersion": version_number,
            }
            assigned.append(
                ModelRecord(
                    id=record.id,
                    name=public_name,
                    display_name=display_name,
                    source=record.source,
                    path=record.path,
                    metadata=metadata,
                )
            )

    ordered = sorted(assigned, key=assigned_record_sort_key)
    latest_by_dimension: dict[str, int] = {}
    representative_by_dimension: dict[str, str] = {}
    for record in ordered:
        group = str(record.metadata.get("dimensionGroup") or "")
        latest_by_dimension[group] = max(latest_by_dimension.get(group, 0), int(record.metadata.get("sortVersion") or 0))
        action_dim = record.metadata.get("actionDim")
        representative = CATALOG_REPRESENTATIVE_RUN_NAMES_BY_ACTION_DIM.get(action_dim)
        if representative:
            representative_by_dimension[group] = representative

    visible_ordered = []
    for record in ordered:
        group = str(record.metadata.get("dimensionGroup") or "")
        representative = representative_by_dimension.get(group)
        raw_run_name = str(record.metadata.get("rawRunName") or record.id)
        is_visible = (
            raw_run_name == representative
            if representative
            else int(record.metadata.get("sortVersion") or 0) == latest_by_dimension.get(group, 0)
        )
        metadata = {
            **record.metadata,
            "catalogVisible": is_visible,
        }
        visible_ordered.append(
            ModelRecord(
                id=record.id,
                name=record.name,
                display_name=record.display_name,
                source=record.source,
                path=record.path,
                metadata=metadata,
            )
        )
    return {record.id: record for record in visible_ordered}


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


def env_kwargs_with_summary_hints(env_kwargs: dict[str, Any], summary: dict[str, Any] | None) -> dict[str, Any]:
    if summary is None:
        return dict(env_kwargs)

    merged = dict(env_kwargs)
    for container in summary_env_hint_containers(summary):
        if not isinstance(container, dict):
            continue
        for key in SUMMARY_ENV_HINT_KEYS:
            if key in container:
                merged[key] = container[key]
        for source_key, target_key in SUMMARY_ENV_KEY_ALIASES.items():
            if source_key in container:
                merged[target_key] = container[source_key]
    return merged


def summary_env_hint_containers(summary: dict[str, Any]) -> list[Any]:
    containers: list[Any] = []
    config = summary.get("config")
    if isinstance(config, dict):
        containers.append(config)
        effective_env = config.get("effective_env")
        if isinstance(effective_env, dict):
            containers.extend(
                [
                    effective_env.get("core"),
                    effective_env.get("reset_randomization"),
                    effective_env.get("controller"),
                ]
            )
    containers.extend([summary.get("env_config"), summary])
    return containers


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
    lines = observation_architecture_lines(observation_dim)
    lines.extend(hidden_layer_line(index, width) for index, width in enumerate(layers, start=1))
    lines.append(f"Policy output: {action_dim} controls" if action_dim else "Policy output")
    lines.append("Value estimate: 1 number")
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
    lines = observation_architecture_lines(observation_dim)
    if isinstance(net_arch, list):
        widths = [value for value in net_arch if isinstance(value, int)]
        lines.extend(hidden_layer_line(index, width) for index, width in enumerate(widths, start=1))
    elif isinstance(net_arch, dict):
        pi_layers = net_arch.get("pi")
        vf_layers = net_arch.get("vf")
        if isinstance(pi_layers, list):
            lines.extend(f"Policy hidden {index}: {value} units" for index, value in enumerate(pi_layers, start=1) if isinstance(value, int))
        if isinstance(vf_layers, list):
            lines.extend(f"Value hidden {index}: {value} units" for index, value in enumerate(vf_layers, start=1) if isinstance(value, int))
    elif class_name:
        lines.append(class_name)
    lines.append(f"Policy output: {action_dim} controls" if action_dim else "Policy output")
    lines.append("Value estimate: 1 number")
    return lines


def observation_architecture_lines(observation_dim: int | None) -> list[str]:
    return [f"Observation input: {observation_dim} values"] if observation_dim else ["Observation input"]


def hidden_layer_line(index: int, width: int) -> str:
    return f"Hidden layer {index}: {width} units"


def action_labels(action_mode: str, action_dim: int | None) -> list[str]:
    if not action_dim or action_dim <= 0:
        return []

    labels = labels_for_action_mode(action_mode)
    if len(labels) < action_dim:
        labels.extend(f"Control {index + 1}" for index in range(len(labels), action_dim))
    return labels[:action_dim]


def normalized_action_labels(summary_labels: list[str] | None, action_mode: str, action_dim: int | None) -> list[str]:
    if not action_dim or action_dim <= 0:
        return []
    labels = list(summary_labels or [])
    fallback = action_labels(action_mode, action_dim)
    while len(labels) < action_dim:
        labels.append(fallback[len(labels)] if len(labels) < len(fallback) else f"Control {len(labels) + 1}")
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


def dimension_group_label(action_dim: Any) -> str:
    return f"{int(action_dim)}D" if isinstance(action_dim, int) and action_dim > 0 else "Unknown D"


def compact_model_detail(raw_run_name: str, model_path: Path) -> str:
    version = version_label_from_name(raw_run_name)
    descriptors = descriptor_words(raw_run_name)

    if raw_run_name == "keep_v39_17d":
        parts = ["Current", version or "v39"]
    elif raw_run_name.startswith(("keep1_", "keep_")):
        parts = ["Keep-up"]
        if version:
            parts.append(version)
    elif raw_run_name.startswith("pmk_cf_self_rally_"):
        parts = ["Contact-frame"]
        if "outward" in raw_run_name:
            parts.append("Outward")
        if version:
            parts.append(version)
    elif raw_run_name.startswith("pmk_cf_zero_init"):
        parts = ["Zero-init eval"]
        if version:
            parts.append(version)
    elif raw_run_name.startswith("ppo_keepup_"):
        parts = ["Early keep-up"]
        if version:
            parts.append(version)
    elif raw_run_name == "ppo_active_hit":
        parts = ["Active hit"]
    elif raw_run_name == "ppo_baseline":
        parts = ["Baseline"]
    else:
        parts = [raw_run_name]

    parts.extend(descriptors)
    if model_path.name.endswith("_best_model.zip"):
        parts.append("Best")
    return title_detail(" ".join(unique_words(parts)))


def unique_words(parts: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for part in parts:
        key = part.lower()
        if not part or key in seen:
            continue
        seen.add(key)
        unique.append(part)
    return unique


def title_detail(value: str) -> str:
    return value[:1].upper() + value[1:] if value else value


def version_label_from_name(raw_run_name: str) -> str | None:
    version = re.search(r"v\d+", raw_run_name)
    return version.group(0) if version else None


def model_chronology_key(record: ModelRecord) -> tuple[int, int, float, str]:
    raw_run_name = str(record.metadata.get("rawRunName") or record.id)
    return (
        version_number_from_name(raw_run_name),
        source_generation_rank(str(record.metadata.get("source") or "")),
        file_modified_time(record.path),
        raw_run_name,
    )


def assigned_record_sort_key(record: ModelRecord) -> tuple[int, int, str]:
    metadata = record.metadata
    return (
        -sort_dimension_value(metadata.get("actionDim")),
        -int(metadata.get("sortVersion") or 0),
        str(metadata.get("displayName") or record.display_name),
    )


def is_catalog_visible(record: ModelRecord) -> bool:
    return bool(record.metadata.get("catalogVisible"))


def sort_dimension_value(value: Any) -> int:
    return int(value) if isinstance(value, int) else -1


def version_number_from_name(raw_run_name: str) -> int:
    versions = [int(match) for match in re.findall(r"v(\d+)", raw_run_name)]
    return max(versions) if versions else 0


def source_generation_rank(source: str) -> int:
    if source == "current":
        return 99
    match = re.fullmatch(r"rl(\d+)", source)
    return int(match.group(1)) if match else 0


def file_modified_time(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def descriptor_words(raw_run_name: str) -> list[str]:
    descriptors: list[str] = []
    keywords = [
        ("first_contact", "First Contact"),
        ("racket_tracking", "Racket Tracking"),
        ("robot_base", "Robot Base"),
        ("balanced", "Balanced"),
        ("tracking_spin", "Tracking Spin"),
        ("tracking_staged", "Tracking Staged"),
        ("distribution", "Distribution"),
        ("strong_axis", "Strong Axis"),
        ("stable", "Stable"),
        ("guarded", "Guarded"),
        ("curriculum", "Curriculum"),
        ("recover", "Recover"),
        ("polish", "Polish"),
        ("long", "Long"),
        ("wide", "Wide"),
        ("mid", "Mid"),
        ("perf", "Performance"),
        ("init", "Init"),
        ("tilt", "Tilt"),
        ("rebound", "Rebound"),
        ("chase", "Chase"),
        ("sector", "Sector"),
        ("disk", "Disk"),
        ("fast", "Fast"),
        ("smoke", "Smoke"),
        ("baseline", "Baseline"),
        ("active_hit", "Active Hit"),
    ]
    for key, label in keywords:
        if key == "racket_tracking" and ("tracking_spin" in raw_run_name or "tracking_staged" in raw_run_name):
            continue
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
