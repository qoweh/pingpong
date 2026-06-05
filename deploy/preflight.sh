#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$(pwd)}"
cd "$PROJECT_ROOT"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

MODEL_PATH="${PINGPONG_POLICY_MODEL_PATH:-rl/artifacts/keep_v39_17d/keep_v39_17d_model.zip}"
SCENE_PATH="${PINGPONG_MUJOCO_SCENE_PATH:-rl/assets/scene.xml}"
RL_SOURCE_ROOT="${PINGPONG_RL_SOURCE_ROOT:-backend/vendor/pingpong_rl2}"

fail() {
  if [ "${GITHUB_ACTIONS:-}" = "true" ]; then
    printf '::error::%s\n' "$1" >&2
  else
    printf 'ERROR: %s\n' "$1" >&2
  fi
  exit 1
}

require_file() {
  local path="$1"
  local label="$2"
  if [ ! -f "$path" ]; then
    fail "$label is missing: $path. Put the runtime asset in the server project before docker compose build."
  fi
}

require_dir() {
  local path="$1"
  local label="$2"
  if [ ! -d "$path" ]; then
    fail "$label is missing: $path. Put the runtime asset in the server project before docker compose build."
  fi
}

require_file "$MODEL_PATH" "Policy model"
require_file "$SCENE_PATH" "MuJoCo scene"
require_dir "$RL_SOURCE_ROOT/src/pingpong_rl2" "Vendored RL package"
require_dir "rl/assets/franka" "Franka assets"
require_file "frontend/public/assets/mujoco/pingpong_scene.mjb" "Compiled MuJoCo scene"

printf 'Runtime asset preflight passed.\n'
printf '  model: %s\n' "$MODEL_PATH"
printf '  scene: %s\n' "$SCENE_PATH"
printf '  rl source: %s\n' "$RL_SOURCE_ROOT/src/pingpong_rl2"
