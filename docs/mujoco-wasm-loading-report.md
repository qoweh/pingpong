# MuJoCo WASM Loading Report

## Current Architecture

The app now uses a Python live RL backend:

```text
FastAPI/Uvicorn
-> pingpong_rl2 original Gym env
-> Stable-Baselines3 PPO model
-> WebSocket qpos/qvel/ctrl/contact frames
-> browser MuJoCo WASM state rendering
-> Three.js viewer
```

This replaces both earlier experiments:

- Browser-side TypeScript policy/controller port.
- Precomputed `rollout.json` replay.

## Why Spring Was Removed

The Spring backend only served static files and `/health`. Once live RL execution moved to Python, keeping Spring would add another server without owning simulation logic.

The Python backend now handles:

- `/api/health`
- `/api/config`
- `/api/live` WebSocket
- built frontend static files from `frontend/dist`

## What Still Runs In The Browser

The browser still loads:

- `@mujoco/mujoco` WASM, pinned to `3.8.0`
- `frontend/public/assets/mujoco/pingpong_scene.mjb`
- Three.js renderer

The browser does not decide actions. It applies live qpos/qvel/ctrl state from Python and calls `mj_forward` so the compiled MuJoCo model buffers update for rendering.

## Required Runtime Data

```text
backend/vendor/pingpong_rl2/src
rl/assets/scene.xml
rl/assets/franka/**
rl/artifacts/<selected_model>/<selected_model>_model.zip
rl/artifacts/<selected_model>/<selected_model>_training_summary.json
frontend/public/assets/mujoco/pingpong_scene.mjb
```

The backend overrides the training summary's old absolute `scene_path` with `rl/assets/scene.xml`, so deployment is not tied to the MacBook path.

## Performance Notes

Homepage load is still affected by static rendering assets:

- MJB: about 38 MB raw
- MuJoCo WASM: about 8.6 MB raw, about 2.2 MB gzip

Live RL frames are small because they contain numeric state, not video. Keep the Python backend warm at process startup so page refreshes do not reload PPO/MuJoCo from scratch.
