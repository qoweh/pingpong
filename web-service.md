# Web Service Architecture

Current direction:

```text
Python FastAPI backend
-> vendored pingpong_rl2 source
-> Stable-Baselines3 PPO model
-> original MuJoCo/Gym environment live step
-> WebSocket /api/live
-> browser MuJoCo WASM + Three.js renderer
```

Spring Boot was removed because it only served static files and health checks. The Python backend is now required for live RL policy execution, so it also serves the built frontend and `/api/health`.

Runtime files needed in this `pingpong` directory:

```text
backend/vendor/pingpong_rl2/src
backend/requirements.txt
rl/assets/scene.xml
rl/assets/franka/**
rl/artifacts/<model>/<model>_model.zip
rl/artifacts/<model>/<model>_training_summary.json
frontend/public/assets/mujoco/pingpong_scene.mjb
```

Environment:

```text
PINGPONG_WEB_PORT=8079
PINGPONG_POLICY_MODEL_PATH=rl/artifacts/pmk_cf_self_rally_v25/pmk_cf_self_rally_v25_model.zip
PINGPONG_POLICY_DETERMINISTIC=true
PINGPONG_LIVE_SEED=251
```

Local development:

```sh
conda run -n mujoco_env python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8079
cd frontend
npm run dev
```

Deployment:

```sh
docker compose build
docker compose up -d
curl http://localhost:8079/api/health
```

The browser still downloads the MJB/WASM rendering assets. The live backend sends compact qpos/qvel/ctrl/contact frames; it does not stream video.
