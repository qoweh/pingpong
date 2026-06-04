# Web Deployment

Development is done on an Apple Silicon MacBook. Deployment target is an ASUS Ubuntu home server with an Intel x86_64 CPU.

Runtime stack:

```text
frontend build
-> Python FastAPI/Uvicorn backend
-> live pingpong_rl2 PPO simulation
-> host port 8079
-> Nginx Proxy Manager
```

Architecture notes:

- Build directly on the ASUS server for a native `linux/amd64` image.
- If building on the M1 MacBook for the ASUS server, use `DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose build`.
- The backend requires Python MuJoCo, Stable-Baselines3, and the selected PPO model artifact.
- The browser still loads MuJoCo WASM and the compiled MJB scene for rendering.
- `@mujoco/mujoco` is pinned to `3.8.0` to match the Python MuJoCo runtime.

Required runtime data:

```text
backend/vendor/pingpong_rl2/src
rl/assets/scene.xml
rl/assets/franka/**
rl/artifacts/<selected_model>/<selected_model>_model.zip
rl/artifacts/<selected_model>/<selected_model>_training_summary.json
frontend/public/assets/mujoco/pingpong_scene.mjb
```

Health check:

```sh
curl http://localhost:8079/api/health
```
