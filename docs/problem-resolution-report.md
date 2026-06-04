# Problem Resolution Report

Date: 2026-06-04

## What Was Wrong

- The browser had drifted away from the original Python RL environment.
- Some behavior was hand-ported to TypeScript or replayed from exported rollout data.
- The Spring backend only served static files and health checks, so it did not help with live MuJoCo/RL execution.
- The initial page could show changing fallback values before MuJoCo WASM and the live policy stream were ready.
- Docs routing conflicted with FastAPI's default Swagger UI at `/docs`.
- Trajectory trail did not know when an episode reset, contact, or floor contact happened.
- The Three.js renderer did not interpret MuJoCo floor reflectance or shadows automatically.

## What Changed

- Replaced the web runtime with a Python FastAPI backend that imports vendored `pingpong_rl2` source and runs the original Gym env plus Stable-Baselines3 PPO model.
- The browser now uses MuJoCo WASM mainly as a renderer: it receives Python `qpos`, `qvel`, `ctrl`, contact, and reset state through `/api/live`.
- Removed the precomputed `rollout.json` replay path and old browser policy JSON path.
- Removed Spring because Python now owns static file serving, health checks, config, and live simulation.
- Moved the policy model path to `.env` through `PINGPONG_POLICY_MODEL_PATH`; current default is v25.
- Disabled FastAPI's built-in docs route so `/docs` opens the React documentation page.
- Added live ball reset controls that call the Python env reset path with `ball_height`, `ball_xy_offset`, and `ball_velocity`.
- Added reset/contact/floor event tracking so trajectory trail and contact markers clear predictably.
- Added a loading overlay and code-split the simulation canvas so the page shell appears earlier and users can see what is still loading.
- Enabled lightweight shadows and a low-resolution floor reflector in Three.js.
- Changed the local `rl/assets/franka/panda.xml` home keyframe to a bent Panda ready pose and recompiled the browser MJB.

## Current Runtime

```text
Browser React app
-> lazy-loaded Three.js + MuJoCo WASM viewer
-> WebSocket /api/live
-> Python FastAPI session
-> vendored pingpong_rl2 env
-> Stable-Baselines3 PPO model
-> Python MuJoCo physics
```

## Remaining Constraints

- First uncached load still includes large assets: WASM, JavaScript, and the compiled MJB scene.
- Shadows and reflection are implemented in Three.js, not by MuJoCo's native renderer, so they are approximations.
- Files under `rl/` are ignored by git. They must be copied to the home server or included in Docker build context/runtime data.
- Changing the Panda home keyframe changes the actual Python env reset state, not only the visual pose.

## Optimization Candidates

- Serve `.wasm`, `.js`, `.css`, and `.mjb` with Brotli or gzip precompression through Nginx.
- Add long-lived immutable cache headers for hashed Vite assets and the MuJoCo WASM bundle.
- Split or shrink the MJB by pruning unused meshes, textures, and high-detail visual geometry.
- Send WebSocket state as binary Float32 buffers instead of JSON arrays.
- Add adaptive graphics quality: disable reflection, lower shadow map size, and reduce renderer pixel ratio on weaker clients.
- Decouple simulation step rate from render rate and interpolate on the browser.
- Keep the Python backend warm instead of cold-starting on first user access.
- Build the Docker image natively on the Intel home server or force `linux/amd64` when building from the M1 Mac.
