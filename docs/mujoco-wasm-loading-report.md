# MuJoCo WASM Loading Report

## Summary

Zalo `mujoco_wasm` and the RoboPianist web demo use the same broad pattern that this project now uses:

- MuJoCo compiled to WebAssembly runs physics in the browser.
- Model files are placed into Emscripten's virtual filesystem.
- MuJoCo `model` and `data` buffers are read from JavaScript.
- Three.js renders MuJoCo geoms and mesh buffers.
- The animation loop advances MuJoCo with `mj_step`.

The major bug in this project was not "MuJoCo WASM is slow." The problem was that the web app loaded a large binary model through `MjVFS.addBuffer()`, which copied the 38 MB model into the WASM-side VFS very slowly in the browser. This blocked first render for about 20 seconds.

The fix was to match the Zalo/RoboPianist style more closely: write model bytes into Emscripten `FS` with `module.FS.writeFile(...)`, then load the model from that filesystem path. After this change, local cold initialization measured about 0.4 seconds instead of about 20 seconds.

## What Zalo Does

Repository: https://github.com/zalo/mujoco_wasm

Observed implementation:

- Loads MuJoCo with `load_mujoco`.
- Creates `/working` in Emscripten FS with `mujoco.FS.mkdir` and `mujoco.FS.mount`.
- Writes XML/assets into that FS with `mujoco.FS.writeFile`.
- Loads a model from `/working/...`.
- Creates `MjData`.
- Recreates MuJoCo geoms in Three.js.
- Uses `OrbitControls` for camera movement.
- Uses `lil-gui` for the right-side Controls panel.
- Runs physics in a render loop with `mj_step`.

Why it appears fast:

- The default demo scene is small.
- It does not load a 38 MB Franka visual model on first paint.
- It uses the Emscripten FS path, not the slow `MjVFS.addBuffer()` path for large binary payloads.

## What RoboPianist Does

Repository: https://github.com/google-research/robopianist

The `google-research/robopianist` repository is mainly the Python benchmark/environment/training code. It uses MuJoCo and `dm_control`/Composer for the actual RoboPianist environments.

The website embeds a separate demo:

- Page: https://kzakka.com/robopianist/
- Demo repo: https://github.com/kevinzakka/robopianist-demo

Observed web-demo implementation:

- Uses `mujoco_wasm` and Three.js.
- Uses the same Zalo-style `FS.mkdir`, `FS.mount`, and model load pattern.
- Uses `OrbitControls`.
- Uses `lil-gui` for the right-side panel.
- Loads a piano/hand scene and precomputed `.npy` action sequences.
- Replays action arrays into `simulation.ctrl()` rather than running the full Python RL environment in the browser.

Important distinction:

RoboPianist's web demo is not running the Python `dm_control` environment directly in the browser. The Python repository defines/exported the environment; the web demo packages a browser-compatible MuJoCo scene plus control/action data.

## What Was Wrong Here

The previous implementation had several separate problems.

1. Model fidelity drifted

The web scene had been simplified too aggressively. That made the arm look unlike the native Franka Emika Panda scene and made the physical/visual environment feel different from the MacBook MuJoCo viewer.

Fix:

- Restored the exact RL MJCF source into the compiled web model.
- Compiled the RL `assets/scene.xml` into `frontend/public/assets/mujoco/pingpong_scene.mjb`.
- The runtime now loads the compiled MJB scene.

2. Asset loading used the wrong large-file path

The slow path was:

```ts
vfs.addBuffer(file, bytes);
MjModel.from_binary_path(scene, vfs);
```

For the 38 MB MJB, browser initialization stalled around 20 seconds at `addBuffer`.

Fix:

```ts
module.FS.writeFile("/pingpong_model/pingpong_scene.mjb", bytes);
MjModel.from_binary_path("/pingpong_model/pingpong_scene.mjb", emptyVfs);
```

Measured result on local dev Chrome:

- Before: about 20 seconds after StrictMode removal, about 42 seconds with React dev double-mount.
- After: about 418 ms to MuJoCo-ready status.

3. React development StrictMode doubled initialization

React StrictMode intentionally runs effects twice in development. That caused the MuJoCo model and policy to initialize twice.

Fix:

- Removed the top-level `StrictMode` wrapper from `frontend/src/main.tsx`.

4. Renderer rebuilt expensive mesh data

The renderer previously did extra Three.js geometry work. MuJoCo compiled models already expose mesh normals.

Fix:

- Use MuJoCo mesh normals directly when creating Three.js geometry.
- Avoid unnecessary vertex merging work.

5. Public assets contained duplicate source meshes

After switching to MJB, the original XML and Franka mesh files in `frontend/public/assets/mujoco` were no longer requested by the browser.

Fix:

- Deleted the public `franka/` source mesh directory.
- Deleted public `scene.xml`.
- Left only:

```text
frontend/public/assets/mujoco/asset-manifest.json
frontend/public/assets/mujoco/pingpong_scene.mjb
```

The regeneration script now reads source assets from:

```text
/Users/pilt/project-collection/ros2/graduation-prj/pingpong_rl2/assets
```

It does not require copied source meshes in `frontend/public`.

## Why First Browser Visit Can Still Feel Slow

After the runtime fix, local initialization is fast. The remaining first-visit delay is mostly network payload size.

Current runtime payload includes:

- `pingpong_scene.mjb`: about 38 MB
- MuJoCo WASM: about 9 MB raw, about 2.3 MB gzip
- Policy JSON: about 0.28 MB
- App JS/CSS

The MJB compresses well:

```text
pingpong_scene.mjb raw: 40,351,869 bytes
pingpong_scene.mjb gzip: 14,672,562 bytes
```

So if the browser's first visit is taking around 5 seconds, the likely cause is that the server is sending the 38 MB MJB uncompressed or with weak cache/compression settings. Zalo and RoboPianist feel faster because their first-load assets are much smaller:

- Zalo assets directory checked locally: about 11 MB total.
- RoboPianist demo scenes checked locally: about 3.5 MB.
- This project's compiled Panda scene MJB: about 38 MB raw.

## Current State

Runtime now follows the important common pattern:

- MuJoCo WASM physics in browser.
- Emscripten FS for model bytes.
- MJB model loading.
- Three.js rendering from MuJoCo model/data.
- No duplicate public mesh bundle.

Verified:

- `npm run compile:mujoco` succeeds from the RL asset directory.
- `npm run build` succeeds.
- Local Chrome smoke test reached MuJoCo-ready status in about 418 ms.

## Remaining Work

To make remote first visit closer to 1 second:

1. Serve `.mjb` with gzip or Brotli compression.
2. Set long-lived cache headers for immutable assets.
3. Optionally precompress `pingpong_scene.mjb.gz` during build and serve it with `Content-Encoding: gzip`.
4. If first visit still needs to be faster, create a lower-size visual mesh/MJB variant while preserving the same collision/physics model, or show an immediate lightweight placeholder while the full MJB streams.

