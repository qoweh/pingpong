# Policy and Training

Current source model:

```text
PINGPONG_POLICY_MODEL_PATH=rl/artifacts/pmk_cf_self_rally_v25/pmk_cf_self_rally_v25_model.zip
```

Browser replay artifact:

```text
frontend/public/assets/demo/rollout.json
```

Export command:

```text
conda run -n mujoco_env python scripts/export_web_rollout_from_env.py
```

Training summary:

| Setting | Value |
| --- | --- |
| Algorithm | PPO |
| Run version | v25 |
| Total timesteps | 500,000 |
| Parallel environments | 4 |
| n_steps | 512 |
| Batch size | 512 |
| Learning rate | 2e-5 |
| Gamma | 0.99 |
| Epochs | 2 |
| Clip range | 0.08 |
| Device | auto |
| Action mode | position_contact_frame_velocity_tilt_lateral_apex_residual |

Browser policy status:

The browser does not run a hand-ported TypeScript copy of the Python Gym environment or PPO policy. The export script imports the original `pingpong_rl2` Python package, loads `PINGPONG_POLICY_MODEL_PATH`, runs the original env/model/policy, and writes the resulting initial MuJoCo state, low-level actuator `ctrl` frames, action metadata, contact info, env config, and reset info to `rollout.json`.

The browser then loads the same compiled MuJoCo scene with WASM and replays the exported control frames with `mj_step`. To switch models, update `PINGPONG_POLICY_MODEL_PATH` and rerun the export command.
