# Policy and Training

Current source model:

```text
PINGPONG_POLICY_MODEL_PATH=rl/artifacts/pmk_cf_self_rally_v25/pmk_cf_self_rally_v25_model.zip
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

Runtime policy status:

The Python backend imports the vendored `pingpong_rl2` source, loads `PINGPONG_POLICY_MODEL_PATH`, creates the original `PingPongKeepUpGymEnv`, and calls the Stable-Baselines3 PPO policy on every environment step.

The browser receives live qpos/qvel/ctrl/contact frames over WebSocket and renders them with the compiled MuJoCo WASM scene.
