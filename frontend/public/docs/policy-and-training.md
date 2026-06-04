# Policy and Training

Current source model:

```text
PINGPONG_POLICY_MODEL_PATH=rl/artifacts/pmk_cf_self_rally_v25/pmk_cf_self_rally_v25_model.zip
```

Browser artifact:

```text
frontend/public/assets/policy/final-policy.json
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

The PPO actor weights are exported to JSON MLP format from `PINGPONG_POLICY_MODEL_PATH`. The browser runs the JSON actor and maps its contact-frame command through a MuJoCo Jacobian racket controller.
