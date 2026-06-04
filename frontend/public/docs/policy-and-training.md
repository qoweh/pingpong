# Policy and Training

Current source model:

```text
pmk_cf_self_rally_v28_robot_base_disk_model.zip
```

Browser artifact:

```text
frontend/public/assets/policy/final-policy.json
```

Training summary:

| Setting | Value |
| --- | --- |
| Algorithm | PPO |
| Run version | v28_robot_base_disk |
| Total timesteps | 2,000,000 |
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

The PPO actor weights are exported to JSON MLP format. Browser execution still needs the Python environment's action mapper/controller logic to be ported before those actions can be applied correctly to MuJoCo control values.
