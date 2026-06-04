# MDP Formulation

The current PPO run uses `position_contact_frame_velocity_tilt_lateral_apex_residual`.

Observation layout:

| Component | Dimension | Meaning |
| --- | ---: | --- |
| Joint positions | 7 | Panda joint angles |
| Joint velocities | 7 | Panda joint velocities |
| Racket position | 3 | `racket_center` world position |
| Racket velocity | 3 | Cartesian racket velocity |
| Target position | 3 | Controller target position |
| Ball position | 3 | Ball world position |
| Ball velocity | 3 | Ball linear velocity |
| Ball relative position | 3 | Ball position relative to racket |
| Predicted intercept relative XY | 2 | Predicted intercept offset |
| Predicted intercept time | 1 | Time-to-intercept estimate |
| Phase one-hot | 4 | Task phase |
| Contact context | 2 | Time since contact and clipped bounce count |
| Next intercept | 6 | Relative XY, time, reachability, distance, readiness |
| Desired outgoing velocity | 3 | Contact target velocity |
| Racket face normal | 3 | Paddle normal vector |
| Target tilt | 2 | Controller tilt target |
| Total | 55 | Policy input |

Action layout:

| Component | Dimension |
| --- | ---: |
| Position residual | 3 |
| Tilt residual | 2 |
| Contact-frame velocity residual | 3 |
| Racket vertical velocity and tilt scale residual | 3 |
| Racket XY velocity residual | 2 |
| Apex timing residual | 2 |
| Total | 15 |
