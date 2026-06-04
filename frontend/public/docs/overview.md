# Overview

Ping-Pong Keep-Up with Reinforcement Learning shows a Franka Panda robot arm with a ping-pong paddle keeping a ball in the air in a MuJoCo simulation.

Current runtime architecture:

```text
Python live backend
-> original pingpong_rl2 Gym env
-> Stable-Baselines3 PPO policy
-> WebSocket qpos/qvel/ctrl/contact stream
-> browser MuJoCo WASM model for rendering state
-> Three.js viewer
-> Nginx Proxy Manager
```

The browser no longer runs a hand-ported TypeScript policy and no longer replays a precomputed rollout. The Python backend runs the original RL environment live and the frontend visualizes the latest MuJoCo state.

Controls:

- Playback
- Reset
- Camera mode
- Visualization toggles

Reference training repository:

https://github.com/qoweh/ros2-study/tree/main/graduation-prj/pingpong_rl2
