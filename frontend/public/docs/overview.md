# Overview

Ping-Pong Keep-Up with Reinforcement Learning shows a Franka Panda robot arm with a ping-pong paddle keeping a ball in the air in a MuJoCo simulation.

The browser demo runs the MuJoCo model with WebAssembly, renders the scene with Three.js, and exposes only the controls needed for the demo:

- Ball position
- Camera mode
- Visualization toggles

System flow:

```text
PPO training artifacts
-> browser-loadable policy export
-> MuJoCo WASM simulation
-> Three.js rendering
-> Spring Boot static serving
-> Nginx Proxy Manager
```

Reference training repository:

https://github.com/qoweh/ros2-study/tree/main/graduation-prj/pingpong_rl2
