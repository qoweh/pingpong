# Simulation Environment

Runtime model:

- MuJoCo scene: `assets/scene.xml`
- Robot: Franka Panda
- End effector: paddle body attached under `hand`
- Ball geom: `ball_geom`
- Racket geom: `racket_head`
- Racket site: `racket_center`

Key physical values:

| Item | Value |
| --- | ---: |
| Control timestep | 0.02 s |
| MuJoCo timestep | 0.002 s |
| Ping-pong ball radius | 0.02 m |
| Ping-pong ball mass | 0.0027 kg |
| Paddle head radius | 0.084 m |
| Paddle head half-depth | 0.006 m |
| Target ball height above racket | 0.30 m |
| Height tolerance | 0.10 m |

The web demo serves only the runtime XML and mesh files required by `scene.xml` and `franka/panda.xml`.
