# Web Deployment

Development is done on an Apple Silicon MacBook. Deployment target is an ASUS Ubuntu home server with an Intel x86_64 CPU.

The project avoids architecture-specific native build output in source control:

- Frontend dependencies are installed inside the build environment.
- Spring Boot runs on a Java 17 runtime image.
- Docker images used by the Dockerfile are multi-architecture images.
- The MuJoCo runtime in the browser uses WASM, not host-native MuJoCo binaries.

Architecture note:

- Building directly on the ASUS server produces a native `linux/amd64` image.
- Building on the M1 MacBook for the ASUS server should use an explicit `linux/amd64` Docker platform.
- Building on the M1 MacBook for local testing can use Docker's default Apple Silicon platform.

Serving flow:

```text
frontend build
-> backend/src/main/resources/static
-> Spring Boot embedded Tomcat
-> host port 8079
-> Nginx Proxy Manager
```

WASM serving notes:

- `.wasm` is served as `application/wasm`.
- The MVP uses single-threaded `@mujoco/mujoco`.
- Multi-threaded WASM is not enabled, so cross-origin isolation headers are not required for the MVP.
