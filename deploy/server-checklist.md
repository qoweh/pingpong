# Server Checklist

Target server:

- Ubuntu on ASUS desktop
- Intel Core i5-8400
- x86_64 / amd64
- No GPU assumed
- External port: 8079

Deploy steps:

```sh
docker compose build
docker compose up -d
curl http://localhost:8079/api/health
```

The preferred deployment path is to build on the ASUS server so Docker creates a native `linux/amd64` image.

If the image is built on the M1 MacBook and then moved to the server, build it with an explicit amd64 platform:

```sh
DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose build
```

For local-only MacBook testing, Docker's default Apple Silicon image is fine. For the ASUS home server, prefer building on the server or using `DOCKER_DEFAULT_PLATFORM=linux/amd64`.
