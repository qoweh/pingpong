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

The same source tree can be built on the M1 MacBook or on the amd64 server because the final runtime is a Java container serving static frontend assets.
