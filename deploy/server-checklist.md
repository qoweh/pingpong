# Server Checklist

이 문서는 Docker Compose로 웹 서비스를 배포할 때 확인할 항목을 정리한다. 현재 배포 기준은 CPU-only `linux/amd64` 서버다.

## 서버 조건

| 항목 | 기준 |
| --- | --- |
| OS | Ubuntu 또는 Docker Compose를 지원하는 Linux |
| CPU architecture | `x86_64` / `amd64` |
| GPU | 필요 없음 |
| 외부 port | 8079 |
| 필수 디렉토리 | project root, `.env`, `rl/assets`, `rl/artifacts` |

## 배포 절차

```sh
docker compose build
docker compose up -d
curl http://localhost:8079/api/health
```

서버에서 직접 build하면 Docker가 해당 서버에 맞는 native image를 만든다.

Apple Silicon Mac에서 서버용 이미지를 만들 때는 platform을 명시한다.

```sh
DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose build
```

로컬 Mac 테스트만 할 때는 기본 Apple Silicon image를 써도 된다. 서버 배포용이면 서버에서 직접 build하거나 `DOCKER_DEFAULT_PLATFORM=linux/amd64`를 사용한다.

## 배포 전 확인

- `.env`의 `PINGPONG_POLICY_MODEL_PATH`가 서버에 존재하는 파일을 가리키는지 확인한다.
- `rl/assets/scene.xml`과 Franka mesh 파일이 있는지 확인한다.
- `frontend/public/assets/mujoco/pingpong_scene.mjb`가 현재 scene과 맞는지 확인한다.
- `/api/live` WebSocket이 proxy에서 차단되지 않는지 확인한다.
- 첫 접속 속도를 위해 `.js`, `.css`, `.wasm`, `.mjb` 파일에 gzip 또는 Brotli 압축을 적용한다.
