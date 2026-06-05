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
bash deploy/preflight.sh
docker compose build
docker compose up -d
curl http://localhost:8079/api/health
```

서버에서 직접 build하면 Docker가 해당 서버에 맞는 native image를 만든다.

직접 명령을 줄이고 싶으면 아래처럼 한 번에 실행해도 된다. Dockerfile에 같은 런타임 파일 검사가 들어 있어서 모델 zip이나 scene이 빠지면 build 단계에서 실패한다.

```sh
docker compose up -d --build --force-recreate
```

preflight를 build 전에 반드시 먼저 실행하고 싶으면 wrapper를 사용한다.

```sh
bash deploy/update.sh
```

Apple Silicon Mac에서 서버용 이미지를 만들 때는 platform을 명시한다.

```sh
DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose build
```

로컬 Mac 테스트만 할 때는 기본 Apple Silicon image를 써도 된다. 서버 배포용이면 서버에서 직접 build하거나 `DOCKER_DEFAULT_PLATFORM=linux/amd64`를 사용한다.

## 배포 전 확인

- `bash deploy/preflight.sh`가 통과하는지 확인한다.
- `.env`의 `PINGPONG_POLICY_MODEL_PATH`가 서버에 존재하는 파일을 가리키는지 확인한다.
- `rl/assets/scene.xml`과 Franka mesh 파일이 있는지 확인한다.
- `frontend/public/assets/mujoco/pingpong_scene.mjb`가 현재 scene과 맞는지 확인한다.
- `/api/live` WebSocket이 proxy에서 차단되지 않는지 확인한다.
- 첫 접속 속도를 위해 `.js`, `.css`, `.wasm`, `.mjb` 파일에 gzip 또는 Brotli 압축을 적용한다.

## 자주 나는 배포 오류

### 모델 zip 누락

서버 로그에 다음과 비슷한 오류가 보이면 모델 파일이 컨테이너 안에 들어가지 않은 상태다.

```text
FileNotFoundError: ... keep_v39_17d_model.zip.zip
```

Stable-Baselines3가 없는 모델 파일을 찾으면서 `.zip` 후보를 한 번 더 붙여 표시할 수 있다. 실제로 확인해야 하는 파일은 `.env`에 적은 `PINGPONG_POLICY_MODEL_PATH`다.

기본 설정이라면 서버 작업 디렉토리에 아래 파일이 있어야 한다.

```text
rl/artifacts/keep_v39_17d/keep_v39_17d_model.zip
```

Dockerfile은 build 시점의 `./rl`을 이미지에 복사한다. 따라서 `rl/assets`와 필요한 모델 zip은 `docker compose build` 전에 서버 작업 디렉토리에 준비되어 있어야 한다.
