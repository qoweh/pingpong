# Ping-Pong Keep-Up

Franka Panda 로봇팔이 탁구채로 공을 계속 받아 올리는 강화학습 시뮬레이션을 웹에서 보여주는 프로젝트다.

웹은 녹화 영상을 재생하지 않는다. 서버에서 원본 `pingpong_rl2` 환경과 선택된 제어 모델을 실행하고, 브라우저는 MuJoCo WebAssembly와 Three.js로 최신 상태를 렌더링한다.

## 프로젝트 구성

| 경로 | 역할 |
| --- | --- |
| `backend/` | 시뮬레이션 서버, WebSocket stream, 정적 파일 serving |
| `backend/vendor/pingpong_rl2/` | 원본 강화학습 패키지의 vendored source |
| `frontend/` | React, Three.js, MuJoCo WebAssembly viewer |
| `rl/assets/` | 런타임 MJCF scene과 Franka mesh asset |
| `rl/artifacts/` | 제어 모델 zip, training summary, 분석 결과 |
| `docs/` | 환경, 모델, 보상, 배포 참고 문서 |
| `deploy/` | 서버 배포 체크리스트와 proxy 설정 메모 |

`rl/` 아래의 모델과 asset은 용량이 크고 환경 의존성이 있으므로 저장소에서 제외되어 있다. 서버에 배포할 때는 이 디렉토리도 함께 준비해야 한다.

## 기본 설정

모델 경로는 `.env`에서 관리한다.

```text
PINGPONG_POLICY_MODEL_PATH=rl/artifacts/keep_v39_17d/keep_v39_17d_model.zip
PINGPONG_POLICY_DETERMINISTIC=true
PINGPONG_LIVE_SEED=251
```

새 모델을 사용할 때는 모델 zip과 training summary를 `rl/artifacts/<run_name>/`에 배치하고 `PINGPONG_POLICY_MODEL_PATH`만 바꾼다.

배포 전에 런타임 파일이 모두 있는지 확인한다.

```sh
bash deploy/preflight.sh
```

`Policy model is missing` 오류가 나면 `.env`의 `PINGPONG_POLICY_MODEL_PATH`가 서버 작업 디렉토리 안에 실제로 존재하는지 확인한다. Stable-Baselines3는 zip이 없을 때 내부적으로 `.zip` 후보를 덧붙일 수 있어서 로그에 `.zip.zip`처럼 보일 수 있지만, 실제 원인은 지정한 모델 zip이 없다는 뜻이다.

`docker compose up -d --build --force-recreate`만 실행해도 Docker build 단계에서 모델 zip, MuJoCo scene, vendored RL package, Franka asset 존재 여부를 검사한다. `deploy/preflight.sh`는 같은 검사를 build 전에 더 빨리 확인하고 싶을 때 사용한다.

preflight와 compose update를 항상 묶어서 실행하려면 wrapper를 사용한다.

```sh
bash deploy/update.sh
```

## 로컬 실행

프론트엔드 개발 서버:

```sh
cd frontend
npm install
npm run dev
```

서버 실행:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8079
```

Docker 실행:

```sh
docker compose up -d --build
```

## MuJoCo asset 갱신

`rl/assets/scene.xml`이나 Franka asset이 바뀌면 웹용 MJB 파일을 다시 만든다.

```sh
cd frontend
npm run compile:mujoco
```

## 서버 배포

대상 서버가 `linux/amd64`이면 서버에서 직접 이미지를 build하는 것이 가장 단순하다.

```sh
docker compose up -d --build --force-recreate
```

Apple Silicon Mac에서 `amd64` 서버용 이미지를 미리 만들 때는 platform을 명시한다.

```sh
DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose build
```

자세한 배포 체크리스트는 `deploy/server-checklist.md`와 `deploy/nginx-proxy-manager-notes.md`를 참고한다.
