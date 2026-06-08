# Ping-Pong Keep-Up Web

Franka Panda 로봇팔이 탁구채로 공을 계속 받아 올리는 MuJoCo 강화학습 모델을 웹에서 실시간으로 보여주는 프로젝트다. 브라우저는 녹화 영상을 재생하지 않고, 백엔드가 실행하는 `pingpong_rl2` 환경의 최신 상태를 WebSocket으로 받아 Three.js/MuJoCo WebAssembly scene에 반영한다.

## 구조

| 경로 | 역할 |
| --- | --- |
| `backend/` | FastAPI 서버, live simulation loop, WebSocket stream, 정적 파일 serving |
| `backend/vendor/pingpong_rl2/` | 원본 RL 패키지의 vendored source |
| `frontend/` | React, Three.js, MuJoCo WASM viewer |
| `rl/assets/` | 런타임 MJCF scene과 Franka mesh asset |
| `rl/artifacts/keep_v39_17d/` | 웹에서 사용하는 v39 17D policy와 분석 결과 |
| `docs/` | 발표/설명용 원본 문서 |
| `frontend/public/docs/` | 웹 DocsPage가 fetch하는 문서 복사본 |
| `deploy/` | preflight, update, server checklist, proxy notes |

## 런타임 파일

기본 런타임은 아래 파일을 사용한다.

```text
PINGPONG_POLICY_MODEL_PATH=rl/artifacts/keep_v39_17d/keep_v39_17d_model.zip
PINGPONG_MUJOCO_SCENE_PATH=rl/assets/scene.xml
PINGPONG_POLICY_DETERMINISTIC=true
PINGPONG_LIVE_SEED=251
```

모델이나 scene을 바꿀 때는 `.env`를 수정한 뒤 preflight를 먼저 통과시킨다.

```sh
bash deploy/preflight.sh
```

## 로컬 실행

프론트엔드:

```sh
cd /Users/pilt/project-collection/pingpong/frontend
npm install
npm run dev
```

백엔드:

```sh
cd /Users/pilt/project-collection/pingpong
conda activate mujoco_env
PYTHONPATH=backend/vendor/pingpong_rl2/src:. \
  python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8079
```

프론트엔드 production build:

```sh
cd /Users/pilt/project-collection/pingpong/frontend
npm run build
```

## MuJoCo web asset 갱신

`rl/assets/scene.xml` 또는 mesh가 바뀌면 브라우저용 MJB와 manifest를 다시 만든다.

```sh
cd /Users/pilt/project-collection/pingpong/frontend
npm run compile:mujoco
```

생성물은 `frontend/public/assets/mujoco/`에 들어간다.

## Docker / 서버 배포

```sh
cd /Users/pilt/project-collection/pingpong
bash deploy/update.sh
```

직접 실행할 때:

```sh
docker compose up -d --build --force-recreate
```

`linux/amd64` 서버용 이미지를 Apple Silicon에서 미리 만들 경우:

```sh
DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose build
```

배포 체크리스트는 `deploy/server-checklist.md`, proxy 설정은 `deploy/nginx-proxy-manager-notes.md`를 참고한다.

## 검증 명령

```sh
bash deploy/preflight.sh
conda run -n mujoco_env env PYTHONPATH=backend/vendor/pingpong_rl2/src:. \
  python -m compileall -q backend/app backend/vendor/pingpong_rl2/src
cd frontend && npm run build
```

현재 기본 모델은 원본 학습 저장소 `/Users/pilt/project-collection/ros2/mujoco/pingpong_rl2`의 `keep1_v39_17d_mid_curriculum_fixed` 계열을 웹 런타임용으로 짧게 복사한 것이다.
