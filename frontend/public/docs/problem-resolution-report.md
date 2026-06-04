# 문제 해결 리포트

작성일: 2026-06-04

## 문제가 있었던 부분

- 브라우저 쪽 시뮬레이션이 원본 Python RL 환경과 달라져 있었다.
- 일부 동작을 TypeScript로 손으로 옮기거나, 미리 뽑아 둔 rollout 데이터를 재생하는 구조가 섞여 있었다.
- Spring 백엔드는 정적 파일과 health check만 제공하고 있어서 live MuJoCo/RL 실행에는 역할이 없었다.
- MuJoCo WASM과 live policy stream이 준비되기 전에도 fallback 값이 움직여서, 처음 시작 시 time/height가 계속 변해 보였다.
- FastAPI 기본 Swagger UI가 `/docs`를 선점해서 React Docs 페이지와 충돌했다.
- trajectory trail이 episode reset, contact, floor contact 같은 이벤트를 알 수 없어 계속 이어져 보였다.
- Three.js 커스텀 렌더러가 MuJoCo의 `reflectance`나 native shadow를 자동으로 해석하지 않는데, 이를 고려하지 않고 렌더링을 붙였다.

## 해결한 내용

- Python FastAPI 백엔드가 vendored `pingpong_rl2` 소스를 import해서 원본 Gym env와 Stable-Baselines3 PPO 모델을 직접 실행하도록 바꿨다.
- 브라우저는 Python에서 받은 `qpos`, `qvel`, `ctrl`, contact, reset 상태를 `/api/live` WebSocket으로 받아 MuJoCo WASM/Three.js에 반영한다.
- precomputed `rollout.json` 재생 경로와 오래된 browser policy JSON 경로를 제거했다.
- Python backend가 정적 파일 서빙, health check, config, live simulation을 모두 담당하므로 Spring 백엔드를 제거했다.
- 정책 모델 경로는 `.env`의 `PINGPONG_POLICY_MODEL_PATH`로 관리한다. 현재 기본값은 v25 모델이다.
- FastAPI 기본 docs route를 꺼서 `/docs`가 React Docs 페이지로 연결되게 했다.
- 공 위치 조절 기능을 live backend 구조에 맞게 복구했다. Ball Reset 컨트롤은 Python env의 `reset(options=...)` 경로로 `ball_height`, `ball_xy_offset`, `ball_velocity`를 보낸다.
- reset, contact, floor contact, episode 전환 시 `resetSerial`을 증가시켜 trajectory trail과 contact marker가 안정적으로 초기화되게 했다.
- 로딩 overlay를 추가하고 simulation canvas를 lazy chunk로 분리해서, 첫 화면 shell은 더 빨리 보이고 사용자가 현재 로딩 상태를 알 수 있게 했다.
- 로딩 완료 전에는 fallback physics loop를 돌리지 않도록 해서 time/height가 준비 전부터 변하지 않게 했다.
- 오른쪽 control panel을 접고 펼 수 있게 했다.
- 초기 Panda 자세를 구부린 ready pose로 변경하고 브라우저용 MJB를 다시 컴파일했다.

## 현재 실행 구조

```text
React 브라우저 앱
-> lazy-loaded Three.js + MuJoCo WASM viewer
-> WebSocket /api/live
-> Python FastAPI session
-> vendored pingpong_rl2 env
-> Stable-Baselines3 PPO model
-> Python MuJoCo physics
```

## 남은 제약

- 첫 uncached load는 여전히 WASM, JavaScript, MJB scene처럼 큰 asset을 내려받아야 한다.
- Three.js 렌더러는 MuJoCo native viewer가 아니므로, 그림자와 반사는 별도 구현이 필요하다.
- `rl/` 디렉토리는 gitignore 대상이다. 홈서버 배포 시 `rl/assets`와 선택한 `rl/artifacts` 모델 파일을 반드시 함께 옮겨야 한다.
- Panda home keyframe을 바꾸면 화면만 바뀌는 것이 아니라 Python env reset 상태도 함께 바뀐다.

## 최적화 후보

- Nginx에서 `.wasm`, `.js`, `.css`, `.mjb`에 Brotli 또는 gzip 압축을 적용한다.
- Vite hashed asset과 MuJoCo WASM bundle에 긴 immutable cache header를 준다.
- MJB에서 쓰지 않는 mesh, texture, visual geom을 줄인다.
- WebSocket state를 JSON 배열 대신 binary Float32 buffer로 보낸다.
- client 성능에 따라 reflection, shadow map size, renderer pixel ratio를 낮추는 graphics quality 옵션을 둔다.
- Python simulation step rate와 browser render rate를 분리하고, 브라우저에서는 보간해서 그린다.
- 첫 접속 때 Python backend가 cold start 되지 않게 프로세스를 항상 warm 상태로 유지한다.
- M1 Mac에서 ASUS Intel 홈서버용 이미지를 만들 때는 `linux/amd64`로 빌드하거나, 가능하면 홈서버에서 native build 한다.
