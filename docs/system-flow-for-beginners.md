# 전체 흐름 이해하기

이 문서는 프론트엔드, FastAPI 백엔드, WebSocket, 강화학습 runtime이 이 프로젝트 안에서 어떻게 이어지는지 큰 흐름부터 설명한다.

한 줄로 요약하면 이렇다.

```text
브라우저 UI -> HTTP/WebSocket -> FastAPI 서버 -> PPO policy + MuJoCo 환경 -> frame JSON -> 브라우저 3D 렌더링
```

## 역할 나누기

| 영역 | 이 프로젝트에서 하는 일 |
| --- | --- |
| React 프론트엔드 | 버튼, 모델 선택, 공 스폰 슬라이더, 3D 장면 렌더링을 담당한다. |
| FastAPI 백엔드 | API route를 열고, 모델 목록을 내려주고, live WebSocket 연결을 관리한다. |
| LiveSimulationHub | 서버 안에서 하나의 공유 live session을 돌리고 모든 접속자에게 frame을 나눠 보낸다. |
| PPO policy | 현재 observation을 보고 다음 action을 계산하는 학습된 신경망이다. |
| MuJoCo 환경 | 로봇, 라켓, 공, 중력, 충돌을 물리적으로 계산한다. |
| 브라우저 MuJoCo/Three.js | 서버가 보낸 `qpos`, `qvel`, `ctrl`을 화면의 3D 장면에 반영한다. |

중요한 점은 브라우저가 PPO action을 계산하지 않는다는 것이다. 정책 추론과 실제 물리 진행은 서버 Python runtime에서 한다. 브라우저는 그 결과를 받아 보여주는 visualization runtime에 가깝다.

## HTTP와 WebSocket

처음 페이지가 열리면 프론트는 짧은 요청은 HTTP로 처리한다.

| API | 용도 |
| --- | --- |
| `GET /api/health` | 서버가 살아 있는지 확인한다. |
| `GET /api/models` | 선택 가능한 policy model 목록과 현재 active model을 받는다. |
| `GET /api/config` | 현재 모델, scene, 공 스폰 범위 같은 초기 설정을 받는다. |
| `POST /api/models/select` | 사용자가 고른 모델로 서버 runtime을 전환한다. |

실시간 frame은 HTTP가 아니라 WebSocket으로 받는다.

| 연결 | 용도 |
| --- | --- |
| `WS /api/live` | 서버가 계속 frame JSON을 보내고, 브라우저가 reset/playback/spawnBall 명령을 보낸다. |

HTTP는 요청 하나를 보내고 응답 하나를 받으면 끝나는 방식이다. 모델 목록이나 모델 전환처럼 가끔 일어나는 작업에 잘 맞는다.

WebSocket은 한 번 연결한 뒤 양쪽이 계속 메시지를 주고받는 방식이다. 이 프로젝트처럼 서버가 0.02초 단위에 가까운 frame을 계속 밀어줘야 할 때 잘 맞는다.

## 소켓과 WebSocket 차이

소켓은 더 일반적인 말이다. TCP나 UDP 같은 네트워크 연결의 낮은 수준 입구를 가리킬 때가 많다.

WebSocket은 웹 브라우저가 사용할 수 있는 표준 프로토콜이다. 처음에는 HTTP 요청처럼 시작하고, 연결이 열린 뒤에는 서버와 브라우저가 양방향으로 계속 메시지를 주고받는다.

그래서 이 프로젝트에서 "소켓"이라고 부르는 것은 대부분 실제로는 WebSocket을 뜻한다고 보면 된다.

```text
일반 소켓: 운영체제/서버 프로그램이 직접 다루는 낮은 수준 네트워크 연결
WebSocket: 브라우저와 서버가 웹 환경에서 쓰기 좋게 만든 지속 연결 프로토콜
```

## 서버 시작 흐름

FastAPI 서버가 뜨면 `backend/app/main.py`의 `lifespan`에서 simulation service와 live hub를 만든다.

```text
load_settings()
-> LiveSimulationService(settings)
-> LiveSimulationHub(service)
-> FastAPI route 대기
```

`LiveSimulationService`는 다음 일을 한다.

- vendored `pingpong_rl2` 패키지를 import할 수 있게 경로를 잡는다.
- 모델 artifact를 스캔해서 카탈로그를 만든다.
- 기본 PPO zip을 Stable-Baselines3로 로드한다.
- 모델에 맞는 `PingPongKeepUpGymEnv` 환경 설정을 복원한다.

`LiveSimulationHub`는 다음 일을 한다.

- WebSocket 구독자 목록을 가진다.
- 하나의 공유 `LiveSimulationSession`을 유지한다.
- reset, playback, spawnBall 명령을 다음 simulation tick에 적용한다.
- 최신 frame을 모든 접속자 queue로 보낸다.

## 브라우저 시작 흐름

브라우저에서 앱이 열리면 대략 이런 순서로 움직인다.

```text
App.tsx
-> GET /api/models
-> 모델 목록과 activeModel 상태 저장
-> SimulationCanvas mount
-> DemoController.initialize()
-> MujocoWorld.initialize()
-> 브라우저용 MuJoCo scene asset 로드
-> WebSocket /api/live 연결
```

프론트는 자체적으로 3D scene을 열지만, 이것은 policy를 실행하기 위한 것이 아니다. 서버가 보낸 `qpos`, `qvel`, `ctrl` 값을 브라우저의 MuJoCo data에 복사하고 `mj_forward`를 호출해서 화면의 geometry 위치를 맞추기 위한 것이다.

## Live frame 흐름

live loop에서 실제로 반복되는 핵심은 다음 순서다.

```text
현재 observation
-> PPO policy.predict(observation)
-> action
-> PingPongKeepUpEnv.step(action)
-> RacketCartesianController.compute_joint_targets()
-> MuJoCo step_with_contact_trace()
-> reward, terminated, info 계산
-> LiveSimulationSession.frame()
-> WebSocket frame JSON 전송
-> 브라우저 MujocoWorld.applyLiveFrame()
-> Three.js render
```

frame JSON에는 다음 정보가 들어간다.

| 필드 | 의미 |
| --- | --- |
| `state.qpos` | MuJoCo generalized position이다. 브라우저 scene에 그대로 복사된다. |
| `state.qvel` | MuJoCo generalized velocity이다. |
| `state.ctrl` | 로봇 actuator control 값이다. |
| `ball.position`, `ball.velocity` | UI와 trail, contact 표시가 쓰는 공 상태다. |
| `racketPosition` | 라켓 중심 위치다. |
| `contact` | 이번 step에서 공과 라켓 접촉이 있었는지, 누적 횟수는 얼마인지 알려준다. |
| `action` | PPO policy가 방금 낸 action vector다. |
| `reward`, `failureReason` | 학습 환경이 계산한 보상과 실패 이유다. |

## 강화학습 관점

강화학습 환경은 매 step마다 다음 네 가지를 다룬다.

| 개념 | 이 프로젝트의 예 |
| --- | --- |
| Observation | 로봇 관절, 라켓 위치/속도, 공 위치/속도, 예측 intercept, phase 정보 등 |
| Action | 라켓 목표 위치, 기울기, contact-frame residual 같은 policy 출력 |
| Reward | 공을 잘 맞췄는지, 다음 공을 다시 받을 수 있는지, 과한 action을 쓰지 않았는지 |
| Done | 바닥 접촉, 로봇 몸체 접촉, 공 이탈, timestep 제한 같은 episode 종료 조건 |

이 live demo에서는 학습을 새로 하지 않는다. 이미 학습된 PPO zip을 읽어서 inference만 한다. 즉 서버는 policy의 파라미터를 업데이트하지 않고, 현재 observation에 대한 action만 계속 계산한다.

## 모델 전환 흐름

사용자가 모델을 바꾸면 프론트는 `POST /api/models/select`를 호출한다.

```text
ModelControls
-> POST /api/models/select
-> LiveSimulationHub.select_model()
-> LiveSimulationService.select_model()
-> PPO.load(model.zip)
-> 새 RuntimeModel 생성 또는 cache 재사용
-> 새 LiveSimulationSession 생성
-> ready 메시지와 첫 frame publish
```

모델마다 observation dimension, action dimension, action mode가 다를 수 있다. 그래서 서버는 새 session을 만든 뒤 policy shape과 env shape가 맞는지 검증한다. 맞지 않으면 `409` 오류를 주고 이전 runtime을 유지한다.

## reset과 spawnBall

프론트의 reset 버튼은 WebSocket으로 `{ "type": "reset" }`을 보낸다. 서버는 다음 live loop tick에서 환경 reset을 수행하고 새 frame을 보낸다.

공 스폰 조작은 `{ "type": "spawnBall", ... }` 형태로 들어온다. 서버는 `backend/app/ball_spawn.py`에서 값을 한 번 더 clamp한 뒤 `LiveSimulationSession.spawn_ball()`로 넘긴다. 이때 환경 전체를 새로 여는 것이 아니라, 현재 racket 기준으로 공 위치와 속도만 다시 배치하고 episode 카운터를 초기화한다.

## 파일을 따라 읽는 순서

처음 구조를 볼 때는 이 순서로 읽으면 덜 헷갈린다.

1. `backend/app/main.py`: FastAPI route, HTTP API, WebSocket 입구
2. `backend/app/live_simulation.py`: 모델 로딩, shared live hub, session frame 생성
3. `backend/app/model_catalog.py`: 모델 목록과 metadata 생성
4. `backend/app/ball_spawn.py`: 공 스폰 UI 값과 환경 option 변환
5. `backend/vendor/pingpong_rl2/src/pingpong_rl2/envs/gym_env.py`: Gymnasium wrapper
6. `backend/vendor/pingpong_rl2/src/pingpong_rl2/envs/keepup_env.py`: observation, action, reward, termination의 핵심 MDP
7. `backend/vendor/pingpong_rl2/src/pingpong_rl2/envs/pingpong_sim.py`: MuJoCo scene과 contact trace
8. `backend/vendor/pingpong_rl2/src/pingpong_rl2/controllers/ee_pose_controller.py`: 라켓 target pose를 Franka 관절 target으로 변환
9. `frontend/src/simulation/mujocoWorld.ts`: WebSocket 연결과 frame 적용
10. `frontend/src/app/App.tsx`: 모델 선택, playback, reset, spawnBall UI 상태

## 가장 중요한 정신 모델

이 프로젝트를 볼 때는 서버와 브라우저를 이렇게 분리해 생각하면 편하다.

```text
서버 Python
  실제 PPO policy 실행
  실제 MuJoCo 환경 step
  reward와 episode 상태 계산
  frame JSON 생성

브라우저
  버튼과 슬라이더 상태 관리
  WebSocket으로 명령 전송
  frame JSON 수신
  받은 MuJoCo state를 3D 화면에 반영
```

그래서 화면이 움직이는 이유는 브라우저가 혼자 시뮬레이션을 학습하거나 제어해서가 아니다. 서버가 계산한 최신 물리 상태를 브라우저가 빠르게 따라 그리기 때문이다.
