# 런타임 구조와 성능

이 문서는 웹 서비스가 실제로 어디에서 계산하고, 어디에서 렌더링하며, 어떤 구조로 최적화되어 있는지 설명한다.

## 역할 분리

서버는 authoritative simulation runtime이다. Stable-Baselines3 PPO policy, 강화학습 환경, Python MuJoCo 시뮬레이션을 실행한다. 매 제어 step마다 observation을 만들고, policy action을 계산하고, MuJoCo 환경을 진행한 뒤 frame을 만든다.

브라우저는 visualization runtime이다. React UI, Three.js 렌더러, MuJoCo WebAssembly를 사용한다. 브라우저의 MuJoCo는 서버가 보낸 `qpos`, `qvel`, `ctrl` 상태를 받아 3D geometry 위치와 회전을 맞추는 데 사용된다. Policy action은 브라우저가 계산하지 않는다.

## 이전 구조

초기 구조에서는 WebSocket 클라이언트마다 `LiveSimulationSession`을 새로 만들었다.

```text
client A -> MuJoCo env A -> PPO predict A -> frame A
client B -> MuJoCo env B -> PPO predict B -> frame B
client C -> MuJoCo env C -> PPO predict C -> frame C
```

이 방식은 각 사용자가 독립적인 시뮬레이션을 갖는 장점이 있지만, 접속자가 늘면 MuJoCo와 PPO 계산도 접속자 수만큼 늘어난다. 홈서버에서는 CPU가 먼저 병목이 된다.

## 현재 구조

현재 구조는 하나의 shared live session을 실행하고, 모든 WebSocket 클라이언트가 그 frame을 구독한다.

```text
shared MuJoCo env -> shared PPO predict -> latest frame
                                      -> client A
                                      -> client B
                                      -> client C
```

서버는 MuJoCo/PPO loop를 한 번만 돌린다. 각 클라이언트에는 최신 frame만 전달한다. 느린 클라이언트가 있으면 오래된 frame을 쌓지 않고 버리고 최신 frame 위주로 받는다.

이 구조에서는 model select, reset, spawn, playback 명령이 shared demo session에 적용된다. 즉 한 사용자가 모델을 바꾸면 같은 live demo를 보고 있는 다른 사용자도 같은 모델 상태를 보게 된다. 공개 데모 목적에서는 서버 비용을 크게 줄이는 쪽이 더 중요하다.

## WebSocket 종료 오류

다음 오류는 이미 닫힌 WebSocket에 서버가 frame을 보내려 할 때 발생한다.

```text
RuntimeError: Unexpected ASGI message 'websocket.send',
after sending 'websocket.close' or response already completed.
```

대표적인 상황은 모델 전환이다. 브라우저가 기존 연결을 닫거나 새 연결로 바꾸는 순간, 서버 loop가 거의 동시에 `send_json`을 호출하면 ASGI runtime이 위 예외를 낸다.

현재 서버는 send loop와 receive loop를 분리하고, `WebSocketDisconnect`, `RuntimeError`, `OSError`를 연결 종료로 처리한다. 따라서 클라이언트가 먼저 끊겨도 stack trace를 남기지 않고 구독만 정리한다.

모델 전환 중 새 런타임 생성이 실패하면 이전 runtime과 session을 복구한다. 예를 들어 legacy 모델이 현재 코드에 없는 action mode를 요구하면 서버는 `409` 응답과 사람이 읽을 수 있는 오류 메시지를 반환하고, live loop는 이전 모델로 계속 유지된다.

## 3D scene 로딩

브라우저는 빠른 시작을 위해 컴파일된 MuJoCo binary model인 `pingpong_scene.mjb`를 먼저 연다. MJB는 MJCF XML과 mesh 자산을 MuJoCo가 바로 열 수 있게 묶어 둔 binary model이다.

`Loading source 3D scene asset 3/70` 같은 메시지는 MJB를 열지 못해 XML source asset fallback으로 넘어갔다는 뜻이다. 여기서 `70`은 `rl/assets` 아래 scene XML, Franka XML, OBJ/STL mesh, license 파일 등을 합친 source asset 개수다. 정상적인 빠른 경로에서는 MJB 한 개만 로드하고, fallback은 진단용 안전망에 가깝다.

## 모델 전환 비용

모델 전환에서 비싼 작업은 두 가지다.

1. PPO zip을 디스크에서 읽고 Stable-Baselines3 policy 객체를 만드는 작업
2. 새 policy 설정으로 Python MuJoCo 환경을 reset하는 작업

3D scene은 모델을 바꿔도 동일하다. 따라서 브라우저 캔버스, MuJoCo WebAssembly, `pingpong_scene.mjb`는 다시 로드하지 않는다.

처음 선택하는 모델은 policy load 때문에 대략 0.5초에서 몇 초까지 걸릴 수 있다. 같은 action dimension끼리 바꾼다고 항상 빨라지는 것은 아니고, 실제로 가장 큰 차이는 서버 runtime cache 여부에서 난다. 한 번 선택한 모델은 서버가 runtime을 캐시하므로 다시 선택할 때 훨씬 빠르다. 다른 action dimension 모델은 action label, env kwargs, policy shape가 달라서 새 runtime 준비가 더 길게 느껴질 수 있다.

## 홈서버 기준 추정

i5-8400 6코어, 16GB RAM 서버에서 이전 구조는 부드러운 live 시뮬레이션 기준 3-6명 정도가 안전권이었다. 현재 shared live 구조는 서버 측 MuJoCo/PPO 계산이 접속자 수에 거의 비례하지 않으므로, 병목은 CPU보다 WebSocket 전송량과 초기 정적 asset 다운로드 쪽으로 옮겨간다.

초기 로딩 자산은 MuJoCo WebAssembly와 3D scene 때문에 크다. 브라우저 캐시가 잡힌 이후에는 반복 방문 비용이 크게 줄어든다.
