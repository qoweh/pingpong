# 시뮬레이션 환경

이 프로젝트의 물리 세계는 MuJoCo로 만들어졌다. MuJoCo는 로봇 관절, 강체, 충돌, 접촉력, 중력 같은 동역학을 계산하는 물리 엔진이다. 강화학습 policy는 공을 "상상"해서 움직이는 것이 아니라, MuJoCo가 계산한 현재 물리 상태를 보고 다음 제어 action을 고른다.

## MuJoCo가 맡는 일

MuJoCo 모델은 XML 기반의 MJCF 형식으로 작성된다. 이 장면에는 로봇팔, 라켓, 공, 바닥, 조명, 재질, 접촉 파라미터가 들어 있다.

| MuJoCo 개념 | 이 프로젝트의 예 |
| --- | --- |
| `body` | 로봇 링크, 라켓, 공처럼 위치와 자세를 가진 물체 |
| `joint` | 로봇 관절, 공의 free joint |
| `geom` | 충돌과 렌더링에 쓰이는 형상. 공 sphere, 라켓 head, 바닥 plane 등 |
| `site` | 관측과 제어를 위한 기준점. 라켓 중심점이 대표적이다. |
| `qpos`, `qvel` | 전체 모델의 generalized position과 velocity |
| `ctrl` | 로봇 actuator에 들어가는 제어 목표 |
| contact | 공과 라켓, 공과 바닥, 공과 로봇 본체의 충돌 정보 |

MuJoCo는 기본 단위를 강제하지 않지만, 일관된 단위계를 사용해야 한다. 이 프로젝트는 길이 m, 질량 kg, 시간 s의 MKS 단위계로 해석한다.

## 장면 구성

| 항목 | 구현 |
| --- | --- |
| 로봇 | Franka Emika Panda 7자유도 로봇팔 |
| 라켓 | Panda hand 아래에 붙은 별도 body |
| 라켓 중심 | `racket_center` site로 추적 |
| 라켓 충돌부 | `racket_head` geom |
| 공 | `ball` body와 `ball_geom` sphere |
| 공 자유도 | `freejoint`로 위치 3축과 회전 3축을 모두 허용 |
| 바닥 | plane geom. 공이 닿으면 실패 조건으로 처리 |

`freejoint`는 공이 로봇에 붙어 있지 않은 독립 물체라는 뜻이다. 그래서 공은 x, y, z 위치를 바꾸고, 회전도 할 수 있다. 현재 학습에서는 공의 선속도가 핵심이고, 공의 회전은 기본적으로 크게 쓰지 않는다.

## 현실 단위와 탁구공

탁구공은 현실 규격을 거의 그대로 넣었다. ITTF 규정의 공은 지름 40mm, 질량 2.7g이다. MuJoCo scene에서는 이를 MKS 단위로 바꿔 반지름 `0.02m`, 질량 `0.0027kg`으로 둔다.

| 항목 | 값 |
| --- | ---: |
| 중력 | `0 0 -9.81` |
| MuJoCo timestep | 0.002 s |
| 제어 주기 | 0.02 s |
| 제어 1 step의 물리 substep | 10 |
| 공 반지름 | 0.02 m |
| 공 질량 | 0.0027 kg |
| 라켓 head 반지름 | 0.084 m |
| 라켓 head half-depth | 0.006 m |

물리 timestep은 아주 짧은 시간 간격으로 동역학을 적분하는 단위다. 제어 주기 0.02초마다 policy action을 한 번 적용하지만, 그 사이에 MuJoCo는 0.002초씩 10번 물리를 진행한다. 접촉이 많은 장면에서는 timestep이 안정성에 중요하다.

## 공 배치와 초기 조건

공은 월드 좌표에 직접 놓기보다 라켓 기준 상대값으로 놓는다. 이렇게 하면 로봇의 초기 자세가 조금 달라도 "라켓 위 몇 cm, 오른쪽 몇 cm" 같은 조작이 직관적으로 유지된다.

| 조작값 | 의미 |
| --- | --- |
| X/Y Position | 라켓 중심 기준 수평 offset |
| Z Position | 라켓 중심 위쪽 높이 |
| X/Y/Z Velocity | 공의 초기 선속도 |

학습 때는 시작 조건을 매번 조금씩 바꿨다. 이를 domain randomization 또는 curriculum으로 볼 수 있다. policy가 한 위치만 외우지 않고, 다양한 시작 위치와 속도에서 공을 회복하는 법을 배우게 하기 위해서다.

## 접촉과 실패 판정

학습 환경은 MuJoCo contact 정보를 읽어 공이 무엇과 부딪혔는지 판단한다.

| 접촉 | 의미 |
| --- | --- |
| 공과 라켓 head | 유효한 타격 후보 |
| 공과 바닥 | episode 실패 |
| 공과 로봇 본체 | 로봇이 몸으로 공을 받은 상황이므로 실패 |
| 공이 장면 범위를 벗어남 | 회복 불가능한 상태로 실패 |

공과 라켓이 닿았다는 사실만으로 성공이 되지는 않는다. 접촉 후 공이 충분히 위로 올라가고, 다음 접촉 위치가 라켓이 따라갈 수 있는 범위에 있어야 유용한 접촉으로 본다.

## 웹 렌더링

서버는 원본 MuJoCo 환경을 실행하고, 브라우저는 전달받은 `qpos`, `qvel`, `ctrl` 상태를 MuJoCo WebAssembly 모델과 Three.js 장면에 반영한다. 따라서 웹은 학습 환경을 새로 구현한 것이 아니라, 서버의 물리 상태를 사람이 보기 좋게 보여주는 뷰어에 가깝다.

웹에서 보이는 공 궤적, 접촉 marker, 목표 높이, 카메라 전환은 이해를 돕기 위한 시각화 요소다. 물리 판정의 기준은 서버의 MuJoCo 상태와 강화학습 환경에 있다.

## 참고 문서

- [MuJoCo Overview](https://mujoco.readthedocs.io/en/stable/overview.html)
- [MuJoCo XML Reference](https://mujoco.readthedocs.io/en/latest/XMLreference.html)
- [ITTF Handbook / Statutes](https://www.ittf.com/statutes/)
