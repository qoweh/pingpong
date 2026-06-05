# 상태와 행동

강화학습 문제는 보통 MDP, 즉 Markov Decision Process로 설명한다. 이름은 딱딱하지만 핵심은 단순하다. agent가 현재 상태를 보고 action을 고르면, 환경이 다음 상태와 reward를 돌려준다. 이 과정을 episode가 끝날 때까지 반복한다.

## MDP 용어 대응

| MDP 용어 | 이 프로젝트에서의 의미 |
| --- | --- |
| State | MuJoCo 내부의 전체 물리 상태. 로봇 관절, 공 위치, 속도, 접촉 상태 등을 모두 포함한다. |
| Observation | policy에게 실제로 보여주는 55차원 feature vector. 전체 state를 학습에 필요한 정보로 요약한 것이다. |
| Action | policy가 출력하는 17차원 residual control. 로봇 관절 토크가 아니라 라켓 목표를 보정하는 값이다. |
| Transition | action을 적용한 뒤 MuJoCo가 다음 물리 상태로 진행하는 과정 |
| Reward | 현재 step의 행동이 keep-up에 얼마나 도움이 됐는지 평가하는 점수 |
| Episode | 공을 계속 받아 올리는 한 번의 시도. 실패 조건이나 평가 step 제한에 도달하면 끝난다. |
| Policy | observation을 action으로 바꾸는 학습된 함수 |

## 관측값

현재 기본 모델의 observation은 55차원이다. 값들은 모두 연속적인 숫자이며, policy network의 입력으로 그대로 들어간다.

| 구성 | 차원 | 의미 |
| --- | ---: | --- |
| Joint positions | 7 | Panda arm의 7개 관절 각도 |
| Joint velocities | 7 | Panda arm의 7개 관절 속도 |
| Racket position | 3 | 라켓 중심의 월드 좌표 |
| Racket velocity | 3 | 라켓 중심의 Cartesian velocity |
| Target position | 3 | 내부 컨트롤러가 보고 있는 라켓 목표 위치 |
| Ball position | 3 | 공의 월드 좌표 |
| Ball velocity | 3 | 공의 선속도 |
| Ball relative position | 3 | 라켓 기준 공 위치 |
| Predicted intercept XY | 2 | 현재 공 궤적으로 예상한 접촉 지점의 수평 offset |
| Predicted intercept time | 1 | 예상 접촉까지 남은 시간 |
| Task phase | 4 | 준비, 타격, 회복 같은 phase one-hot |
| Contact context | 2 | 최근 접촉 이후 시간, 성공 접촉 count 요약 |
| Next intercept | 6 | 다음 접촉의 상대 위치, 시간, 도달 가능성, 회복 거리, 준비도 |
| Desired outgoing velocity | 3 | 접촉 후 공이 가져야 할 목표 속도 |
| Racket face normal | 3 | 라켓 면이 향하는 방향 |
| Target tilt | 2 | 내부 컨트롤러의 목표 라켓 기울기 |
| Total | 55 | policy 입력 전체 |

여기서 중요한 점은 observation이 "화면에 보이는 모든 것"이 아니라는 것이다. 예를 들어 카메라 위치나 UI 상태는 policy가 알지 못한다. 반대로 화면에서는 잘 보이지 않는 예상 접촉 시간, 다음 공의 도달 가능성 같은 계산 feature는 policy 입력에 포함된다.

## 행동값

현재 action mode는 `position_contact_frame_velocity_tilt_lateral_apex_tracking_residual`이다. 이름이 길지만, 뜻은 "기본 컨트롤러가 만든 라켓 목표를 접촉 좌표계 기준으로 조금씩 보정한다"에 가깝다.

| 구성 | 차원 | 의미 |
| --- | ---: | --- |
| Contact-frame position residual | 3 | 예상 접촉 지점 기준 라켓 위치 보정. radial, tangent, strike 높이로 나뉜다. |
| Tilt residual | 2 | 라켓 면 기울기 보정 |
| Velocity residual | 3 | 접촉 전후 속도 계획 보정 |
| Vertical velocity / tilt scale residual | 3 | 라켓 수직 속도와 tilt scale 보정 |
| Lateral velocity residual | 2 | 라켓 수평 속도 보정 |
| Apex timing residual | 2 | 목표 apex 높이와 strike plane 높이 보정 |
| Tracking residual | 2 | 다음 접촉 지점 추적 속도 보정 |
| Total | 17 | policy 출력 전체 |

Residual action을 쓰는 이유는 학습 난도를 낮추기 위해서다. 로봇팔 전체를 처음부터 끝까지 직접 제어하게 하면 action 하나의 의미가 너무 복잡해진다. 대신 기본 컨트롤러가 "대체로 그럴듯한 라켓 목표"를 만들고, policy는 그 목표를 상황에 맞게 미세 조정한다.

## 한 step에서 일어나는 일

1. 환경이 MuJoCo 상태에서 55차원 observation을 만든다.
2. policy가 17차원 action을 낸다.
3. action이 라켓 목표 위치, 기울기, 속도 목표, 다음 접촉 계획에 반영된다.
4. Cartesian controller가 라켓 목표를 로봇 관절 target으로 바꾼다.
5. MuJoCo가 물리를 진행한다.
6. 환경이 contact, reward, 종료 여부, 다음 observation을 계산한다.

이 구조 덕분에 policy는 로봇팔의 모든 세부 물리식을 직접 알 필요가 없다. 대신 "공이 어디 있고, 라켓이 어디 있고, 다음 접촉이 얼마나 쉬운가"를 보고 적절한 보정량을 고르는 데 집중한다.

## Episode 종료 조건

episode는 공이 바닥에 닿거나, 로봇 본체와 충돌하거나, 공이 장면 범위를 벗어나거나, 회복하기 어려운 낮은 apex 접촉이 반복될 때 끝난다. 평가에서는 정해진 step 수까지 버티는 것도 하나의 종료 방식이다.

웹 뷰어의 reset, contact marker, trail 초기화는 이 episode 상태를 기준으로 동작한다.
