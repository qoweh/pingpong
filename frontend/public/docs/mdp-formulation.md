# 상태와 행동

현재 선택된 제어 모델은 공의 위치, 속도, 라켓 상태, 예상 접촉 정보를 입력으로 받아 라켓의 다음 목표를 보정한다. 내부 action mode는 `position_contact_frame_velocity_tilt_lateral_apex_tracking_residual`이다.

이 이름은 길지만 의미는 단순하다. 기본 컨트롤러가 예측한 라켓 목표 위치와 자세가 있고, 학습 모델은 그 목표를 조금씩 보정하면서 공을 다시 칠 수 있는 상태로 만든다.

## 관측값

| 구성 | 차원 | 의미 |
| --- | ---: | --- |
| Joint positions | 7 | Panda joint 각도 |
| Joint velocities | 7 | Panda joint 속도 |
| Racket position | 3 | `racket_center`의 world position |
| Racket velocity | 3 | 라켓의 Cartesian velocity |
| Target position | 3 | 기본 컨트롤러가 잡은 목표 위치 |
| Ball position | 3 | 공의 world position |
| Ball velocity | 3 | 공의 선속도 |
| Ball relative position | 3 | 라켓 기준 공 위치 |
| Predicted intercept XY | 2 | 예상 접촉 지점의 XY offset |
| Predicted intercept time | 1 | 예상 접촉까지 남은 시간 |
| Task phase | 4 | 준비, 타격, 회복 등 phase one-hot |
| Contact context | 2 | 최근 접촉 이후 시간과 접촉 횟수 |
| Next intercept | 6 | 다음 접촉의 상대 위치, 시간, 도달 가능성 |
| Desired outgoing velocity | 3 | 접촉 후 원하는 공 속도 |
| Racket face normal | 3 | 라켓 면의 normal vector |
| Target tilt | 2 | 목표 라켓 기울기 |
| Total | 55 | 모델 입력 전체 |

## 행동값

| 구성 | 차원 | 의미 |
| --- | ---: | --- |
| Position residual | 3 | 라켓 목표 위치 보정 |
| Tilt residual | 2 | 라켓 기울기 보정 |
| Contact-frame velocity residual | 3 | 접촉 기준 속도 보정 |
| Racket vertical velocity / tilt scale residual | 3 | 라켓 수직 속도와 tilt scale 보정 |
| Racket XY velocity residual | 2 | 라켓 수평 속도 보정 |
| Apex timing residual | 2 | 목표 apex와 strike plane 보정 |
| Tracking XY residual | 2 | 다음 접촉 지점 추적 속도 보정 |
| Total | 17 | 모델 출력 전체 |

## episode 종료 조건

episode는 공이 바닥에 닿거나, 너무 낮은 apex로 회복이 어려운 접촉이 반복되거나, 최대 step에 도달하면 종료된다. 웹 뷰어는 episode reset, contact, floor contact를 받으면 trail과 contact marker 같은 보조 시각화를 초기화한다.
