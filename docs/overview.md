# 개요

Ping-Pong Keep-Up은 Franka Panda 로봇팔에 탁구채를 붙인 뒤, 탁구공을 계속 받아 올리는 제어 정책을 웹에서 관찰하는 강화학습 시뮬레이션이다.

서버가 MuJoCo 물리 환경과 Stable-Baselines3 PPO 모델을 실행하고, 브라우저는 매 순간의 로봇, 공, 라켓 상태를 받아 3D 장면으로 렌더링한다.

## 프로젝트 구성

| 개념 | 이 프로젝트에서의 역할 |
| --- | --- |
| MuJoCo | 로봇 관절, 라켓, 공, 중력, 충돌을 계산하는 물리 엔진 |
| 로봇팔 | 7자유도 Franka Panda arm. 학습 정책이 직접 토크를 내지 않고 라켓 목표를 보정하면, 내부 컨트롤러가 로봇 관절 명령으로 바꾼다. |
| 강화학습 환경 | MuJoCo 상태를 관측값으로 요약하고, policy action을 적용하고, 보상과 episode 종료 여부를 계산하는 프로그램 |
| PPO | policy network를 안정적으로 업데이트하기 위해 사용한 Stable-Baselines3의 강화학습 알고리즘 |
| Policy network | 현재 관측값을 받아 다음 라켓 보정 action을 내는 신경망. 웹 runtime에서 실제로 action을 계산하는 부분이다. |
| Reward function | "공을 맞혔는가"뿐 아니라 "다음 공을 다시 칠 수 있는가"를 수치로 평가하는 학습 목표 |
| 웹 뷰어 | policy, 물리 상태, 접촉 이벤트, 공 궤적, action 값을 사람이 볼 수 있게 시각화하는 인터페이스 |

## 무엇을 학습했나

학습된 대상은 MuJoCo 물리 엔진도, 로봇팔의 기구 구조도 아니다. MuJoCo는 주어진 action이 들어왔을 때 다음 물리 상태를 계산하고, 로봇 모델은 관절과 링크의 구조를 제공한다.

학습된 것은 PPO로 최적화한 ActorCriticPolicy의 파라미터다. 그 안에는 두 가지 네트워크가 있다.

- Actor 또는 policy: 관측값을 action으로 바꾸는 제어기
- Critic 또는 value function: 현재 상태가 앞으로 얼마나 좋은 return을 낼지 추정하는 학습 보조 모델

웹에서 매 step 사용되는 것은 actor 쪽이다. critic은 학습 중 PPO 업데이트를 돕는 역할이 크고, 배포된 시뮬레이션에서는 주로 모델 구조를 이해할 때 의미가 있다.

## 실행 루프

1. MuJoCo가 현재 로봇, 라켓, 공의 위치와 속도를 가진다.
2. 강화학습 환경이 이 물리 상태를 55차원 observation으로 요약한다.
3. PPO policy가 observation을 받아 17차원 residual action을 낸다.
4. residual action은 라켓의 목표 위치, 속도, 기울기, 다음 접촉 목표를 조금씩 보정한다.
5. 내부 컨트롤러가 보정된 라켓 목표를 Franka Panda 관절 명령으로 바꾼다.
6. MuJoCo가 0.02초 제어 구간을 작은 물리 timestep들로 적분한다.
7. 서버가 새 상태, action, reward, contact event를 브라우저로 보낸다.

이 과정을 반복하면 화면에서는 로봇이 공을 받아 올리는 것처럼 보인다. 실제로는 "관측 -> policy action -> 컨트롤러 -> 물리 적분 -> 새 관측"의 닫힌 루프가 계속 도는 것이다.


## 참고 문서

- [MuJoCo Overview](https://mujoco.readthedocs.io/en/stable/overview.html)
- [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/v2.8.0/modules/ppo.html)
- [Stable-Baselines3 Policy Networks](https://stable-baselines3.readthedocs.io/en/v2.8.0/guide/custom_policy.html)
