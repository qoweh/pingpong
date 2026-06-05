# 보상 설계

보상 함수는 policy에게 "무엇이 좋은 행동인가"를 알려주는 점수 체계다. 이 프로젝트의 목표는 단순히 공과 라켓이 한 번 닿게 하는 것이 아니다. 좋은 타격은 공을 다시 칠 수 있는 위치와 속도로 돌려보내야 한다.

그래서 v39 보상은 접촉 자체보다 "유용한 접촉"과 "안정적인 반복"에 더 초점을 둔다.

## 왜 단순 contact 보상만으로 부족한가

공을 아무 방향으로 세게 치기만 해도 contact count는 올라갈 수 있다. 하지만 공이 옆으로 멀리 날아가거나 너무 낮게 뜨면 다음 step에서 로봇이 회복할 수 없다. 이런 policy는 보기에는 공을 맞힌 것 같지만 keep-up 문제를 풀지 못한다.

따라서 보상은 다음 질문들을 함께 본다.

- 공이 라켓에 닿았는가?
- 접촉 후 공이 목표 높이 근처까지 올라가는가?
- 다음 접촉 위치가 라켓이 도달 가능한 범위에 남는가?
- 공의 수평 속도가 너무 커서 회복이 어려워지지는 않는가?
- 라켓이 불필요하게 크게 흔들리거나 급격히 기울지는 않는가?
- 안정적인 접촉이 여러 번 이어지는가?

## 주요 보상 항목

| 항목 | 목적 |
| --- | --- |
| Tracking reward | 공이 내려오는 동안 라켓이 타격 가능한 위치로 이동하도록 유도 |
| Contact bonus | 유효한 라켓-공 접촉을 장려 |
| Apex match reward | 접촉 후 공이 목표 apex 높이에 가깝게 올라가도록 유도 |
| Return target XY reward | 공의 apex 또는 다음 접촉 위치가 중앙에 가깝게 돌아오도록 유도 |
| Next intercept reachable bonus | 다음 공을 라켓이 따라갈 수 있으면 보상 |
| Easy next ball reward | 다음 공의 시간, 위치, 속도가 쉬운 상태가 되도록 유도 |
| Stable contact reward | 높이와 lateral 안정성을 모두 만족하는 접촉을 보상 |
| Stable cycle reward | 좋은 접촉이 연속될수록 추가 보상 |
| Trajectory match reward | 접촉 후 실제 공 속도가 planner가 원하는 속도와 가까워지도록 유도 |

`Easy next ball`은 특히 중요하다. 이것은 현재 접촉만 보는 점수가 아니라, 그 접촉 이후의 공이 다음에 다시 치기 쉬운지를 평가한다. keep-up은 한 번의 성공보다 반복 가능성이 중요하기 때문이다.

## 패널티

| 항목 | 줄이려는 행동 |
| --- | --- |
| Action penalty | 너무 큰 residual action |
| Tilt angle penalty | 과도한 라켓 기울기 |
| Tilt delta penalty | step 사이의 급격한 기울기 변화 |
| Lateral velocity penalty | 접촉 후 공이 옆으로 너무 빨리 움직이는 상황 |
| Contact XY error penalty | 공이 라켓 중심에서 너무 벗어나 맞는 상황 |
| Outward velocity penalty | 라켓이 공을 바깥쪽으로 밀어내는 상황 |
| Non-useful contact penalty | 맞기는 했지만 다음 공으로 이어지지 않는 접촉 |
| Low apex penalty | 공이 목표 높이까지 충분히 올라가지 못하는 접촉 |

패널티는 policy를 소극적으로 만들기 위한 것이 아니라, "계속 칠 수 있는 형태의 타격"으로 좁혀 주는 역할을 한다.

## 실패와 종료

| 상황 | 처리 |
| --- | --- |
| Floor contact | 공이 바닥에 닿으면 실패 |
| Robot body contact | 공이 라켓이 아닌 로봇 본체에 닿으면 실패 |
| Ball out of bounds | 공이 장면의 유효 범위를 벗어나면 실패 |
| Ball speed limit | 비정상적으로 빠른 공 속도는 실패 |
| Low apex contact 반복 | 회복 불가능한 낮은 타격이 반복되면 종료 |
| Evaluation time limit | 평가용 step 제한까지 버티면 episode 종료 |

웹에서 contact count가 높아 보여도 항상 좋은 episode라는 뜻은 아니다. 공 높이, 공 궤적, 라켓 움직임, 다음 접촉 가능성을 같이 봐야 한다. v39 모델의 핵심 성능 지표도 단순 contact 수보다 useful bounce와 stable cycle에 더 가깝다.

## v39 보상의 방향

v39는 이전 checkpoint에서 이어 학습하면서 "공을 계속 중앙 근처로 되돌리고, 다음 접촉이 쉬운 공을 만드는 것"을 강화했다. 목표 apex는 라켓 위 약 0.30m이고, 성공 접촉은 목표 높이, 도달 가능한 다음 접촉, lateral 안정성을 함께 만족해야 한다.

즉 policy가 배운 것은 탁구 랠리의 화려한 스윙이 아니라, 작은 라켓 보정으로 공을 잃지 않는 안정적인 keep-up 습관이다.
