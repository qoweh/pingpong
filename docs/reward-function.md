# 보상 설계

v25 보상은 단순히 공을 맞히는 것보다, 맞힌 뒤에도 다음 공을 다시 칠 수 있는 상태를 만드는 데 초점을 둔다. 따라서 접촉 자체, 목표 높이, 다음 예상 접촉의 도달 가능성, 라켓의 불필요한 움직임을 함께 평가한다.

## 주요 보상 항목

| 항목 | 목적 |
| --- | --- |
| Contact bonus | 라켓과 공의 유효 접촉을 장려한다. |
| Apex match reward | 공이 목표 apex 근처까지 올라가도록 유도한다. |
| Easy next ball reward | 다음 공이 라켓이 도달하기 쉬운 위치로 오도록 유도한다. |
| Stable contact reward | 충분한 높이와 lateral 안정성을 가진 접촉을 장려한다. |
| Stable cycle reward | 유효한 접촉이 반복될수록 보상한다. |
| Trajectory match reward | 접촉 후 공의 궤적이 planner target과 맞도록 유도한다. |
| Action penalty | 지나치게 큰 residual action을 줄인다. |
| Tilt penalties | 불필요하거나 급격한 라켓 기울기 변화를 줄인다. |
| Lateral velocity penalties | 회복하기 어려운 수평 방향 속도를 줄인다. |
| Non-useful contact penalty | 다음 공으로 이어지지 않는 접촉을 억제한다. |

## 실패와 종료

| 상황 | 처리 |
| --- | --- |
| Floor contact | 공이 바닥에 닿으면 episode를 종료한다. |
| Robot body contact | 공이 로봇 본체와 부딪히면 실패로 처리한다. |
| Low apex contact | 낮은 apex 접촉이 반복되면 회복이 어렵다고 보고 종료한다. |
| Max episode steps | step 제한에 도달하면 episode를 끝낸다. |

웹 화면의 contact count와 trail 초기화는 이 episode 상태를 기준으로 동작한다.

## 해석 기준

높은 contact count가 항상 좋은 episode를 뜻하지는 않는다. 중요한 것은 접촉 이후 공이 목표 높이에 가깝게 올라가고, 다음 접촉 위치가 라켓이 따라갈 수 있는 범위에 남는 것이다.

따라서 웹에서 결과를 볼 때는 contact count, 공 높이, 공 궤적, 라켓 움직임을 함께 확인해야 한다.
