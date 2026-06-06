# 13D/17D 모델 검토 메모

기준 시점: 2026-06-06

이 문서는 13D 모델이 약 10 contacts 전후에서 종료되는 현상, 17D policy action의 차원 필요성, 프로젝트 전반 리스크를 발표 Q&A에 답할 수 있게 정리한 메모다.

## 결론 요약

1. 13D가 10 contacts 근처에서 리셋되는 것은 contact 수 제한 때문이 아니다. `low_apex_contact` 실패가 누적되어 episode가 `terminated`되고, 라이브 세션이 다음 tick에서 reset을 수행하기 때문이다.
2. 13D 대표 모델 `pmk_cf_self_rally_v20`은 evaluation에서 평균 useful bounce가 1.62, 최대 useful bounce가 8이고, 실패 100회 중 80회가 `low_apex_contact`다. UI의 raw contact count가 10 근처까지 보여도, 학습 기준으로 성공적인 bounce는 훨씬 적다.
3. 17D action은 로봇 관절 17개를 제어하는 값이 아니다. 7자유도 Panda arm 위에 있는 inverse kinematics/controller가 실제 관절 목표를 만들고, policy는 task-space 타격 계획의 residual parameter를 출력한다.
4. 7D policy도 설계할 수는 있지만, 현재 구조에서는 15D에서 추가된 `Target Apex Z`, `Strike Plane Z`가 성능 향상에 직접 관련되어 보인다. 특히 `Strike Plane Z`는 current 17D 로그에서 매우 강하게 쓰인다.
5. 다만 17D의 모든 차원이 같은 정도로 중요하지는 않다. `centering_tilt_scale`, `velocity_scale`, `racket_vz`, `racket_vy`, `tracking_x/y`는 현재 contact 로그 기준으로 사용량이 작아 ablation 후보가 된다.

## 0. 모델 artifact에 들어있는 것

Stable-Baselines3 PPO의 `*.zip` 모델 파일에는 주로 policy/optimizer/space/algorithm metadata가 들어 있다.

13D v20과 17D v39 모델 zip의 공통 파일:

- `data`
- `pytorch_variables.pth`
- `policy.pth`
- `policy.optimizer.pth`
- `_stable_baselines3_version`
- `system_info.txt`

여기서 핵심은 `policy.pth`와 `data`다.

- `policy.pth`: actor-critic policy network의 PyTorch weight. observation을 받아 action distribution과 value estimate를 만드는 신경망 파라미터다.
- `policy.optimizer.pth`: 학습 재개에 필요한 optimizer state.
- `data`: observation/action space, PPO hyperparameter, policy class, learning-rate/clip schedule 같은 metadata.
- `pytorch_variables.pth`: SB3가 별도로 저장하는 torch 변수.
- `_stable_baselines3_version`, `system_info.txt`: 로드 호환성과 실행 환경 정보.

모델 zip 자체에 Python reward function이나 termination function 코드가 들어있는 것은 아니다. reward/termination은 환경 코드에 있고, 모델은 그 reward를 통해 학습된 policy weight를 담는다.

다만 이 프로젝트에서는 모델 zip 옆에 `*_training_summary.json`이 있고, 서버가 이 summary의 `env_config`를 읽어 해당 모델이 학습된 환경 설정을 복원한다. 그래서 실질적으로는 "모델 선택"이 "policy weight 선택 + 그 모델의 env/reward/termination 설정 복원"에 가깝다.

관련 코드:

- `backend/vendor/pingpong_rl2/src/pingpong_rl2/utils/ppo_runs.py`: 모델 경로에서 training summary의 `env_config`를 로드한다.
- `backend/app/live_simulation.py`: `_resolve_env_kwargs()`에서 summary 기반 env kwargs를 만든다.
- `backend/app/model_catalog.py`: summary hint를 모델 metadata에 반영한다.

13D v20과 17D v39의 model zip 차이는 대표적으로 action space에 드러난다.

| 모델 | observation space | action space |
| --- | ---: | ---: |
| 13D v20 | 55D | 13D |
| 17D v39 | 55D | 17D |

17D는 13D action 범위 뒤에 `Target Apex Z`, `Strike Plane Z`, `Tracking X`, `Tracking Y` 4개 residual이 추가되어 있다.

## 1. 13D가 종료되는 원인

### 런타임 종료 흐름

`PingPongKeepUpEnv.step()`은 action을 적용한 뒤 실패 사유를 계산한다.

- 먼저 공의 일반 실패 사유를 확인한다: floor contact, out-of-bounds, speed limit 등.
- contact event가 발생하면 outgoing trajectory를 계산한다.
- contact는 있었지만 성공 bounce로 인정되지 않고, 예측 apex가 기준보다 낮으면 `low_apex_contact_observed`가 된다.
- `low_apex_contact_observed`가 grace count보다 많이 연속 발생하면 `failure_reason = "low_apex_contact"`가 된다.
- `failure_reason`이 있으면 `terminated = True`가 된다.
- 라이브 세션은 `terminated` 또는 `truncated` 프레임을 한 번 보낸 뒤 `reset_pending = True`로 바꾸고, 다음 step에서 reset한다.

즉 UI에서는 "10 contacts 정도 치다가 갑자기 초기화"처럼 보이지만, 실제로는 "낮은 apex의 비성공 contact가 누적되어 episode failure 처리"가 된 것이다.

관련 코드:

- `backend/vendor/pingpong_rl2/src/pingpong_rl2/envs/keepup_env.py`
  - action slice 적용: `step()`의 `applied_action[0:17]`
  - low apex 누적과 실패 처리: `low_apex_contact_failure`
  - `terminated = failure_reason is not None`
- `backend/app/live_simulation.py`
  - `LiveSimulationSession.step()`에서 `terminated/truncated` 후 `reset_pending` 설정

### 13D 대표 모델의 평가 지표

현재 카탈로그에 노출되는 13D 대표는 `rl/artifacts/legacy_models/rl2/pmk_cf_self_rally_v20/pmk_cf_self_rally_v20_model.zip`이다.

13D v20 summary:

| 항목 | 값 |
| --- | --- |
| action mode | `position_contact_frame_velocity_tilt_lateral_residual` |
| 학습 timesteps | 1,000,000 |
| reset xy range | 0.028 |
| reset velocity xy range | 0.0 |
| reset velocity z range | [-0.01, 0.01] |
| low apex termination | true |
| low apex grace count | 2 |
| mean return | -11.76 |
| mean useful bounces | 1.62 |
| max useful bounces | 8 |
| failure counts | `low_apex_contact: 80`, `ball_out_of_bounds: 6`, `ball_speed_limit: 4`, `floor_contact: 1`, `time_limit: 9` |

이 수치는 13D v20이 "어느 정도 잘 치다가 10회에서 끊기는 모델"이라기보다, evaluation 기준으로 이미 낮은 apex failure가 주된 실패 모드였다는 뜻이다.

### 왜 13D가 특히 낮은 apex에 취약한가

13D action mode에는 다음 계열의 residual이 있다.

- 0-2: contact-frame target position residual
- 3-4: racket tilt residual
- 5-7: outgoing velocity residual
- 8-10: racket vz, trajectory/centering tilt scale
- 11-12: racket xy velocity residual

하지만 15D/17D에서 추가된 두 차원이 없다.

- 13: `Target Apex Z`
- 14: `Strike Plane Z`

이 둘은 낮게 떠버리는 contact를 직접 고치는 데 중요한 차원이다. 13D도 z 위치와 outgoing velocity를 간접 조정할 수는 있지만, "다음 공이 어느 apex까지 올라가야 하는가"와 "어느 높이의 strike plane에서 맞출 것인가"를 직접 보정하지 못한다.

평가 지표도 이 해석과 맞다.

| 모델 | action mode | mean useful | max useful | low apex failures |
| --- | --- | ---: | ---: | ---: |
| 13D v20 | lateral residual | 1.62 | 8 | 80 |
| 15D v31 | lateral + apex residual | 35.47 | 95 | 23 |
| 17D v32 | apex + tracking residual | 49.67 | 96 | 33 |
| 17D v36 | wider curriculum | 106.86 | 180 | 2 |
| 17D v39 current | current | 119.52 | 181 | 1 |

15D/17D 성능 향상에는 action dimension뿐 아니라 curriculum, reset distribution, reward/termination 설정 변화도 함께 들어갔다. 따라서 "13D 대 15D의 순수 ablation"이라고 단정하면 안 된다. 그래도 low-apex 실패를 줄이는 데 apex/timing residual이 중요한 축이라는 근거는 강하다.

### 13D의 low-apex 종료 조건은 더 빡빡한가

현재 UI에 노출되는 13D v20은 current 17D v39보다 low-apex 종료 기준이 실제로 더 빡빡하다.

| 모델 | low apex threshold | grace count | mean useful | low apex failures |
| --- | ---: | ---: | ---: | ---: |
| 13D v18 | 0.14 | 3 | 2.89 | 18 |
| 13D v19 | 0.20 | 2 | 1.83 | 74 |
| 13D v20 | 0.20 | 2 | 1.62 | 80 |
| 15D v31 | 0.14 | 3 | 35.47 | 23 |
| 17D v32 | 0.14 | 3 | 49.67 | 33 |
| 17D v36 | 0.14 | 6 | 106.86 | 2 |
| 17D v39 | 0.14 | 6 | 119.52 | 1 |

`low_apex_contact_height_threshold`는 공이 racket 기준으로 최소 어느 높이까지 올라갈 것으로 예측되어야 하는지를 뜻한다. 값이 높을수록 더 엄격하다. grace count는 낮은 apex contact를 몇 번까지 봐줄지의 여유 횟수다. 값이 낮을수록 더 엄격하다.

따라서 13D v20은 두 방향 모두 엄격하다.

- threshold: 0.20m로, 17D v39의 0.14m보다 높다.
- grace count: 2로, 17D v39의 6보다 낮다.

그래서 13D v20이 더 빨리 종료되는 데 종료조건 차이가 영향을 준 것은 맞다. 하지만 종료조건만이 원인은 아니다. 13D v18은 0.14/3으로 15D v31, 17D v32와 같은 기준을 쓰는데도 mean useful가 2.89에 그친다. 즉 13D 계열은 action mode와 policy 성능 자체도 약했고, v19/v20에서 종료조건까지 더 엄격해지면서 UI에서 더 금방 죽는 것처럼 보이게 된 것이다.

## 2. 17D action을 7D로 줄일 수 있었나

### 로봇 자유도와 policy action dimension은 다르다

Panda arm은 7자유도지만, 이 프로젝트의 policy는 관절 7개를 직접 출력하지 않는다. action은 task-space residual이다.

현재 17D action label:

| idx | label | 의미 |
| ---: | --- | --- |
| 0 | Radial X | contact frame 기준 radial target offset |
| 1 | Tangent Y | contact frame 기준 tangent target offset |
| 2 | Strike Z | strike 높이 residual |
| 3 | Tilt X | pitch tilt residual |
| 4 | Tilt Y | roll tilt residual |
| 5 | Velocity Scale | outgoing z velocity scale residual |
| 6 | Outgoing X | outgoing x velocity residual |
| 7 | Outgoing Y | outgoing y velocity residual |
| 8 | Racket VZ | racket z velocity residual |
| 9 | Tilt Scale X | trajectory tilt scale residual |
| 10 | Tilt Scale Y | centering tilt scale residual |
| 11 | Racket VX | racket x velocity residual |
| 12 | Racket VY | racket y velocity residual |
| 13 | Target Apex Z | target apex z residual |
| 14 | Strike Plane Z | strike plane z residual |
| 15 | Tracking X | tracking x velocity residual |
| 16 | Tracking Y | tracking y velocity residual |

환경은 이 action들을 controller target position, target tilt, target velocity로 바꾸고, controller가 다시 joint target을 만든다. 그래서 17D는 "관절 수보다 많은 제어 입력"이 아니라 "고수준 타격 계획을 보정하는 feature knobs"에 가깝다.

### 현재 17D 로그 기준 차원별 사용량

`rl/artifacts/keep_v39_17d/analysis/*_contacts.csv` 10개 파일, 총 20,034 contact를 집계했다. 값은 각 action limit으로 나눈 normalized mean abs 기준이다.

| idx | label | norm mean abs | p95 norm | 80% 이상 saturation | 해석 |
| ---: | --- | ---: | ---: | ---: | --- |
| 0 | radial_x | 0.254 | 0.776 | 3.9% | 위치 보정에 의미 있음 |
| 1 | tangent_y | 0.106 | 0.190 | 0.0% | 작지만 완전 무시 수준은 아님 |
| 2 | strike_z | 0.280 | 0.518 | 0.1% | 높이 보정에 의미 있음 |
| 3 | tilt_x | 0.745 | 0.993 | 35.1% | 매우 중요, 거의 한계까지 사용 |
| 4 | tilt_y | 0.530 | 0.715 | 0.3% | 중요 |
| 5 | velocity_scale | 0.038 | 0.086 | 0.0% | contact 기준 사용 작음 |
| 6 | outgoing_x | 0.549 | 0.576 | 0.0% | 매우 중요 |
| 7 | outgoing_y | 0.137 | 0.166 | 0.0% | 보조적 |
| 8 | racket_vz | 0.033 | 0.078 | 0.0% | contact 기준 사용 작음 |
| 9 | tilt_scale_x | 0.071 | 0.104 | 0.0% | 작지만 일관된 bias |
| 10 | tilt_scale_y | 0.011 | 0.020 | 0.0% | 거의 안 씀 |
| 11 | racket_vx | 0.156 | 0.261 | 0.0% | 보조적/의미 있음 |
| 12 | racket_vy | 0.036 | 0.067 | 0.0% | contact 기준 작음 |
| 13 | target_apex_z | 0.379 | 0.455 | 0.0% | 중요 |
| 14 | strike_plane_z | 0.813 | 1.000 | 62.8% | 매우 중요, action limit까지 자주 사용 |
| 15 | tracking_x | 0.048 | 0.057 | 0.0% | contact 기준 작음 |
| 16 | tracking_y | 0.052 | 0.059 | 0.0% | contact 기준 작음 |

### "안 쓰는 차원" 후보

현재 로그만 보면 10번 `centering_tilt_scale`은 가장 강한 제거 후보다. normalized mean이 0.011이고 near-zero 비율이 100%였다.

그 다음 후보는 5번 `velocity_scale`, 8번 `racket_vz`, 12번 `racket_vy`다. 다만 이들은 controller target velocity나 recovery 동작에 섞이기 때문에 완전 제거 전에는 ablation이 필요하다.

15/16번 `tracking_x/y`도 contact 로그 기준으로는 작다. 하지만 이 residual은 공이 하강하는 tracking phase에서 target velocity에 더해지는 구조라, contact 순간 로그만으로는 효과를 과소평가할 수 있다. 삭제 확정이 아니라 ablation 우선 후보로 두는 것이 맞다.

### 줄인다면 어떤 차원 구성이 그럴듯한가

7D로 줄인다면 다음처럼 "핵심 residual만 남기는" 구성이 그나마 설득력 있다.

| 후보 | 남기는 차원 | 의도 |
| --- | --- | --- |
| 7D-core | 0, 2, 3, 4, 6, 13, 14 | 위치 x/z, tilt, outgoing x, apex/strike plane 핵심만 유지 |
| 9D-balanced | 0, 1, 2, 3, 4, 6, 7, 13, 14 | y/tangent와 outgoing y까지 유지 |
| 11D-practical | 0, 1, 2, 3, 4, 6, 7, 9, 11, 13, 14 | 현재 로그상 의미 있는 보조축까지 유지 |
| 15D-no-tracking | 0-14 | 현 17D에서 tracking residual만 제거 |

추천 실험 순서는 `17D -> 15D-no-tracking -> 11D-practical -> 9D-balanced -> 7D-core`다. 바로 7D로 줄이면 어떤 기능이 사라져 성능이 깨졌는지 원인 분리가 어렵다.

## 3. 프로젝트 전체 리뷰에서 보이는 문제

### A. 13D 대표 모델 선택이 성능 기준이 아니다

카탈로그는 차원별 대표 모델을 대부분 "해당 차원의 최신 version"으로 고른다. 5D만 수동 대표가 지정되어 있고, 13D는 수동 지정이 없다. 그래서 13D에서 v20이 대표로 노출된다.

문제는 v20이 13D 중에서도 좋은 대표라고 보기 어렵다는 점이다.

| 13D 모델 | mean useful | max useful | low apex failures |
| --- | ---: | ---: | ---: |
| v18 | 2.89 | 9 | 18 |
| v19 | 1.83 | 8 | 74 |
| v20 | 1.62 | 8 | 80 |

개선안:

- 13D를 legacy/unstable로 표시한다.
- 13D 대표를 v18로 바꾸거나, 아예 발표 UI에서는 13D를 숨긴다.
- 카탈로그 대표 선정 기준에 evaluation score를 반영한다.

### B. UI가 실패 사유를 충분히 보여주지 않는다

백엔드는 프레임에 `failureReason`, `successReason`, `reward`, contact count 등을 보낸다. 프론트도 `failureReason`, `terminated`, `truncated`를 state로 들고 있다. 그러나 화면의 주요 metric은 Height, Contacts, Time, Controller뿐이다.

이 때문에 13D가 종료될 때 사용자는 `low_apex_contact`인지, `floor_contact`인지, `ball_out_of_bounds`인지 바로 알기 어렵다.

개선안:

- metric panel에 `Failure` 또는 `Episode End` 필드를 추가한다.
- contact count와 useful bounce count를 분리해서 보여준다.
- `low_apex_contact` 발생 시 projected apex/threshold를 표시한다.
- reward term breakdown을 디버그 토글로 보여준다.

### C. 라이브 런타임이 `max_episode_steps`를 항상 0으로 덮어쓴다

`backend/app/live_simulation.py`에서 `_resolve_env_kwargs()`가 summary/env kwargs를 읽은 뒤 `max_episode_steps = 0`으로 고정한다. 따라서 UI는 time limit으로 자주 끊기지 않고 계속 보여주려는 의도에 가깝다.

이 자체가 버그는 아니지만, 학습/evaluation summary의 `max_episode_steps`와 라이브 데모 조건이 다르다는 점은 발표 때 언급해야 한다.

해석:

- 13D 리셋은 time limit 때문이 아니다.
- evaluation의 time limit 성공률과 라이브 데모의 지속 시간은 완전히 같은 조건이 아니다.

### D. runtimeCompatible 메타데이터가 shape까지 검증하지 않는다

카탈로그의 `runtimeCompatible`은 action mode가 지원 목록에 있는지만 본다. 실제 action/observation shape mismatch는 모델 선택 후 runtime load/session validation에서 잡는다.

현재 visible 대표 모델들은 큰 문제 없어 보이지만, hidden legacy 모델 중에는 메타데이터가 혼란스러운 사례가 있다. future catalog 노출이 바뀌면 UI에서는 compatible로 보이는데 선택 시 실패하는 모델이 생길 수 있다.

개선안:

- 카탈로그 생성 단계에서 `actionDim`과 `actionMode`가 기대 차원과 일치하는지 검증한다.
- 실패 시 `runtimeCompatible=false`와 구체적 compatibility message를 부여한다.

### E. docs route가 코드상 비활성화되어 있다

`frontend/src/app/App.tsx`에서 `const isDocsPage = false`로 고정되어 있고, Docs nav 링크도 주석 처리되어 있다. docs 파일은 존재하지만 앱에서 `/docs`를 실제 docs page로 렌더링하지 않는다.

발표 자료 관점에서는 괜찮을 수 있지만, "문서가 앱에 연결되어 있다"고 말하면 현재 코드와 맞지 않는다.

개선안:

- `/docs` path 판별을 복구한다.
- docs 링크를 다시 노출한다.
- 아니면 발표에서는 docs를 내부 개발 문서로 설명한다.

## 4. 발표 Q&A용 답변 초안

### Q. 13D는 왜 10 contacts 정도 치고 초기화되나요?

contact 수 제한으로 종료되는 것은 아닙니다. 13D 대표 모델은 낮은 apex로 공을 충분히 띄우지 못하는 contact가 연속으로 발생하는 경향이 있고, 환경은 이런 contact가 grace count를 넘으면 `low_apex_contact` failure로 episode를 종료합니다. 라이브 서버는 종료 프레임을 보낸 뒤 다음 tick에서 reset하기 때문에 UI에서는 10 contacts 전후로 갑자기 초기화되는 것처럼 보입니다.

### Q. 로봇팔은 7자유도인데 왜 action은 17D인가요?

이 action은 7개 관절 명령이 아니라, 타격 계획을 보정하는 task-space residual입니다. 실제 관절 목표는 controller/IK가 만들고, policy는 목표 접촉 위치, 라켓 기울기, outgoing velocity, apex, strike plane, tracking residual 같은 상위 파라미터를 조절합니다. 그래서 로봇 자유도와 policy action dimension이 1:1로 대응하지 않습니다.

### Q. 7D만으로도 가능했을까요?

가능한 설계는 있습니다. 하지만 현재 실험 흐름에서는 13D에서 없던 apex/timing residual이 15D부터 추가되면서 low-apex failure가 크게 줄었습니다. 특히 current 17D 로그에서 `Strike Plane Z`는 action limit까지 자주 쓰이는 핵심 차원입니다. 따라서 단순히 7D로 줄이면 학습은 가능하더라도 현재 성능을 유지하기 어렵고, 줄이려면 15D-no-tracking, 11D, 9D, 7D 순서로 ablation하는 것이 안전합니다.

### Q. 안 쓰는 차원이 있나요?

contact 로그 기준으로는 `centering_tilt_scale`이 거의 안 쓰이고, `velocity_scale`, `racket_vz`, `racket_vy`, `tracking_x/y`도 작게 쓰입니다. 다만 tracking residual은 접촉 순간보다 하강 tracking phase에서 의미가 있을 수 있어 로그만 보고 삭제 결론을 내리면 안 됩니다. 발표에서는 "삭제 후보"가 아니라 "ablation 후보"라고 말하는 것이 정확합니다.

## 5. 다음 검증 실험 후보

1. current 17D policy에서 특정 action 차원을 runtime에서 0으로 마스킹하고 evaluation을 돌린다.
2. `17D -> 15D-no-tracking` mask 실험으로 tracking residual의 기여를 먼저 확인한다.
3. `Strike Plane Z`만 0으로 마스킹해 성능이 얼마나 무너지는지 본다. 현재 로그상 가장 중요한 차원 중 하나라 큰 하락이 예상된다.
4. 13D v18/v20을 동일 reset 조건에서 비교한다. v20이 정말 대표로 부적절한지 확인할 수 있다.
5. UI에 `failureReason`, `successfulBounceCount`, `low_apex_contact` threshold/projection을 추가해 데모 중 원인 분석이 보이게 한다.
