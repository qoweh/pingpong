# 모델 학습 과정 서술형 정리

작성 기준: 2026-06-09 로컬 저장소, git log, `docs/report/`, `docs/rl_presentation_pack/`, training summary, analysis artifact 기준.

이 문서는 [model_evolution_to_v39.md](mujoco/pingpong_rl2/docs/model_evolution_to_v39.md)를 더 서술형으로 풀어쓴 것이다. 표로 버전을 빠르게 비교하려면 기존 문서를 보고, "처음에 무엇을 생각했고, 왜 안 됐고, 다음에 무엇을 바꿨는지"를 발표나 설명용으로 이해하려면 이 문서를 보면 된다.

## 조사한 근거

확인한 자료:

- git commit 흐름: `PingPongSim` 구현, keep-up 재정의, contact trace, contact-frame primitive, residual action 확장, v25 long horizon, v26 unlimited reset, v30/v31, v32 17D transfer, v34/v36/v39 계열 커밋
- 실험 보고서: [docs/report/00_index.md](mujoco/pingpong_rl2/docs/report/00_index.md), 특히 `01`부터 `54`까지의 keep-up 진단/학습 보고서
- 발표용 재정리 문서: [docs/rl_presentation_pack/00_pre_v25_trial_history.md](mujoco/pingpong_rl2/docs/rl_presentation_pack/00_pre_v25_trial_history.md), [01_experiment_story.md](mujoco/pingpong_rl2/docs/rl_presentation_pack/01_experiment_story.md), [07_v35_training_review_and_next_plan.md](mujoco/pingpong_rl2/docs/rl_presentation_pack/07_v35_training_review_and_next_plan.md), [08_v36_wider_domain_review.md](mujoco/pingpong_rl2/docs/rl_presentation_pack/08_v36_wider_domain_review.md)
- 학습 entrypoint와 env 코드: [scripts/run_ppo_learning.py](mujoco/pingpong_rl2/scripts/run_ppo_learning.py), [src/pingpong_rl2/envs/keepup_env.py](mujoco/pingpong_rl2/src/pingpong_rl2/envs/keepup_env.py), [src/pingpong_rl2/controllers/ee_pose_controller.py](mujoco/pingpong_rl2/src/pingpong_rl2/controllers/ee_pose_controller.py), [src/pingpong_rl2/training/bootstrap.py](mujoco/pingpong_rl2/src/pingpong_rl2/training/bootstrap.py)
- 최종 artifact: [keep1_v39_17d_mid_curriculum_fixed_training_summary.json](mujoco/pingpong_rl2/artifacts/ppo_runs/keep1_v39_17d_mid_curriculum_fixed/keep1_v39_17d_mid_curriculum_fixed_training_summary.json), [keep1_v39_oldbase_long7200_eval20_summary.json](mujoco/pingpong_rl2/artifacts/ppo_runs/keep1_v39_17d_mid_curriculum_fixed/analysis/keep1_v39_oldbase_long7200_eval20_summary.json)

## 한 문장 결론

처음에는 "PPO가 로봇팔 목표 위치를 조금씩 움직이면 공을 계속 칠 수 있지 않을까"에서 시작했지만, 실제로는 공을 맞히는 것보다 "적절한 높이와 방향으로 다시 칠 수 있는 상태를 만드는 것"이 훨씬 어려웠다. 그래서 단순 reward 튜닝에서 contact trace와 feasibility 분석으로 넘어갔고, 최종적으로는 contact-frame primitive와 Cartesian controller 위에 PPO가 17D residual action을 학습하는 hybrid 구조가 되었다.

## 용어를 먼저 분리하기

### heuristic

여기서 heuristic은 학습된 neural policy가 아니라 사람이 짠 규칙 기반 policy다. 공 위치와 속도, 예측 접촉점 등을 보고 "대충 이렇게 치면 될 것 같다"는 action을 만든다.

이 프로젝트에서 heuristic은 세 가지 역할로 쓰였다.

1. PPO 전에 환경과 action mode가 말이 되는지 보는 baseline
2. scripted policy가 최소 gate를 넘는지 보는 diagnostic
3. 일부 초기 run에서 PPO actor를 사전 모방학습시키는 bootstrap teacher

중요한 점은 최종 v39 학습 자체에는 heuristic bootstrap이 직접 들어가지 않았다는 것이다. v39 summary에는 `training_mode=resume`, `starting_model_path=keep1_v36_17d_balanced_xyz012_model.zip`, `bootstrap=null`이 기록되어 있다.

### contact-frame

contact-frame은 heuristic과 다른 개념이다. heuristic은 action을 만드는 주체이고, contact-frame은 action을 해석하는 좌표계와 primitive 구조다.

초기에는 action의 x/y가 world 좌표의 단순 이동처럼 쓰였다. 그런데 탁구공은 매번 다른 방향에서 오기 때문에 world x/y만으로는 "공을 라켓 중심 방향으로 돌려보내기"를 안정적으로 표현하기 어렵다. contact-frame은 예측 접촉점과 controller anchor를 기준으로 radial/tangent 방향을 만들고, 그 방향에서 residual을 해석한다.

그래서 contact-frame 모드에서 zero residual은 아무 행동도 안 하는 것이 아니다. 기본 planner가 만든 중심 타격을 수행하고, PPO는 그 위에서 보정량만 낸다.

### bootstrap

bootstrap은 PPO 학습 전에 heuristic rollout으로 observation/action sample을 모으고, PPO actor가 그 action을 흉내 내도록 MSE로 사전학습하는 단계다.

코드상 조건은 [run_ppo_learning.py](mujoco/pingpong_rl2/scripts/run_ppo_learning.py)에서 `starting_model_path is None and bootstrap_heuristic_episodes > 0 and bootstrap_epochs > 0`일 때만 탄다. 따라서 checkpoint에서 resume하는 v25 이후 keep1 계열, 특히 최종 v39는 이 블록을 타지 않았다.

### residual

residual은 "기본 계획을 완전히 대체하는 action"이 아니라 "기본 계획에 더하는 보정값"이다.

예를 들어 최종 17D action에서 PPO가 내는 값은 관절 7개를 직접 움직이는 torque나 joint position이 아니다. contact-frame planner가 만든 목표 위치, 라켓 기울기, 목표 outgoing velocity, target apex, strike plane, tracking velocity 위에 작은 correction을 더하는 값이다.

## 1. 처음 생각: 로봇팔 목표점을 PPO가 조금씩 움직이면 될 것 같았다

초기 `pingpong_rl` 단계에서는 Panda 로봇팔의 end-effector, 즉 라켓 쪽 목표 위치를 PPO action으로 조금씩 움직이는 구조였다. "공 위치를 observation으로 보고, PPO가 라켓을 어디로 보낼지 배우면 되지 않을까"라는 단순한 문제 설정에 가까웠다.

초기 구조의 특징:

- action은 주로 end-effector position delta였다.
- 이후 `position_tilt`처럼 라켓 tilt도 일부 열었다.
- `tracking_assist_weight`가 들어간 run도 있었다.
- 쉬운 reset에서 넓은 reset으로 가는 curriculum도 시도했다.

이 단계에서 어느 정도 "공을 맞히는 장면"은 만들 수 있었지만, 계속 치는 self-rally로 이어지지 않았다. 이유는 단순했다. 공을 한 번 맞히는 것과, 공을 적절한 높이와 방향으로 보내서 다음에도 다시 칠 수 있게 만드는 것은 전혀 다른 목표였다.

특히 tracking assist가 섞인 경우에는 정책이 혼자 다 한 것이 아니라 analytic target이 함께 섞였다. 그래서 최종 발표에서 "PPO가 관절을 처음부터 끝까지 알아서 제어했다"고 말하면 부정확하다. 이 프로젝트의 최종 구조는 순수 end-to-end joint policy가 아니라 hybrid residual RL이다.

## 2. pingpong_rl2로 다시 만든 이유: 보상보다 측정이 먼저였다

`pingpong_rl2`는 단순히 파일명을 바꾼 것이 아니라, 문제를 다시 쪼개기 위해 만든 재설계였다. 초반 보고서들을 보면 reward를 막 추가하기보다, 먼저 실패를 관찰하고 측정하는 도구를 만들었다.

추가된 핵심 도구:

- contact trace: 공과 라켓이 닿기 전/후 속도, 접촉 위치, 라켓 속도, face normal을 기록
- rebound analysis: episode별 useful bounce, stable cycle, failure reason을 집계
- heuristic diagnostic: hand-coded policy가 어느 정도까지 되는지 확인
- contact feasibility map: scripted controller 조합으로 물리적으로 가능한 영역을 탐색
- next-intercept metric: 공을 친 뒤 다음에 다시 칠 수 있는 위치로 돌아오는지 측정

이때 얻은 중요한 교훈은 "reward가 부족해서 못 배우는 것"과 "현재 controller/action 구조로는 좋은 접촉 자체가 잘 안 나오는 것"을 구분해야 한다는 점이었다.

예를 들어 contact feasibility map에서는 scripted controller가 좋은 조합을 찾아도 장기적으로 `3+` useful bounce gate를 안정적으로 넘지 못했다. 반대로 contact oracle 실험에서는 outgoing velocity를 일부 보정하면 더 많은 반복이 가능했다. 이 둘을 함께 보면 물리 자체가 불가능한 것은 아니지만, 당시의 action/control surface가 부족했다는 결론이 나온다.

## 3. observation도 다시 정의했다: 공만 보면 안 됐다

초기에는 공 위치와 속도 중심으로 observation을 보면 될 것처럼 보였다. 하지만 self-rally에서는 현재 공만 보는 것으로 부족했다.

그래서 observation에 들어가는 정보가 늘었다.

- 지금 task phase가 무엇인지
- 직전 contact context가 어떤지
- 다음 intercept가 어디쯤 생기는지
- 다음 intercept가 다시 칠 수 있는 위치인지
- 원하는 outgoing velocity가 무엇인지
- controller target과 contact-frame plan이 어떤 상태인지

발표용 문서에서는 v34 이후 계열을 `55D observation -> PPO policy -> 17D action residual` 구조로 설명한다. 여기서 55D는 출력이 아니라 입력 상태다. 정책은 이 55차원 상태를 보고, 최종적으로 17차원 residual action을 낸다.

중요한 해석:

- observation 차원을 늘린 것은 "정보를 많이 넣으면 좋아지겠지"가 아니라, MDP 상태를 더 정직하게 만들기 위한 것이다.
- 공을 한 번 맞히는 상태와 다음 공을 다시 칠 수 있는 상태는 다르다.
- 그래서 phase, contact context, next intercept, desired outgoing velocity가 필요해졌다.

## 4. reward도 그냥 많이 넣은 것이 아니라 목표를 다시 정의했다

초기 목표는 "공을 맞혔다"에 가까웠다. 하지만 self-rally에서 중요한 것은 "공을 맞힌 뒤 다시 칠 수 있는 공으로 남겼는가"다.

그래서 reward와 success 기준은 다음 방향으로 계속 바뀌었다.

- 단순 contact 보상에서 useful contact 보상으로 이동
- useful contact에 apex height window를 포함
- projected apex XY가 return target 근처인지 확인
- next intercept가 reachable한지 확인
- lateral velocity가 너무 커서 밖으로 나가지 않는지 확인
- low-apex loop를 성공으로 착각하지 않도록 제한
- stable cycle, 즉 유효 타격 이후 다음 공도 쉬운 상태로 돌아오는지를 따로 집계

여기서 "reward 차원을 늘렸다"는 표현은 엄밀히 말하면 reward vector를 PPO가 따로 보는 것이 아니라, reward term과 metric을 세분화했다는 뜻이다. 실제 `env.step()`이 반환하는 reward는 term들의 합이고, 각 term은 분석과 디버깅을 위해 info에 남긴다.

## 5. heuristic은 처음부터 최종 해답이 아니었다

heuristic은 초반에 꼭 필요했지만, 그 자체가 장기 랠리 정답은 아니었다.

처음에는 heuristic diagnostic으로 "이 환경이 말이 되는가", "action mode가 제대로 연결됐는가", "scripted policy가 최소한의 keep-up을 할 수 있는가"를 확인했다. 그런데 heuristic 자체도 반복 랠리에서 한계가 있었다. 특히 contact feasibility와 heuristic gate가 `2` 근처에서 자주 막혔고, 이것이 contact-frame primitive와 residual action 확장의 근거가 됐다.

이후 일부 run에서는 heuristic bootstrap을 사용했다. 예를 들어 v13, v18, v21, v24, v28, v29 계열에는 bootstrap 기록이 남아 있다. 이때 bootstrap은 다음 순서로 작동했다.

1. `HeuristicKeepUpPolicy`가 episode를 실행한다.
2. 성공 기준을 만족한 episode에서 observation/action sample을 모은다.
3. PPO actor가 heuristic action을 deterministic action으로 따라가도록 MSE 학습을 한다.
4. 그 다음 PPO rollout과 policy update를 진행한다.

하지만 이 bootstrap도 만능은 아니었다. bootstrap 후 PPO continuation이 기존 skill을 망가뜨린 기록도 있다. 그래서 후반에는 무작정 새로 배우기보다, 이미 안정적인 checkpoint에서 낮은 learning rate와 작은 clip range로 보수적으로 fine-tune하는 방향이 됐다.

## 6. contact-frame primitive가 큰 전환점이었다

초반 position/tilt 계열 action은 라켓 target을 직접 조금 움직이는 방식이었다. 이 방식은 공이 매번 다른 위치와 방향에서 떨어지는 상황을 표현하기에 부족했다.

contact-frame primitive는 이 문제를 바꿨다.

기본 아이디어:

- 공의 하강 교차점과 controller anchor를 예측한다.
- 그 관계에서 radial/tangent 방향을 만든다.
- planner가 기본 contact target, target apex, desired outgoing velocity를 만든다.
- PPO는 그 위에 residual을 더한다.

초기 contact-frame action은 5D였다.

```text
[radial residual, tangent residual, z residual, pitch residual, roll residual]
```

여기서 radial/tangent는 world x/y가 아니라 contact-frame 기준 방향이다. 그래서 "contact-frame 모드"는 heuristic과 다르다. heuristic이 아니어도 PPO action이 이 좌표계에서 해석될 수 있고, heuristic도 이 action schema에 맞춰 action을 낼 수 있다.

이 전환 이후 zero residual은 빈 action이 아니라 기본 타격이 됐다. 이것이 최종 구조의 핵심 골조다.

## 7. v1-v15: 5D contact-frame만으로는 부족했다

contact-frame primitive가 생긴 뒤에도 바로 성공한 것은 아니었다. `pmk_cf_self_rally_v1`부터 v15까지는 대부분 실패 원인을 하나씩 줄이는 구간이었다.

초기 self-rally 목표는 "공을 위로 띄우는 것"이 아니라 "다시 받을 수 있는 위치와 높이로 보내는 것"으로 재정의됐다. 그래서 target XY, target apex, desired outgoing velocity, strict useful contact 기준이 들어갔다.

하지만 v1/v2는 평균 useful bounce가 1회도 안 되는 수준이었다. 주요 실패는 `ball_out_of_bounds`, `floor_contact`, `robot_body_contact`, 낮은 apex였다. 보고서 흐름을 보면 다음과 같은 수정을 반복했다.

- 라켓이 몸에 부딪히지 않게 body clearance와 nullspace posture를 넣음
- outward racket scene과 state-based tilt를 시도
- tilt timing을 앞당기고 orientation gain을 조정
- low apex를 막기 위해 height reward와 recovery lift를 조정
- stable cycle objective와 material sanity를 추가
- low-apex contact를 너무 빨리 실패로 끊지 않게 recovery memory와 grace를 조정
- v14/v15에서 lateral out-of-bounds를 줄이기 위해 height reward를 easy-next-ball로 gate하고 lateral stability reward를 추가

이 구간의 핵심 결론은 5D action이 너무 좁다는 것이었다. PPO가 contact point와 tilt residual만 조정할 수 있으면, "공을 어느 속도와 방향으로 보내야 하는지", "라켓이 접촉 순간 어떤 속도로 움직여야 하는지", "목표 apex와 strike plane을 어떻게 바꿔야 하는지"를 직접 표현하기 어렵다.

## 8. action ownership를 단계적으로 넘겼다

v15 이후의 흐름은 "PPO에게 무엇을 맡길 것인가"를 계속 다시 나눈 과정이다.

### 5D: position_contact_frame

초기 contact-frame은 5D였다.

```text
[radial, tangent, z, pitch, roll]
```

policy는 접촉점과 라켓 기울기만 보정했다. lift, outgoing velocity, timing은 대부분 hand-coded primitive가 담당했다.

### 8D: velocity residual

v16에서는 8D가 됐다.

```text
[radial, tangent, z, pitch, roll,
 vz_scale, outgoing_x_residual, outgoing_y_residual]
```

이유는 v15에서 `ball_out_of_bounds`가 너무 컸기 때문이다. contact point만 바꿔서는 공을 어느 XY 방향으로 보내야 할지 충분히 직접 제어하기 어려웠다.

결과적으로 v16은 v15보다 좋아졌지만, 새 velocity residual 축은 충분히 쓰이지 않았다. 기록상 `vz_scale`과 outgoing residual이 0 근처에 묶이는 경향이 있었고, 여전히 ball-out과 낮은 apex가 남았다.

### 11D: racket velocity와 tilt scale

v17에서는 11D가 됐다.

```text
[5D 기본,
 vz_scale, outgoing_x, outgoing_y,
 racket_vz_residual, trajectory_tilt_scale, centering_tilt_scale]
```

이유는 v16에서 desired outgoing z는 충분히 높게 잡혔지만, 실제 racket velocity가 그만큼 올라오지 못했기 때문이다. 즉 목표 outgoing velocity를 바꾸는 간접 경로만으로는 부족했고, 라켓 z velocity target에 직접 개입할 필요가 있었다.

또한 pitch/roll residual이 자주 포화되어, raw tilt를 더 크게 여는 대신 trajectory/centering tilt primitive의 scale을 policy가 조정하게 했다.

### 13D: lateral racket velocity residual

v18에서는 13D가 됐다.

```text
[11D 기본, racket_vx_residual, racket_vy_residual]
```

이유는 v17에서 공을 안쪽으로 보내려는 desired outgoing x가 있어도, 접촉 순간 실제 라켓 lateral velocity가 바깥쪽이면 공이 밖으로 튀었기 때문이다.

또 v17에서 발견한 중요한 문제는 per-axis action std였다. 작은 position/tilt action bound와 큰 Gaussian std가 맞지 않아 앞 5개 축이 전부 clipping되는 문제가 있었다. v18에서는 action limit별로 log_std를 맞추는 초기화를 넣었고, 앞 5축 saturation이 내려갔다.

### v19-v20: action을 더 늘리기보다 reward와 timing을 다시 봤다

v18은 v17보다 확실히 좋아졌지만, 낮은 통통 loop가 생겼다. 낮지만 중앙 근처에 공을 남기면 보상을 계속 받는 local optimum이 생긴 것이다.

v19는 action dimension을 늘리지 않고 height-qualified reward를 넣었다. 낮은 apex에서는 lateral stability나 stable contact 보상을 덜 주고, target apex 쪽으로 회복하는 potential shaping을 추가했다.

v20은 boundary-out을 줄이기 위해 contact target offset과 lateral brake를 넣었다. 하지만 low-apex가 악화되어 v20 자체를 계속 밀기보다, v19 기반에서 apex/timing을 직접 열기로 했다.

### 15D: apex와 strike plane residual

v21부터 15D가 됐다.

```text
[13D 기본, target_apex_z_residual, strike_plane_z_residual]
```

이유는 낮은 apex 문제를 z velocity residual로 간접 제어하는 것보다, "이번 공을 어느 높이까지 보내고 어느 높이에서 치는가"를 policy가 직접 조정하게 하는 편이 명확했기 때문이다.

v21은 낮지만 안정적인 루프를 꽤 만들었으나, useful 기준이 너무 엄격해서 viewer에서 보이는 느낌과 metric이 달랐다. v22에서는 useful height window를 실제 양방향 window로 고쳤다. 즉 `0.30m` 이상만 useful로 보는 것이 아니라, `0.20m~0.40m` 범위를 안정적인 성공 높이로 보게 했다.

v23에서는 남은 `ball_out_of_bounds` 원인을 분석했다. 정책이 공을 바깥으로 보내고 싶어 한 것이 아니라, 접촉 순간 라켓이 바깥 방향으로 움직여 공을 밀어낸 것이 문제였다. 그래서 outward racket velocity penalty와 약한 lateral brake를 추가했다.

## 9. v25: 30회 이상 유지 목표가 성능을 드러냈다

v23은 이미 꽤 안정적인 정책이었지만, episode horizon이 짧았다. `max_episode_steps=600`이면 정책이 더 오래 칠 수 있어도 episode가 먼저 끝난다. v25에서는 horizon을 `1800`으로 늘리고, checkpoint 선택 기준에 `10+`, `20+`, `30+` useful bounce rate를 넣었다.

v25의 의미:

- 새 reward 하나가 갑자기 모든 것을 해결한 것이 아니다.
- v23에서 만들어진 안정 정책을 더 긴 목표와 맞는 평가 기준으로 드러낸 것이다.
- `time_limit`이 늘어난 것은 실패가 아니라 safety cap까지 살아남은 episode가 많아졌다는 뜻이다.

v25 결과는 mean useful 약 30회, max 약 48-51회, 30+ rate 60%대까지 올라갔다. 이때부터 "발표 가능한 keep-up"이 보이기 시작했다.

## 10. v26-v31: reset distribution은 천천히 넓혀야 했다

v26에서는 시연과 웹 서비스 관점에서 "실패할 때까지 계속 실행"하는 unlimited horizon을 넣고, reset XY/Z 분포를 조금 넓혔다.

이후 v28/v29에서는 17D tracking residual, 넓은 XY, 초기 속도, spin을 한 번에 크게 키웠다. 결과는 좋지 않았다. 일부 긴 episode는 살아남았지만, mean useful과 30+ rate가 크게 떨어졌고 tracking residual도 처음에는 거의 쓰이지 않았다. v29에서 tracking residual 사용량은 늘었지만 lateral drift가 커졌다.

이 실험의 교훈은 중요하다.

- action 차원을 늘리면 무조건 좋아지는 것이 아니다.
- reset 난도, 초기 속도, spin, action dimension을 한 번에 키우면 기존 안정성이 깨진다.
- 새 residual 축이 있어도, 그 축을 어떻게 안정적으로 탐색하고 보상받을지 따로 설계해야 한다.

그래서 다시 v26 안정 모델로 돌아가, action mode는 유지하고 reset XY만 조금 넓힌 v30을 만들었다. v30은 성공적이었다. 반면 v31은 XY를 더 넓혀 `0.14m`까지 가면서 성능이 떨어졌다. 이 흐름은 curriculum을 한 번에 크게 주면 실패한다는 근거가 됐다.

## 11. v32: 15D 정책을 버리지 않고 17D로 확장했다

17D는 15D 뒤에 tracking velocity residual x/y를 추가한 구조다.

```text
15D + [tracking_vx_residual, tracking_vy_residual]
```

처음 v28처럼 17D를 scratch로 학습하면 기존 안정성이 크게 흔들렸다. 그래서 v32에서는 [scripts/expand_ppo_action_space.py](mujoco/pingpong_rl2/scripts/expand_ppo_action_space.py)를 사용해 기존 15D v30 policy를 17D로 확장했다.

핵심 방법:

- observation space는 그대로 둔다.
- 기존 policy/value network weight를 복사한다.
- action head의 앞 15개 출력은 기존 weight를 복사한다.
- 새 2개 action head는 0으로 초기화한다.
- 그래서 17D 모델이 시작할 때는 기존 15D 정책과 거의 같은 행동을 한다.

이 방식은 heuristic bootstrap이 아니다. 기존 PPO checkpoint를 action space에 맞춰 확장한 transfer다.

v32 17D fine-tune은 v30보다 mean useful과 30+ rate를 조금 올리고, ball-out 실패를 줄였다. 이후 v33/v34/v35/v36/v39 계열의 기반이 됐다.

## 12. v34-v36: 장기 랠리 목표와 trade-off를 다시 봤다

v25-v32에서는 `30+ useful`이 중요한 지표였다. 하지만 사용자가 목표로 잡은 "거의 계속 치는 느낌"은 1800-step eval만으로는 부족했다. 그래서 v33/v34 이후에는 7200-step long eval로 `contacts 300 / useful 100`, `contacts 400 / useful 150` 같은 장기 랠리 목표를 봤다.

v34는 큰 개선이었다.

- reset XY, height, 초기 velocity 범위를 넓혔다.
- stable cycle reward cap을 늘렸다.
- low-apex threshold를 무작정 낮추지 않고, recovery grace를 늘렸다.
- 7200-step long eval에서 평균 useful 100회 이상, max useful 170회 수준을 보였다.

v35는 body contact와 lateral stability를 더 강하게 밀어본 실험이다. training summary만 보면 좋아 보였지만, long horizon에서는 v34보다 약했다. body contact는 줄었지만 next-intercept/easy-next-ball 품질이 떨어졌고, 장기 랠리 목표 달성률도 낮아졌다.

v36은 v34에서 v35의 장점을 약하게 가져온 balanced fine-tune이었다. training summary 기준으로는 mean useful 106.86, max 180, 30+ rate 0.87로 좋아 보였다. 다만 같은 seed의 7200-step long eval에서는 v34보다 낮았다. 대신 더 넓은 위치/속도 stress 조건에서는 v34보다 나은 면을 보였다.

이 구간의 결론은 "평가 기준 하나만 보면 모델 선택을 잘못할 수 있다"였다. short eval, long eval, stress eval을 함께 봐야 했다.

## 13. v37-v39: 넓히는 과정에서 무너진 뒤, v39로 정리했다

v37은 v36에서 시작해 reset distribution을 더 넓히는 guarded curriculum이었다. 하지만 `ball_out_of_bounds`가 크게 늘어 안정성이 떨어졌다.

v38은 중간 curriculum recover 성격이었지만, low-apex failure가 다시 커졌다. 즉 넓은 조건으로 확장하면서 기본 랠리 안정성이 흔들린 것이다.

최종 v39는 v36 checkpoint에서 다시 시작해 mid curriculum을 고친 fine-tune이다.

v39 summary 기준:

```text
training_mode = resume
starting_model_path = keep1_v36_17d_balanced_xyz012_model.zip
completed_timesteps = 700,000
action_mode = position_contact_frame_velocity_tilt_lateral_apex_tracking_residual
bootstrap = null
```

v39 train eval 100 episode:

| metric | value |
| --- | ---: |
| mean useful bounces | 119.52 |
| max useful bounces | 181 |
| 30+ useful rate | 0.83 |
| low-apex failures | 1 |

v39 long eval 20 episode, 7200-step:

| metric | value |
| --- | ---: |
| mean useful bounces | 130.95 |
| max useful bounces | 182 |
| 30+ useful rate | 0.90 |

따라서 최종 설명은 이렇게 해야 정확하다.

> 최종 v39는 heuristic으로 직접 움직이는 모델이 아니라, v36 PPO checkpoint에서 이어 학습한 17D residual PPO 모델이다. 다만 contact-frame primitive와 Cartesian controller는 여전히 남아 있으며, PPO는 그 위에서 residual을 학습한다.

## 14. 코드에서 실제 학습은 어떻게 흐르는가

학습 시작점은 [scripts/run_ppo_learning.py](mujoco/pingpong_rl2/scripts/run_ppo_learning.py)다.

큰 흐름:

1. CLI 또는 config file을 읽는다.
2. preset과 `--set` override를 적용한다.
3. `env_kwargs_from_args()`로 환경 설정을 만든다.
4. `PingPongKeepUpGymEnv`를 만들어 action/observation space와 training config를 확정한다.
5. 여러 MuJoCo env를 vector env로 만든다.
6. 새 run이면 `PPO("MlpPolicy", ...)`를 생성한다.
7. resume run이면 `PPO.load(starting_model_path, env=monitored_env)`로 checkpoint를 불러온다.
8. bootstrap 조건이 맞으면 heuristic rollout dataset으로 actor를 MSE 사전학습한다.
9. `model.learn()`이 rollout을 모으고 PPO update를 반복한다.
10. checkpoint evaluation과 final evaluation을 저장한다.
11. training summary에 env config, evaluation, bootstrap 여부, starting model path를 남긴다.

코드상 중요한 조건:

```text
if starting_model_path is None:
    model = PPO(...)
else:
    model = PPO.load(starting_model_path, env=monitored_env)

if starting_model_path is None and bootstrap_heuristic_episodes > 0 and bootstrap_epochs > 0:
    collect_heuristic_bootstrap_dataset(...)
    bootstrap_actor_from_dataset(...)
```

그래서 v39처럼 `starting_model_path`가 있는 resume run에서는 bootstrap을 직접 실행하지 않는다.

## 15. 로봇팔은 최종적으로 어떻게 움직이는가

최종 action flow는 다음과 같다.

```text
observation
  -> PPO policy
  -> 17D residual action
  -> PingPongKeepUpEnv.step()
  -> contact-frame plan + residual 적용
  -> target position / target tilt / target velocity
  -> RacketCartesianController.compute_joint_targets()
  -> Jacobian 기반 differential IK
  -> 7개 joint target
  -> MuJoCo mj_step substeps
  -> 다음 observation / reward / failure reason
```

[keepup_env.py](mujoco/pingpong_rl2/src/pingpong_rl2/envs/keepup_env.py)에서는 action mode에 따라 action vector 길이를 만든다. 최종 17D 모드에서는 위치/tilt/velocity/lateral/apex/tracking residual slot을 차례대로 붙인다.

`step()` 안에서는 policy action을 clip한 뒤 다음처럼 해석한다.

- `0:3`: contact-frame position residual
- `3:5`: pitch/roll tilt residual
- `5:8`: desired outgoing velocity residual
- `8`: racket z velocity residual
- `9:11`: trajectory/centering tilt scale residual
- `11:13`: racket x/y velocity residual
- `13`: target apex z residual
- `14`: strike plane z residual
- `15:17`: tracking x/y velocity residual

그 다음 contact-frame planner가 만든 기본 target에 residual을 더하고, controller target position/tilt/velocity를 만든다.

[ee_pose_controller.py](mujoco/pingpong_rl2/src/pingpong_rl2/controllers/ee_pose_controller.py)는 이 Cartesian target을 7개 관절 target으로 바꾼다. 여기서 쓰는 것은 엄밀한 closed-form IK가 아니라 Jacobian 기반 differential IK다.

수식으로는 대략 다음 형태다.

```text
delta_q = J^T (J J^T + lambda I)^-1 error
```

여기서:

- `q`: 7개 관절 위치
- `delta_q`: 이번 control step에서 관절을 얼마나 바꿀지
- `J`: MuJoCo `mj_jacSite()`로 얻은 라켓 site의 position/rotation Jacobian
- `error`: 라켓 목표 위치 오차, 목표 속도 step, face normal 방향 오차를 합친 task error
- `lambda`: damping

그 뒤 nullspace posture와 body clearance 보정을 더하고, joint limit으로 clip한 뒤 MuJoCo에 joint target을 넣는다. MuJoCo는 `mj_step()` substep을 돌며 물리 상태를 적분하고, contact trace는 그 substep 중 공과 라켓의 접촉 정보를 기록한다.

## 16. 발표나 질문에 대한 짧은 답

### "heuristic을 썼나요?"

예전 일부 run에서는 썼다. 하지만 최종 v39 학습 자체에는 쓰지 않았다. v39는 v36 PPO checkpoint에서 resume한 모델이고, bootstrap summary가 null이다.

### "contact-frame이 heuristic인가요?"

아니다. contact-frame은 action을 해석하는 좌표계와 기본 타격 primitive다. heuristic은 hand-coded action generator다. 둘은 같이 쓰일 수도 있고, PPO policy가 contact-frame action을 낼 수도 있다.

### "PPO가 7개 관절을 직접 움직이나요?"

아니다. 최종 모델에서 PPO output은 17D residual action이다. 이 residual이 target position/tilt/velocity를 보정하고, Cartesian controller가 Jacobian 기반 IK로 7개 joint target을 만든다.

### "왜 observation을 늘렸나요?"

공 위치/속도만으로는 다음 접촉 가능성과 성공 조건이 충분히 표현되지 않았기 때문이다. 그래서 task phase, contact context, next intercept, desired outgoing velocity 같은 정보를 넣어 MDP 상태를 더 명확히 만들었다.

### "왜 action 차원을 늘렸나요?"

초기 5D는 contact point와 tilt만 보정했다. 하지만 실패 원인이 height, outgoing velocity, lateral racket velocity, apex/timing, tracking으로 분리되면서, 해당 축을 PPO가 직접 보정할 수 있게 8D, 11D, 13D, 15D, 17D로 늘렸다.

### "최종적으로 PPO가 한 일은 뭔가요?"

controller가 만든 기본 타격 계획을 상황별로 보정하는 residual policy를 학습했다. 즉 "공을 어디로 보낼지, 얼마나 높게 살릴지, 접촉 타이밍과 라켓 속도를 어떻게 보정할지"를 55D observation에서 판단해 17D action으로 출력한 것이다.

## 전체 흐름 요약

```text
초기 pingpong_rl
  - EE delta / position_tilt PPO
  - tracking assist 일부 사용
  - 공을 맞히는 장면은 만들었지만 self-rally로 부족

pingpong_rl2 재설계
  - contact trace, rebound analysis, feasibility map
  - observation에 phase/contact/next-intercept 추가
  - reward보다 control/action abstraction 병목 확인

heuristic과 bootstrap
  - baseline, diagnostic, 일부 actor warm-start
  - 최종 v39에서는 직접 사용 안 함

contact-frame primitive
  - zero residual = 기본 중심 타격
  - PPO는 contact-frame residual을 학습

action ownership 확장
  - 5D position/tilt
  - 8D velocity residual
  - 11D racket_vz / tilt scale
  - 13D racket_xy velocity
  - 15D target apex / strike plane
  - 17D tracking x/y residual

학습 운영 개선
  - horizon 600 -> 1800 -> unlimited/safety cap
  - 30+ useful, 7200-step long eval
  - reset distribution curriculum
  - 15D -> 17D weight transfer

최종 v39
  - v36 checkpoint에서 700k resume
  - heuristic bootstrap 없음
  - 17D residual PPO
  - train eval mean useful 119.52, max 181
  - long eval mean useful 130.95, max 182
```

