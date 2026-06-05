# 제어 모델과 학습

현재 기본 제어 모델은 `keep_v39_17d`다. 모델 파일 이름은 `keep_v39_17d_model.zip`이고, Stable-Baselines3 2.8.0의 PPO 형식으로 저장되어 있다. 초기 active model은 `.env`의 `PINGPONG_POLICY_MODEL_PATH`로 정하지만, 실행 중에는 웹 UI의 Action Dimension 섹션에서 `rl/artifacts` 아래의 dimension별 최신 대표 PPO 모델로 전환할 수 있다.

중요한 구분이 있다. PPO는 학습 알고리즘이고, 학습된 제어기는 PPO가 최적화한 policy network다. 흔히 "PPO 모델"이라고 부르지만, 웹에서 매 step action을 계산하는 것은 그 안의 actor, 즉 policy 쪽 네트워크다.

## SB3 PPO zip 안에 들어 있는 것

Stable-Baselines3의 저장 파일은 단순한 신경망 가중치만 담지 않는다.

| 구성 | 의미 |
| --- | --- |
| Observation space | policy 입력 차원과 범위 정보. 현재 55D |
| Action space | policy 출력 차원과 bounds. 현재 17D |
| Policy parameters | actor와 critic 신경망 가중치 |
| `log_std` | 연속 action 분포의 표준편차 파라미터 |
| PPO metadata | `n_steps`, `gamma`, `clip_range` 같은 학습 설정 |
| Optimizer state | 이어서 학습할 때 필요한 optimizer 내부 상태 |

서버 runtime은 모델을 `PPO.load(...)`로 읽고, 매 step `predict(observation, deterministic=True)` 형태로 action을 얻는다. deterministic inference에서는 학습 때처럼 action을 무작위로 샘플링하기보다 현재 policy가 고른 대표 action을 사용한다.

## 현재 policy network 구조

`keep_v39_17d_model.zip`의 `policy.pth`에서 확인한 tensor shape 기준 구조는 아래와 같다.

```text
Observation (55)
  |
  +-- Actor / policy path
  |     Linear(55 -> 64) + Tanh
  |     Linear(64 -> 64) + Tanh
  |     Linear(64 -> 17) = action mean
  |     log_std(17)      = stochastic training/inference용 표준편차
  |
  +-- Critic / value path
        Linear(55 -> 64) + Tanh
        Linear(64 -> 64) + Tanh
        Linear(64 -> 1) = state value
```

Actor는 observation을 보고 17차원 action 평균을 낸다. Critic은 같은 observation을 보고 "이 상태가 앞으로 얼마나 좋은 보상을 기대할 수 있는지"를 1개의 값으로 추정한다. PPO 학습 중에는 actor와 critic이 함께 업데이트되지만, 웹에서 로봇을 움직이는 직접 출력은 actor의 action이다.

## v39 학습 카드

| 항목 | 값 |
| --- | --- |
| Algorithm | PPO |
| Library | Stable-Baselines3 2.8.0 |
| Policy class | `ActorCriticPolicy` |
| Run name | `keep_v39_17d` |
| Action mode | `position_contact_frame_velocity_tilt_lateral_apex_tracking_residual` |
| Observation | 55D |
| Action | 17D |
| 추가 학습 step | 700,000 |
| 병렬 환경 수 | 4 |
| Rollout length | `n_steps=512` per env |
| Batch size | 512 |
| Learning rate | 8e-7 |
| Gamma | 0.99 |
| GAE lambda | 0.95 |
| PPO epochs | 1 |
| Clip range | 0.01 |
| Seed | 7 |

이 설정은 안정성을 꽤 보수적으로 잡은 형태다. learning rate와 clip range가 작고, 한 번에 크게 바꾸기보다 이전에 학습된 v36 계열 checkpoint에서 이어 학습했다.

## 무엇을 학습시켰나

학습 데이터는 사람이 라벨링한 정답 action 목록이 아니다. PPO는 policy가 직접 환경을 실행하며 모은 rollout으로부터 학습한다.

1. 여러 개의 MuJoCo 환경을 병렬로 실행한다.
2. 현재 policy가 observation을 보고 action을 낸다.
3. 환경이 action을 적용하고 reward, 다음 observation, 종료 여부를 계산한다.
4. 일정 길이의 rollout을 모은다.
5. reward 흐름으로 advantage와 return을 추정한다.
6. PPO loss로 actor와 critic을 업데이트한다.
7. 업데이트된 policy로 다시 환경을 실행한다.

이 과정을 반복하면서 policy는 "어떤 observation에서 어떤 residual action을 내면 이후 reward가 좋아지는지"를 배운다.

## 학습 분포

v39는 공 시작 조건을 고정하지 않았다. 학습 중 시작 위치와 속도를 바꿔 policy가 다양한 상황에 적응하도록 했다.

| 항목 | 학습 분포 |
| --- | --- |
| XY 시작 offset | 라켓 기준 반경 0.13m disk sampling |
| 공 시작 높이 | 라켓 위 0.22m ~ 0.52m |
| X/Y 초기 속도 | -0.045m/s ~ +0.045m/s |
| Z 초기 속도 | -0.14m/s ~ +0.04m/s |

웹 조작 패널은 이 학습 안전 범위를 우선 사용한다. 학습 분포가 XY disk sampling인 모델은 X/Y를 독립 사각형으로 두지 않고 반경 기준으로 clamp한다. 같은 clamp는 range slider, 숫자 입력, backend `parse_ball_spawn_options`에 모두 적용된다.

## 평가 요약

100 episode evaluation 기준 v39 결과는 아래와 같다.

| 지표 | 값 |
| --- | ---: |
| Mean return | 1076.64 |
| Mean useful bounces | 119.52 |
| Max useful bounces | 181 |
| 1+ useful bounce rate | 0.87 |
| 10+ useful bounce rate | 0.86 |
| 20+ useful bounce rate | 0.85 |
| 30+ useful bounce rate | 0.83 |

긴 7200 step 분석에서는 평균 contact 353.5, 평균 useful bounce 130.9를 기록했다. 여기서 useful bounce는 단순 접촉이 아니라 다음 공으로 이어질 수 있는 조건을 만족한 접촉에 가깝다.

## 웹 서비스에서의 모델 사용

웹 서비스는 브라우저에서 action을 계산하지 않는다. 서버가 PPO 모델과 MuJoCo 환경을 함께 실행한다.

```text
MuJoCo state
  -> 55D observation
  -> PPO policy predict
  -> 17D residual action
  -> racket target/controller
  -> robot control and MuJoCo step
  -> rendered frame
```

브라우저는 action 값을 시각화하고, 공 시작 조건이나 재생 상태 같은 사용자의 조작을 서버에 전달한다. 따라서 웹 UI를 바꿔도 policy가 배운 행동 자체가 바뀌지는 않는다. policy를 바꾸려면 새 모델을 학습하거나 다른 모델 zip을 선택해야 한다.

## 다중 모델 선택

웹 서버는 `rl/artifacts` 아래의 대표 PPO zip을 자동으로 수집한다. `checkpoints/` 아래 중간 저장본은 기본 목록에서 제외하고, 각 run 폴더의 `<run>_model.zip` 또는 대표 best model만 노출한다.

`GET /api/models`는 사용 가능한 모델 목록과 현재 active model을 반환한다. `POST /api/models/select`는 선택한 PPO zip을 로드하고 새 env kwargs, 공 시작값 config, policy metadata를 적용한다. 서버는 shared live session을 새 runtime으로 교체하고, 브라우저는 기존 캔버스와 WebSocket을 유지한 채 새 frame을 받아 화면만 갱신한다.

모델 metadata는 가능한 범위에서 아래 정보를 포함한다.

| 필드 | 설명 |
| --- | --- |
| `id` | 서버 전환에 쓰는 안정적인 run id |
| `name` / `displayName` | UI에 표시하는 짧은 이름 |
| `algorithm` | PPO |
| `observationDim` / `actionDim` | SB3 zip metadata의 observation/action space shape |
| `actionMode` | training summary에서 복원한 action mode |
| `runtimeCompatible` / `compatibilityMessage` | 현재 서버 런타임에서 열 수 있는 모델인지 여부와 비호환 사유 |
| `ballSpawn` | 모델별 학습/검증 공 시작값 범위 |
| `policy.architecture` | 가능한 경우 policy network 요약 |
| `trainingSummaryPath` | 내부 추적용 summary JSON 위치 |

예시 응답:

```json
{
  "activeModel": "keep_v39_17d",
  "models": [
    {
      "id": "keep_v39_17d",
      "name": "17D V11",
      "displayName": "17D V11 · Current v39",
      "dimensionGroup": "17D",
      "versionLabel": "V11",
      "observationDim": 55,
      "actionDim": 17,
      "runtimeCompatible": true
    }
  ]
}
```

`rl1`, `rl2`, `rl3`처럼 실험 시점별로 나뉜 legacy run은 action dimension별로 묶고, 같은 dimension 안에서는 오래된 run부터 `V1`, `V2`를 부여한다. 현재 UI에는 각 dimension에서 가장 최신 대표 모델 하나만 노출한다. 버튼은 `17D`, `5D`, `3D`처럼 dimension을 크게 보여주고, 보조 라벨로 `V11`, `V27`, `V9` 같은 내부 순번만 보여준다. 원본 run 이름과 path는 내부 metadata로만 유지하고 선택 UI에는 노출하지 않는다.

3D/5D legacy 모델은 최신 환경보다 policy observation 길이가 짧다. 서버는 모델 ZIP metadata의 observation shape를 보고, 최신 35D+ observation에서 예전 policy가 학습 때 보던 26D/29D 필드만 추려 `predict(...)`에 넣는다. 이 adapter는 웹 표시용 상태를 바꾸지 않고 policy 입력만 맞춘다.

## Policy output/action 시각화

프레임 payload에는 서버 PPO가 실제로 낸 `action` 배열과 `modelId`가 포함된다. 프론트엔드는 선택된 모델의 `actionDim`과 `actionLabels`를 기준으로 action bar를 동적으로 생성한다.

| 모델 계열 | 표시 방식 |
| --- | --- |
| 3D | 3개 action bar |
| 5D | 5개 action bar |
| 8D | 8개 action bar |
| 11D | 11개 action bar |
| 13D | 13개 action bar |
| 15D | 15개 action bar |
| 17D | 17개 action bar |
| 새 차원 | metadata의 `actionDim`만 맞으면 같은 컴포넌트로 표시 |

Action label은 training summary에 명시된 값이 있으면 우선 사용하고, 없으면 action mode 기반 라벨을 생성한다. 둘 다 없거나 차원이 맞지 않으면 `Control 1`, `Control 2` 형태로 채운다. Live UI에서는 사용자가 모델 선택과 action 크기를 빠르게 읽는 것이 중요하므로 policy architecture chip은 표시하지 않는다. 발표용으로는 SB3 policy 객체에서 linear layer를 읽어 `Observation -> hidden layers -> policy output/value` 다이어그램을 별도 시각화할 수 있다.

## 추가 학습 명령어

학습 안전 범위를 유지하면서 v39를 조금 더 안정화하는 command는 아래처럼 둘 수 있다.

```sh
export PYTHONPATH=src
export MUJOCO_GL=osmesa
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python scripts/run_ppo_learning.py \
  --config-file configs/keep1_v32_17d_transfer.json \
  --run-version v40_17d_safe_range_refresh \
  --resume-from artifacts/ppo_runs/keep1_v39_17d_mid_curriculum_fixed/keep1_v39_17d_mid_curriculum_fixed_model.zip \
  --set total_timesteps=500000 \
  --set reset_xy_range=0.13 \
  --set reset_xy_sampling=disk \
  --set reset_velocity_xy_range=0.045 \
  --set reset_velocity_z_range='[-0.14,0.04]' \
  --set reset_ball_height_bounds='[0.22,0.52]' \
  --set learning_rate=5e-7 \
  --set n_epochs=1 \
  --set clip_range=0.01 \
  --set eval_episodes=100 \
  --set evaluation_step_limit=7200 \
  --set bootstrap_heuristic_episodes=0 \
  --set bootstrap_epochs=0 \
  --set bootstrap_followup_epochs=0
```

## 참고 문서

- [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/v2.8.0/modules/ppo.html)
- [Stable-Baselines3 Policy Networks](https://stable-baselines3.readthedocs.io/en/v2.8.0/guide/custom_policy.html)
