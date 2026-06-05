# 제어 모델과 학습

현재 웹 서비스의 기본 모델은 `keep_v39_17d` run을 사용한다. 모델 파일 경로는 코드에 흩어져 있지 않고 `.env`에서 한 번만 관리한다.

```text
PINGPONG_POLICY_MODEL_PATH=rl/artifacts/keep_v39_17d/keep_v39_17d_model.zip
```

## 모델 실행 방식

서버는 vendored `pingpong_rl2` 소스를 import하고, 선택된 모델 파일과 같은 폴더의 training summary에서 환경 설정을 복원한다. 그 다음 원본 Gym 환경을 만들고 매 제어 step마다 PPO policy의 action을 계산한다.

브라우저는 action을 직접 계산하지 않는다. 서버에서 받은 MuJoCo 상태를 웹 뷰어에 반영하고, 화면 조작값을 서버에 명령으로 전달한다.

## v39 학습 설정

| 설정 | 값 |
| --- | --- |
| Algorithm | PPO |
| Run name | `keep_v39_17d` |
| Preset | `contact_frame_self_rally_v32_17d_v30_transfer` |
| 시작 checkpoint | v36 17D balanced model |
| 추가 학습 step | 700,000 |
| 병렬 환경 수 | 4 |
| n_steps | 512 |
| Batch size | 512 |
| Learning rate | 8e-7 |
| Gamma | 0.99 |
| Epochs | 1 |
| Clip range | 0.01 |
| Seed | 7 |
| Observation | 55D |
| Action | 17D |
| Action mode | `position_contact_frame_velocity_tilt_lateral_apex_tracking_residual` |

## v39 도메인

학습 summary에 저장된 기본 분포는 아래와 같다.

| 항목 | 범위 |
| --- | --- |
| XY 시작 offset | 반경 0.13 m disk sampling |
| 공 시작 높이 | 0.22 m ~ 0.52 m above racket |
| XY 초기 속도 | -0.045 m/s ~ +0.045 m/s |
| Z 초기 속도 | -0.14 m/s ~ +0.04 m/s |

웹 조작 패널은 v39 분석에서 따로 확인한 확장 분포까지 열어 둔다.

| 항목 | 웹 조작 범위 |
| --- | --- |
| X/Y 시작 offset | -0.16 m ~ +0.16 m |
| 공 시작 높이 | 0.18 m ~ 0.56 m above racket |
| X/Y 초기 속도 | -0.06 m/s ~ +0.06 m/s |
| Z 초기 속도 | -0.18 m/s ~ +0.04 m/s |

확장 범위는 `cov_v39_*`와 `stress_v39_*` 분석에서 평가된 값이다. 다만 학습 분포는 XY disk sampling이므로 X와 Y를 동시에 끝까지 밀면 반경 기준으로는 학습 때보다 더 어려운 상태가 된다.

## v39 평가 요약

100 episode evaluation 기준 결과다.

| 지표 | 값 |
| --- | ---: |
| Mean return | 1076.64 |
| Mean useful bounces | 119.52 |
| Max useful bounces | 181 |
| 1+ useful bounce rate | 0.87 |
| 10+ useful bounce rate | 0.86 |
| 20+ useful bounce rate | 0.85 |
| 30+ useful bounce rate | 0.83 |

긴 7200 step oldbase 분석에서는 평균 contact 353.5, 평균 useful bounce 130.9를 기록했고, 20 episode 중 16 episode가 contact 300회와 useful bounce 100회를 동시에 넘겼다.

## 모델 교체 절차

1. 새 모델 zip과 training summary를 `rl/artifacts/<run_name>/` 아래에 둔다.
2. 모델 파일명은 `<run_name>_model.zip`, training summary는 `<run_name>_training_summary.json`로 맞춘다.
3. `.env`의 `PINGPONG_POLICY_MODEL_PATH`를 새 모델 zip 경로로 바꾼다.
4. 모델이 다른 scene이나 asset을 요구하면 `rl/assets`와 compiled `frontend/public/assets/mujoco/pingpong_scene.mjb`도 함께 갱신한다.
5. 서버를 다시 시작해서 새 모델과 환경 설정을 로드한다.

모델 파일만 바꾸는 경우에는 프론트엔드 코드를 수정하지 않는다. 다만 observation/action 차원, action mode, scene의 joint 수, geom/site 이름이 바뀌면 backend env 설정과 웹 뷰어의 identifier 매핑을 함께 확인해야 한다.

## 서버에서 추가 학습

학습은 웹 프로젝트가 아니라 원본 RL 프로젝트에서 실행한다.

원본 artifacts에는 `keep1_v40_17d_v39_polish`도 있지만, 저장된 eval100 기준 mean useful bounces가 `106.11`로 v39의 `119.52`보다 낮다. 그래서 현재 웹 기본 모델은 v39를 유지하고, 추가 개선은 v39에서 새 run으로 이어 학습하는 쪽을 기준으로 한다.

```sh
cd /home/pilt/pingpong_rl2
```

Ubuntu 계열 CPU 서버라면 MuJoCo 런타임 라이브러리를 먼저 준비한다.

```sh
sudo apt-get update
sudo apt-get install -y python3-venv libegl1 libgl1 libglib2.0-0 libosmesa6
```

가상환경은 conda를 이미 쓰고 있으면 기존 `mujoco_env`를 써도 되고, 서버에서는 venv로도 충분하다.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel
python -m pip install -e .
```

v39에서 이어서 확장 범위를 더 학습시키는 보수적인 command는 아래 형태다.

```sh
export PYTHONPATH=src
export MUJOCO_GL=osmesa
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python scripts/run_ppo_learning.py \
  --config-file configs/keep1_v32_17d_transfer.json \
  --run-version v40_17d_server_polish \
  --resume-from artifacts/ppo_runs/keep1_v39_17d_mid_curriculum_fixed/keep1_v39_17d_mid_curriculum_fixed_model.zip \
  --set total_timesteps=700000 \
  --set reset_xy_range=0.15 \
  --set reset_xy_curriculum_enabled=true \
  --set reset_xy_curriculum_start=0.13 \
  --set reset_xy_curriculum_end=0.15 \
  --set reset_xy_curriculum_fraction=0.90 \
  --set reset_velocity_xy_range=0.06 \
  --set reset_velocity_xy_curriculum_start=0.045 \
  --set reset_velocity_xy_curriculum_end=0.06 \
  --set reset_velocity_z_range='[-0.18,0.04]' \
  --set reset_velocity_z_curriculum_start='[-0.14,0.04]' \
  --set reset_velocity_z_curriculum_end='[-0.18,0.04]' \
  --set reset_ball_height_bounds='[0.18,0.56]' \
  --set learning_rate=5e-7 \
  --set n_epochs=1 \
  --set clip_range=0.01 \
  --set eval_episodes=100 \
  --set evaluation_step_limit=7200 \
  --set bootstrap_heuristic_episodes=0 \
  --set bootstrap_epochs=0 \
  --set bootstrap_followup_epochs=0
```

오래 걸리는 서버 학습은 `tmux`나 `nohup`으로 실행한다.

```sh
nohup env PYTHONPATH=src MUJOCO_GL=osmesa OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/run_ppo_learning.py \
  --config-file configs/keep1_v32_17d_transfer.json \
  --run-version v40_17d_server_polish \
  --resume-from artifacts/ppo_runs/keep1_v39_17d_mid_curriculum_fixed/keep1_v39_17d_mid_curriculum_fixed_model.zip \
  --set total_timesteps=700000 \
  --set reset_xy_range=0.15 \
  --set reset_velocity_xy_range=0.06 \
  --set reset_velocity_z_range='[-0.18,0.04]' \
  --set reset_ball_height_bounds='[0.18,0.56]' \
  > artifacts/ppo_runs/v40_17d_server_polish.log 2>&1 &
```

학습 후에는 같은 분포로 rebound analysis를 돌려서 웹에 올릴 모델을 고른다.

```sh
python scripts/run_ppo_rebound_analysis.py \
  --model-path artifacts/ppo_runs/keep1_v40_17d_server_polish/keep1_v40_17d_server_polish_model.zip \
  --episodes 100 \
  --seed 251 \
  --episode-step-limit 7200 \
  --reset-xy-range 0.15 \
  --reset-velocity-xy-range 0.06 \
  --reset-velocity-z-range -0.18 0.04 \
  --reset-ball-height-bounds 0.18 0.56 \
  --analysis-name keep1_v40_17d_server_polish_eval100
```

결과가 더 좋으면 모델 zip과 training summary를 웹 프로젝트의 `rl/artifacts/<run_name>/`에 배치하고 `.env`의 `PINGPONG_POLICY_MODEL_PATH`만 새 경로로 바꾼다.
