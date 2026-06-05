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

| 항목 | 범위 |
| --- | --- |
| XY 시작 offset | 반경 0.13 m disk sampling |
| 공 시작 높이 | 0.22 m ~ 0.52 m above racket |
| XY 초기 속도 | -0.045 m/s ~ +0.045 m/s |
| Z 초기 속도 | -0.14 m/s ~ +0.04 m/s |

웹 조작 패널은 v39 검증 범위를 기준으로 조금 넓게 열어 둔다. X/Y position은 각각 -0.15 m ~ +0.15 m까지 조절할 수 있지만, 학습 도메인은 원형 disk sampling이므로 X와 Y를 동시에 끝까지 밀면 학습 때보다 더 어려운 상태가 된다.

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
