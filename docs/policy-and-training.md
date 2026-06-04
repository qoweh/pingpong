# 제어 모델과 학습

현재 웹 서비스의 기본 모델은 v25 run을 사용한다. 모델 파일 경로는 코드에 흩어져 있지 않고 `.env`에서 한 번만 관리한다.

```text
PINGPONG_POLICY_MODEL_PATH=rl/artifacts/pmk_cf_self_rally_v25/pmk_cf_self_rally_v25_model.zip
```

## 모델 실행 방식

서버는 vendored `pingpong_rl2` 소스를 import하고, 선택된 모델 파일의 training summary에서 환경 설정을 복원한다. 그 다음 원본 Gym 환경을 만들고 매 제어 step마다 모델의 action을 계산한다.

브라우저는 action을 직접 계산하지 않는다. 서버에서 받은 MuJoCo 상태를 웹 뷰어에 반영하고, 화면 조작값을 서버에 명령으로 전달한다.

## v25 학습 설정

| 설정 | 값 |
| --- | --- |
| Algorithm | PPO |
| Run name | `pmk_cf_self_rally_v25` |
| Preset | `contact_frame_self_rally_v25_long_horizon_30_bounce` |
| 시작 checkpoint | v23 model |
| 추가 학습 step | 500,000 |
| 병렬 환경 수 | 4 |
| n_steps | 512 |
| Batch size | 512 |
| Learning rate | 2e-5 |
| Gamma | 0.99 |
| Epochs | 2 |
| Clip range | 0.08 |
| Seed | 7 |
| Action mode | `position_contact_frame_velocity_tilt_lateral_apex_residual` |

## v25 평가 요약

100 episode rebound analysis 기준 결과다.

| 지표 | 값 |
| --- | ---: |
| Mean return | 255.06 |
| Mean useful bounces | 28.51 |
| Max useful bounces | 51 |
| 1+ useful bounce rate | 0.98 |
| 10+ useful bounce rate | 0.84 |
| 20+ useful bounce rate | 0.72 |
| 30+ useful bounce rate | 0.61 |

현재 모델은 발표와 데모 후보로 쓰기 좋은 수준까지 도달했지만, 긴 episode 후반의 공 이탈과 낮은 apex 접촉은 아직 개선 여지가 있다.

## 모델 교체 절차

1. 새 모델 zip과 training summary를 `rl/artifacts/<run_name>/` 아래에 둔다.
2. `.env`의 `PINGPONG_POLICY_MODEL_PATH`를 새 모델 zip 경로로 바꾼다.
3. 모델이 다른 scene이나 asset을 요구하면 `rl/assets`와 compiled `frontend/public/assets/mujoco/pingpong_scene.mjb`도 함께 갱신한다.
4. 서버를 다시 시작해서 새 모델과 환경 설정을 로드한다.

모델 파일만 바꾸는 경우에는 프론트엔드 코드를 수정하지 않는다. 환경 구조, joint 수, geom/site 이름이 바뀌면 웹 뷰어의 identifier 매핑도 함께 확인해야 한다.
