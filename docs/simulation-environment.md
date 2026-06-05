# 시뮬레이션 환경

웹에서 사용하는 물리 장면은 원본 강화학습 프로젝트의 `assets/scene.xml`과 Franka Panda asset을 기준으로 한다. 배포 디렉토리에서는 같은 내용을 `rl/assets`로 가져와 사용한다.

## 장면 구성

| 항목 | 설명 |
| --- | --- |
| 로봇 | Franka Emika Panda 7자유도 로봇팔 |
| 라켓 | `hand` 아래에 부착된 `racket` body |
| 공 | free joint를 가진 구 형태의 `ball` body |
| 라켓 기준점 | `racket_center` site |
| 접촉 판정 | 공 geom과 라켓 head geom의 접촉 정보를 사용 |
| 바닥 | MuJoCo plane geom, checker material, 반사 값 포함 |

## 주요 물리 값

| 항목 | 값 |
| --- | ---: |
| 제어 주기 | 0.02 s |
| MuJoCo timestep | 0.002 s |
| 중력 | `0 0 -9.81` |
| 공 반지름 | 0.02 m |
| 공 질량 | 0.0027 kg |
| 라켓 head 반지름 | 0.084 m |
| 라켓 head half-depth | 0.006 m |
| 기본 공 시작 높이 | 0.34 m above racket |
| 목표 공 높이 | 0.30 m above racket |
| 높이 허용 범위 | 0.10 m |
| episode 최대 step | 웹 runtime은 고정 제한 없음, v39 분석 기준 7200 step |

## 초기화와 공 배치

초기화는 로봇, 공, 내부 episode 카운터를 함께 되돌린다. 공 시작 위치 조절은 로봇 자세를 다시 세우지 않고 공 free joint만 이동시키도록 분리되어 있다.

공 위치 조절값은 라켓 기준 상대값으로 해석한다.

| 조작값 | 의미 |
| --- | --- |
| X Position | 라켓 기준 X 방향 시작 위치 |
| Y Position | 라켓 기준 Y 방향 시작 위치 |
| Z Position | 라켓 위쪽 시작 높이 |
| X/Y/Z Velocity | 공의 초기 선속도 |

## 웹 렌더링 범위

브라우저는 MuJoCo 상태 배열인 `qpos`, `qvel`, `ctrl`을 받아 같은 WebAssembly 모델에 반영한다. 화면의 카메라, 바닥 타일, trail, contact marker는 웹 뷰어가 추가로 그리는 시각화 요소다.

MuJoCo native viewer의 그림자와 반사는 MuJoCo 렌더러가 직접 처리한다. 현재 웹 뷰어는 Three.js로 장면을 다시 구성하므로, 반사와 그림자는 Three.js 쪽 구현 품질에 영향을 받는다.
