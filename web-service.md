# 웹 서비스 - 개요 수정본

이 문서는 로봇팔 탁구공 Keep-Up 웹 데모를 실제로 구현하기 위한 작업 지시서다. 기능 후보를 많이 늘리기보다, 현재 필요한 범위만 남겨서 정리한다.

---

# 1. 프로젝트 목표

로봇팔 끝에 탁구채를 붙이고, MuJoCo 환경에서 PPO로 학습한 policy가 탁구공을 계속 위로 띄우는 모습을 웹에서 보여준다.

사용자는 웹에서 시뮬레이션을 보면서 아래 정도만 직접 조정한다. 

```text
공 위치
카메라 시점
```

추가로 화면 이해를 돕기 위해 아래 시각화는 포함한다. (on/off 가능하게, 기본값 : off)

```text
공 궤적 trail
목표 높이 band
현재 공 높이 label
```

contact marker는 있으면 좋지만 MVP 필수는 아니다.

---

# 2. 전체 실행 구조

## 2.1 브라우저 역할

```text
MuJoCo WASM 실행
Three.js 렌더링
학습된 PPO policy 실행
사용자 조작 UI 실행
공 위치 조정
카메라 시점 전환
공 궤적 / 목표 높이 / 현재 높이 시각화
4-camera view 표시 (하나의 viewer에서 on/off 해서 4개로 나눠지는 것으로 추가하면 좋을듯, 기본값 : off)
```

## 2.2 홈서버 역할

홈서버는 Spring Boot 애플리케이션을 실행하고, 브라우저가 필요한 정적 파일을 제공한다.

```text
React build 결과 제공
MuJoCo WASM 파일 제공
MuJoCo 모델 asset 제공
최종 PPO policy 실행 파일 제공
Docs page 제공
GitHub 링크 제공
```

## 2.3 MuJoCo 모델 asset 제공이 필요한 이유

GitHub에 모델 파일을 올려두는 것과, 웹 데모가 브라우저에서 실행되는 것은 별개다.

브라우저에서 MuJoCo WASM이 실제로 모델을 로드하려면 XML, mesh, texture 등 실행에 필요한 파일이 웹 앱의 정적 asset으로 접근 가능해야 한다.

따라서 아래 파일들은 GitHub 참고용 링크와 별개로 배포물에 포함한다.

```text
MJCF/XML 파일
mesh 파일
texture 파일
MuJoCo WASM 파일
최종 PPO policy 실행 파일
```

단, 이것들을 별도 API로 제공할 필요는 없다. Spring Boot가 정적 파일로 제공하면 된다.

---

# 3. 개발/배포 환경

```text
개발 머신: MacBook
학습 환경 격리: conda
배포 서버: Ubuntu가 설치된 ASUS 데스크톱
서버 CPU: Intel Core i5-8400
서버 architecture: x86_64 / amd64
서버 core: 6 cores / 6 threads
GPU: 없음으로 가정
외부 접근 포트 계획: 8079
Reverse proxy: 기존 Nginx Proxy Manager 사용
```

MacBook이 Apple Silicon이므로, 서버 배포 시 amd64 아키텍처 차이를 주의한다.

---

# 4. 기술 스택

## 4.1 Frontend

```text
React + Vite
TypeScript
Three.js
@mujoco/mujoco
```

Next.js도 가능하지만, Spring Boot에서 정적 파일로 배포하기 쉽게 하려면 React + Vite가 단순하다.

## 4.2 Simulation

```text
MuJoCo WASM
MJCF/XML model
single-threaded @mujoco/mujoco 우선 사용
```

MVP에서는 multi-threaded WASM을 고려하지 않는다.

## 4.3 Policy 실행

PPO로 학습된 최종 policy만 웹에서 실행한다.

권장 방식:

```text
Python PPO actor export
→ ONNX 또는 JSON weight 형태로 변환
→ 브라우저에서 TypeScript로 observation 생성
→ policy 실행
→ action을 MuJoCo data.ctrl에 적용
```

구현 우선순위:

```text
1순위: ONNX Runtime Web
2순위: JSON weight + TypeScript MLP runner
```

기존 PPO 모델을 ONNX로 안정적으로 변환할 수 있다면 ONNX Runtime Web을 우선 사용한다. 변환이 번거롭고 policy가 작은 MLP라면 JSON weight와 TypeScript MLP runner를 사용한다.

## 4.4 Backend / Serving

```text
Spring Boot
Spring Boot 내장 Tomcat
Docker Compose
기존 Nginx Proxy Manager
```

별도 nginx 컨테이너는 추가하지 않는다.

---

# 5. 페이지 구성

## 5.1 Header

```text
Project title
Demo
Docs
GitHub
```

## 5.2 Hero

필요 요소:

```text
프로젝트 제목
짧은 설명
Try Demo 버튼
Read Docs 버튼
GitHub 버튼
```

제목 예시:

```text
Ping-Pong Keep-Up with Reinforcement Learning
```

부제 예시:

```text
A browser-based MuJoCo WebAssembly demo of a PPO policy keeping a ping-pong ball in the air.
```

## 5.3 Live Demo

필수 UI:

```text
MuJoCo WASM canvas
Play
Pause
Reset
공 위치 조정 패널
카메라 선택 패널
현재 공 높이 표시
Contact count 표시
```

Step 버튼은 디버깅용으로만 필요하면 추가한다.

## 5.4 User Controls

MVP에서 사용자가 직접 만지는 항목은 최소화한다.

```text
Ball position x/y/z
Camera mode
Visualization toggle
```

Visualization toggle:

```text
Trajectory trail on/off
Target height band on/off
Current height label on/off
Contact marker on/off
```

---

# 6. 카메라 기능

## 6.1 카메라 모드

```text
Free View
North View
South View
East View
West View
Top View
4-Camera View
```

Follow Ball, Follow Paddle은 v1 이후에 추가한다.

## 6.2 4-Camera View 배치

```text
North View | South View
East View  | West View
```

기본은 Free View 또는 Single View로 시작한다. 사용자가 4-Camera View를 켜면 화면을 4분할한다.

---

# 7. 웹 시각화 기능

시각화는 MuJoCo 물리에 영향을 주지 않는 Three.js overlay로 구현한다.

## 7.1 Ball trajectory trail

목적:

```text
공이 이동한 궤적 표시
공이 얼마나 안정적으로 유지되는지 확인
```

구현:

```text
매 frame 또는 일정 step마다 공 위치 저장
최근 100~300개 point만 유지
Three.js line으로 표시
```

성능 조건:

```text
전체 history 무한 저장 금지
최근 1~3초 정도만 표시
4-camera view에서도 렌더링 부담 관리
```

## 7.2 Target height band

목적:

```text
공이 유지되면 좋은 높이 범위 표시
현재 공 높이가 목표 범위 안에 있는지 확인
```

구현:

```text
targetHeight 기준 반투명 plane 표시
targetHeight ± heightTolerance 범위를 반투명 box로 표시
```

예시:

```text
targetHeight = 0.85m
heightTolerance = 0.15m
표시 범위 = 0.70m ~ 1.00m
```

## 7.3 Current ball height label

목적:

```text
현재 공 높이 표시
target height와의 차이 표시
```

표시 예시:

```text
Ball height: 0.87m
Target: 0.85m
Error: +0.02m
```

## 7.4 Contact marker

MVP 필수는 아니지만 구현 난이도가 크지 않으면 추가한다.

목적:

```text
공과 탁구채가 접촉한 위치 표시
접촉 횟수 확인
```

구현:

```text
MuJoCo contact 정보 확인
공 geom과 paddle geom 접촉 여부 확인
접촉 위치에 marker 생성
0.2~0.5초 뒤 fade out
```

UI 표시:

```text
Contact count
Last contact time
```

---

# 8. Docs 페이지 구성

문서는 길게 늘리지 말고, 프로젝트를 이해하는 데 필요한 항목만 정리한다.

```text
docs/
  overview.md
  simulation-environment.md
  mdp-formulation.md
  reward-function.md
  policy-and-training.md
  web-deployment.md
  results.md
```

## 8.1 overview.md

역할:

```text
프로젝트 전체 소개
웹 데모에서 볼 수 있는 것 설명
시스템 전체 구조 설명
```

포함 내용:

```text
문제 정의
로봇팔 + 탁구채 + 탁구공 환경 설명
PPO를 사용한 이유
브라우저 데모 기능 요약
시스템 구조 그림
GitHub 링크 (https://github.com/qoweh/ros2-study/tree/main/graduation-prj/pingpong_rl2)
```

## 8.2 simulation-environment.md

역할:

```text
MuJoCo 물리 환경 설명
```

포함 내용:

```text
Robot arm DOF
Paddle geometry
Paddle mass
Ping-pong ball radius
Ping-pong ball mass
Ball restitution
MuJoCo model structure
Control frequency
Episode length
Termination conditions
Camera configuration
```

## 8.3 mdp-formulation.md

역할:

```text
강화학습 문제 정의 설명
```

포함 내용:

```text
Observation vector
Action space
Episode initialization
Episode termination
```

Observation 표 예시:

| Component | Dimension | Meaning |
| --- | ---: | --- |
| Joint positions | 구현값 기준 | 로봇팔 관절 각도 |
| Joint velocities | 구현값 기준 | 로봇팔 관절 속도 |
| Paddle position | 3 | 탁구채 위치 |
| Paddle orientation | 4 | 탁구채 quaternion |
| Ball position | 3 | 공의 x/y/z 위치 |
| Ball velocity | 3 | 공의 x/y/z 속도 |
| Previous action | 구현값 기준 | 직전 action |
| Total | 구현값 기준 | policy input |

## 8.4 reward-function.md

역할:

```text
로봇이 어떤 행동을 하도록 보상을 설계했는지 설명
```

Reward term 후보:

| Term | Purpose |
| --- | --- |
| Target height reward | 공을 목표 높이 근처에 유지 |
| Contact reward | 탁구채와 공의 적절한 접촉 유도 |
| Upward velocity reward | 공을 위로 띄우는 행동 유도 |
| Ball-paddle distance reward | 탁구채가 공을 따라가도록 유도 |
| Energy penalty | 과도한 토크 사용 억제 |
| Smoothness penalty | action 변화량 억제 |
| Joint limit penalty | 관절 한계 근처 움직임 억제 |

실제 사용하지 않은 reward term은 문서에서 제거한다.

## 8.5 policy-and-training.md

역할:

```text
PPO로 학습한 최종 policy의 핵심 설정 설명
```

포함 내용:

```text
Algorithm: PPO
Policy network 구조
Activation function
Value network 구조
Training steps
Parallel environments
주요 hyperparameters
Observation normalization 여부
Evaluation protocol
최종 policy export 방식
```

주의:

```text
학습 과정 전체를 장황하게 쓰지 않는다.
중간 모델 비교 내용은 넣지 않는다.
실제로 결과에 영향을 준 설정만 정리한다.
```

## 8.6 web-deployment.md

역할:

```text
PPO 학습 결과를 웹에서 실행하는 구조 설명
```

포함 내용:

```text
Conda 기반 Python 학습 환경
최종 policy export
Browser policy execution
MuJoCo WASM simulation
Three.js rendering
Spring Boot static serving
Nginx Proxy Manager routing
MacBook 개발 환경과 Ubuntu amd64 배포 차이
WASM serving 주의사항
```

흐름:

```text
Python conda env + MuJoCo + PPO
→ 최종 policy export
→ web-loadable format 변환
→ browser에서 policy 실행
→ MuJoCo WASM simulation
→ Three.js rendering
→ Spring Boot serving
→ Nginx Proxy Manager
```

## 8.7 results.md

역할:

```text
학습 결과와 데모 성능 요약
```

포함 내용:

```text
Average keep-up duration
Success rate
Contacts per episode
Training reward curve
Height error over time
Failure cases
```

그래프 후보:

```text
episode reward over time
episode length / keep-up duration
contacts per episode
success rate
height error
```

---

# 9. 추천 디렉터리 구조

```text
project-root/
  README.md
  docker-compose.yml
  .env.example
  .gitignore

  frontend/
    package.json
    vite.config.ts
    tsconfig.json
    src/
      app/
      components/
      simulation/
      policy/
      visualization/
      controls/
      docs/
    public/
      assets/
        mujoco/
          scene.xml
          meshes/
          textures/
        policy/
          final-policy.onnx
          final-policy.json
        images/

  backend/
    Dockerfile
    build.gradle
    src/main/java/...
    src/main/resources/
      application.yml
      static/

  docs/
    overview.md
    simulation-environment.md
    mdp-formulation.md
    reward-function.md
    policy-and-training.md
    web-deployment.md
    results.md

  training-artifacts/
    final-policy/
    reward-curves/
    evaluation-results/

  deploy/
    server-checklist.md
    nginx-proxy-manager-notes.md
```

---

# 10. 디렉터리별 역할

## 10.1 project-root/

역할:

```text
프로젝트 전체 관리
Docker Compose 실행
README 관리
환경 변수 예시 관리
```

파일:

```text
README.md
docker-compose.yml
.env.example
.gitignore
```

## 10.2 frontend/

역할:

```text
웹 화면 구현
MuJoCo WASM 로드
Three.js 렌더링
PPO policy 실행
공 위치 조정 UI
카메라 UI
시각화 overlay 구현
Docs page 렌더링
```

## 10.3 frontend/src/simulation/

역할:

```text
MuJoCo WASM과 직접 연결되는 코드
```

파일 예시:

```text
mujocoLoader.ts
simulationLoop.ts
episodeReset.ts
ballState.ts
contactReader.ts
forceApplier.ts
mujocoTypes.ts
```

주요 기능:

```text
MuJoCo 로드
MJCF/XML 로드
mj_step 실행
mj_forward 호출
qpos/qvel 접근
공 위치 변경
contact 정보 읽기
```

## 10.4 frontend/src/policy/

역할:

```text
최종 PPO policy 실행 코드
```

파일 예시:

```text
policyLoader.ts
onnxRunner.ts
mlpRunner.ts
observationBuilder.ts
actionMapper.ts
normalization.ts
```

주요 기능:

```text
최종 policy 파일 로드
observation vector 생성
normalization 적용
action 계산
action clipping
data.ctrl에 action 적용
```

## 10.5 frontend/src/visualization/

역할:

```text
Three.js overlay 시각화 코드
```

파일 예시:

```text
BallTrail.ts
TargetHeightBand.ts
BallHeightLabel.ts
ContactMarkerManager.ts
CameraRig.ts
```

주요 기능:

```text
공 궤적 표시
목표 높이 band 표시
현재 공 높이 표시
접촉 marker 표시
4-camera view 지원
```

## 10.6 frontend/src/controls/

역할:

```text
사용자 조작 UI
```

파일 예시:

```text
DemoControls.tsx
CameraControls.tsx
BallControls.tsx
VisualizationToggles.tsx
PlaybackControls.tsx
```

주요 기능:

```text
Play / Pause / Reset
공 위치 조정
카메라 선택
시각화 on/off
```

## 10.7 frontend/public/assets/mujoco/

역할:

```text
브라우저가 로드할 MuJoCo 모델 파일 보관
```

파일 예시:

```text
scene.xml
meshes/*.stl
meshes/*.obj
textures/*
```

## 10.8 frontend/public/assets/policy/

역할:

```text
브라우저에서 실행할 최종 PPO policy 파일 보관
```

파일 예시:

```text
final-policy.onnx
final-policy.json
```

둘 다 둘 필요는 없다. 실제 구현에서 선택한 방식 하나만 둔다.

## 10.9 frontend/public/assets/images/

역할:

```text
문서와 메인 페이지에 사용할 이미지 보관
```

파일 예시:

```text
architecture-diagram.png
rl-loop-diagram.png
observation-vector.png
reward-breakdown.png
training-curve.png
failure-case-1.png
```

## 10.10 backend/

역할:

```text
Spring Boot 애플리케이션
정적 파일 제공
Docs page 제공
간단한 health check 제공
```

MVP에서 복잡한 API는 만들지 않는다.

## 10.11 backend/src/main/resources/static/

역할:

```text
frontend build 결과가 들어가는 위치
Spring Boot가 정적 파일로 제공하는 위치
```

배포 시:

```text
frontend/dist/*
→ backend/src/main/resources/static/*
```

또는 Docker build 과정에서 복사한다.

## 10.12 docs/

역할:

```text
Docs 페이지 원본 markdown 관리
GitHub에서도 읽을 수 있는 프로젝트 설명 문서 관리
```

## 10.13 training-artifacts/

역할:

```text
학습 결과 요약 자료 보관
웹 문서에 넣을 그래프 원본 보관
최종 policy export 파일 보관
```

파일 예시:

```text
final-policy/
reward-curves/episode_reward.csv
evaluation-results/summary.json
```

## 10.14 deploy/

역할:

```text
서버 배포 메모
Nginx Proxy Manager 설정 메모
Spring Boot 8079 포트 설정 메모
```

파일 예시:

```text
server-checklist.md
nginx-proxy-manager-notes.md
```

---

# 11. Spring Boot + Nginx Proxy Manager 배포 구조

별도 nginx 컨테이너는 사용하지 않는다.

권장 구조:

```text
Browser
→ Nginx Proxy Manager
→ Spring Boot embedded Tomcat
→ static assets + web app
```

## 11.1 Spring Boot 포트

Spring Boot는 8079 포트로 실행한다.

`application.yml` 예시:

```yaml
server:
  port: 8079
```

## 11.2 Nginx Proxy Manager 설정

기본 설정 방향:

```text
Forward Hostname / IP: Spring Boot가 실행 중인 호스트 또는 컨테이너
Forward Port: 8079
Scheme: http
SSL: Nginx Proxy Manager에서 처리
```

주의사항:

```text
Nginx Proxy Manager가 HTTPS를 담당하면 Spring Boot는 HTTP로 두는 것이 단순하다.
Spring Boot와 Nginx Proxy Manager가 같은 host port 8079를 동시에 사용할 수 없다.
외부에서 https://192.168.219.46:8079 로 접근하게 만들려면, 8079 포트를 누가 listen할지 먼저 정해야 한다.
```

권장안:

```text
Nginx Proxy Manager가 외부 HTTPS를 담당
Spring Boot는 내부 또는 host의 HTTP 8079에서 실행
Nginx Proxy Manager가 Spring Boot 8079로 reverse proxy
```

만약 Nginx Proxy Manager가 외부 8079 포트를 직접 listen한다면, Spring Boot는 host의 다른 포트나 Docker 내부 네트워크 포트로 연결되게 구성해야 한다.

## 11.3 Spring Boot로 정적 파일 serving 시 유의사항

```text
.wasm Content-Type이 application/wasm인지 확인
React Router 사용 시 새로고침 fallback 처리
큰 정적 파일 cache header 설정 검토
브라우저 Network 탭에서 WASM / XML / mesh / policy 파일 200 응답 확인
CORS 문제 없도록 같은 origin에서 asset 제공
```

## 11.4 gzip / br 압축

MVP 필수는 아니다.

```text
gzip은 Spring Boot 또는 프록시 레이어에서 적용 가능
br Brotli 압축은 Spring Boot 단독보다 Nginx Proxy Manager 또는 별도 프록시 레이어에서 처리하는 편이 단순
WASM 파일은 압축 여부와 별개로 Content-Type 확인이 더 중요
```

---

# 12. Docker Compose 구성

Nginx Proxy Manager는 이미 실행 중이므로 compose에는 Spring Boot 앱만 둔다.

예시:

```yaml
services:
  app:
    build:
      context: .
      dockerfile: backend/Dockerfile
    container_name: pingpong-rl-web
    ports:
      - "8079:8079"
    restart: unless-stopped
```

단, Nginx Proxy Manager가 host의 8079 포트를 직접 listen하도록 구성할 경우 위 port mapping은 충돌할 수 있다. 그 경우 Spring Boot의 host port를 바꾸거나, Nginx Proxy Manager와 같은 Docker network에 붙여 container name으로 프록시한다.

---

# 13. MacBook → Ubuntu amd64 배포 주의사항

```text
Mac의 node_modules를 서버에 복사하지 않는다.
Mac의 conda 환경 폴더를 서버에 복사하지 않는다.
서버에서 npm install 또는 pnpm install을 다시 실행한다.
Docker를 사용하면 linux/amd64로 빌드한다.
Apple Silicon Mac에서 빌드한 이미지를 그대로 Ubuntu amd64 서버에 올리지 않는다.
```

Docker build 예시:

```bash
docker buildx build --platform linux/amd64 -t pingpong-rl-web .
```

학습 환경 재현이 필요하면 conda 환경 자체를 복사하지 말고 `environment.yml`로 재생성한다.

```bash
conda env export > environment.yml
conda env create -f environment.yml
```

---

# 14. 구현 Task Breakdown

## 14.1 Frontend

```text
React + Vite 프로젝트 세팅
라우팅 구성
메인 페이지 Hero 구현
MuJoCo WASM 로더 구현
MJCF/XML asset 로딩 구현
Three.js scene 구성
로봇팔 / 탁구채 / 탁구공 렌더링
시뮬레이션 step loop 구현
최종 PPO policy 실행 loop 구현
Play / Pause / Reset 구현
공 위치 조정 UI 구현
카메라 모드 구현
4-camera layout 구현
trajectory trail 구현
target height band 구현
current ball height label 구현
contact marker 구현 여부 판단 후 구현
Docs page 렌더링 구현
GitHub 링크 연결
```

## 14.2 Simulation

```text
ball body / geom id 추적
paddle geom id 추적
qpos/qvel 접근 함수 구현
ball position setter 구현
mj_forward 호출 처리
mj_step loop 처리
contact 정보 추출
초기 상태 reset 로직 구현
```

## 14.3 Policy

```text
최종 PPO policy export 방식 결정
ONNX 또는 JSON 중 실제 사용할 형식 결정
browser loader 구현
observation builder 구현
normalization 처리
action output mapping 구현
action clipping 처리
data.ctrl 적용 처리
```

## 14.4 Visualization

```text
BallTrail 구현
TargetHeightBand 구현
BallHeightLabel 구현
ContactMarkerManager 구현
CameraRig 구현
4-Camera renderer 구현
```

## 14.5 Docs

```text
overview.md 작성
simulation-environment.md 작성
mdp-formulation.md 작성
reward-function.md 작성
policy-and-training.md 작성
web-deployment.md 작성
results.md 작성
필요한 시각화 이미지 제작
필요한 결과 그래프 제작
```

## 14.6 Backend / Deploy

```text
Spring Boot 프로젝트 구성
server.port=8079 설정
React build 결과를 Spring Boot static으로 포함
Dockerfile 작성
docker-compose.yml 작성
.wasm MIME type 확인
static asset path 확인
SPA fallback 처리
health check 추가
Ubuntu amd64 서버에서 compose 실행 확인
Nginx Proxy Manager reverse proxy 설정 확인
```

---

# 15. Acceptance Criteria

## 15.1 Demo

```text
브라우저에서 MuJoCo WASM이 정상 로드된다.
로봇팔, 탁구채, 탁구공이 렌더링된다.
최종 PPO policy가 실행된다.
Play / Pause / Reset이 동작한다.
사용자가 공 위치를 조정할 수 있다.
공 궤적 trail이 표시된다.
Target height band가 표시된다.
공 높이 label이 표시된다.
4-camera view가 동작한다.
카메라 전환이 동작한다.
```

선택 acceptance:

```text
공-탁구채 contact marker가 표시된다.
Contact count가 표시된다.
```

## 15.2 Docs

```text
Overview가 작성되어 있다.
Simulation Environment가 작성되어 있다.
MDP formulation이 표로 정리되어 있다.
Observation dimension이 설명되어 있다.
Action space가 설명되어 있다.
Reward term이 항별로 설명되어 있다.
PPO 학습 설정이 정리되어 있다.
Results 지표가 포함되어 있다.
Browser Deployment 구조가 설명되어 있다.
GitHub 링크가 포함되어 있다.
```

## 15.3 Deployment

```text
docker compose up -d 로 Spring Boot 앱이 실행된다.
Spring Boot가 8079 포트로 실행된다.
Ubuntu amd64 서버에서 실행된다.
Nginx Proxy Manager에서 Spring Boot 8079로 reverse proxy가 설정된다.
.wasm 파일이 정상 서빙된다.
Content-Type이 application/wasm으로 설정된다.
MJCF/XML asset이 정상 로드된다.
mesh / texture asset이 정상 로드된다.
최종 policy 실행 파일이 정상 로드된다.
브라우저 콘솔에 critical error가 없다.
React route 새로고침 시 404가 발생하지 않는다.
```

---

# 16. 에이전트 작업 프롬프트

```text
너는 senior frontend engineer + simulation engineer + reinforcement learning technical writer 역할을 맡는다.

목표:
로봇팔 끝에 탁구채를 붙이고 MuJoCo 환경에서 PPO로 학습된 policy가 탁구공을 계속 위로 띄우는 웹 데모를 설계하고 구현 계획을 작성한다.

개발/배포 환경:
- 개발 머신은 MacBook이다.
- PPO 학습 환경은 Python conda 환경으로 격리되어 있다.
- 배포 서버는 Ubuntu가 설치된 ASUS 데스크톱이다.
- 서버 CPU는 Intel Core i5-8400이다.
- 서버 architecture는 x86_64 / amd64다.
- 서버는 6 cores / 6 threads다.
- GPU는 없는 것으로 가정한다.
- 기존 Nginx Proxy Manager를 reverse proxy로 사용한다.
- 별도 nginx 컨테이너는 추가하지 않는다.
- Spring Boot 앱은 8079 포트로 실행한다.

기술 스택:
- Frontend: React + Vite, TypeScript
- Rendering: Three.js
- Simulation: single-threaded @mujoco/mujoco WASM
- Policy execution: 최종 PPO policy를 ONNX Runtime Web 또는 JSON weight + TypeScript MLP runner로 실행
- Backend: Spring Boot embedded Tomcat
- Deployment: Docker Compose

Main page:
- Header
- Hero
- Live Demo
- Play / Pause / Reset
- Ball position control
- Camera selector
- 4-Camera View
- Current ball height
- Trajectory trail
- Target height band
- GitHub link
- Docs link

User controls:
- Ball position x/y/z
- Camera mode
- Visualization on/off

Camera:
- Free View
- North View
- South View
- East View
- West View
- Top View
- 4-Camera View

Visualization:
- Ball trajectory trail
- Target height band
- Current ball height label
- Contact marker는 구현 난이도 확인 후 추가

Docs:
- overview.md
- simulation-environment.md
- mdp-formulation.md
- reward-function.md
- policy-and-training.md
- web-deployment.md
- results.md

Deployment:
- docker-compose.yml 작성
- backend Dockerfile 작성
- Spring Boot server.port=8079 설정
- React build 결과를 Spring Boot static으로 포함
- .wasm 파일을 application/wasm으로 서빙되는지 확인
- MJCF/XML asset serving 확인
- mesh/texture asset serving 확인
- 최종 policy 실행 파일 serving 확인
- Nginx Proxy Manager가 Spring Boot 8079로 reverse proxy하도록 설정
- 외부 8079 포트를 Nginx Proxy Manager와 Spring Boot가 동시에 bind하지 않도록 주의
- Ubuntu amd64 서버에서 실행 가능하게 구성
- MacBook이 Apple Silicon일 수 있으므로 linux/amd64 빌드 전략 포함

산출물:
1. 전체 정보구조 IA
2. 페이지별 wireframe 설명
3. 주요 UI 컴포넌트 목록
4. 브라우저 중심 실행 아키텍처
5. Spring Boot + Nginx Proxy Manager 배포 아키텍처
6. i5-8400 Ubuntu amd64 서버 기준 배포 전략
7. 웹 시각화 구현 전략
8. MacBook 개발 환경과 Ubuntu 서버 배포 시 주의사항
9. Spring Boot WASM 서빙 체크리스트
10. Docs 페이지 목차와 각 섹션 초안
11. 구현 task breakdown
12. acceptance criteria

최종 답변은 한국어로 작성하라.
```
