FROM node:24-bookworm AS frontend-build
WORKDIR /workspace/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app

ARG PINGPONG_POLICY_MODEL_PATH=rl/artifacts/keep_v39_17d/keep_v39_17d_model.zip
ARG PINGPONG_MUJOCO_SCENE_PATH=rl/assets/scene.xml

ENV PYTHONUNBUFFERED=1
ENV PINGPONG_PROJECT_ROOT=/app
ENV SERVER_PORT=8079
ENV PINGPONG_POLICY_MODEL_PATH=${PINGPONG_POLICY_MODEL_PATH}
ENV PINGPONG_MUJOCO_SCENE_PATH=${PINGPONG_MUJOCO_SCENE_PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      libegl1 \
      libgl1 \
      libglib2.0-0 \
      libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY backend ./backend
COPY rl ./rl
RUN set -e; \
    test -f "$PINGPONG_POLICY_MODEL_PATH" \
      || { echo "Missing policy model in image: $PINGPONG_POLICY_MODEL_PATH" >&2; exit 1; }; \
    test -f "$PINGPONG_MUJOCO_SCENE_PATH" \
      || { echo "Missing MuJoCo scene in image: $PINGPONG_MUJOCO_SCENE_PATH" >&2; exit 1; }; \
    test -d "backend/vendor/pingpong_rl2/src/pingpong_rl2" \
      || { echo "Missing vendored RL package in image: backend/vendor/pingpong_rl2/src/pingpong_rl2" >&2; exit 1; }; \
    test -d "rl/assets/franka" \
      || { echo "Missing Franka assets in image: rl/assets/franka" >&2; exit 1; }
COPY --from=frontend-build /workspace/frontend/dist ./frontend/dist

EXPOSE 8079
CMD ["sh", "-c", "python -m uvicorn backend.app.main:app --host 0.0.0.0 --port ${SERVER_PORT:-8079}"]
