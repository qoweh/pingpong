FROM node:24-bookworm AS frontend-build
WORKDIR /workspace/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PINGPONG_PROJECT_ROOT=/app
ENV SERVER_PORT=8079

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
COPY --from=frontend-build /workspace/frontend/dist ./frontend/dist

EXPOSE 8079
CMD ["sh", "-c", "python -m uvicorn backend.app.main:app --host 0.0.0.0 --port ${SERVER_PORT:-8079}"]
