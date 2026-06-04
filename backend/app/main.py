from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .live_simulation import LiveCommandState, LiveSimulationService
from .settings import load_settings


settings = load_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.simulation = LiveSimulationService(settings)
    yield


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "pingpong-python-live",
        "model": str(settings.model_path),
        "scene": str(settings.scene_path),
    }


@app.get("/api/config")
async def config() -> dict[str, Any]:
    return app.state.simulation.config_payload()


@app.websocket("/api/live")
async def live(websocket: WebSocket) -> None:
    await websocket.accept()
    command_state = LiveCommandState()
    session = await asyncio.to_thread(app.state.simulation.create_session)
    receiver = asyncio.create_task(receive_commands(websocket, command_state))

    try:
        await websocket.send_json({"type": "ready", "config": app.state.simulation.config_payload()})
        await websocket.send_json(session.frame(reset=True))

        while True:
            if command_state.reset_requested:
                command_state.reset_requested = False
                reset_options = command_state.reset_options
                command_state.reset_options = None
                frame = await asyncio.to_thread(session.reset, reset_options)
            elif command_state.ball_spawn_requested:
                command_state.ball_spawn_requested = False
                spawn_options = command_state.ball_spawn_options or {}
                command_state.ball_spawn_options = None
                frame = await asyncio.to_thread(session.spawn_ball, spawn_options)
            elif command_state.playback == "playing":
                frame = await asyncio.to_thread(session.step)
            else:
                frame = await asyncio.to_thread(session.frame)

            await websocket.send_json(frame)
            await asyncio.sleep(session.control_dt if command_state.playback == "playing" else 0.1)
    except WebSocketDisconnect:
        pass
    finally:
        receiver.cancel()
        await asyncio.gather(receiver, return_exceptions=True)


async def receive_commands(websocket: WebSocket, state: LiveCommandState) -> None:
    try:
        while True:
            raw_message = await websocket.receive_text()
            try:
                message = json.loads(raw_message)
            except json.JSONDecodeError:
                continue

            message_type = message.get("type")
            if message_type == "reset":
                state.reset_requested = True
            elif message_type == "playback":
                playback = message.get("playback")
                if playback in {"playing", "paused"}:
                    state.playback = playback
            elif message_type == "spawnBall":
                state.ball_spawn_options = parse_ball_spawn_options(message)
                state.ball_spawn_requested = True
    except WebSocketDisconnect:
        return


def parse_ball_spawn_options(message: dict[str, Any]) -> dict[str, Any]:
    x_offset = clamp_float(message.get("xOffset"), -0.2, 0.2, 0.0)
    y_offset = clamp_float(message.get("yOffset"), -0.2, 0.2, 0.0)
    z_offset = clamp_float(message.get("zOffset"), 0.08, 0.9, 0.34)
    velocity_x = clamp_float(message.get("velocityX"), -1.0, 1.0, 0.0)
    velocity_y = clamp_float(message.get("velocityY"), -1.0, 1.0, 0.0)
    velocity_z = clamp_float(message.get("velocityZ"), -1.0, 1.0, 0.0)
    return {
        "ball_height": z_offset,
        "ball_xy_offset": [x_offset, y_offset],
        "ball_velocity": [velocity_x, velocity_y, velocity_z],
    }


def clamp_float(value: Any, low: float, high: float, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return min(max(parsed, low), high)


frontend_dist = settings.frontend_dist
if frontend_dist.exists():
    assets_dir = frontend_dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")


@app.get("/docs/{doc_path:path}")
async def docs_or_spa(doc_path: str):
    docs_dir = frontend_dist / "docs"
    requested = (docs_dir / doc_path).resolve()
    if doc_path and requested.is_file() and docs_dir in requested.parents:
        return FileResponse(requested)
    return spa_index_or_status()


@app.get("/{path:path}")
async def static_spa(path: str):
    return spa_index_or_status(path)


def spa_index_or_status(path: str = ""):
    if not frontend_dist.exists():
        return JSONResponse(
            {
                "status": "ok",
                "message": "Frontend dist is not built. Run npm run build in frontend or use Vite dev server.",
            }
        )

    requested = (frontend_dist / path).resolve()
    if requested.is_file() and frontend_dist in requested.parents:
        return FileResponse(requested)

    index = frontend_dist / "index.html"
    if index.exists():
        return FileResponse(index)

    return JSONResponse({"status": "error", "message": "index.html is missing"}, status_code=404)
