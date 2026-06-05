from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .ball_spawn import BallSpawnConfig, parse_ball_spawn_options
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


@app.middleware("http")
async def cache_static_assets(request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path.endswith("/asset-manifest.json"):
        response.headers["Cache-Control"] = "no-cache"
    elif path.startswith(("/assets/", "/runtime-mujoco-assets/")):
        response.headers["Cache-Control"] = "public, max-age=604800"
    return response


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "pingpong-simulation",
        "model": str(settings.model_path),
        "scene": str(settings.scene_path),
    }


@app.get("/api/config")
async def config() -> dict[str, Any]:
    return app.state.simulation.config_payload()


@app.get("/api/models")
async def models() -> dict[str, Any]:
    return app.state.simulation.models_payload()


@app.post("/api/models/select")
async def select_model(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    model_id = payload.get("modelId") or payload.get("id") or payload.get("model")
    if not isinstance(model_id, str) or not model_id:
        raise HTTPException(status_code=400, detail="modelId is required.")

    try:
        return await asyncio.to_thread(app.state.simulation.select_model, model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}") from exc


@app.websocket("/api/live")
async def live(websocket: WebSocket) -> None:
    await websocket.accept()
    simulation = app.state.simulation
    command_state = LiveCommandState()
    session = await asyncio.to_thread(simulation.create_session)
    receiver = asyncio.create_task(receive_commands(websocket, command_state, session.ball_spawn_config))

    try:
        await websocket.send_json({"type": "ready", "config": simulation.config_payload(session.runtime)})
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


async def receive_commands(websocket: WebSocket, state: LiveCommandState, ball_spawn_config: BallSpawnConfig) -> None:
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
                state.ball_spawn_options = parse_ball_spawn_options(message, ball_spawn_config)
                state.ball_spawn_requested = True
    except WebSocketDisconnect:
        return


frontend_dist = settings.frontend_dist
runtime_mujoco_assets = settings.project_root / "rl" / "assets"
if runtime_mujoco_assets.exists():
    app.mount("/runtime-mujoco-assets", StaticFiles(directory=runtime_mujoco_assets), name="runtime-mujoco-assets")

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
                "message": "Web app is not built yet. Build the frontend before serving this page.",
            }
        )

    requested = (frontend_dist / path).resolve()
    if requested.is_file() and frontend_dist in requested.parents:
        return FileResponse(requested)

    index = frontend_dist / "index.html"
    if index.exists():
        return FileResponse(index)

    return JSONResponse({"status": "error", "message": "Web app entry file is missing."}, status_code=404)
