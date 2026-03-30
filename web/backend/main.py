"""
web/backend/main.py — FastAPI application entry point.

Run with:
    uvicorn web.backend.main:app --reload --host 0.0.0.0 --port 8000

API design:
  - All responses: {"data": ..., "error": null} envelope
  - Long operations return {"data": {"task_id": "<uuid>"}, "error": null}
  - Connect to ws://localhost:8000/ws/tasks/{task_id} for live progress
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from core.session_manager import SessionManager
from web.backend.routes import (
    audit, calibration, dataset, extraction, sessions,
    autolabel, depth, slam, reconstruction,
    active_learning, analytics, augmentation, training,
)
from web.backend.routes.inference import router as inference_router
from web.backend.routes.annotation_review import router as annotation_review_router, init_sm as ar_init_sm
from web.backend.routes.continuous_learning import router as cl_router, init_sm as cl_init_sm
# Phase 3 perception + infra
from web.backend.routes.segmentation import router as segmentation_router
from web.backend.routes.occupancy import router as occupancy_router
from web.backend.routes.lanes import router as lanes_router
from web.backend.routes.tracking import router as tracking_router
from web.backend.routes.versioning import router as versioning_router
from web.backend.routes.experiments import router as experiments_router
from web.backend.routes.edge_export import router as edge_export_router
from web.backend.routes.insta360 import router as insta360_router
from web.backend.websocket import task_progress_ws

# ---------------------------------------------------------------------------
# Sessions root (matches desktop default)
# ---------------------------------------------------------------------------

_SESSIONS_ROOT = Path.home() / ".roverdatakit" / "data" / "sessions"


# ---------------------------------------------------------------------------
# App factory with lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    _SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)
    sm = SessionManager(_SESSIONS_ROOT)
    app.state.session_manager = sm
    # Inject SM into Phase 3 routes
    ar_init_sm(sm)
    cl_init_sm(sm)
    yield
    # Shutdown — nothing to clean up


app = FastAPI(
    title="RoverDataKit API",
    version="3.0.0",
    description=(
        "Data pipeline API for autonomous rover dataset collection.\n\n"
        "GoPro Hero 11 · Insta360 X4 · Android Sensor Logger"
    ),
    lifespan=lifespan,
)

# CORS — allow the Vite dev server (port 5173) and any localhost origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(sessions.router)
app.include_router(audit.router)
app.include_router(extraction.router)
app.include_router(calibration.router)
app.include_router(dataset.router)
# Phase 2 routes
app.include_router(autolabel.router)
app.include_router(depth.router)
app.include_router(slam.router)
app.include_router(reconstruction.router)
app.include_router(active_learning.router)
app.include_router(analytics.router)
app.include_router(augmentation.router)
app.include_router(training.router)
# Phase 3 routes (inference / review / continuous learning)
app.include_router(inference_router)
app.include_router(annotation_review_router)
app.include_router(cl_router)
# Phase 3 perception + infra
app.include_router(segmentation_router)
app.include_router(occupancy_router)
app.include_router(lanes_router)
app.include_router(tracking_router)
app.include_router(versioning_router)
app.include_router(experiments_router)
app.include_router(edge_export_router)
# 360° Camera
app.include_router(insta360_router)


# ---------------------------------------------------------------------------
# WebSocket — task progress streaming
# ---------------------------------------------------------------------------

@app.websocket("/ws/tasks/{task_id}")
async def ws_task_progress(websocket: WebSocket, task_id: str) -> None:
    await task_progress_ws(websocket, task_id)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"data": {"status": "ok"}, "error": None}


@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "data": {
            "name": "RoverDataKit API",
            "version": "1.0.0",
            "docs": "/docs",
        },
        "error": None,
    }
