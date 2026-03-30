"""
web/backend/routes/occupancy.py — Occupancy grid endpoints.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from core.session_manager import SessionNotFoundError

router = APIRouter(prefix="/occupancy", tags=["occupancy"])


def _sm(request: Request):
    return request.app.state.session_manager


class OccupancyRequest(BaseModel):
    grid_resolution_m: float = 0.1
    grid_width_m: float = 20.0
    grid_height_m: float = 30.0
    camera_height_m: float = 1.0
    max_depth_m: float = 30.0
    temporal_fusion_window: int = 5
    decay_factor: float = 0.95


@router.post("/{session_id}/run", summary="Generate occupancy grid for session")
def start_occupancy(session_id: str, body: OccupancyRequest, request: Request) -> dict:
    try:
        _sm(request).get_session(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    task_id = str(uuid.uuid4())
    from web.backend.tasks import run_occupancy
    run_occupancy.apply_async(
        args=[session_id, body.model_dump(), task_id],
        task_id=task_id,
    )
    return {"data": {"task_id": task_id}, "error": None}


@router.get("/{session_id}/results", summary="Get occupancy grid results")
def get_occupancy_results(session_id: str, request: Request) -> dict:
    try:
        _sm(request).get_session(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    import json
    from pathlib import Path
    sm = _sm(request)
    output_dir = sm.session_folder(session_id) / "occupancy"
    summary_path = output_dir / "occupancy_summary.json"
    if not summary_path.exists():
        return {"data": {"status": "not_run"}, "error": None}

    summary = json.loads(summary_path.read_text())
    return {"data": {"status": "done", "summary": summary}, "error": None}
