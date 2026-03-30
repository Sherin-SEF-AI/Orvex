"""
web/backend/routes/tracking.py — Multi-object tracking endpoints.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from core.session_manager import SessionNotFoundError

router = APIRouter(prefix="/tracking", tags=["tracking"])


def _sm(request: Request):
    return request.app.state.session_manager


class TrackingRequest(BaseModel):
    track_thresh: float = 0.5
    track_buffer: int = 30
    match_thresh: float = 0.8
    frame_rate: float = 30.0
    class_filter: list[str] = []    # empty = track all classes


@router.post("/{session_id}/run", summary="Run ByteTrack on session annotations")
def start_tracking(session_id: str, body: TrackingRequest, request: Request) -> dict:
    try:
        _sm(request).get_session(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    task_id = str(uuid.uuid4())
    from web.backend.tasks import run_tracking
    run_tracking.apply_async(
        args=[session_id, body.model_dump(), task_id],
        task_id=task_id,
    )
    return {"data": {"task_id": task_id}, "error": None}


@router.get("/{session_id}/results", summary="Get tracking results")
def get_tracking_results(session_id: str, request: Request) -> dict:
    try:
        _sm(request).get_session(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    import json
    from pathlib import Path
    sm = _sm(request)
    output_dir = sm.session_folder(session_id) / "tracking"
    summary_path = output_dir / "tracking_summary.json"
    if not summary_path.exists():
        return {"data": {"status": "not_run"}, "error": None}

    summary = json.loads(summary_path.read_text())
    return {"data": {"status": "done", "summary": summary}, "error": None}


@router.get("/{session_id}/mot-export", summary="Export tracks in MOT Challenge format")
def export_mot(session_id: str, request: Request) -> dict:
    try:
        _sm(request).get_session(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    from pathlib import Path
    sm = _sm(request)
    mot_path = sm.session_folder(session_id) / "tracking" / "tracks_mot.csv"
    if not mot_path.exists():
        raise HTTPException(status_code=404, detail="Tracking not run yet")

    from fastapi.responses import FileResponse
    return FileResponse(str(mot_path), filename=f"{session_id}_tracks_mot.csv")
