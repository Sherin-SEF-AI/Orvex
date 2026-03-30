"""
web/backend/routes/extraction.py — Telemetry + frame extraction endpoints.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from core.models import ExtractionConfig
from core.session_manager import SessionNotFoundError
from web.backend.tasks import run_extraction

router = APIRouter(prefix="/extraction", tags=["extraction"])


def _sm(request: Request):
    return request.app.state.session_manager


class ExtractionRequest(BaseModel):
    frame_fps: float = 5.0
    frame_format: str = "jpg"
    frame_quality: int = 95
    output_format: str = "euroc"
    sync_devices: bool = True
    imu_interpolation: bool = True


@router.post("/{session_id}/run", summary="Start extraction for a session")
def start_extraction(session_id: str, body: ExtractionRequest, request: Request) -> dict:
    try:
        _sm(request).get_session(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    config = ExtractionConfig(session_id=session_id, **body.model_dump())
    task_id = str(uuid.uuid4())
    run_extraction.apply_async(
        args=[session_id, config.model_dump(), task_id],
        task_id=task_id,
    )
    return {"data": {"task_id": task_id}, "error": None}


@router.get("/{session_id}/status", summary="Get extraction status for a session")
def get_extraction_status(session_id: str, request: Request) -> dict:
    try:
        s = _sm(request).get_session(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return {"data": {"status": s.extraction_status}, "error": None}
