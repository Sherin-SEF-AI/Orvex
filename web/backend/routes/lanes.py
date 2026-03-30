"""
web/backend/routes/lanes.py — Lane detection endpoints.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from core.session_manager import SessionNotFoundError

router = APIRouter(prefix="/lanes", tags=["lanes"])


def _sm(request: Request):
    return request.app.state.session_manager


class LaneRequest(BaseModel):
    use_ufld: bool = False          # default False — classical is always available
    ufld_conf_threshold: float = 0.5
    classical_fallback: bool = True
    roi_top_percent: float = 0.55
    camera_height_m: float = 1.0
    camera_pitch_deg: float = 0.0
    model_path: str | None = None   # path to UFLD weights if use_ufld=True


@router.post("/{session_id}/run", summary="Run lane detection on session frames")
def start_lanes(session_id: str, body: LaneRequest, request: Request) -> dict:
    try:
        _sm(request).get_session(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    task_id = str(uuid.uuid4())
    from web.backend.tasks import run_lane_detection
    run_lane_detection.apply_async(
        args=[session_id, body.model_dump(), task_id],
        task_id=task_id,
    )
    return {"data": {"task_id": task_id}, "error": None}


@router.get("/{session_id}/results", summary="Get lane detection results")
def get_lane_results(session_id: str, request: Request) -> dict:
    try:
        _sm(request).get_session(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    import json
    from pathlib import Path
    sm = _sm(request)
    output_dir = sm.session_folder(session_id) / "lanes"
    summary_path = output_dir / "lane_summary.json"
    if not summary_path.exists():
        return {"data": {"status": "not_run"}, "error": None}

    summary = json.loads(summary_path.read_text())
    return {"data": {"status": "done", "summary": summary}, "error": None}


@router.get("/health", summary="Check lane detection dependencies")
def lanes_health() -> dict:
    deps: dict[str, bool] = {}
    for pkg in ("cv2", "numpy", "scipy"):
        try:
            __import__(pkg)
            deps[pkg] = True
        except ImportError:
            deps[pkg] = False
    return {"data": deps, "error": None}
