"""
web/backend/routes/autolabel.py — YOLOv8 auto-labeling endpoints.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from core.session_manager import SessionNotFoundError
from web.backend.tasks import run_autolabel

router = APIRouter(prefix="/autolabel", tags=["autolabel"])


def _sm(request: Request):
    return request.app.state.session_manager


class AutoLabelRequest(BaseModel):
    model_path: str = "yolov8n.pt"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    batch_size: int = 16
    device: str = "auto"
    export_format: str = "both"   # "cvat", "yolo", "both"


@router.post("/{session_id}/run", summary="Run YOLOv8 inference on session frames")
def start_autolabel(session_id: str, body: AutoLabelRequest, request: Request) -> dict:
    try:
        _sm(request).get_session(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    task_id = str(uuid.uuid4())
    run_autolabel.apply_async(
        args=[session_id, body.model_dump(), task_id],
        task_id=task_id,
    )
    return {"data": {"task_id": task_id}, "error": None}


@router.get("/{session_id}/results", summary="Get auto-label annotation results")
def get_autolabel_results(session_id: str, request: Request) -> dict:
    try:
        _sm(request).get_session(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    import json
    from pathlib import Path
    sm = _sm(request)
    output_dir = sm.session_folder(session_id) / "autolabel"
    stats_path = output_dir / "annotation_stats.json"
    if not stats_path.exists():
        return {"data": {"status": "not_run", "annotations": []}, "error": None}

    stats = json.loads(stats_path.read_text())
    return {"data": {"status": "done", "stats": stats}, "error": None}


@router.get("/health", summary="Check auto-label dependencies")
def autolabel_health() -> dict:
    try:
        import ultralytics  # noqa: F401
        return {"data": {"ultralytics": True}, "error": None}
    except ImportError:
        return {"data": {"ultralytics": False}, "error": "ultralytics not installed"}
