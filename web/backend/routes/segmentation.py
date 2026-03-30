"""
web/backend/routes/segmentation.py — Semantic segmentation endpoints.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from core.session_manager import SessionNotFoundError

router = APIRouter(prefix="/segmentation", tags=["segmentation"])


def _sm(request: Request):
    return request.app.state.session_manager


class SegmentationRequest(BaseModel):
    model_name: str = "nvidia/segformer-b2-finetuned-cityscapes-512-1024"
    batch_size: int = 4
    device: str = "auto"
    overlay_alpha: float = 0.5
    mark_invalid: bool = True


@router.post("/{session_id}/run", summary="Run semantic segmentation on session frames")
def start_segmentation(session_id: str, body: SegmentationRequest, request: Request) -> dict:
    try:
        _sm(request).get_session(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    task_id = str(uuid.uuid4())
    from web.backend.tasks import run_segmentation
    run_segmentation.apply_async(
        args=[session_id, body.model_dump(), task_id],
        task_id=task_id,
    )
    return {"data": {"task_id": task_id}, "error": None}


@router.get("/{session_id}/results", summary="Get segmentation results")
def get_segmentation_results(session_id: str, request: Request) -> dict:
    try:
        _sm(request).get_session(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    import json
    from pathlib import Path
    sm = _sm(request)
    output_dir = sm.session_folder(session_id) / "segmentation"
    summary_path = output_dir / "segmentation_summary.json"
    if not summary_path.exists():
        return {"data": {"status": "not_run"}, "error": None}

    summary = json.loads(summary_path.read_text())
    return {"data": {"status": "done", "summary": summary}, "error": None}


@router.get("/health", summary="Check segmentation dependencies")
def segmentation_health() -> dict:
    deps: dict[str, bool] = {}
    for pkg in ("transformers", "torch", "cv2", "numpy"):
        try:
            __import__(pkg)
            deps[pkg] = True
        except ImportError:
            deps[pkg] = False
    all_ok = all(deps.values())
    return {"data": deps, "error": None if all_ok else "Missing dependencies"}
