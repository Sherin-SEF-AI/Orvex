"""
web/backend/routes/depth.py — Depth estimation endpoints.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from core.session_manager import SessionNotFoundError
from web.backend.tasks import run_depth_estimation

router = APIRouter(prefix="/depth", tags=["depth"])


def _sm(request: Request):
    return request.app.state.session_manager


class DepthRequest(BaseModel):
    model_variant: str = "small"   # "small", "base", "large"
    batch_size: int = 8
    device: str = "auto"
    colorize: bool = True


@router.post("/{session_id}/run", summary="Run depth estimation on session frames")
def start_depth(session_id: str, body: DepthRequest, request: Request) -> dict:
    try:
        _sm(request).get_session(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    task_id = str(uuid.uuid4())
    run_depth_estimation.apply_async(
        args=[session_id, body.model_dump(), task_id],
        task_id=task_id,
    )
    return {"data": {"task_id": task_id}, "error": None}


@router.get("/{session_id}/results", summary="Get depth estimation results")
def get_depth_results(session_id: str, request: Request) -> dict:
    try:
        _sm(request).get_session(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    import json
    from pathlib import Path
    sm = _sm(request)
    output_dir = sm.session_folder(session_id) / "depth"
    summary_path = output_dir / "depth_summary.json"
    if not summary_path.exists():
        return {"data": {"status": "not_run"}, "error": None}

    summary = json.loads(summary_path.read_text())
    return {"data": {"status": "done", "summary": summary}, "error": None}


@router.get("/health", summary="Check depth estimation dependencies")
def depth_health() -> dict:
    deps: dict[str, bool] = {}
    for pkg in ("transformers", "torch", "PIL"):
        try:
            __import__(pkg)
            deps[pkg] = True
        except ImportError:
            deps[pkg] = False
    all_ok = all(deps.values())
    return {"data": deps, "error": None if all_ok else "Missing dependencies"}
