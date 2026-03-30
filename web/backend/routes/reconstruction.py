"""
web/backend/routes/reconstruction.py — 3D reconstruction (COLMAP) endpoints.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from core.session_manager import SessionNotFoundError
from web.backend.tasks import run_reconstruction

router = APIRouter(prefix="/reconstruction", tags=["reconstruction"])


def _sm(request: Request):
    return request.app.state.session_manager


class ReconstructionRequest(BaseModel):
    every_nth: int = 6
    use_gpu: bool = True
    camera_model: str = "OPENCV_FISHEYE"
    max_image_size: int = 1600


@router.post("/{session_id}/run", summary="Run COLMAP 3D reconstruction")
def start_reconstruction(session_id: str, body: ReconstructionRequest, request: Request) -> dict:
    try:
        _sm(request).get_session(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    task_id = str(uuid.uuid4())
    run_reconstruction.apply_async(
        args=[session_id, body.model_dump(), task_id],
        task_id=task_id,
    )
    return {"data": {"task_id": task_id}, "error": None}


@router.get("/{session_id}/results", summary="Get reconstruction results")
def get_reconstruction_results(session_id: str, request: Request) -> dict:
    try:
        _sm(request).get_session(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    import json
    from pathlib import Path
    sm = _sm(request)
    result_path = sm.session_folder(session_id) / "reconstruction" / "colmap_result.json"
    if not result_path.exists():
        return {"data": {"status": "not_run"}, "error": None}

    result = json.loads(result_path.read_text())
    return {"data": {"status": "done", "result": result}, "error": None}


@router.get("/installation", summary="Check COLMAP installation")
def colmap_installation() -> dict:
    from core.reconstructor import check_colmap_installation
    installed = check_colmap_installation()
    return {"data": {"colmap": installed}, "error": None}
