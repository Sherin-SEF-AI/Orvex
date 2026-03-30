"""
web/backend/routes/augmentation.py — Data augmentation endpoints.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from core.session_manager import SessionNotFoundError
from web.backend.tasks import run_augmentation

router = APIRouter(prefix="/augmentation", tags=["augmentation"])


def _sm(request: Request):
    return request.app.state.session_manager


class AugmentationRequest(BaseModel):
    horizontal_flip: bool = True
    vertical_flip: bool = False
    random_rotate_90: bool = True
    brightness_contrast: bool = True
    hue_saturation: bool = True
    gaussian_noise: bool = True
    motion_blur: bool = True
    jpeg_compression: bool = True
    mosaic: bool = True
    rain_simulation: bool = False
    fog_simulation: bool = False
    multiplier: int = 3


@router.post("/{session_id}/run", summary="Run data augmentation on annotated frames")
def start_augmentation(session_id: str, body: AugmentationRequest, request: Request) -> dict:
    try:
        _sm(request).get_session(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    # Check auto-label results exist
    from pathlib import Path
    sm = _sm(request)
    autolabel_dir = sm.session_folder(session_id) / "autolabel"
    if not autolabel_dir.exists():
        raise HTTPException(
            status_code=400,
            detail="Run auto-labeling first to get annotated frames."
        )

    task_id = str(uuid.uuid4())
    run_augmentation.apply_async(
        args=[session_id, body.model_dump(), task_id],
        task_id=task_id,
    )
    return {"data": {"task_id": task_id}, "error": None}


@router.get("/{session_id}/results", summary="Get augmentation results")
def get_augmentation_results(session_id: str, request: Request) -> dict:
    try:
        _sm(request).get_session(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    import json
    from pathlib import Path
    sm = _sm(request)
    result_path = sm.session_folder(session_id) / "augmented_dataset" / "augmentation_result.json"
    if not result_path.exists():
        return {"data": {"status": "not_run"}, "error": None}

    result = json.loads(result_path.read_text())
    return {"data": {"status": "done", "result": result}, "error": None}


@router.get("/health", summary="Check augmentation dependencies")
def augmentation_health() -> dict:
    try:
        import albumentations  # noqa: F401
        return {"data": {"albumentations": True}, "error": None}
    except ImportError:
        return {"data": {"albumentations": False}, "error": "albumentations not installed"}
