"""
web/backend/routes/calibration.py — Calibration workflow endpoints.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from core.calibration import get_step_result, is_step_complete
from core.models import CalibrationSession, DeviceType
from core.session_manager import SessionNotFoundError

router = APIRouter(prefix="/calibration", tags=["calibration"])

# In-memory store for calibration sessions (keyed by cal_session_id).
# A production version would persist to disk; for now the desktop
# already handles persistence via core.calibration._save_step().
_cal_sessions: dict[str, CalibrationSession] = {}


def _sm(request: Request):
    return request.app.state.session_manager


class CreateCalibrationRequest(BaseModel):
    camera_device: str      # DeviceType value
    session_type: str       # "imu_static" | "camera_intrinsic" | "camera_imu_extrinsic"
    file_path: str


@router.post("", summary="Create a calibration session")
def create_calibration(body: CreateCalibrationRequest) -> dict:
    try:
        device = DeviceType(body.camera_device)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid camera_device: {body.camera_device!r}"
        )

    cal_id = str(uuid.uuid4())
    cal = CalibrationSession(
        id=cal_id,
        camera_device=device,
        session_type=body.session_type,
        file_path=body.file_path,
        status="pending",
        results={},
        reprojection_error_px=None,
    )
    _cal_sessions[cal_id] = cal
    return {"data": cal.model_dump(), "error": None}


@router.get("/{cal_id}", summary="Get calibration session status")
def get_calibration(cal_id: str) -> dict:
    cal = _cal_sessions.get(cal_id)
    if cal is None:
        raise HTTPException(status_code=404, detail=f"Calibration {cal_id} not found")
    return {"data": cal.model_dump(), "error": None}


@router.get("/{cal_id}/steps", summary="Get per-step completion status")
def get_step_status(cal_id: str) -> dict:
    if cal_id not in _cal_sessions:
        raise HTTPException(status_code=404, detail=f"Calibration {cal_id} not found")

    steps = {}
    for step_key in ("imu_static", "camera_intrinsic", "camera_imu_extrinsic"):
        result = get_step_result(cal_id, step_key)
        steps[step_key] = {
            "complete": is_step_complete(cal_id, step_key),
            "result": result,
        }
    return {"data": steps, "error": None}


class RunStepRequest(BaseModel):
    step: str       # "imu_static" | "camera_intrinsic" | "camera_imu_extrinsic"
    file_path: str
    extra: dict = {}


@router.post("/{cal_id}/run-step", summary="Run a calibration step (async)")
def run_step(cal_id: str, body: RunStepRequest) -> dict:
    """Enqueue a calibration step as a Celery task and return task_id."""
    if cal_id not in _cal_sessions:
        raise HTTPException(status_code=404, detail=f"Calibration {cal_id} not found")

    from web.backend.tasks import celery_app

    task_id = str(uuid.uuid4())

    # Dynamically dispatch to the calibration worker task
    celery_app.send_task(
        "tasks.run_calibration_step",
        args=[cal_id, body.step, body.file_path, body.extra, task_id],
        task_id=task_id,
    )
    return {"data": {"task_id": task_id}, "error": None}
