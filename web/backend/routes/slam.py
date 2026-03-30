"""
web/backend/routes/slam.py — SLAM validation endpoints.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from core.session_manager import SessionNotFoundError
from web.backend.tasks import run_slam_validation

router = APIRouter(prefix="/slam", tags=["slam"])


def _sm(request: Request):
    return request.app.state.session_manager


class SLAMRequest(BaseModel):
    vocabulary_path: str
    config_yaml_path: str
    mode: str = "mono_inertial"   # "mono", "mono_inertial", "stereo"


@router.post("/{session_id}/run", summary="Run ORBSLAM3 on EuRoC session")
def start_slam(session_id: str, body: SLAMRequest, request: Request) -> dict:
    try:
        _sm(request).get_session(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    task_id = str(uuid.uuid4())
    run_slam_validation.apply_async(
        args=[session_id, body.model_dump(), task_id],
        task_id=task_id,
    )
    return {"data": {"task_id": task_id}, "error": None}


@router.get("/{session_id}/results", summary="Get SLAM trajectory results")
def get_slam_results(session_id: str, request: Request) -> dict:
    try:
        _sm(request).get_session(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    import json
    from pathlib import Path
    sm = _sm(request)
    result_path = sm.session_folder(session_id) / "slam" / "slam_result.json"
    if not result_path.exists():
        return {"data": {"status": "not_run"}, "error": None}

    result = json.loads(result_path.read_text())
    return {"data": {"status": "done", "result": result}, "error": None}


@router.get("/installation", summary="Check ORBSLAM3 installation")
def slam_installation() -> dict:
    from core.slam_validator import check_slam_installation
    info = check_slam_installation()
    return {"data": info, "error": None}


@router.post("/{session_id}/generate-config",
             summary="Auto-generate ORBSLAM3 config from calibration")
def generate_slam_config(session_id: str, request: Request) -> dict:
    try:
        _sm(request).get_session(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    from pathlib import Path
    from core.slam_validator import generate_orbslam3_config

    sm = _sm(request)
    cal_dir = sm.session_folder(session_id) / "calibration"
    intrinsics_path = cal_dir / "camera_intrinsics.json"
    if not intrinsics_path.exists():
        raise HTTPException(
            status_code=400,
            detail="Run camera calibration first to generate intrinsics."
        )

    try:
        import json
        from core.models import CalibrationResult
        cal_data = json.loads(intrinsics_path.read_text())
        cal_result = CalibrationResult(**cal_data)
        output_path = str(sm.session_folder(session_id) / "slam" / "orbslam3_config.yaml")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        config_path = generate_orbslam3_config(
            calibration_result=cal_result,
            session_fps=30.0,
            imu_rate_hz=200.0,
            output_path=output_path,
        )
        return {"data": {"config_path": config_path}, "error": None}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
