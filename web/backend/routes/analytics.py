"""
web/backend/routes/analytics.py — Dataset analytics endpoints.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from core.session_manager import SessionNotFoundError
from web.backend.tasks import run_analytics

router = APIRouter(prefix="/analytics", tags=["analytics"])


def _sm(request: Request):
    return request.app.state.session_manager


class AnalyticsRequest(BaseModel):
    sample_n: int = 500


@router.post("/{session_id}/run", summary="Compute scene analytics for a session")
def start_analytics(session_id: str, body: AnalyticsRequest, request: Request) -> dict:
    try:
        _sm(request).get_session(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    task_id = str(uuid.uuid4())
    run_analytics.apply_async(
        args=[session_id, body.model_dump(), task_id],
        task_id=task_id,
    )
    return {"data": {"task_id": task_id}, "error": None}


@router.get("/{session_id}/results", summary="Get analytics results")
def get_analytics_results(session_id: str, request: Request) -> dict:
    try:
        _sm(request).get_session(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    import json
    from pathlib import Path
    sm = _sm(request)
    result_path = sm.session_folder(session_id) / "analytics" / "analytics_result.json"
    if not result_path.exists():
        return {"data": {"status": "not_run"}, "error": None}

    result = json.loads(result_path.read_text())
    return {"data": {"status": "done", "result": result}, "error": None}


@router.post("/{session_id}/report", summary="Generate HTML analytics report")
def generate_report(session_id: str, request: Request) -> dict:
    try:
        _sm(request).get_session(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    from pathlib import Path
    from core.road_analytics import generate_dataset_report

    sm = _sm(request)
    output_dir = str(sm.session_folder(session_id) / "analytics")
    try:
        path = generate_dataset_report(
            session_ids=[session_id],
            annotations=[],
            gps_samples=[],
            output_dir=output_dir,
        )
        return {"data": {"report_path": path}, "error": None}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
