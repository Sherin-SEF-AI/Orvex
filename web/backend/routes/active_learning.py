"""
web/backend/routes/active_learning.py — Active learning frame selection endpoints.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from core.session_manager import SessionNotFoundError
from web.backend.tasks import run_active_learning

router = APIRouter(prefix="/active-learning", tags=["active-learning"])


def _sm(request: Request):
    return request.app.state.session_manager


class ActiveLearningRequest(BaseModel):
    method: str = "entropy"          # "entropy", "margin", "least_confidence"
    n_frames: int = 100
    uncertainty_weight: float = 0.6
    diversity_weight: float = 0.4


@router.post("/{session_id}/run", summary="Score and select frames for labeling")
def start_active_learning(
    session_id: str, body: ActiveLearningRequest, request: Request
) -> dict:
    try:
        _sm(request).get_session(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    # Check annotations exist
    from pathlib import Path
    sm = _sm(request)
    autolabel_dir = sm.session_folder(session_id) / "autolabel"
    if not autolabel_dir.exists():
        raise HTTPException(
            status_code=400,
            detail="Run auto-labeling first to generate confidence scores."
        )

    task_id = str(uuid.uuid4())
    run_active_learning.apply_async(
        args=[session_id, body.model_dump(), task_id],
        task_id=task_id,
    )
    return {"data": {"task_id": task_id}, "error": None}


@router.get("/{session_id}/results", summary="Get active learning selection results")
def get_active_learning_results(session_id: str, request: Request) -> dict:
    try:
        _sm(request).get_session(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    import json
    from pathlib import Path
    sm = _sm(request)
    result_path = sm.session_folder(session_id) / "active_learning" / "selection_result.json"
    if not result_path.exists():
        return {"data": {"status": "not_run"}, "error": None}

    result = json.loads(result_path.read_text())
    return {"data": {"status": "done", "result": result}, "error": None}
