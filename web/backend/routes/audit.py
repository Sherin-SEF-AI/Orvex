"""
web/backend/routes/audit.py — File audit endpoints.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request

from core.session_manager import SessionNotFoundError
from web.backend.tasks import run_audit

router = APIRouter(prefix="/audit", tags=["audit"])


def _sm(request: Request):
    return request.app.state.session_manager


@router.post("/{session_id}/run", summary="Start an audit task for a session")
def start_audit(session_id: str, request: Request) -> dict:
    try:
        _sm(request).get_session(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    task_id = str(uuid.uuid4())
    run_audit.apply_async(args=[session_id, task_id], task_id=task_id)
    return {"data": {"task_id": task_id}, "error": None}


@router.get("/{session_id}/results", summary="Get audit results for a session")
def get_audit_results(session_id: str, request: Request) -> dict:
    try:
        s = _sm(request).get_session(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    results = [r.model_dump() for r in s.audit_results]
    return {"data": results, "error": None}
