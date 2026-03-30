"""
web/backend/routes/sessions.py — Session CRUD endpoints.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.session_manager import SessionManager, SessionNotFoundError

router = APIRouter(prefix="/sessions", tags=["sessions"])

# SessionManager is initialised once and injected via app.state in main.py
# Each route receives it through a FastAPI dependency.
from fastapi import Request


def _sm(request: Request) -> SessionManager:
    return request.app.state.session_manager


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    name: str
    environment: str
    location: str
    notes: str = ""


class AddFileRequest(BaseModel):
    file_path: str


class SessionResponse(BaseModel):
    id: str
    name: str
    environment: str
    location: str
    created_at: str
    extraction_status: str
    notes: str
    files: list[str]
    audit_results_count: int


def _to_response(s) -> SessionResponse:
    return SessionResponse(
        id=s.id,
        name=s.name,
        environment=s.environment,
        location=s.location,
        created_at=s.created_at.isoformat(),
        extraction_status=s.extraction_status,
        notes=s.notes,
        files=s.files,
        audit_results_count=len(s.audit_results),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("", summary="List all sessions")
def list_sessions(request: Request) -> dict:
    sessions = _sm(request).list_sessions()
    return {"data": [_to_response(s).model_dump() for s in sessions], "error": None}


@router.post("", summary="Create a new session")
def create_session(body: CreateSessionRequest, request: Request) -> dict:
    s = _sm(request).create_session(
        name=body.name,
        environment=body.environment,
        location=body.location,
        notes=body.notes,
    )
    return {"data": _to_response(s).model_dump(), "error": None}


@router.get("/{session_id}", summary="Get a session by ID")
def get_session(session_id: str, request: Request) -> dict:
    try:
        s = _sm(request).get_session(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return {"data": _to_response(s).model_dump(), "error": None}


@router.delete("/{session_id}", summary="Delete a session")
def delete_session(session_id: str, request: Request) -> dict:
    try:
        _sm(request).delete_session(session_id, delete_files=False)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return {"data": {"deleted": session_id}, "error": None}


@router.post("/{session_id}/files", summary="Add a file to a session")
def add_file(session_id: str, body: AddFileRequest, request: Request) -> dict:
    try:
        sm = _sm(request)
        sm.add_file(session_id, body.file_path)
        s = sm.get_session(session_id)
        return {"data": _to_response(s).model_dump(), "error": None}
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
