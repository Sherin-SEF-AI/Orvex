"""
labelox/web/backend/routes/users.py — User/annotator management endpoints.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from labelox.web.backend.auth import create_token, get_current_user
from labelox.web.backend.response import err, ok

router = APIRouter()


class LoginRequest(BaseModel):
    name: str
    role: str = "annotator"


class CreateAnnotatorRequest(BaseModel):
    name: str
    role: str = "annotator"
    color: str = "#e94560"


@router.post("/login")
async def login(req: LoginRequest):
    """Simple login — returns a JWT. No password for local dev."""
    import uuid
    user_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, req.name))
    token = create_token(user_id, req.name, req.role)
    return ok({"token": token, "user_id": user_id, "name": req.name, "role": req.role})


@router.get("/me")
async def me(user: dict = Depends(get_current_user)):
    return ok(user)


@router.get("/annotators/{project_id}")
async def list_annotators(project_id: str, user: dict = Depends(get_current_user)):
    from labelox.core.database import get_session, DBAnnotator
    from sqlalchemy import select
    db = get_session()
    try:
        stmt = select(DBAnnotator).where(DBAnnotator.project_id == project_id)
        result = db.execute(stmt).scalars().all()
        return ok([
            {"id": a.id, "name": a.name, "role": a.role, "color": a.color, "is_active": a.is_active}
            for a in result
        ])
    finally:
        db.close()


@router.post("/annotators/{project_id}")
async def create_annotator(
    project_id: str,
    req: CreateAnnotatorRequest,
    user: dict = Depends(get_current_user),
):
    from labelox.core.database import get_session, DBAnnotator
    import uuid
    db = get_session()
    try:
        ann = DBAnnotator(
            id=str(uuid.uuid4()),
            project_id=project_id,
            name=req.name,
            role=req.role,
            color=req.color,
        )
        db.add(ann)
        db.commit()
        return ok({"id": ann.id, "name": ann.name})
    finally:
        db.close()
