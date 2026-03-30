"""
labelox/web/backend/routes/projects.py — Project CRUD endpoints.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from labelox.web.backend.auth import get_current_user
from labelox.web.backend.response import err, ok

router = APIRouter()


class CreateProjectRequest(BaseModel):
    name: str
    description: str = ""
    label_classes: list[dict] = []


class UpdateProjectRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    label_classes: list[dict] | None = None


@router.get("")
async def list_projects(user: dict = Depends(get_current_user)):
    from labelox.core.database import list_projects as db_list
    projects = db_list()
    return ok([
        {
            "id": p.id,
            "name": p.name,
            "description": p.description,
            "image_count": p.image_count,
            "annotated_count": p.annotated_count,
            "created_at": p.created_at.isoformat() if p.created_at else None,
        }
        for p in projects
    ])


@router.post("")
async def create_project(req: CreateProjectRequest, user: dict = Depends(get_current_user)):
    from labelox.core.database import create_project as db_create
    proj = db_create(req.name, req.description, req.label_classes)
    return ok({"id": proj.id, "name": proj.name})


@router.get("/{project_id}")
async def get_project(project_id: str, user: dict = Depends(get_current_user)):
    from labelox.core.database import get_project as db_get
    proj = db_get(project_id)
    if not proj:
        return err("Project not found", 404)
    return ok({
        "id": proj.id,
        "name": proj.name,
        "description": proj.description,
        "label_classes": proj.label_classes,
        "image_count": proj.image_count,
        "annotated_count": proj.annotated_count,
        "reviewed_count": proj.reviewed_count,
        "created_at": proj.created_at.isoformat() if proj.created_at else None,
    })


@router.put("/{project_id}")
async def update_project(project_id: str, req: UpdateProjectRequest, user: dict = Depends(get_current_user)):
    from labelox.core.database import get_project, get_session
    db = get_session()
    try:
        from labelox.core.database import DBProject
        proj = db.get(DBProject, project_id)
        if not proj:
            return err("Project not found", 404)
        if req.name is not None:
            proj.name = req.name
        if req.description is not None:
            proj.description = req.description
        if req.label_classes is not None:
            import json
            proj.label_classes_json = json.dumps(req.label_classes)
        db.commit()
        return ok({"id": proj.id, "name": proj.name})
    finally:
        db.close()


@router.delete("/{project_id}")
async def delete_project(project_id: str, user: dict = Depends(get_current_user)):
    from labelox.core.database import delete_project as db_delete
    db_delete(project_id)
    return ok({"deleted": True})
