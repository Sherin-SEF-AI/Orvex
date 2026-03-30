"""
labelox/web/backend/routes/export.py — Export endpoints.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from labelox.core.models import ExportConfig, ExportFormat
from labelox.web.backend.auth import get_current_user
from labelox.web.backend.response import err, ok
from labelox.web.backend.tasks import get_task, submit_task

router = APIRouter()


class ExportRequest(BaseModel):
    project_id: str
    format: str = "yolo"
    output_dir: str
    include_images: bool = True
    only_reviewed: bool = False
    split_train_val: bool = True
    train_ratio: float = 0.8


@router.post("")
async def start_export(req: ExportRequest, user: dict = Depends(get_current_user)):
    config = ExportConfig(
        project_id=req.project_id,
        format=ExportFormat(req.format),
        output_dir=req.output_dir,
        include_images=req.include_images,
        only_reviewed=req.only_reviewed,
        split_train_val=req.split_train_val,
        train_ratio=req.train_ratio,
    )
    from labelox.core.exporter import export_project
    task_id = submit_task("export", export_project, config)
    return ok({"task_id": task_id})


@router.get("/status/{task_id}")
async def export_status(task_id: str, user: dict = Depends(get_current_user)):
    task = get_task(task_id)
    if not task:
        return err("Task not found", 404)
    return ok({
        "task_id": task.id,
        "status": task.status.value,
        "progress": task.progress,
        "total": task.total,
        "message": task.message,
        "error": task.error,
        "result": task.result if hasattr(task.result, "__dict__") else str(task.result) if task.result else None,
    })
