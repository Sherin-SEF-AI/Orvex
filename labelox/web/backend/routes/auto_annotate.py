"""
labelox/web/backend/routes/auto_annotate.py — YOLOv8 auto-annotation endpoints.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from labelox.web.backend.auth import get_current_user
from labelox.web.backend.response import err, ok
from labelox.web.backend.tasks import get_task, submit_task

router = APIRouter()


class AutoAnnotateRequest(BaseModel):
    project_id: str
    model_path: str = "yolov8n.pt"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    batch_size: int = 16


@router.post("")
async def start_auto_annotate(req: AutoAnnotateRequest, user: dict = Depends(get_current_user)):
    """Start auto-annotation as a background task."""
    from labelox.core.auto_annotator import run_auto_annotate_project
    task_id = submit_task(
        "auto_annotate",
        run_auto_annotate_project,
        req.project_id,
        model_path=req.model_path,
        conf_threshold=req.conf_threshold,
        iou_threshold=req.iou_threshold,
        batch_size=req.batch_size,
    )
    return ok({"task_id": task_id})


@router.get("/status/{task_id}")
async def auto_annotate_status(task_id: str, user: dict = Depends(get_current_user)):
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
    })
