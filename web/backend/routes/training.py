"""
web/backend/routes/training.py — YOLOv8 model training endpoints.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from core.session_manager import SessionNotFoundError
from web.backend.tasks import run_training

router = APIRouter(prefix="/training", tags=["training"])


def _sm(request: Request):
    return request.app.state.session_manager


class TrainingRequest(BaseModel):
    dataset_dir: str
    model_variant: str = "yolov8n"
    pretrained_weights: str = "yolov8n.pt"
    epochs: int = 100
    batch_size: int = 16
    image_size: int = 640
    learning_rate: float = 0.01
    device: str = "auto"
    project_name: str = "rover_detection"
    run_name: str = "run1"


class CancelRequest(BaseModel):
    task_id: str


@router.post("/run", summary="Start YOLOv8 model training")
def start_training(body: TrainingRequest, request: Request) -> dict:
    task_id = str(uuid.uuid4())
    run_training.apply_async(
        args=[body.model_dump(), task_id],
        task_id=task_id,
    )
    return {"data": {"task_id": task_id}, "error": None}


@router.post("/cancel", summary="Cancel a running training job")
def cancel_training(body: CancelRequest) -> dict:
    try:
        from core.trainer import cancel_training as core_cancel
        core_cancel()
        return {"data": {"cancelled": True}, "error": None}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/export", summary="Export trained model weights")
def export_model(request: Request) -> dict:
    from pydantic import BaseModel as BM

    class ExportBody(BM):
        weights_path: str
        format: str = "onnx"
        image_size: int = 640

    raise HTTPException(status_code=400, detail="Pass weights_path, format, image_size in body.")


@router.post("/export-model", summary="Export trained model to target format")
def export_model_v2(body: dict, request: Request) -> dict:
    weights_path = body.get("weights_path")
    fmt = body.get("format", "onnx")
    imgsz = body.get("image_size", 640)
    if not weights_path:
        raise HTTPException(status_code=400, detail="weights_path is required")
    try:
        from core.trainer import export_model as core_export
        path = core_export(weights_path, fmt, imgsz)
        return {"data": {"exported_path": path}, "error": None}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/health", summary="Check training dependencies")
def training_health() -> dict:
    deps: dict[str, bool] = {}
    for pkg in ("ultralytics", "torch"):
        try:
            __import__(pkg)
            deps[pkg] = True
        except ImportError:
            deps[pkg] = False
    all_ok = all(deps.values())
    return {"data": deps, "error": None if all_ok else "Missing dependencies"}
