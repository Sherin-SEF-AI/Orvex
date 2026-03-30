"""
web/backend/routes/inference.py — Model registry + live inference endpoints.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.inference_server import (
    REGISTRY_FILE,
    compare_models,
    delete_model,
    get_active_model,
    load_registry,
    register_model,
    run_inference,
    run_inference_batch,
    set_active_model,
)
from core.models import InferenceRequest, InferenceResult

router = APIRouter(prefix="/inference", tags=["inference"])


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class RegisterModelRequest(BaseModel):
    weights_path: str
    name: str
    model_variant: str = "yolov8n"
    training_run_id: str | None = None
    metrics: dict = {}


class ActivateModelRequest(BaseModel):
    model_id: str


# ---------------------------------------------------------------------------
# Registry CRUD
# ---------------------------------------------------------------------------

@router.get("/models")
def list_models():
    models = load_registry(REGISTRY_FILE)
    return {"data": [m.model_dump() for m in models], "error": None}


@router.post("/models/register")
def register_weights(req: RegisterModelRequest):
    try:
        entry = register_model(
            weights_path=req.weights_path,
            name=req.name,
            model_variant=req.model_variant,
            training_run_id=req.training_run_id,
            metrics=req.metrics,
            registry_path=REGISTRY_FILE,
        )
        return {"data": entry.model_dump(), "error": None}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/models/{model_id}/activate")
def activate_model(model_id: str):
    try:
        set_active_model(model_id, REGISTRY_FILE)
        return {"data": {"model_id": model_id, "activated": True}, "error": None}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.delete("/models/{model_id}")
def remove_model(model_id: str):
    try:
        delete_model(model_id, REGISTRY_FILE)
        return {"data": {"deleted": model_id}, "error": None}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@router.post("/predict")
def predict_single(req: InferenceRequest):
    try:
        result = run_inference(req, REGISTRY_FILE)
        return {"data": result.model_dump(), "error": None}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class BatchInferenceRequest(BaseModel):
    requests: list[InferenceRequest]


@router.post("/batch")
def predict_batch(req: BatchInferenceRequest):
    try:
        results = run_inference_batch(req.requests, REGISTRY_FILE)
        return {"data": [r.model_dump() for r in results], "error": None}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/health")
def inference_health():
    try:
        active = get_active_model(REGISTRY_FILE)
        return {
            "data": {
                "status": "ok",
                "active_model": active.name,
                "model_id": active.model_id,
                "variant": active.model_variant,
            },
            "error": None,
        }
    except RuntimeError:
        return {
            "data": {"status": "no_model", "active_model": None},
            "error": None,
        }
