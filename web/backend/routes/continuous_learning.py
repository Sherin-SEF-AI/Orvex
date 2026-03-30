"""
web/backend/routes/continuous_learning.py — Continuous learning loop endpoints.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.continuous_learning import check_and_maybe_trigger, get_learning_log
from core.models import AugmentationConfig, TrainingConfig
from core.session_manager import SessionManager

router = APIRouter(prefix="/learning", tags=["continuous_learning"])

_sm: SessionManager | None = None


def init_sm(sm: SessionManager) -> None:
    global _sm
    _sm = sm


def _get_sm() -> SessionManager:
    if _sm is None:
        raise RuntimeError("SessionManager not initialised")
    return _sm


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TriggerCycleRequest(BaseModel):
    trigger_type: str = "manual"
    threshold: int = 50
    auto_promote: bool = True
    augmentation_config: dict = {}
    training_config: dict = {}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/log")
def learning_log():
    entries = get_learning_log()
    raw = []
    for e in entries:
        d = e.model_dump()
        d["triggered_at"] = d["triggered_at"].isoformat() if d.get("triggered_at") else None
        raw.append(d)
    return {"data": raw, "error": None}


@router.post("/{session_id}/check")
def check_trigger(session_id: str, threshold: int = 50):
    try:
        ready = check_and_maybe_trigger(session_id, _get_sm(), threshold)
        return {"data": {"ready": ready, "threshold": threshold}, "error": None}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/{session_id}/trigger")
def trigger_cycle(session_id: str, req: TriggerCycleRequest):
    """Enqueue a full learning cycle as a Celery task."""
    try:
        from web.backend.tasks import run_learning_cycle_task
        task_id = str(uuid.uuid4())
        aug_cfg = AugmentationConfig(**req.augmentation_config) if req.augmentation_config else AugmentationConfig()
        train_cfg = TrainingConfig(**req.training_config) if req.training_config else TrainingConfig(dataset_dir="")

        run_learning_cycle_task.apply_async(
            kwargs={
                "task_id": task_id,
                "session_id": session_id,
                "trigger_type": req.trigger_type,
                "aug_config": aug_cfg.model_dump(),
                "train_config": train_cfg.model_dump(),
                "auto_promote": req.auto_promote,
            }
        )
        return {"data": {"task_id": task_id}, "error": None}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/{session_id}/status")
def cycle_status(session_id: str):
    """Return latest learning log entry for this session."""
    entries = [e for e in get_learning_log() if e.session_id == session_id]
    if not entries:
        return {"data": {"status": "no_history"}, "error": None}
    latest = entries[-1]
    d = latest.model_dump()
    d["triggered_at"] = d["triggered_at"].isoformat() if d.get("triggered_at") else None
    return {"data": d, "error": None}
