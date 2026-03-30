"""
labelox/web/backend/routes/stats.py — Project statistics endpoints.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from labelox.web.backend.auth import get_current_user
from labelox.web.backend.response import ok

router = APIRouter()


@router.get("/{project_id}")
async def project_stats(project_id: str, user: dict = Depends(get_current_user)):
    from labelox.core.stats_engine import compute_project_stats
    stats = compute_project_stats(project_id)
    return ok(stats)


@router.get("/{project_id}/class-balance")
async def class_balance(project_id: str, user: dict = Depends(get_current_user)):
    from labelox.core.stats_engine import compute_class_balance
    balance = compute_class_balance(project_id)
    return ok(balance)


@router.get("/{project_id}/quality")
async def quality_metrics(project_id: str, user: dict = Depends(get_current_user)):
    from labelox.core.stats_engine import compute_annotation_quality_metrics
    metrics = compute_annotation_quality_metrics(project_id)
    return ok(metrics)


@router.get("/{project_id}/daily")
async def daily_progress(project_id: str, user: dict = Depends(get_current_user)):
    from labelox.core.stats_engine import compute_daily_progress
    progress = compute_daily_progress(project_id)
    return ok(progress)
