"""
web/backend/routes/experiments.py — MLflow experiment tracking endpoints.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.get("/health", summary="Check MLflow installation")
def experiments_health() -> dict:
    from core.experiment_tracker import check_mlflow_installation
    ok = check_mlflow_installation()
    return {"data": {"mlflow": ok}, "error": None if ok else "MLflow not installed. pip install mlflow"}


@router.get("/runs", summary="List all training runs")
def list_runs(experiment_name: str = Query(default="rover_detection")) -> dict:
    from core.experiment_tracker import get_all_runs
    runs = get_all_runs(experiment_name)
    return {"data": [r.model_dump() for r in runs], "error": None}


@router.get("/runs/{run_id}", summary="Get single run details")
def get_run(run_id: str) -> dict:
    from core.experiment_tracker import get_all_runs
    runs = get_all_runs()
    match = next((r for r in runs if r.run_id == run_id), None)
    if not match:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return {"data": match.model_dump(), "error": None}


@router.get("/compare", summary="Compare multiple runs")
def compare_runs(runs: str = Query(...)) -> dict:
    """runs = comma-separated run IDs"""
    run_ids = [r.strip() for r in runs.split(",") if r.strip()]
    if len(run_ids) < 2:
        raise HTTPException(status_code=400, detail="Provide at least 2 run IDs")
    from core.experiment_tracker import compare_runs as _compare
    result = _compare(run_ids)
    if result is None:
        raise HTTPException(status_code=404, detail="Runs not found or MLflow unavailable")
    return {"data": result.model_dump(), "error": None}


@router.post("/ui/launch", summary="Launch MLflow UI")
def launch_ui(port: int = Query(default=5000)) -> dict:
    from core.experiment_tracker import launch_mlflow_ui
    proc = launch_mlflow_ui(port)
    if proc is None:
        raise HTTPException(status_code=503, detail="MLflow not installed")
    return {"data": {"url": f"http://localhost:{port}", "pid": proc.pid}, "error": None}
