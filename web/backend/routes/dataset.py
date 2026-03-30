"""
web/backend/routes/dataset.py — Dataset build endpoints.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/dataset", tags=["dataset"])


class DatasetBuildRequest(BaseModel):
    session_ids: list[str]
    output_format: str = "euroc"     # "euroc" | "rosbag2" | "hdf5"
    output_dir: str                  # server-side absolute path


@router.post("/build", summary="Start a dataset build task")
def start_build(body: DatasetBuildRequest, request: Request) -> dict:
    if not body.session_ids:
        raise HTTPException(status_code=400, detail="session_ids must not be empty")
    if body.output_format not in ("euroc", "rosbag2", "hdf5"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid output_format: {body.output_format!r}. "
                   "Use 'euroc', 'rosbag2', or 'hdf5'."
        )

    from web.backend.tasks import run_dataset_build

    task_id = str(uuid.uuid4())
    run_dataset_build.apply_async(
        args=[body.session_ids, body.output_format, body.output_dir, task_id],
        task_id=task_id,
    )
    return {"data": {"task_id": task_id}, "error": None}
