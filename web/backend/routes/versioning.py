"""
web/backend/routes/versioning.py — Dataset versioning (DVC) endpoints.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

router = APIRouter(prefix="/versioning", tags=["versioning"])


class CommitVersionRequest(BaseModel):
    dataset_dir: str
    version_tag: str
    message: str
    metadata: dict = {}


class DiffRequest(BaseModel):
    dataset_dir: str
    version_a: str
    version_b: str


@router.get("/health", summary="Check DVC installation status")
def versioning_health() -> dict:
    from core.versioning import check_dvc_installation
    dvc_ok = check_dvc_installation()
    return {"data": {"dvc": dvc_ok}, "error": None if dvc_ok else "DVC not installed"}


@router.post("/init", summary="Initialize DVC in dataset directory")
def init_dvc(dataset_dir: str = Query(...)) -> dict:
    try:
        from core.versioning import init_dvc_repo
        ok = init_dvc_repo(dataset_dir)
        return {"data": {"initialized": ok}, "error": None}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/versions", summary="List dataset versions")
def list_versions(dataset_dir: str = Query(...)) -> dict:
    try:
        from core.versioning import list_dataset_versions
        versions = list_dataset_versions(dataset_dir)
        return {"data": [v.model_dump() for v in versions], "error": None}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/versions", summary="Commit new dataset version")
def commit_version(body: CommitVersionRequest) -> dict:
    try:
        from core.versioning import commit_dataset_version
        version = commit_dataset_version(
            body.dataset_dir, body.version_tag, body.message, body.metadata
        )
        return {"data": version.model_dump(), "error": None}
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/diff", summary="Compare two dataset versions")
def diff_versions(body: DiffRequest) -> dict:
    try:
        from core.versioning import diff_dataset_versions
        diff = diff_dataset_versions(body.dataset_dir, body.version_a, body.version_b)
        return {"data": diff.model_dump(), "error": None}
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/restore", summary="Restore dataset to a named version")
def restore_version(dataset_dir: str = Query(...), version_tag: str = Query(...)) -> dict:
    try:
        from core.versioning import restore_dataset_version
        ok = restore_dataset_version(dataset_dir, version_tag)
        return {"data": {"restored": ok, "version": version_tag}, "error": None}
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
