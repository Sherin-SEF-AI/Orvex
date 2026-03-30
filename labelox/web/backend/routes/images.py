"""
labelox/web/backend/routes/images.py — Image management endpoints.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from labelox.web.backend.auth import get_current_user
from labelox.web.backend.response import err, ok

router = APIRouter()

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


@router.get("/file")
async def serve_image_file(
    path: str = Query(..., description="Absolute path to image file"),
    user: dict = Depends(get_current_user),
):
    """Serve an image file from the local filesystem."""
    file_path = Path(path)
    if not file_path.exists():
        return err("File not found", 404)
    if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return err("Not an allowed image type", 400)
    return FileResponse(str(file_path))


@router.get("/{project_id}")
async def list_images(
    project_id: str,
    status: str | None = None,
    offset: int = 0,
    limit: int = 100,
    user: dict = Depends(get_current_user),
):
    from labelox.core.database import get_images, get_session
    db = get_session()
    try:
        images = get_images(project_id, limit=limit, db=db)
        result = []
        for img in images[offset:offset + limit]:
            if status and img.status != status:
                continue
            result.append({
                "id": img.id,
                "file_name": img.file_name,
                "file_path": img.file_path,
                "width": img.width,
                "height": img.height,
                "status": img.status,
                "blur_score": img.blur_score,
                "thumbnail_path": img.thumbnail_path,
            })
        return ok(result)
    finally:
        db.close()


@router.post("/{project_id}/upload")
async def upload_images(
    project_id: str,
    files: list[UploadFile] = File(...),
    user: dict = Depends(get_current_user),
):
    """Upload images to a project — streams to disk, no memory buffering."""
    upload_dir = Path.home() / ".labelox" / "uploads" / project_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for f in files:
        dest = upload_dir / f.filename
        with open(dest, "wb") as out:
            while chunk := await f.read(1024 * 1024):
                out.write(chunk)
        saved_paths.append(dest)

    # Import into project
    from labelox.core.image_manager import import_images
    from labelox.core.database import get_session
    db = get_session()
    try:
        thumb_dir = Path.home() / ".labelox" / "projects" / project_id / "thumbs"
        created = import_images(saved_paths, project_id, thumb_dir=thumb_dir, db=db)
        return ok({"imported": len(created)})
    finally:
        db.close()


@router.get("/{project_id}/{image_id}")
async def get_image(project_id: str, image_id: str, user: dict = Depends(get_current_user)):
    from labelox.core.database import get_session, DBImage
    db = get_session()
    try:
        img = db.get(DBImage, image_id)
        if not img or img.project_id != project_id:
            return err("Image not found", 404)
        return ok({
            "id": img.id,
            "file_name": img.file_name,
            "file_path": img.file_path,
            "width": img.width,
            "height": img.height,
            "status": img.status,
            "blur_score": img.blur_score,
            "sequence_id": img.sequence_id,
            "frame_index": img.frame_index,
        })
    finally:
        db.close()


@router.get("/{project_id}/next-unlabeled")
async def next_unlabeled(project_id: str, user: dict = Depends(get_current_user)):
    from labelox.core.database import get_next_unlabeled_image, get_session
    db = get_session()
    try:
        img = get_next_unlabeled_image(project_id, db=db)
        if not img:
            return ok(None)
        return ok({
            "id": img.id,
            "file_name": img.file_name,
            "file_path": img.file_path,
        })
    finally:
        db.close()
