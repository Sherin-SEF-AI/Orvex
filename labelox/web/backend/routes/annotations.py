"""
labelox/web/backend/routes/annotations.py — Annotation CRUD endpoints.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from labelox.web.backend.auth import get_current_user
from labelox.web.backend.response import err, ok

router = APIRouter()


class SaveAnnotationsRequest(BaseModel):
    annotations: list[dict]
    annotator_id: str = "web_user"


@router.get("/{image_id}")
async def get_annotations(image_id: str, user: dict = Depends(get_current_user)):
    from labelox.core.annotation_engine import get_annotations
    anns = get_annotations(image_id)
    return ok([ann.model_dump(mode="json") for ann in anns])


@router.post("/{image_id}")
async def save_annotations(
    image_id: str,
    req: SaveAnnotationsRequest,
    user: dict = Depends(get_current_user),
):
    from labelox.core.annotation_engine import save_annotations
    from labelox.core.models import Annotation
    anns = [Annotation(**a) for a in req.annotations]
    saved = save_annotations(image_id, anns, req.annotator_id)
    return ok({"saved": saved})


@router.delete("/{image_id}/{annotation_id}")
async def delete_annotation(
    image_id: str,
    annotation_id: str,
    user: dict = Depends(get_current_user),
):
    from labelox.core.annotation_engine import delete_annotation
    delete_annotation(image_id, annotation_id)
    return ok({"deleted": True})


@router.post("/{image_id}/copy-to-next")
async def copy_to_next(image_id: str, user: dict = Depends(get_current_user)):
    from labelox.core.annotation_engine import copy_annotations_to_next_frame
    count = copy_annotations_to_next_frame(image_id)
    return ok({"copied": count})
