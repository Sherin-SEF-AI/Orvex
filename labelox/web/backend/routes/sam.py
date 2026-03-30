"""
labelox/web/backend/routes/sam.py — SAM (Segment Anything) endpoints.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from labelox.web.backend.auth import get_current_user
from labelox.web.backend.response import err, ok

router = APIRouter()


class SAMPointRequest(BaseModel):
    image_path: str
    x: float
    y: float
    label: int = 1  # 1=foreground, 0=background


class SAMBoxRequest(BaseModel):
    image_path: str
    box: list[float]  # [x1, y1, x2, y2] in pixels


class SAMMultiPointRequest(BaseModel):
    image_path: str
    points: list[list[float]]  # [[x,y], ...]
    labels: list[int]          # [1, 0, 1, ...]


@router.post("/point")
async def predict_from_point(req: SAMPointRequest, user: dict = Depends(get_current_user)):
    try:
        from labelox.core.sam_engine import load_sam, set_image_for_sam, predict_mask_from_point
        load_sam()
        set_image_for_sam(req.image_path)
        ann = predict_mask_from_point(req.image_path, req.x, req.y, label=req.label)
        if ann:
            return ok(ann.model_dump(mode="json"))
        return ok(None)
    except Exception as exc:
        return err(str(exc))


@router.post("/box")
async def predict_from_box(req: SAMBoxRequest, user: dict = Depends(get_current_user)):
    try:
        from labelox.core.sam_engine import load_sam, set_image_for_sam, predict_mask_from_box
        load_sam()
        set_image_for_sam(req.image_path)
        ann = predict_mask_from_box(req.image_path, req.box)
        if ann:
            return ok(ann.model_dump(mode="json"))
        return ok(None)
    except Exception as exc:
        return err(str(exc))


@router.post("/points")
async def predict_from_points(req: SAMMultiPointRequest, user: dict = Depends(get_current_user)):
    try:
        from labelox.core.sam_engine import load_sam, set_image_for_sam, predict_mask_from_points
        load_sam()
        set_image_for_sam(req.image_path)
        ann = predict_mask_from_points(req.image_path, req.points, req.labels)
        if ann:
            return ok(ann.model_dump(mode="json"))
        return ok(None)
    except Exception as exc:
        return err(str(exc))
