"""
labelox/core/annotation_engine.py — Core annotation CRUD, merge, format conversion.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any

import numpy as np
from loguru import logger
from sqlalchemy.orm import Session

from labelox.core.database import DBAnnotation, DBImage, get_session
from labelox.core.models import (
    Annotation,
    AnnotationType,
    BBoxAnnotation,
    MaskAnnotation,
    Point,
    PolylineAnnotation,
)


# ─── CRUD ────────────────────────────────────────────────────────────────────

def save_annotations(
    image_id: str,
    annotations: list[Annotation],
    annotator_name: str,
    db: Session | None = None,
) -> None:
    """Atomic save: delete existing, bulk insert new. Update image status."""
    close = False
    if db is None:
        db = get_session()
        close = True
    try:
        # Delete existing
        db.query(DBAnnotation).filter(DBAnnotation.image_id == image_id).delete()

        # Insert new
        now = datetime.utcnow()
        for ann in annotations:
            data = _annotation_to_data_json(ann)
            db_ann = DBAnnotation(
                id=ann.id,
                image_id=image_id,
                label_id=ann.label_id,
                label_name=ann.label_name,
                annotation_type=ann.annotation_type.value,
                data_json=json.dumps(data),
                confidence=ann.confidence,
                is_auto=ann.is_auto,
                is_reviewed=ann.is_reviewed,
                track_id=ann.track_id,
                attributes_json=json.dumps(ann.attributes),
                created_by=annotator_name,
                created_at=ann.created_at or now,
                updated_at=now,
                comment=ann.comment,
            )
            db.add(db_ann)

        # Update image status
        img = db.get(DBImage, image_id)
        if img and annotations:
            img.status = "annotated"
            img.updated_at = now

        db.commit()
        logger.debug("Saved {} annotations for image {}", len(annotations), image_id)
    finally:
        if close:
            db.close()


def get_annotations(image_id: str, db: Session | None = None) -> list[Annotation]:
    """Fetch all annotations for an image, ordered by created_at."""
    close = False
    if db is None:
        db = get_session()
        close = True
    try:
        rows = (
            db.query(DBAnnotation)
            .filter(DBAnnotation.image_id == image_id)
            .order_by(DBAnnotation.created_at)
            .all()
        )
        return [_db_to_annotation(r) for r in rows]
    finally:
        if close:
            db.close()


def delete_annotation(annotation_id: str, db: Session | None = None) -> bool:
    close = False
    if db is None:
        db = get_session()
        close = True
    try:
        ann = db.get(DBAnnotation, annotation_id)
        if ann is None:
            return False
        db.delete(ann)
        db.commit()
        return True
    finally:
        if close:
            db.close()


# ─── Copy / Propagate ────────────────────────────────────────────────────────

def copy_annotations_to_next_frame(
    source_image_id: str,
    target_image_id: str,
    db: Session | None = None,
) -> list[Annotation]:
    """Copy all annotations from one frame to another. Preserves track_id."""
    source_anns = get_annotations(source_image_id, db)
    copied: list[Annotation] = []
    for ann in source_anns:
        new = ann.model_copy(deep=True)
        new.id = str(uuid.uuid4())
        new.image_id = target_image_id
        new.is_auto = False
        new.created_at = datetime.utcnow()
        new.updated_at = datetime.utcnow()
        copied.append(new)
    return copied


# ─── Merge ───────────────────────────────────────────────────────────────────

def merge_auto_with_manual(
    image_id: str,
    auto_annotations: list[Annotation],
    db: Session | None = None,
    conflict_strategy: str = "keep_manual",
) -> list[Annotation]:
    """Merge AI-generated annotations with existing manual ones.

    conflict_strategy:
    - "keep_manual": manual annotations take precedence
    - "keep_auto": replace all with auto
    - "merge_iou": keep manual if IoU > 0.5 with auto, else add auto
    """
    manual = get_annotations(image_id, db)

    if conflict_strategy == "keep_auto":
        return auto_annotations

    if conflict_strategy == "keep_manual":
        # Add auto annotations that don't overlap with manual
        merged = list(manual)
        for auto_ann in auto_annotations:
            if auto_ann.bbox is None:
                merged.append(auto_ann)
                continue
            overlaps = False
            for man_ann in manual:
                if man_ann.bbox and compute_iou(auto_ann.bbox, man_ann.bbox) > 0.5:
                    overlaps = True
                    break
            if not overlaps:
                merged.append(auto_ann)
        return merged

    if conflict_strategy == "merge_iou":
        merged = list(manual)
        for auto_ann in auto_annotations:
            if auto_ann.bbox is None:
                merged.append(auto_ann)
                continue
            has_match = False
            for man_ann in manual:
                if man_ann.bbox and compute_iou(auto_ann.bbox, man_ann.bbox) > 0.5:
                    has_match = True
                    break
            if not has_match:
                merged.append(auto_ann)
        return merged

    return manual + auto_annotations


# ─── Geometry Helpers ────────────────────────────────────────────────────────

def compute_iou(a: BBoxAnnotation, b: BBoxAnnotation) -> float:
    """Intersection over Union for two normalised bboxes."""
    ax1, ay1 = a.x, a.y
    ax2, ay2 = a.x + a.width, a.y + a.height
    bx1, by1 = b.x, b.y
    bx2, by2 = b.x + b.width, b.y + b.height

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = a.width * a.height
    area_b = b.width * b.height
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def bbox_to_yolo(bbox: BBoxAnnotation, label_id: int) -> str:
    """Convert normalised bbox to YOLO label line: class cx cy w h."""
    cx = bbox.x + bbox.width / 2
    cy = bbox.y + bbox.height / 2
    return f"{label_id} {cx:.6f} {cy:.6f} {bbox.width:.6f} {bbox.height:.6f}"


def bbox_from_yolo(line: str) -> tuple[int, BBoxAnnotation]:
    """Parse a YOLO label line. Returns (class_id, BBoxAnnotation)."""
    parts = line.strip().split()
    cls = int(parts[0])
    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    return cls, BBoxAnnotation(x=cx - w / 2, y=cy - h / 2, width=w, height=h)


def bbox_to_abs(bbox: BBoxAnnotation, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Convert normalised bbox to absolute pixel coords: (x, y, w, h)."""
    return (
        round(bbox.x * img_w),
        round(bbox.y * img_h),
        round(bbox.width * img_w),
        round(bbox.height * img_h),
    )


def polygon_to_rle(points: list[Point], height: int, width: int) -> dict:
    """Convert polygon points to COCO RLE mask format.

    Uses numpy rasterisation (no pycocotools dependency required).
    """
    import cv2

    pts = np.array(
        [[round(p.x * width), round(p.y * height)] for p in points],
        dtype=np.int32,
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)

    # Run-length encode
    flat = mask.flatten()
    changes = np.diff(flat)
    change_idx = np.where(changes != 0)[0] + 1
    runs = np.concatenate([[0], change_idx, [len(flat)]])
    lengths = np.diff(runs).tolist()

    # COCO RLE starts with zero-count if mask starts with 0
    if flat[0] == 1:
        lengths = [0] + lengths

    return {"counts": lengths, "size": [height, width]}


# ─── Serialisation Helpers ───────────────────────────────────────────────────

def _annotation_to_data_json(ann: Annotation) -> dict:
    """Extract the type-specific data from an Annotation for DB storage."""
    data: dict[str, Any] = {}
    if ann.bbox:
        data["bbox"] = ann.bbox.model_dump()
    if ann.mask:
        data["mask"] = ann.mask.model_dump()
    if ann.polyline:
        data["polyline"] = ann.polyline.model_dump()
    if ann.keypoints:
        data["keypoints"] = ann.keypoints.model_dump()
    if ann.cuboid_3d:
        data["cuboid_3d"] = ann.cuboid_3d.model_dump()
    if ann.classification is not None:
        data["classification"] = ann.classification
    return data


def _db_to_annotation(row: DBAnnotation) -> Annotation:
    """Convert a DBAnnotation row to a Pydantic Annotation."""
    data = json.loads(row.data_json) if row.data_json else {}
    attrs = json.loads(row.attributes_json) if row.attributes_json else {}

    kwargs: dict[str, Any] = {
        "id": row.id,
        "image_id": row.image_id,
        "label_id": row.label_id,
        "label_name": row.label_name,
        "annotation_type": AnnotationType(row.annotation_type),
        "confidence": row.confidence,
        "is_auto": row.is_auto,
        "is_reviewed": row.is_reviewed,
        "track_id": row.track_id,
        "attributes": attrs,
        "created_by": row.created_by or "",
        "created_at": row.created_at,
        "updated_at": row.updated_at,
        "comment": row.comment,
    }

    if "bbox" in data:
        kwargs["bbox"] = BBoxAnnotation(**data["bbox"])
    if "mask" in data:
        kwargs["mask"] = MaskAnnotation(**data["mask"])
    if "polyline" in data:
        kwargs["polyline"] = PolylineAnnotation(**data["polyline"])
    if "keypoints" in data:
        from labelox.core.models import KeypointAnnotation
        kwargs["keypoints"] = KeypointAnnotation(**data["keypoints"])
    if "cuboid_3d" in data:
        from labelox.core.models import Cuboid3DAnnotation
        kwargs["cuboid_3d"] = Cuboid3DAnnotation(**data["cuboid_3d"])
    if "classification" in data:
        kwargs["classification"] = data["classification"]

    return Annotation(**kwargs)
