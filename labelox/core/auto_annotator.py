"""
labelox/core/auto_annotator.py — YOLOv8 auto-annotation engine.

Model loaded once and cached. Batch inference with progress callbacks.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from labelox.core.models import Annotation, AnnotationType, BBoxAnnotation

ProgressCB = Callable[[int, int], None]

# ─── Model Cache ─────────────────────────────────────────────────────────────

_MODEL_CACHE: dict[str, Any] = {}

# COCO to common road-scene label mapping
COCO_TO_ROVER: dict[str, str] = {
    "person": "person",
    "bicycle": "bicycle",
    "car": "car",
    "motorcycle": "motorcycle",
    "bus": "bus",
    "truck": "truck",
    "traffic light": "traffic_light",
    "stop sign": "stop_sign",
    "dog": "dog",
    "cat": "cat",
    "cow": "cow",
}


def _resolve_device(device: str) -> str:
    import torch
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def load_yolo_model(
    model_path: str = "yolov8n.pt",
    device: str = "auto",
) -> Any:
    """Load YOLOv8 model. Cached globally — subsequent calls are instant."""
    resolved = _resolve_device(device)
    key = f"{model_path}:{resolved}"

    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    from ultralytics import YOLO

    logger.info("Loading YOLO model {} on {}", model_path, resolved)
    model = YOLO(model_path)
    model.to(resolved)
    _MODEL_CACHE[key] = model
    return model


# ─── Single Image Inference ──────────────────────────────────────────────────

def run_yolo_on_image(
    image_path: str,
    model: Any,
    label_classes: list[dict],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    class_mapping: dict[str, str] | None = None,
    annotator_name: str = "auto",
) -> list[Annotation]:
    """Run YOLOv8 on a single image. Returns list of Annotation."""
    results = model(image_path, conf=conf_threshold, iou=iou_threshold, verbose=False)
    if not results:
        return []

    mapping = class_mapping or COCO_TO_ROVER
    project_names = {cls["name"] for cls in label_classes}
    project_by_name = {cls["name"]: cls for cls in label_classes}
    now = datetime.utcnow()

    annotations: list[Annotation] = []
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for i in range(len(boxes)):
            coco_cls = r.names[int(boxes.cls[i])]
            mapped = mapping.get(coco_cls, coco_cls)
            if mapped not in project_names:
                continue

            xyxyn = boxes.xyxyn[i].cpu().tolist()
            x1, y1, x2, y2 = xyxyn
            conf = float(boxes.conf[i])
            cls_info = project_by_name[mapped]

            annotations.append(Annotation(
                image_id="",  # caller sets this
                label_id=cls_info.get("id", mapped),
                label_name=mapped,
                annotation_type=AnnotationType.BBOX,
                bbox=BBoxAnnotation(
                    x=x1, y=y1,
                    width=x2 - x1, height=y2 - y1,
                ),
                confidence=conf,
                is_auto=True,
                created_by=annotator_name,
                created_at=now,
                updated_at=now,
            ))

    return annotations


# ─── Batch Inference ─────────────────────────────────────────────────────────

def run_yolo_batch(
    image_paths: list[str],
    model: Any,
    label_classes: list[dict],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    batch_size: int = 16,
    class_mapping: dict[str, str] | None = None,
    progress_callback: ProgressCB | None = None,
) -> dict[str, list[Annotation]]:
    """Batch inference. Returns {image_path: [annotations]}."""
    all_results: dict[str, list[Annotation]] = {}
    n = len(image_paths)

    for start in range(0, n, batch_size):
        batch = image_paths[start: start + batch_size]
        results = model(batch, conf=conf_threshold, iou=iou_threshold, verbose=False)

        mapping = class_mapping or COCO_TO_ROVER
        project_names = {cls["name"] for cls in label_classes}
        project_by_name = {cls["name"]: cls for cls in label_classes}
        now = datetime.utcnow()

        for j, r in enumerate(results):
            path = batch[j]
            anns: list[Annotation] = []
            boxes = r.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    coco_cls = r.names[int(boxes.cls[i])]
                    mapped = mapping.get(coco_cls, coco_cls)
                    if mapped not in project_names:
                        continue

                    xyxyn = boxes.xyxyn[i].cpu().tolist()
                    x1, y1, x2, y2 = xyxyn
                    conf = float(boxes.conf[i])
                    cls_info = project_by_name[mapped]

                    anns.append(Annotation(
                        image_id="",
                        label_id=cls_info.get("id", mapped),
                        label_name=mapped,
                        annotation_type=AnnotationType.BBOX,
                        bbox=BBoxAnnotation(
                            x=x1, y=y1,
                            width=x2 - x1, height=y2 - y1,
                        ),
                        confidence=conf,
                        is_auto=True,
                        created_by="auto",
                        created_at=now,
                        updated_at=now,
                    ))
            all_results[path] = anns

        if progress_callback:
            progress_callback(min(start + batch_size, n), n)

    return all_results


# ─── Full Pipeline ───────────────────────────────────────────────────────────

def run_auto_annotate_project(
    project_id: str,
    image_ids: list[str] | None = None,
    model_path: str = "yolov8n.pt",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    batch_size: int = 16,
    device: str = "auto",
    class_mapping: dict[str, str] | None = None,
    db=None,
    progress_callback: ProgressCB | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Full auto-annotation pipeline for a project.

    Returns: {processed, skipped, total_detections, errors}
    """
    from labelox.core.annotation_engine import save_annotations
    from labelox.core.database import DBImage, DBProject, get_session

    close = db is None
    if db is None:
        db = get_session()

    try:
        proj = db.get(DBProject, project_id)
        if proj is None:
            raise ValueError(f"Project not found: {project_id}")

        label_classes = proj.label_classes

        # Get target images
        q = db.query(DBImage).filter(DBImage.project_id == project_id)
        if image_ids:
            q = q.filter(DBImage.id.in_(image_ids))
        else:
            q = q.filter(DBImage.status == "unlabeled")
        images = list(q.all())

        if not images:
            return {"processed": 0, "skipped": 0, "total_detections": 0, "errors": []}

        if status_callback:
            status_callback(f"Loading model {model_path}...")
        model = load_yolo_model(model_path, device)

        paths = [img.file_path for img in images]
        if status_callback:
            status_callback(f"Running inference on {len(paths)} images...")

        batch_results = run_yolo_batch(
            paths, model, label_classes,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            batch_size=batch_size,
            class_mapping=class_mapping,
            progress_callback=progress_callback,
        )

        processed = 0
        total_dets = 0
        errors: list[str] = []

        for img in images:
            anns = batch_results.get(img.file_path, [])
            if not anns:
                continue
            # Set image_id on annotations
            for ann in anns:
                ann.image_id = img.id
            try:
                save_annotations(img.id, anns, "auto", db)
                img.auto_annotation_status = "done"
                processed += 1
                total_dets += len(anns)
            except Exception as exc:
                errors.append(f"{img.file_name}: {exc}")
                img.auto_annotation_status = "failed"

        db.commit()

        return {
            "processed": processed,
            "skipped": len(images) - processed,
            "total_detections": total_dets,
            "errors": errors,
        }
    finally:
        if close:
            db.close()
