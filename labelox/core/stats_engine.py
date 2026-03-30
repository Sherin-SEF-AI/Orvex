"""
labelox/core/stats_engine.py — Project and annotator statistics.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any

from loguru import logger
from sqlalchemy import func
from sqlalchemy.orm import Session

from labelox.core.database import (
    DBAnnotation,
    DBAnnotationSession,
    DBImage,
    DBProject,
    DBReviewHistory,
    get_session,
)
from labelox.core.models import AnnotatorStats, ProjectStats


def compute_project_stats(
    project_id: str,
    db: Session | None = None,
) -> ProjectStats:
    """Compute comprehensive project statistics."""
    close = db is None
    if db is None:
        db = get_session()
    try:
        proj = db.get(DBProject, project_id)
        if proj is None:
            raise ValueError(f"Project not found: {project_id}")

        total = db.query(func.count(DBImage.id)).filter(
            DBImage.project_id == project_id,
        ).scalar() or 0

        by_status: dict[str, int] = {}
        for status in ("unlabeled", "in_progress", "annotated", "reviewed", "rejected", "skipped"):
            by_status[status] = db.query(func.count(DBImage.id)).filter(
                DBImage.project_id == project_id,
                DBImage.status == status,
            ).scalar() or 0

        total_ann = db.query(func.count(DBAnnotation.id)).join(DBImage).filter(
            DBImage.project_id == project_id,
        ).scalar() or 0

        auto_count = db.query(func.count(DBAnnotation.id)).join(DBImage).filter(
            DBImage.project_id == project_id,
            DBAnnotation.is_auto.is_(True),
        ).scalar() or 0

        # Per-class counts
        class_rows = (
            db.query(DBAnnotation.label_name, func.count(DBAnnotation.id))
            .join(DBImage)
            .filter(DBImage.project_id == project_id)
            .group_by(DBAnnotation.label_name)
            .all()
        )
        per_class = {name: count for name, count in class_rows}

        # Per-annotator stats
        annotator_rows = (
            db.query(DBAnnotation.created_by, func.count(DBAnnotation.id))
            .join(DBImage)
            .filter(
                DBImage.project_id == project_id,
                DBAnnotation.created_by != "",
                DBAnnotation.created_by != "auto",
            )
            .group_by(DBAnnotation.created_by)
            .all()
        )
        annotator_stats = []
        for name, ann_count in annotator_rows:
            img_count = (
                db.query(func.count(func.distinct(DBAnnotation.image_id)))
                .join(DBImage)
                .filter(
                    DBImage.project_id == project_id,
                    DBAnnotation.created_by == name,
                )
                .scalar() or 0
            )

            # Average time per image from annotation sessions
            avg_time = (
                db.query(func.avg(DBAnnotationSession.duration_seconds))
                .filter(DBAnnotationSession.annotator_name == name)
                .join(DBImage, DBAnnotationSession.image_id == DBImage.id)
                .filter(DBImage.project_id == project_id)
                .scalar()
            )

            annotator_stats.append(AnnotatorStats(
                annotator_name=name,
                images_annotated=img_count,
                annotations_created=ann_count,
                avg_time_per_image_seconds=float(avg_time) if avg_time else 0.0,
            ))

        annotated = by_status.get("annotated", 0)
        reviewed = by_status.get("reviewed", 0)
        completion = (annotated + reviewed) / total * 100 if total > 0 else 0

        return ProjectStats(
            project_id=project_id,
            total_images=total,
            annotated_images=annotated,
            reviewed_images=reviewed,
            approved_images=reviewed,  # reviewed = approved in our flow
            rejected_images=by_status.get("rejected", 0),
            total_annotations=total_ann,
            annotations_per_class=per_class,
            auto_annotation_percent=(auto_count / total_ann * 100) if total_ann > 0 else 0,
            completion_percent=completion,
            annotators=annotator_stats,
        )
    finally:
        if close:
            db.close()


def compute_class_balance(
    project_id: str,
    db: Session | None = None,
) -> dict[str, int]:
    """Per-class annotation counts. Useful for detecting class imbalance."""
    close = db is None
    if db is None:
        db = get_session()
    try:
        rows = (
            db.query(DBAnnotation.label_name, func.count(DBAnnotation.id))
            .join(DBImage)
            .filter(DBImage.project_id == project_id)
            .group_by(DBAnnotation.label_name)
            .order_by(func.count(DBAnnotation.id).desc())
            .all()
        )
        return {name: count for name, count in rows}
    finally:
        if close:
            db.close()


def compute_annotation_quality_metrics(
    project_id: str,
    db: Session | None = None,
) -> dict[str, Any]:
    """Quality metrics: avg annotations per image, flagged count, etc."""
    close = db is None
    if db is None:
        db = get_session()
    try:
        total_images = db.query(func.count(DBImage.id)).filter(
            DBImage.project_id == project_id,
            DBImage.status.in_(["annotated", "reviewed"]),
        ).scalar() or 0

        total_anns = db.query(func.count(DBAnnotation.id)).join(DBImage).filter(
            DBImage.project_id == project_id,
        ).scalar() or 0

        avg_per_image = total_anns / total_images if total_images > 0 else 0

        # Images with very few annotations (potential misses)
        sparse_count = 0
        if total_images > 0:
            images = (
                db.query(DBImage.id, func.count(DBAnnotation.id).label("ann_count"))
                .outerjoin(DBAnnotation)
                .filter(
                    DBImage.project_id == project_id,
                    DBImage.status.in_(["annotated", "reviewed"]),
                )
                .group_by(DBImage.id)
                .having(func.count(DBAnnotation.id) < max(1, avg_per_image * 0.3))
                .all()
            )
            sparse_count = len(images)

        # Blurry annotated images
        blurry_count = db.query(func.count(DBImage.id)).filter(
            DBImage.project_id == project_id,
            DBImage.status.in_(["annotated", "reviewed"]),
            DBImage.blur_score.isnot(None),
            DBImage.blur_score < 50.0,
        ).scalar() or 0

        # Rejection rate
        rejected = db.query(func.count(DBImage.id)).filter(
            DBImage.project_id == project_id,
            DBImage.status == "rejected",
        ).scalar() or 0

        reviewed = db.query(func.count(DBImage.id)).filter(
            DBImage.project_id == project_id,
            DBImage.status.in_(["reviewed", "rejected"]),
        ).scalar() or 0

        return {
            "avg_annotations_per_image": round(avg_per_image, 1),
            "sparse_images": sparse_count,
            "blurry_annotated_images": blurry_count,
            "rejection_rate": (rejected / reviewed * 100) if reviewed > 0 else 0,
            "total_annotated": total_images,
            "total_annotations": total_anns,
        }
    finally:
        if close:
            db.close()


def compute_daily_progress(
    project_id: str,
    days: int = 30,
    db: Session | None = None,
) -> list[dict[str, Any]]:
    """Annotations created per day for the last N days."""
    from datetime import timedelta

    close = db is None
    if db is None:
        db = get_session()
    try:
        cutoff = datetime.utcnow() - timedelta(days=days)
        rows = (
            db.query(
                func.date(DBAnnotation.created_at).label("day"),
                func.count(DBAnnotation.id).label("count"),
            )
            .join(DBImage)
            .filter(
                DBImage.project_id == project_id,
                DBAnnotation.created_at >= cutoff,
            )
            .group_by(func.date(DBAnnotation.created_at))
            .order_by(func.date(DBAnnotation.created_at))
            .all()
        )
        return [{"date": str(row.day), "count": row.count} for row in rows]
    finally:
        if close:
            db.close()
