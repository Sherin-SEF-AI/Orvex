"""
labelox/core/review_engine.py — QA review workflow: queue, submit, accuracy.
"""
from __future__ import annotations

import random
from datetime import datetime
from typing import Any

from loguru import logger
from sqlalchemy import func
from sqlalchemy.orm import Session

from labelox.core.database import (
    DBAnnotation,
    DBImage,
    DBReviewHistory,
    get_session,
)
from labelox.core.models import Annotation, ReviewDecision


def get_review_queue(
    project_id: str,
    reviewer_name: str | None = None,
    sample_percent: float = 100.0,
    db: Session | None = None,
) -> list[DBImage]:
    """Get images awaiting review.

    Prioritises low-confidence auto-annotated images first.
    Excludes images already reviewed by this reviewer.
    """
    close = db is None
    if db is None:
        db = get_session()
    try:
        q = (
            db.query(DBImage)
            .filter(
                DBImage.project_id == project_id,
                DBImage.status == "annotated",
            )
        )

        # Exclude already reviewed by this reviewer
        if reviewer_name:
            reviewed_ids = (
                db.query(DBReviewHistory.image_id)
                .filter(DBReviewHistory.reviewer_name == reviewer_name)
                .subquery()
            )
            q = q.filter(~DBImage.id.in_(reviewed_ids))

        images = list(q.order_by(DBImage.file_name).all())

        # Sample if needed
        if sample_percent < 100.0 and images:
            k = max(1, int(len(images) * sample_percent / 100))
            images = random.sample(images, k)

        # Sort: auto-annotated with lowest confidence first
        def _sort_key(img: DBImage) -> float:
            min_conf = (
                db.query(func.min(DBAnnotation.confidence))
                .filter(
                    DBAnnotation.image_id == img.id,
                    DBAnnotation.is_auto.is_(True),
                    DBAnnotation.confidence.isnot(None),
                )
                .scalar()
            )
            return min_conf if min_conf is not None else 1.0

        images.sort(key=_sort_key)
        return images
    finally:
        if close:
            db.close()


def submit_review(
    image_id: str,
    decision: ReviewDecision,
    reviewer_name: str,
    comment: str | None = None,
    corrected_annotations: list[Annotation] | None = None,
    db: Session | None = None,
) -> None:
    """Record a review decision for an image."""
    close = db is None
    if db is None:
        db = get_session()
    try:
        img = db.get(DBImage, image_id)
        if img is None:
            raise ValueError(f"Image not found: {image_id}")

        # Update image status
        if decision == ReviewDecision.APPROVED:
            img.status = "reviewed"
            img.review_decision = "approved"
        elif decision == ReviewDecision.REJECTED:
            img.status = "rejected"
            img.review_decision = "rejected"
        elif decision == ReviewDecision.NEEDS_WORK:
            img.status = "in_progress"
            img.review_decision = "needs_work"

        img.review_comment = comment
        img.reviewed_by = reviewer_name
        img.updated_at = datetime.utcnow()

        # Save corrected annotations if provided
        if corrected_annotations is not None:
            from labelox.core.annotation_engine import save_annotations
            save_annotations(image_id, corrected_annotations, reviewer_name, db)

        # Record in history
        history = DBReviewHistory(
            image_id=image_id,
            reviewer_name=reviewer_name,
            decision=decision.value,
            comment=comment,
        )
        db.add(history)
        db.commit()

        logger.info("Review {} for image {} by {}", decision.value, image_id, reviewer_name)
    finally:
        if close:
            db.close()


def compute_annotator_accuracy(
    annotator_name: str,
    project_id: str,
    db: Session | None = None,
) -> dict[str, Any]:
    """Accuracy = approved / (approved + rejected) for this annotator."""
    close = db is None
    if db is None:
        db = get_session()
    try:
        # Get images annotated by this person that have been reviewed
        images = (
            db.query(DBImage)
            .filter(
                DBImage.project_id == project_id,
                DBImage.status.in_(["reviewed", "rejected"]),
            )
            .all()
        )

        # Filter to those with annotations by this annotator
        annotator_images = []
        for img in images:
            anns = (
                db.query(DBAnnotation)
                .filter(
                    DBAnnotation.image_id == img.id,
                    DBAnnotation.created_by == annotator_name,
                )
                .first()
            )
            if anns:
                annotator_images.append(img)

        approved = sum(1 for im in annotator_images if im.review_decision == "approved")
        rejected = sum(1 for im in annotator_images if im.review_decision == "rejected")
        total = approved + rejected

        return {
            "annotator_name": annotator_name,
            "total_reviewed": total,
            "approved": approved,
            "rejected": rejected,
            "accuracy_percent": (approved / total * 100) if total > 0 else None,
        }
    finally:
        if close:
            db.close()


def get_review_history(
    image_id: str,
    db: Session | None = None,
) -> list[DBReviewHistory]:
    """Get all review records for an image."""
    close = db is None
    if db is None:
        db = get_session()
    try:
        return list(
            db.query(DBReviewHistory)
            .filter(DBReviewHistory.image_id == image_id)
            .order_by(DBReviewHistory.created_at.desc())
            .all()
        )
    finally:
        if close:
            db.close()


def get_rejection_reasons(
    project_id: str,
    db: Session | None = None,
) -> dict[str, int]:
    """Parse review comments for common rejection patterns."""
    close = db is None
    if db is None:
        db = get_session()
    try:
        reviews = (
            db.query(DBReviewHistory)
            .join(DBImage, DBReviewHistory.image_id == DBImage.id)
            .filter(
                DBImage.project_id == project_id,
                DBReviewHistory.decision == "rejected",
                DBReviewHistory.comment.isnot(None),
            )
            .all()
        )

        reasons: dict[str, int] = {}
        for rev in reviews:
            comment = (rev.comment or "").lower().strip()
            if not comment:
                continue
            # Simple bucketing by first few words
            key = comment[:50]
            reasons[key] = reasons.get(key, 0) + 1

        return dict(sorted(reasons.items(), key=lambda x: x[1], reverse=True))
    finally:
        if close:
            db.close()
