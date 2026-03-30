"""
web/backend/routes/annotation_review.py — Human-in-the-loop review endpoints.
"""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel

from core.annotation_review import (
    export_corrected_dataset,
    get_review_stats,
    load_reviews,
    save_review,
    save_reviews_bulk,
)
from core.models import AnnotationReview, ReviewStatus
from core.session_manager import SessionManager

router = APIRouter(prefix="/review", tags=["annotation_review"])

# Module-level session manager — injected at app startup
_sm: SessionManager | None = None


def init_sm(sm: SessionManager) -> None:
    global _sm
    _sm = sm


def _get_sm() -> SessionManager:
    if _sm is None:
        raise RuntimeError("SessionManager not initialised")
    return _sm


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SaveReviewRequest(BaseModel):
    review: dict   # AnnotationReview serialised


class BulkReviewRequest(BaseModel):
    reviews: list[dict]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/{session_id}/frames")
def list_frames(session_id: str):
    try:
        reviews = load_reviews(session_id, _get_sm())
        return {"data": [r.model_dump() for r in reviews], "error": None}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/{session_id}/frames/{idx}")
def get_frame(session_id: str, idx: int):
    try:
        reviews = load_reviews(session_id, _get_sm())
        if idx < 0 or idx >= len(reviews):
            raise HTTPException(status_code=404, detail=f"Frame index {idx} out of range")
        return {"data": reviews[idx].model_dump(), "error": None}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/{session_id}/frames/{idx}")
def save_frame_review(session_id: str, idx: int, req: SaveReviewRequest):
    try:
        review = AnnotationReview(**req.review)
        save_review(review, session_id, _get_sm())
        return {"data": {"saved": True}, "error": None}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/{session_id}/bulk")
def save_bulk_reviews(session_id: str, req: BulkReviewRequest):
    try:
        reviews = [AnnotationReview(**r) for r in req.reviews]
        save_reviews_bulk(reviews, session_id, _get_sm())
        return {"data": {"saved": len(reviews)}, "error": None}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/{session_id}/export")
def export_dataset(session_id: str, output_dir: str = Query(...)):
    """Synchronous export (small datasets). For large sessions use Celery task."""
    try:
        result = export_corrected_dataset(session_id, _get_sm(), output_dir)
        return {"data": result.model_dump(), "error": None}
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/{session_id}/stats")
def review_stats(session_id: str):
    try:
        stats = get_review_stats(session_id, _get_sm())
        return {"data": stats, "error": None}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
