"""
labelox/web/backend/routes/review.py — Review workflow endpoints.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from labelox.core.models import ReviewDecision
from labelox.web.backend.auth import get_current_user
from labelox.web.backend.response import err, ok

router = APIRouter()


class SubmitReviewRequest(BaseModel):
    image_id: str
    decision: str  # "approved", "rejected", "needs_work"
    comment: str | None = None


@router.get("/queue/{project_id}")
async def review_queue(project_id: str, user: dict = Depends(get_current_user)):
    from labelox.core.review_engine import get_review_queue
    queue = get_review_queue(project_id)
    return ok(queue)


@router.post("/submit")
async def submit_review(req: SubmitReviewRequest, user: dict = Depends(get_current_user)):
    from labelox.core.review_engine import submit_review as do_review
    try:
        decision = ReviewDecision(req.decision)
    except ValueError:
        return err(f"Invalid decision: {req.decision}")
    result = do_review(
        image_id=req.image_id,
        decision=decision,
        reviewer_id=user.get("sub", "anonymous"),
        comment=req.comment,
    )
    return ok(result)


@router.get("/history/{project_id}")
async def review_history(project_id: str, user: dict = Depends(get_current_user)):
    from labelox.core.review_engine import get_review_history
    history = get_review_history(project_id)
    return ok(history)


@router.get("/accuracy/{project_id}")
async def annotator_accuracy(project_id: str, user: dict = Depends(get_current_user)):
    from labelox.core.review_engine import compute_annotator_accuracy
    accuracy = compute_annotator_accuracy(project_id)
    return ok(accuracy)
