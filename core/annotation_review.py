"""
core/annotation_review.py — Human-in-the-loop annotation review.

Loads auto-label results per session, persists human corrections
(accept / reject / correct), and exports a corrected YOLO dataset
containing only accepted + corrected frames.

No UI imports — pure Python business logic.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from core.models import (
    AnnotationReview, AugmentationResult, Detection,
    FrameAnnotation, ReviewStatus,
)
from core.session_manager import SessionManager

# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def _reviews_path(session_id: str, sm: SessionManager) -> Path:
    return sm.session_folder(session_id) / "reviews" / "reviews.json"


def load_reviews(session_id: str, sm: SessionManager) -> list[AnnotationReview]:
    """Load all reviews for a session. Returns empty list if none saved yet."""
    path = _reviews_path(session_id, sm)
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    result = []
    for item in raw:
        try:
            result.append(AnnotationReview(**item))
        except Exception as exc:
            logger.warning(f"Skipping malformed review entry: {exc}")
    return result


def save_review(
    review: AnnotationReview,
    session_id: str,
    sm: SessionManager,
) -> None:
    """Persist a single review, replacing any existing entry for same frame_path."""
    reviews = load_reviews(session_id, sm)
    # Replace existing entry for this frame, or append
    replaced = False
    for i, r in enumerate(reviews):
        if r.frame_path == review.frame_path:
            reviews[i] = review
            replaced = True
            break
    if not replaced:
        reviews.append(review)
    _write_reviews(reviews, session_id, sm)


def save_reviews_bulk(
    reviews: list[AnnotationReview],
    session_id: str,
    sm: SessionManager,
) -> None:
    """Bulk-save a list of reviews, merging with any existing entries."""
    existing = {r.frame_path: r for r in load_reviews(session_id, sm)}
    for r in reviews:
        existing[r.frame_path] = r
    _write_reviews(list(existing.values()), session_id, sm)


def _write_reviews(
    reviews: list[AnnotationReview],
    session_id: str,
    sm: SessionManager,
) -> None:
    path = _reviews_path(session_id, sm)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = []
    for r in reviews:
        d = r.model_dump()
        if d.get("reviewed_at"):
            d["reviewed_at"] = d["reviewed_at"].isoformat()
        raw.append(d)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(raw, indent=2))
    tmp.replace(path)
    logger.debug(f"Saved {len(reviews)} review(s) for session {session_id}")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def get_review_stats(session_id: str, sm: SessionManager) -> dict:
    """
    Return counts by status and overall coverage.
    Also checks how many auto-labeled frames exist for this session.
    """
    reviews = load_reviews(session_id, sm)
    counts: dict[str, int] = {s.value: 0 for s in ReviewStatus}
    for r in reviews:
        counts[r.status.value] += 1

    # Total auto-labeled frames available for review
    autolabel_dir = sm.session_folder(session_id) / "autolabel"
    total_frames = 0
    if (autolabel_dir / "yolo").exists():
        total_frames = len(list((autolabel_dir / "yolo").glob("*.txt")))
    elif autolabel_dir.exists():
        ann_stats = autolabel_dir / "annotation_stats.json"
        if ann_stats.exists():
            total_frames = json.loads(ann_stats.read_text()).get("total_frames", 0)

    reviewed = len(reviews)
    coverage_pct = (reviewed / max(total_frames, 1)) * 100
    usable = counts["accepted"] + counts["corrected"]

    return {
        "total_frames": total_frames,
        "reviewed": reviewed,
        "pending": counts["pending"],
        "accepted": counts["accepted"],
        "corrected": counts["corrected"],
        "rejected": counts["rejected"],
        "coverage_percent": round(coverage_pct, 1),
        "usable_for_training": usable,
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_corrected_dataset(
    session_id: str,
    sm: SessionManager,
    output_dir: str,
    status_callback=None,
) -> AugmentationResult:
    """
    Export accepted + corrected frames to a YOLO-format dataset.
    Rejected frames are excluded. Original auto-labels are never modified.

    Output layout:
        output_dir/
            images/train/   ← JPEG frames
            labels/train/   ← YOLO .txt label files
            classes.txt
            dataset.yaml
    """
    import shutil
    from core.autolabel import ROVER_CLASSES

    reviews = load_reviews(session_id, sm)
    usable = [r for r in reviews if r.status in (ReviewStatus.accepted, ReviewStatus.corrected)]

    if not usable:
        raise RuntimeError(
            "No accepted or corrected frames to export.\n"
            "Review at least one frame first."
        )

    out = Path(output_dir)
    img_dir = out / "images" / "train"
    lbl_dir = out / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    if status_callback:
        status_callback(f"Exporting {len(usable)} reviewed frames…")

    copied = 0
    for review in usable:
        frame = Path(review.frame_path)
        if not frame.exists():
            logger.warning(f"Frame missing, skipping: {frame}")
            continue

        dest_img = img_dir / frame.name
        shutil.copy2(frame, dest_img)

        # Use corrected_detections (which equals original_detections for "accepted")
        dets = review.corrected_detections
        lbl_lines = []
        for det in dets:
            cx, cy, w, h = det.bbox_xywhn
            lbl_lines.append(f"{det.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        (lbl_dir / (frame.stem + ".txt")).write_text("\n".join(lbl_lines))
        copied += 1

    # Write classes.txt and dataset.yaml
    (out / "classes.txt").write_text("\n".join(ROVER_CLASSES))
    yaml_content = (
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/train\n"
        f"nc: {len(ROVER_CLASSES)}\n"
        f"names: {ROVER_CLASSES}\n"
    )
    (out / "dataset.yaml").write_text(yaml_content)

    logger.info(f"Exported corrected dataset: {copied} frames → {out}")
    if status_callback:
        status_callback(f"Export complete: {copied} frames → {output_dir}")

    return AugmentationResult(
        original_count=len(usable),
        augmented_count=copied,
        output_dir=str(out),
        per_transform_counts={"corrected": copied},
    )


# ---------------------------------------------------------------------------
# Continuous learning trigger
# ---------------------------------------------------------------------------

def check_learning_trigger(
    session_id: str,
    sm: SessionManager,
    threshold: int = 50,
) -> bool:
    """
    Return True if enough corrections have accumulated to justify retraining.
    Threshold counts accepted + corrected frames (not pending or rejected).
    """
    stats = get_review_stats(session_id, sm)
    usable = stats["usable_for_training"]
    logger.debug(
        f"Session {session_id}: {usable} usable reviews, threshold={threshold}"
    )
    return usable >= threshold
