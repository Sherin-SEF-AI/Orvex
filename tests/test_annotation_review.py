"""
tests/test_annotation_review.py — Unit tests for core/annotation_review.py.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from core.annotation_review import (
    check_learning_trigger,
    export_corrected_dataset,
    get_review_stats,
    load_reviews,
    save_review,
    save_reviews_bulk,
)
from core.models import AnnotationReview, Detection, ReviewStatus
from core.session_manager import SessionManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sm(tmp_path):
    """SessionManager pointing to a temporary directory."""
    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir()
    return SessionManager(sessions_root)


@pytest.fixture
def session_id(sm):
    """Create a real session and return its ID."""
    from datetime import datetime, timezone
    sess = sm.create_session(
        name="test_review_session",
        environment="test",
        location="lab",
        notes="",
    )
    return sess.id


@pytest.fixture
def sample_detection():
    return Detection(
        class_id=0,
        class_name="pothole",
        confidence=0.85,
        bbox_xyxy=[10.0, 20.0, 100.0, 80.0],
        bbox_xywhn=[0.5, 0.5, 0.2, 0.2],
    )


@pytest.fixture
def sample_review(sample_detection, tmp_path):
    # Create a dummy frame file
    img = tmp_path / "frame_0001.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)   # minimal JPEG header
    return AnnotationReview(
        frame_path=str(img),
        original_detections=[sample_detection],
        corrected_detections=[sample_detection],
        status=ReviewStatus.pending,
        reviewed_at=None,
    )


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------

def test_load_reviews_empty(session_id, sm):
    assert load_reviews(session_id, sm) == []


def test_save_and_load_review(session_id, sm, sample_review):
    save_review(sample_review, session_id, sm)
    reviews = load_reviews(session_id, sm)
    assert len(reviews) == 1
    assert reviews[0].frame_path == sample_review.frame_path
    assert reviews[0].status == ReviewStatus.pending


def test_save_review_replaces_existing(session_id, sm, sample_review):
    save_review(sample_review, session_id, sm)
    # Update status
    updated = sample_review.model_copy(update={"status": ReviewStatus.accepted})
    save_review(updated, session_id, sm)
    reviews = load_reviews(session_id, sm)
    assert len(reviews) == 1
    assert reviews[0].status == ReviewStatus.accepted


def test_save_reviews_bulk(session_id, sm, tmp_path):
    reviews = []
    for i in range(3):
        img = tmp_path / f"frame_{i:04d}.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        reviews.append(AnnotationReview(
            frame_path=str(img),
            original_detections=[],
            corrected_detections=[],
            status=ReviewStatus.pending,
        ))
    save_reviews_bulk(reviews, session_id, sm)
    loaded = load_reviews(session_id, sm)
    assert len(loaded) == 3


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def test_review_stats_empty(session_id, sm):
    stats = get_review_stats(session_id, sm)
    assert stats["reviewed"] == 0
    assert stats["usable_for_training"] == 0


def test_review_stats_counts(session_id, sm, tmp_path):
    statuses = [ReviewStatus.accepted, ReviewStatus.corrected, ReviewStatus.rejected, ReviewStatus.pending]
    for i, st in enumerate(statuses):
        img = tmp_path / f"f{i}.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 50)
        save_review(AnnotationReview(
            frame_path=str(img),
            original_detections=[],
            corrected_detections=[],
            status=st,
            reviewed_at=datetime.now(timezone.utc) if st != ReviewStatus.pending else None,
        ), session_id, sm)

    stats = get_review_stats(session_id, sm)
    assert stats["accepted"] == 1
    assert stats["corrected"] == 1
    assert stats["rejected"] == 1
    assert stats["pending"] == 1
    assert stats["usable_for_training"] == 2


# ---------------------------------------------------------------------------
# Learning trigger
# ---------------------------------------------------------------------------

def test_check_learning_trigger_not_ready(session_id, sm):
    assert check_learning_trigger(session_id, sm, threshold=50) is False


def test_check_learning_trigger_ready(session_id, sm, tmp_path):
    # Add 5 accepted reviews, threshold=5
    for i in range(5):
        img = tmp_path / f"acc_{i}.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 50)
        save_review(AnnotationReview(
            frame_path=str(img),
            original_detections=[],
            corrected_detections=[],
            status=ReviewStatus.accepted,
            reviewed_at=datetime.now(timezone.utc),
        ), session_id, sm)
    assert check_learning_trigger(session_id, sm, threshold=5) is True
    assert check_learning_trigger(session_id, sm, threshold=6) is False


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def test_export_raises_when_no_usable(session_id, sm, tmp_path):
    with pytest.raises(RuntimeError, match="No accepted or corrected"):
        export_corrected_dataset(session_id, sm, str(tmp_path / "out"))


def test_export_writes_yolo_dataset(session_id, sm, tmp_path):
    """Export 2 accepted frames — verify dataset.yaml and label files are created."""
    from PIL import Image
    import numpy as np

    out_dir = tmp_path / "out"
    accepted = []
    for i in range(2):
        img_path = tmp_path / f"frame_{i:04d}.jpg"
        Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(str(img_path))
        det = Detection(
            class_id=0, class_name="pothole", confidence=1.0,
            bbox_xyxy=[0, 0, 10, 10],
            bbox_xywhn=[0.5, 0.5, 0.1, 0.1],
        )
        review = AnnotationReview(
            frame_path=str(img_path),
            original_detections=[det],
            corrected_detections=[det],
            status=ReviewStatus.accepted,
            reviewed_at=datetime.now(timezone.utc),
        )
        save_review(review, session_id, sm)

    result = export_corrected_dataset(session_id, sm, str(out_dir))
    assert result.augmented_count == 2
    assert (out_dir / "dataset.yaml").exists()
    assert (out_dir / "classes.txt").exists()
    lbl_dir = out_dir / "labels" / "train"
    assert lbl_dir.exists()
    assert len(list(lbl_dir.glob("*.txt"))) == 2
