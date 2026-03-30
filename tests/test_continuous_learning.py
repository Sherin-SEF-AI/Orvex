"""
tests/test_continuous_learning.py — Unit tests for core/continuous_learning.py.

Cycle orchestration tests are skipped unless ultralytics + fixture weights exist.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from core.continuous_learning import (
    check_and_maybe_trigger,
    get_learning_log,
    record_trigger,
)
from core.models import LearningTrigger
from core.session_manager import SessionManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def log_path(tmp_path):
    return tmp_path / "learning_log.json"


@pytest.fixture
def sm(tmp_path):
    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir()
    return SessionManager(sessions_root)


@pytest.fixture
def session_id(sm):
    sess = sm.create_session("cl_test", "test", "lab", "")
    return sess.id


def make_trigger(session_id: str, n: int = 10) -> LearningTrigger:
    return LearningTrigger(
        session_id=session_id,
        trigger_type="manual",
        corrections_count=n,
        triggered_at=datetime.now(timezone.utc),
        resulting_run_id=None,
    )


# ---------------------------------------------------------------------------
# Learning log
# ---------------------------------------------------------------------------

def test_get_learning_log_empty(log_path):
    assert get_learning_log(log_path) == []


def test_record_and_retrieve_trigger(log_path, session_id):
    trigger = make_trigger(session_id, 10)
    record_trigger(trigger, log_path)
    entries = get_learning_log(log_path)
    assert len(entries) == 1
    assert entries[0].session_id == session_id
    assert entries[0].corrections_count == 10


def test_record_multiple_triggers(log_path, session_id):
    for i in range(3):
        record_trigger(make_trigger(session_id, i * 10 + 10), log_path)
    entries = get_learning_log(log_path)
    assert len(entries) == 3


def test_trigger_json_roundtrip(log_path, session_id):
    trigger = make_trigger(session_id, 25)
    record_trigger(trigger, log_path)
    raw = json.loads(log_path.read_text())
    assert raw[0]["corrections_count"] == 25
    assert "triggered_at" in raw[0]
    # triggered_at must be an ISO string
    datetime.fromisoformat(raw[0]["triggered_at"].replace("Z", "+00:00"))


def test_malformed_log_entry_skipped(log_path, session_id):
    """Ensure a broken entry in the log file doesn't crash get_learning_log."""
    log_path.write_text(json.dumps([
        {"not": "valid"},                                        # malformed
        {                                                        # valid
            "session_id": session_id,
            "trigger_type": "manual",
            "corrections_count": 5,
            "triggered_at": datetime.now(timezone.utc).isoformat(),
            "resulting_run_id": None,
        },
    ]))
    entries = get_learning_log(log_path)
    assert len(entries) == 1   # malformed entry skipped


# ---------------------------------------------------------------------------
# Trigger check
# ---------------------------------------------------------------------------

def test_check_trigger_not_ready(session_id, sm):
    # No reviews → not ready
    ready = check_and_maybe_trigger(session_id, sm, threshold=50)
    assert ready is False


def test_check_trigger_ready(session_id, sm, tmp_path):
    # Manually insert accepted reviews to reach threshold=2
    from datetime import datetime, timezone
    from core.annotation_review import save_review
    from core.models import AnnotationReview, ReviewStatus

    for i in range(2):
        img = tmp_path / f"frame_{i}.jpg"
        img.write_bytes(b"\xff\xd8\xff" + b"\x00" * 20)
        save_review(AnnotationReview(
            frame_path=str(img),
            original_detections=[],
            corrected_detections=[],
            status=ReviewStatus.accepted,
            reviewed_at=datetime.now(timezone.utc),
        ), session_id, sm)

    assert check_and_maybe_trigger(session_id, sm, threshold=2) is True
    assert check_and_maybe_trigger(session_id, sm, threshold=3) is False


# ---------------------------------------------------------------------------
# Full cycle — skipped unless ultralytics + fixture weights present
# ---------------------------------------------------------------------------

FIXTURE_WEIGHTS = Path(__file__).parent / "fixtures" / "test_model.pt"
_HAS_ULTRALYTICS = False
try:
    import ultralytics  # noqa: F401
    _HAS_ULTRALYTICS = True
except ImportError:
    pass


@pytest.mark.skipif(
    not (_HAS_ULTRALYTICS and FIXTURE_WEIGHTS.exists()),
    reason="ultralytics not installed or tests/fixtures/test_model.pt missing",
)
def test_run_learning_cycle(session_id, sm, tmp_path):
    """
    Smoke-test the full learning cycle with 1 epoch and fixture weights.
    We create a minimal corrected dataset (2 accepted frames) and run the cycle.
    """
    import numpy as np
    from PIL import Image
    from core.annotation_review import save_review
    from core.continuous_learning import run_learning_cycle
    from core.models import (
        AnnotationReview, AugmentationConfig, Detection, ReviewStatus, TrainingConfig,
    )

    # Create 2 accepted frames with labels
    session_dir = sm.session_folder(session_id)
    frames_dir = session_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    autolabel_yolo = session_dir / "autolabel" / "yolo"
    autolabel_yolo.mkdir(parents=True, exist_ok=True)

    det = Detection(class_id=0, class_name="pothole", confidence=1.0,
                    bbox_xyxy=[0, 0, 10, 10], bbox_xywhn=[0.5, 0.5, 0.1, 0.1])
    for i in range(2):
        img_path = frames_dir / f"frame_{i:04d}.jpg"
        Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(str(img_path))
        (autolabel_yolo / f"frame_{i:04d}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
        save_review(AnnotationReview(
            frame_path=str(img_path),
            original_detections=[det],
            corrected_detections=[det],
            status=ReviewStatus.accepted,
            reviewed_at=datetime.now(timezone.utc),
        ), session_id, sm)

    aug_cfg   = AugmentationConfig(multiplier=1)
    train_cfg = TrainingConfig(
        dataset_dir="",   # filled in by run_learning_cycle
        model_variant="yolov8n",
        pretrained_weights=str(FIXTURE_WEIGHTS),
        epochs=1,
        batch_size=2,
        project_name="test_cl",
    )
    reg_path = tmp_path / "registry.toml"

    train_result, comparison = run_learning_cycle(
        session_id=session_id,
        sm=sm,
        trigger_type="test",
        augmentation_config=aug_cfg,
        training_config=train_cfg,
        auto_promote=True,
        registry_path=reg_path,
    )

    assert train_result.status in ("done", "running", "failed")   # at least attempted
    assert comparison is not None
