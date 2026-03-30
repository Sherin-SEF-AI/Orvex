"""tests/test_tracker.py — Unit tests for core/tracker.py"""
import pytest
from core.tracker import TrackingResult, TrackingStats


def test_tracking_result_model():
    result = TrackingResult(
        track_id=1,
        detections=[],
        frames=[],
    )
    assert result.track_id == 1
    assert result.detections == []


def test_tracking_stats_model():
    stats = TrackingStats(
        total_tracks=10,
        avg_track_length=25.3,
        total_frames=500,
    )
    assert stats.total_tracks == 10
    assert stats.avg_track_length == 25.3
