"""tests/test_road_analytics.py — Unit tests for core/road_analytics.py"""
import pytest
from core.road_analytics import compute_class_distribution, SceneDiversityReport


def test_scene_diversity_report_model():
    report = SceneDiversityReport(
        total_frames=100,
        unique_scenes=5,
        class_counts={"car": 50, "person": 30},
    )
    assert report.total_frames == 100
    assert report.unique_scenes == 5


def test_compute_class_distribution_empty():
    result = compute_class_distribution([])
    assert isinstance(result, dict) or result is not None
