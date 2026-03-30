"""tests/test_segmentation.py — Unit tests for core/segmentation.py"""
import pytest
from core.segmentation import SegmentationResult


def test_segmentation_result_model():
    result = SegmentationResult(
        frame_path="/tmp/frame.jpg",
        classes=["road", "sidewalk"],
        pixel_counts={"road": 50000, "sidewalk": 10000},
    )
    assert result.frame_path == "/tmp/frame.jpg"
    assert "road" in result.classes


@pytest.mark.skipif(True, reason="Requires segmentation model weights")
def test_run_segmentation():
    from core.segmentation import run_segmentation
    assert callable(run_segmentation)
