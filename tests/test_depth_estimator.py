"""tests/test_depth_estimator.py — Unit tests for core/depth_estimator.py"""
import pytest
from core.depth_estimator import DepthResult


def test_depth_result_model():
    result = DepthResult(min_depth=0.5, max_depth=80.0, mean_depth=12.3)
    assert result.min_depth == 0.5
    assert result.max_depth == 80.0


@pytest.mark.skipif(True, reason="Requires GPU and depth model weights")
def test_load_depth_model():
    from core.depth_estimator import load_depth_model
    model = load_depth_model()
    assert model is not None
