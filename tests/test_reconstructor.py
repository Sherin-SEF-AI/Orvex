"""tests/test_reconstructor.py — Unit tests for core/reconstructor.py"""
import pytest
from core.reconstructor import ColmapResult


def test_colmap_result_model():
    result = ColmapResult(
        num_images=50,
        num_points=10000,
        mean_reprojection_error=0.8,
    )
    assert result.num_images == 50
    assert result.num_points == 10000


@pytest.mark.skipif(True, reason="Requires COLMAP installation")
def test_run_colmap_sfm():
    from core.reconstructor import run_colmap_sfm
    assert callable(run_colmap_sfm)
