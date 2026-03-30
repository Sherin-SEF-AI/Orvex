"""tests/test_calibration.py — Unit tests for core/calibration.py"""
import pytest
from core.calibration import CalibrationResult


def test_calibration_result_model():
    result = CalibrationResult(
        fx=500.0, fy=500.0, cx=320.0, cy=240.0,
        distortion=[0.1, -0.2, 0.0, 0.0, 0.01],
        reprojection_error=0.3,
    )
    assert result.fx == 500.0
    assert result.reprojection_error == 0.3


def test_calibration_result_valid_reprojection():
    result = CalibrationResult(
        fx=500.0, fy=500.0, cx=320.0, cy=240.0,
        distortion=[], reprojection_error=0.1,
    )
    assert result.reprojection_error < 0.5
