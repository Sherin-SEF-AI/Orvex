"""tests/test_slam_validator.py — Unit tests for core/slam_validator.py"""
import pytest
from core.slam_validator import check_slam_installation, SLAMResult


def test_check_slam_installation():
    result = check_slam_installation()
    assert isinstance(result, (bool, dict))


def test_slam_result_model():
    result = SLAMResult(
        trajectory_length=125.5,
        num_keyframes=200,
        loop_closures=3,
    )
    assert result.trajectory_length == 125.5
    assert result.num_keyframes == 200
