"""tests/test_lane_detector.py — Unit tests for core/lane_detector.py"""
import pytest
from core.lane_detector import LaneConfig, LaneFrame


def test_lane_config_defaults():
    cfg = LaneConfig()
    assert cfg is not None


def test_lane_frame_model():
    frame = LaneFrame(frame_index=0, lanes=[])
    assert frame.frame_index == 0
    assert frame.lanes == []
