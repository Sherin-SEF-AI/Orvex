"""tests/test_occupancy.py — Unit tests for core/occupancy.py"""
import pytest
from core.occupancy import OccupancyConfig, OccupancyFrame


def test_occupancy_config_defaults():
    cfg = OccupancyConfig()
    assert cfg is not None


def test_occupancy_frame_model():
    frame = OccupancyFrame(frame_index=0, grid=[])
    assert frame.frame_index == 0
    assert frame.grid == []
