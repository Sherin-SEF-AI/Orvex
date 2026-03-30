"""tests/test_insta360_processor.py — Unit tests for core/insta360_processor.py"""
import pytest
from core.insta360_processor import Insta360ProcessingConfig


def test_processing_config_defaults():
    cfg = Insta360ProcessingConfig()
    assert cfg is not None


def test_processing_config_custom():
    cfg = Insta360ProcessingConfig(extract_imu=True, extract_gps=False)
    assert cfg.extract_imu is True
    assert cfg.extract_gps is False
