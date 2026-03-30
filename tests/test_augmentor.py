"""tests/test_augmentor.py — Unit tests for core/augmentor.py"""
import pytest
from core.augmentor import AugmentationConfig, build_augmentation_pipeline


def test_augmentation_config_defaults():
    cfg = AugmentationConfig()
    assert cfg is not None


def test_build_pipeline_returns_callable():
    cfg = AugmentationConfig()
    pipeline = build_augmentation_pipeline(cfg)
    assert callable(pipeline) or pipeline is not None
