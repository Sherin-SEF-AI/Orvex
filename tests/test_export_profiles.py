"""tests/test_export_profiles.py — Unit tests for core/export_profiles.py"""
import pytest
from core.export_profiles import ExportProfile


def test_export_profile_creation():
    profile = ExportProfile(name="yolo_default", format="yolo")
    assert profile.name == "yolo_default"
    assert profile.format == "yolo"


def test_export_profile_has_name():
    profile = ExportProfile(name="coco_test", format="coco")
    assert hasattr(profile, "name")
    assert hasattr(profile, "format")
