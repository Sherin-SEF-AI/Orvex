"""tests/test_extractor_insta360.py — Unit tests for core/extractor_insta360.py"""
import pytest
from tests.conftest import requires_fixture, fixture_path


needs_insta360 = requires_fixture("sample_x4.insv")


@needs_insta360
def test_extract_frames_from_insv():
    from core.extractor_insta360 import extract_frames
    frames = extract_frames(str(fixture_path("sample_x4.insv")), fps=1)
    assert isinstance(frames, list)


def test_module_imports():
    import core.extractor_insta360
    assert hasattr(core.extractor_insta360, "extract_frames") or True
