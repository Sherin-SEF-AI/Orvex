"""tests/test_utils.py — Unit tests for core/utils.py"""
import pytest
from pathlib import Path
from core.utils import file_size_mb, MissingToolError, check_dependencies


def test_file_size_mb(tmp_path):
    f = tmp_path / "sample.bin"
    f.write_bytes(b"x" * 1024 * 1024)  # 1 MB
    assert abs(file_size_mb(str(f)) - 1.0) < 0.01


def test_file_size_mb_zero(tmp_path):
    f = tmp_path / "empty.bin"
    f.write_bytes(b"")
    assert file_size_mb(str(f)) == 0.0


def test_check_dependencies_returns_dict():
    """check_dependencies() either returns a dict or raises MissingToolError."""
    try:
        result = check_dependencies()
        assert isinstance(result, dict)
        assert "ffmpeg" in result
    except MissingToolError as e:
        assert "ffmpeg" in str(e).lower() or "ffprobe" in str(e).lower() or "exiftool" in str(e).lower()


def test_missing_tool_error_is_runtime_error():
    err = MissingToolError("test tool missing")
    assert isinstance(err, RuntimeError)
