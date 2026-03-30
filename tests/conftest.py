"""
tests/conftest.py — Shared fixtures and skip markers.

Fixture files expected in tests/fixtures/:
  sample_hero11.MP4          — GoPro Hero 11 file (user provides)
  sample_x4.insv             — Insta360 X4 file (user provides)
  sample_sensorlogger/       — Sensor Logger output folder (user provides)

Tests that require these files are automatically skipped when absent.
"""
import pytest
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def fixture_path(name: str) -> Path:
    return FIXTURES / name


def requires_fixture(name: str):
    """Skip test if fixture file/folder is absent."""
    return pytest.mark.skipif(
        not fixture_path(name).exists(),
        reason=f"Fixture not found: tests/fixtures/{name}",
    )


# Convenience marks
needs_gopro        = requires_fixture("sample_hero11.MP4")
needs_insta360     = requires_fixture("sample_x4.insv")
needs_sensorlogger = requires_fixture("sample_sensorlogger")
