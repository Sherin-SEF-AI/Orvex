"""tests/test_audit.py — Unit tests for core/audit.py"""
import pytest
import csv
import json
from pathlib import Path

from core.audit import audit_file, audit_sensor_logger, _GOPRO_CHAPTER_RE
from core.models import DeviceType
from tests.conftest import needs_gopro, needs_insta360, needs_sensorlogger, fixture_path


# ---------------------------------------------------------------------------
# Chapter regex
# ---------------------------------------------------------------------------

def test_chapter_regex_matches_gopro():
    assert _GOPRO_CHAPTER_RE.match("GH010001.MP4")
    assert _GOPRO_CHAPTER_RE.match("GH020001.MP4")
    assert _GOPRO_CHAPTER_RE.match("GX010001.MP4")


def test_chapter_regex_no_match_regular():
    assert _GOPRO_CHAPTER_RE.match("video.mp4") is None
    assert _GOPRO_CHAPTER_RE.match("GH010001.MOV") is None


# ---------------------------------------------------------------------------
# Sensor Logger (synthetic CSV)
# ---------------------------------------------------------------------------

def _write_sensor_csv(path: Path, rows=200) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "accel_x", "accel_y", "accel_z",
                         "gyro_x", "gyro_y", "gyro_z",
                         "latitude", "longitude", "altitude"])
        for i in range(rows):
            t = f"2024-01-01T00:00:{i//60:02d}.{(i%60)*1000:06d}Z" if i < 60 else \
                f"2024-01-01T00:00:{i:02d}.000000Z"
            writer.writerow([
                f"2024-01-01T00:00:{i:06.3f}Z",
                0.1 * i, 0.2, 9.81,
                0.01, 0.02, 0.03,
                10.5, 76.3, 15.0,
            ])


def test_audit_sensor_logger_synthetic(tmp_path):
    csv_file = tmp_path / "sensor_data.csv"
    _write_sensor_csv(csv_file, rows=200)
    result = audit_sensor_logger(str(csv_file))
    assert result.device_type == DeviceType.SENSOR_LOGGER
    assert result.has_imu
    assert result.imu_sample_count == 200
    assert result.file_path == str(csv_file)


def test_audit_file_dispatches_csv(tmp_path):
    csv_file = tmp_path / "data.csv"
    _write_sensor_csv(csv_file, rows=50)
    result = audit_file(str(csv_file))
    assert result.device_type == DeviceType.SENSOR_LOGGER


# ---------------------------------------------------------------------------
# Real fixture tests (skipped if files absent)
# ---------------------------------------------------------------------------

@needs_gopro
def test_audit_gopro_real():
    result = audit_file(str(fixture_path("sample_hero11.MP4")))
    assert result.device_type == DeviceType.GOPRO
    assert result.duration_seconds > 0
    assert result.video_fps > 0
    assert result.video_resolution[0] > 0


@needs_insta360
def test_audit_insta360_real():
    result = audit_file(str(fixture_path("sample_x4.insv")))
    assert result.device_type == DeviceType.INSTA360
    assert result.duration_seconds > 0


@needs_sensorlogger
def test_audit_sensorlogger_real():
    folder = fixture_path("sample_sensorlogger")
    csv_files = list(folder.glob("*.csv")) + list(folder.glob("*.json"))
    assert csv_files, "No CSV/JSON files in sample_sensorlogger/"
    result = audit_file(str(csv_files[0]))
    assert result.device_type == DeviceType.SENSOR_LOGGER
    assert result.imu_sample_count > 0
