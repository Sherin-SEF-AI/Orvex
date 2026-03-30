"""tests/test_extractor_sensorlogger.py — Unit tests for core/extractor_sensorlogger.py"""
import csv
import json
import pytest
from pathlib import Path

from core.extractor_sensorlogger import extract_sensor_logger
from core.models import ExtractionConfig, DeviceType
from tests.conftest import needs_sensorlogger, fixture_path


def _write_csv(path: Path, rows: int = 100, ts_format: str = "iso") -> None:
    import datetime
    base = datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "accel_x", "accel_y", "accel_z",
                         "gyro_x", "gyro_y", "gyro_z",
                         "latitude", "longitude", "altitude"])
        for i in range(rows):
            if ts_format == "iso":
                ts = (base + datetime.timedelta(milliseconds=i * 5)).strftime(
                    "%Y-%m-%dT%H:%M:%S.%f"
                ) + "Z"
            elif ts_format == "epoch_s":
                ts = str(1700000000.0 + i * 0.005)
            else:
                ts = str(i * 5_000_000)   # nanoseconds
            writer.writerow([ts, 0.1 * i, 0.2, 9.81, 0.01, 0.02, 0.03,
                             10.5, 76.3, 15.0])


@pytest.fixture
def iso_csv(tmp_path):
    f = tmp_path / "sensor.csv"
    _write_csv(f, rows=200, ts_format="iso")
    return f


@pytest.fixture
def epoch_csv(tmp_path):
    f = tmp_path / "sensor.csv"
    _write_csv(f, rows=200, ts_format="epoch_s")
    return f


def test_extract_iso_timestamps(iso_csv, tmp_path):
    cfg = ExtractionConfig(session_id="s1")
    result = extract_sensor_logger(str(iso_csv), cfg, tmp_path / "out")
    assert result.device_type == DeviceType.SENSOR_LOGGER
    assert len(result.imu_samples) == 200
    # All timestamps must be int nanoseconds
    for s in result.imu_samples:
        assert isinstance(s.timestamp_ns, int)


def test_extract_epoch_timestamps(epoch_csv, tmp_path):
    cfg = ExtractionConfig(session_id="s1")
    result = extract_sensor_logger(str(epoch_csv), cfg, tmp_path / "out")
    assert len(result.imu_samples) == 200


def test_output_imu_csv_written(iso_csv, tmp_path):
    cfg = ExtractionConfig(session_id="s1")
    out = tmp_path / "out"
    extract_sensor_logger(str(iso_csv), cfg, out)
    imu_csv = out / "imu0" / "data.csv"
    assert imu_csv.exists()
    lines = imu_csv.read_text().splitlines()
    assert lines[0].startswith("#timestamp")


def test_extract_directory_input(tmp_path):
    """extract_sensor_logger should accept a directory containing a CSV."""
    csv_file = tmp_path / "Accelerometer.csv"
    _write_csv(csv_file, rows=50, ts_format="iso")
    cfg = ExtractionConfig(session_id="s1")
    out = tmp_path / "out"
    result = extract_sensor_logger(str(tmp_path), cfg, out)
    assert result.device_type == DeviceType.SENSOR_LOGGER


@needs_sensorlogger
def test_extract_sensorlogger_real(tmp_path):
    folder = fixture_path("sample_sensorlogger")
    cfg = ExtractionConfig(session_id="real_sl")
    result = extract_sensor_logger(str(folder), cfg, tmp_path / "out")
    assert len(result.imu_samples) > 0
