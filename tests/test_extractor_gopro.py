"""tests/test_extractor_gopro.py — Unit tests for core/extractor_gopro.py"""
import pytest
from pathlib import Path

from core.extractor_gopro import _interp_xyz, _write_euroc_imu, _write_gps_csv
from core.models import IMUSample, GPSSample, ExtractionConfig, DeviceType
from tests.conftest import needs_gopro, fixture_path

_IMU_HEADER = (
    "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
    "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]"
)


def test_interp_xyz_exact():
    src_ts  = [0, 1_000_000_000, 2_000_000_000]
    src_xyz = [(0.0, 0.0, 0.0), (1.0, 2.0, 3.0), (2.0, 4.0, 6.0)]
    dst_ts  = [500_000_000, 1_500_000_000]
    result  = _interp_xyz(src_ts, src_xyz, dst_ts)
    assert len(result) == 2
    assert abs(result[0][0] - 0.5) < 1e-9
    assert abs(result[1][1] - 3.0) < 1e-9


def test_interp_xyz_clamp_extrapolation():
    src_ts  = [1_000_000_000, 2_000_000_000]
    src_xyz = [(1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
    # dst before first src → clamp to first
    result = _interp_xyz(src_ts, src_xyz, [0])
    assert result[0][0] == 1.0
    # dst after last src → clamp to last
    result = _interp_xyz(src_ts, src_xyz, [3_000_000_000])
    assert result[0][0] == 2.0


def test_write_euroc_imu_header(tmp_path):
    samples = [
        IMUSample(timestamp_ns=1_000_000_000,
                  accel_x=1.0, accel_y=2.0, accel_z=9.81,
                  gyro_x=0.1, gyro_y=0.2, gyro_z=0.3),
    ]
    out = tmp_path / "imu0" / "data.csv"
    out.parent.mkdir(parents=True)
    _write_euroc_imu(samples, out)
    lines = out.read_text().splitlines()
    assert lines[0] == _IMU_HEADER
    assert len(lines) == 2  # header + 1 data row


def test_write_gps_csv(tmp_path):
    samples = [
        GPSSample(timestamp_ns=1_000_000_000,
                  latitude=10.5, longitude=76.3, altitude_m=15.0,
                  speed_mps=2.0, fix_type=3),
    ]
    out = tmp_path / "gps.csv"
    _write_gps_csv(samples, out)
    text = out.read_text()
    assert "10.5" in text
    assert "76.3" in text


@needs_gopro
def test_extract_gopro_real(tmp_path):
    from core.extractor_gopro import extract_gopro
    config = ExtractionConfig(session_id="test", frame_fps=1.0)
    result = extract_gopro(
        str(fixture_path("sample_hero11.MP4")),
        config,
        tmp_path / "out",
    )
    assert result.device_type == DeviceType.GOPRO
    assert len(result.imu_samples) > 0
    assert result.duration_seconds > 0
