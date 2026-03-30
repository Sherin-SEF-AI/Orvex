"""tests/test_models.py — Unit tests for core/models.py"""
import datetime
import pytest
from core.models import (
    DeviceType, IMUSample, GPSSample, AuditResult,
    Session, CalibrationSession, ExtractionConfig, ExtractedSession,
)


def test_device_type_values():
    assert DeviceType.GOPRO == "gopro"
    assert DeviceType.INSTA360 == "insta360"
    assert DeviceType.SENSOR_LOGGER == "sensor_logger"


def test_imu_sample_fields():
    s = IMUSample(timestamp_ns=1_000_000_000,
                  accel_x=1.0, accel_y=2.0, accel_z=3.0,
                  gyro_x=0.1, gyro_y=0.2, gyro_z=0.3)
    assert s.timestamp_ns == 1_000_000_000
    assert isinstance(s.timestamp_ns, int)
    assert s.accel_x == 1.0


def test_gps_sample_fields():
    g = GPSSample(timestamp_ns=2_000_000_000,
                  latitude=10.5, longitude=76.3, altitude_m=15.0,
                  speed_mps=5.0, fix_type=3)
    assert g.fix_type == 3


def test_audit_result_issues_default():
    r = AuditResult(
        file_path="/tmp/x.mp4", device_type=DeviceType.GOPRO,
        duration_seconds=60.0, has_imu=True, has_gps=True,
        imu_sample_count=12000, imu_rate_hz=200.0,
        gps_sample_count=60, gps_rate_hz=1.0,
        video_fps=59.94, video_resolution=(3840, 2160),
        file_size_mb=500.0, issues=[],
    )
    assert r.issues == []
    assert r.video_resolution == (3840, 2160)


def test_session_defaults():
    import uuid
    s = Session(
        id=str(uuid.uuid4()), name="Test",
        created_at=datetime.datetime.now(datetime.timezone.utc),
        environment="road", location="Loc",
        files=[], audit_results=[],
        extraction_status="pending", notes="",
    )
    assert s.extraction_status == "pending"
    assert s.files == []


def test_calibration_session_nullable_error():
    c = CalibrationSession(
        id="cal1", camera_device=DeviceType.GOPRO,
        session_type="camera_intrinsic",
        file_path="/tmp/calib.mp4",
        status="pending", results={},
        reprojection_error_px=None,
    )
    assert c.reprojection_error_px is None


def test_extraction_config_defaults():
    cfg = ExtractionConfig(session_id="abc")
    assert cfg.frame_fps == 5.0
    assert cfg.output_format == "euroc"
    assert cfg.sync_devices is True


def test_extracted_session():
    e = ExtractedSession(
        session_id="s1", device_type=DeviceType.GOPRO,
        imu_samples=[], gps_samples=[],
        frame_paths=[], frame_timestamps_ns=[],
        duration_seconds=0.0, stats={},
    )
    assert e.imu_samples == []
