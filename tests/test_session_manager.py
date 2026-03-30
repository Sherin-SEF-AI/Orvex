"""tests/test_session_manager.py — Unit tests for core/session_manager.py"""
import datetime
import tempfile
import pytest
from pathlib import Path

from core.session_manager import SessionManager, SessionNotFoundError
from core.models import AuditResult, DeviceType


@pytest.fixture
def sm(tmp_path):
    return SessionManager(tmp_path)


def test_create_and_get(sm):
    s = sm.create_session(name="S1", environment="road", location="Loc", notes="n")
    assert s.name == "S1"
    fetched = sm.get_session(s.id)
    assert fetched.id == s.id


def test_list_sessions(sm):
    sm.create_session("A", "road", "X", "")
    sm.create_session("B", "indoor", "Y", "")
    assert len(sm.list_sessions()) == 2


def test_get_missing_raises(sm):
    with pytest.raises(SessionNotFoundError):
        sm.get_session("nonexistent")


def test_add_file(sm, tmp_path):
    f = tmp_path / "test.mp4"
    f.write_bytes(b"fake")
    s = sm.create_session("S", "road", "L", "")
    sm.add_file(s.id, str(f))
    assert str(f) in sm.get_session(s.id).files


def test_add_file_duplicate_warning(sm, tmp_path, caplog):
    f = tmp_path / "test.mp4"
    f.write_bytes(b"fake")
    s = sm.create_session("S", "road", "L", "")
    sm.add_file(s.id, str(f))
    sm.add_file(s.id, str(f))   # second add — should warn, not raise
    assert len(sm.get_session(s.id).files) == 1


def test_set_extraction_status(sm):
    s = sm.create_session("S", "road", "L", "")
    sm.set_extraction_status(s.id, "done")
    assert sm.get_session(s.id).extraction_status == "done"


def test_set_audit_results(sm):
    s = sm.create_session("S", "road", "L", "")
    r = AuditResult(
        file_path="/tmp/x.mp4", device_type=DeviceType.GOPRO,
        duration_seconds=10.0, has_imu=True, has_gps=False,
        imu_sample_count=2000, imu_rate_hz=200.0,
        gps_sample_count=0, gps_rate_hz=0.0,
        video_fps=60.0, video_resolution=(1920, 1080),
        file_size_mb=100.0, issues=[],
    )
    sm.set_audit_results(s.id, [r])
    assert len(sm.get_session(s.id).audit_results) == 1


def test_delete_session(sm):
    s = sm.create_session("S", "road", "L", "")
    sm.delete_session(s.id)
    with pytest.raises(SessionNotFoundError):
        sm.get_session(s.id)


def test_persistence(tmp_path):
    """Sessions persisted to disk survive a fresh SessionManager instance."""
    sm1 = SessionManager(tmp_path)
    s = sm1.create_session("Persist", "road", "X", "")
    sm2 = SessionManager(tmp_path)
    assert sm2.get_session(s.id).name == "Persist"


def test_session_folder(sm):
    s = sm.create_session("S", "road", "L", "")
    folder = sm.session_folder(s.id)
    assert folder.exists()
