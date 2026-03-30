"""tests/test_dataset_builder.py — Unit tests for core/dataset_builder.py"""
import csv
import pytest
from pathlib import Path

from core.dataset_builder import (
    build_dataset, DatasetBuildResult,
    _read_euroc_imu_rows, _write_euroc_imu_rows,
    _IMU_HEADER,
)
from core.session_manager import SessionManager


@pytest.fixture
def sm(tmp_path):
    return SessionManager(tmp_path / "sessions")


@pytest.fixture
def session(sm):
    return sm.create_session("TestSess", "road", "Loc", "")


def test_build_empty_session_euroc(sm, session, tmp_path):
    result = build_dataset([session.id], sm, "euroc", tmp_path / "out")
    assert isinstance(result, DatasetBuildResult)
    assert result.total_frames == 0
    assert result.total_imu_samples == 0
    assert result.manifest_path.exists()
    assert result.format == "euroc"


def test_build_no_sessions_raises(sm, tmp_path):
    with pytest.raises(ValueError, match="No session IDs"):
        build_dataset([], sm, "euroc", tmp_path / "out")


def test_build_unknown_format_raises(sm, session, tmp_path):
    with pytest.raises(ValueError, match="Unknown output_format"):
        build_dataset([session.id], sm, "badformat", tmp_path / "out")


def test_build_session_skipped_if_no_extraction(sm, session, tmp_path):
    result = build_dataset([session.id], sm, "euroc", tmp_path / "out")
    assert len(result.warnings) >= 1
    assert any("no extracted data" in w for w in result.warnings)


def test_euroc_imu_roundtrip(tmp_path):
    csv_path = tmp_path / "imu.csv"
    rows = [["1000000000", "0.1", "0.2", "0.3", "1.0", "2.0", "3.0"]]
    _write_euroc_imu_rows(csv_path, rows)
    # Check header is exactly correct
    lines = csv_path.read_text().splitlines()
    assert lines[0] == _IMU_HEADER
    read_back = _read_euroc_imu_rows(csv_path)
    assert read_back == rows


def test_build_with_extraction_output(sm, session, tmp_path):
    """Simulate an existing extraction output dir and verify frames are copied."""
    # Set up fake extraction tree
    ext_dir = sm.session_folder(session.id) / "extraction_gopro"
    cam_dir = ext_dir / "cam0" / "data"
    imu_dir = ext_dir / "imu0"
    cam_dir.mkdir(parents=True)
    imu_dir.mkdir(parents=True)

    # Write 3 fake frames
    for ts in [1000000000, 1200000000, 1400000000]:
        (cam_dir / f"{ts}.jpg").write_bytes(b"\xff\xd8\xff\xe0fake_jpeg")

    # Write EuRoC IMU CSV
    rows = [[str(1000000000 + i * 5000000)] + ["0.0"] * 6 for i in range(10)]
    _write_euroc_imu_rows(imu_dir / "data.csv", rows)

    out = tmp_path / "dataset"
    result = build_dataset([session.id], sm, "euroc", out)

    assert result.total_frames == 3
    assert result.total_imu_samples == 10
    assert result.manifest_path.exists()

    # Verify frames exist in output
    frame_dir = out / session.id / "cam0" / "data"
    assert len(list(frame_dir.glob("*.jpg"))) == 3

    # Verify IMU CSV header is EuRoC-exact
    imu_out = out / session.id / "imu0" / "data.csv"
    assert imu_out.read_text().splitlines()[0] == _IMU_HEADER
