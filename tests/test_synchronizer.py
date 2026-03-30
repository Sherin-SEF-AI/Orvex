"""tests/test_synchronizer.py — Unit tests for core/synchronizer.py"""
import math
import pytest
from core.synchronizer import (
    sync_by_gps, sync_by_xcorr, sync_manual, SyncResult,
)
from core.models import IMUSample, GPSSample, ExtractedSession, DeviceType

STEP_NS = 5_000_000  # 200 Hz


def _make_session(key: str, n: int = 400, imu_t0_ns: int = 0,
                  gps_t0_ns: int = 0) -> ExtractedSession:
    abs_start = imu_t0_ns // STEP_NS
    # Use a Gaussian pulse for accel_x so the signal is uniquely localised —
    # accel magnitude varies clearly and xcorr finds the correct unique lag.
    peak = n // 2
    imu = [
        IMUSample(
            timestamp_ns=imu_t0_ns + i * STEP_NS,
            accel_x=10.0 * math.exp(-((abs_start + i - (abs_start + peak)) ** 2) / (2 * 15 ** 2)),
            accel_y=0.0,
            accel_z=0.0,
            gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
        )
        for i in range(n)
    ]
    gps = [
        GPSSample(
            timestamp_ns=gps_t0_ns + i * 1_000_000_000,
            latitude=10.0 + i * 0.001, longitude=76.0,
            altitude_m=10.0, speed_mps=1.0, fix_type=3,
        )
        for i in range(5)
    ]
    return ExtractedSession(
        session_id=key, device_type=DeviceType.GOPRO,
        imu_samples=imu, gps_samples=gps,
        frame_paths=[], frame_timestamps_ns=[],
        duration_seconds=n * STEP_NS / 1e9, stats={},
    )


def test_sync_manual():
    sessions = {"a": _make_session("a"), "b": _make_session("b")}
    result = sync_manual(sessions, {"a": 0, "b": 50_000_000})
    assert result.method == "manual"
    assert result.offsets_ns["b"] == 50_000_000


def test_sync_manual_apply_shifts_timestamps():
    sessions = {"a": _make_session("a", imu_t0_ns=0),
                "b": _make_session("b", imu_t0_ns=0)}
    offset_ns = 100_000_000
    result = sync_manual(sessions, {"a": 0, "b": offset_ns})
    # apply() returns a new dict — it does not mutate in place
    shifted = result.apply(sessions)
    first_b = shifted["b"].imu_samples[0].timestamp_ns
    assert first_b == offset_ns


def test_sync_by_gps():
    t0_ref = 1_700_000_000_000_000_000
    sessions = {
        "gopro":  _make_session("gopro",  gps_t0_ns=t0_ref),
        "sensor": _make_session("sensor", gps_t0_ns=t0_ref + 200_000_000),
    }
    result = sync_by_gps(sessions, master_key="gopro")
    assert result.method == "gps"
    assert result.offsets_ns["gopro"] == 0
    assert result.offsets_ns["sensor"] == -200_000_000


def test_sync_by_xcorr_finds_offset():
    lag_ns = 10 * STEP_NS   # 50 ms
    sessions = {
        "ref":  _make_session("ref",  imu_t0_ns=0),
        "late": _make_session("late", imu_t0_ns=lag_ns),
    }
    result = sync_by_xcorr(sessions, reference_key="ref")
    assert result.method == "xcorr"
    # Tolerance: within ±1 sample (5 ms)
    assert abs(result.offsets_ns.get("late", 0)) <= STEP_NS


def test_sync_result_confidence():
    sessions = {"a": _make_session("a")}
    result = sync_manual(sessions, {"a": 0})
    assert 0.0 <= result.confidence <= 1.0
