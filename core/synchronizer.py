"""
core/synchronizer.py — Multi-device temporal alignment.

Three synchronisation methods:

  Method 1 — GPS time anchor
    Use GoPro GPS timestamps (UTC) as master clock.
    Map Sensor Logger GPS timestamps onto it.  Compute per-device offset.

  Method 2 — IMU cross-correlation
    Cross-correlate acceleration magnitude signals between devices.
    Finds the lag that maximises correlation.  Works without GPS.
    If computed lag > 2 s, emits a warning — likely a sync failure.

  Method 3 — Manual offset
    User supplies offset_ns for each device explicitly.

All offsets are in nanoseconds (int).
Output: SyncResult with per-device offset_ns applied to their IMU/GPS samples.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from loguru import logger

from core.models import DeviceType, ExtractedSession, GPSSample, IMUSample

# Maximum cross-correlation lag before a warning is emitted (seconds)
_MAX_SAFE_LAG_S = 2.0


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SyncResult:
    """Outcome of a synchronisation run."""
    method: Literal["gps", "xcorr", "manual"]
    offsets_ns: dict[str, int]          # device_key → offset_ns
    confidence: float                   # 0.0–1.0 (1.0 = GPS anchor, xcorr peak ratio)
    lag_warning: bool = False           # True if any lag > _MAX_SAFE_LAG_S
    notes: list[str] = field(default_factory=list)

    def apply(self, sessions: dict[str, ExtractedSession]) -> dict[str, ExtractedSession]:
        """Return new ExtractedSession instances with timestamps shifted by offset_ns."""
        result: dict[str, ExtractedSession] = {}
        for key, session in sessions.items():
            offset = self.offsets_ns.get(key, 0)
            if offset == 0:
                result[key] = session
                continue
            shifted_imu = [
                IMUSample(
                    timestamp_ns=s.timestamp_ns + offset,
                    accel_x=s.accel_x, accel_y=s.accel_y, accel_z=s.accel_z,
                    gyro_x=s.gyro_x, gyro_y=s.gyro_y, gyro_z=s.gyro_z,
                )
                for s in session.imu_samples
            ]
            shifted_gps = [
                GPSSample(
                    timestamp_ns=s.timestamp_ns + offset,
                    latitude=s.latitude, longitude=s.longitude,
                    altitude_m=s.altitude_m, speed_mps=s.speed_mps,
                    fix_type=s.fix_type,
                )
                for s in session.gps_samples
            ]
            result[key] = ExtractedSession(
                session_id=session.session_id,
                device_type=session.device_type,
                imu_samples=shifted_imu,
                gps_samples=shifted_gps,
                frame_paths=session.frame_paths,
                frame_timestamps_ns=[t + offset for t in session.frame_timestamps_ns],
                duration_seconds=session.duration_seconds,
                stats=session.stats,
            )
        return result


# ---------------------------------------------------------------------------
# Method 1 — GPS anchor
# ---------------------------------------------------------------------------

def sync_by_gps(
    sessions: dict[str, ExtractedSession],
    master_key: str,
) -> SyncResult:
    """Compute per-device offsets using GPS timestamps as the master clock.

    The device named *master_key* is treated as the reference (offset = 0).
    All other devices with GPS data have their first GPS timestamp aligned
    to the master's first GPS timestamp.

    Args:
        sessions:   Dict of device_key → ExtractedSession.
        master_key: Key of the master (reference) device.

    Returns:
        SyncResult with per-device offsets.
    """
    if master_key not in sessions:
        raise ValueError(
            f"Master device '{master_key}' not found in sessions. "
            f"Available keys: {list(sessions.keys())}"
        )

    master = sessions[master_key]
    if not master.gps_samples:
        raise RuntimeError(
            f"Master device '{master_key}' has no GPS data — "
            "cannot use GPS anchor synchronisation. "
            "Use IMU cross-correlation instead."
        )

    master_t0 = master.gps_samples[0].timestamp_ns
    offsets: dict[str, int] = {master_key: 0}
    notes: list[str] = [f"Master: {master_key} (GPS t0 = {master_t0} ns)"]

    for key, session in sessions.items():
        if key == master_key:
            continue
        if not session.gps_samples:
            offsets[key] = 0
            notes.append(
                f"{key}: no GPS — offset set to 0 (manual calibration recommended)"
            )
            logger.warning(
                "Device '{}' has no GPS data — offset set to 0 ns.", key
            )
            continue

        device_t0 = session.gps_samples[0].timestamp_ns
        offset_ns = master_t0 - device_t0
        offsets[key] = offset_ns
        lag_s = abs(offset_ns) / 1e9
        notes.append(f"{key}: offset = {offset_ns} ns ({lag_s:.3f} s)")
        logger.info("GPS sync: {} offset = {} ns ({:.3f} s)", key, offset_ns, lag_s)

    lag_warning = any(
        abs(v) / 1e9 > _MAX_SAFE_LAG_S for k, v in offsets.items() if k != master_key
    )
    if lag_warning:
        msg = (
            f"GPS sync lag > {_MAX_SAFE_LAG_S} s detected — "
            "verify GPS clocks are consistent across devices."
        )
        notes.append(f"WARNING: {msg}")
        logger.warning("{}", msg)

    return SyncResult(
        method="gps",
        offsets_ns=offsets,
        confidence=1.0,
        lag_warning=lag_warning,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Method 2 — IMU cross-correlation
# ---------------------------------------------------------------------------

def sync_by_xcorr(
    sessions: dict[str, ExtractedSession],
    reference_key: str,
    resample_hz: float = 200.0,
) -> SyncResult:
    """Estimate per-device time offsets by cross-correlating accel magnitude.

    Resamples all devices to a common grid at *resample_hz*, then finds the
    lag that maximises the normalised cross-correlation between each device
    and the reference device.

    Args:
        sessions:      Dict of device_key → ExtractedSession.
        reference_key: Key of the reference device (offset = 0).
        resample_hz:   Common resampling rate (default 200 Hz).

    Returns:
        SyncResult with per-device offsets.
    """
    if reference_key not in sessions:
        raise ValueError(
            f"Reference device '{reference_key}' not found. "
            f"Available keys: {list(sessions.keys())}"
        )

    ref_session = sessions[reference_key]
    if not ref_session.imu_samples:
        raise RuntimeError(
            f"Reference device '{reference_key}' has no IMU data — "
            "cannot perform cross-correlation sync."
        )

    step_ns = int(1e9 / resample_hz)
    ref_mag = _accel_magnitude_resampled(ref_session.imu_samples, step_ns)

    offsets: dict[str, int] = {reference_key: 0}
    confidence_vals: list[float] = []
    notes: list[str] = [f"Reference: {reference_key}, resample={resample_hz} Hz"]
    lag_warning = False

    for key, session in sessions.items():
        if key == reference_key:
            continue
        if not session.imu_samples:
            offsets[key] = 0
            notes.append(f"{key}: no IMU — offset set to 0")
            logger.warning("Device '{}' has no IMU data — offset set to 0 ns.", key)
            continue

        dev_mag = _accel_magnitude_resampled(session.imu_samples, step_ns)
        lag_samples, peak_ratio = _cross_correlate(ref_mag, dev_mag)
        lag_ns = lag_samples * step_ns
        lag_s = abs(lag_ns) / 1e9

        offsets[key] = -lag_ns  # shift device to align with reference
        confidence_vals.append(peak_ratio)
        notes.append(
            f"{key}: lag={lag_samples} samples ({lag_s:.3f} s), "
            f"xcorr peak ratio={peak_ratio:.3f}"
        )
        logger.info(
            "xcorr sync: {} lag={} samples ({:.3f} s), confidence={:.3f}",
            key, lag_samples, lag_s, peak_ratio,
        )

        if lag_s > _MAX_SAFE_LAG_S:
            lag_warning = True
            msg = (
                f"Cross-correlation lag for '{key}' is {lag_s:.1f} s "
                f"(>{_MAX_SAFE_LAG_S} s) — likely a sync failure. "
                "Verify devices were recording the same motion simultaneously."
            )
            notes.append(f"WARNING: {msg}")
            logger.warning("{}", msg)

    avg_confidence = (sum(confidence_vals) / len(confidence_vals)) if confidence_vals else 0.0

    return SyncResult(
        method="xcorr",
        offsets_ns=offsets,
        confidence=avg_confidence,
        lag_warning=lag_warning,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Method 3 — Manual offset
# ---------------------------------------------------------------------------

def sync_manual(
    sessions: dict[str, ExtractedSession],
    offsets_ns: dict[str, int],
) -> SyncResult:
    """Apply user-supplied offsets directly.

    Any device not present in *offsets_ns* gets offset 0.

    Args:
        sessions:   Dict of device_key → ExtractedSession.
        offsets_ns: Dict of device_key → offset in nanoseconds.

    Returns:
        SyncResult wrapping the supplied offsets.
    """
    full_offsets = {key: offsets_ns.get(key, 0) for key in sessions}
    lag_warning = any(abs(v) / 1e9 > _MAX_SAFE_LAG_S for v in full_offsets.values())
    notes = [f"{k}: {v} ns ({abs(v)/1e9:.3f} s)" for k, v in full_offsets.items()]
    return SyncResult(
        method="manual",
        offsets_ns=full_offsets,
        confidence=1.0,
        lag_warning=lag_warning,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Signal processing helpers
# ---------------------------------------------------------------------------

def _accel_magnitude_resampled(
    samples: list[IMUSample], step_ns: int
) -> list[float]:
    """Resample accel magnitude to a uniform grid via linear interpolation."""
    if not samples:
        return []

    t0 = samples[0].timestamp_ns
    t1 = samples[-1].timestamp_ns
    n_out = max(1, (t1 - t0) // step_ns + 1)

    # Build source arrays
    src_ts = [s.timestamp_ns for s in samples]
    src_mag = [
        math.sqrt(s.accel_x ** 2 + s.accel_y ** 2 + s.accel_z ** 2)
        for s in samples
    ]

    out: list[float] = []
    n_src = len(src_ts)
    src_idx = 0

    for i in range(n_out):
        t = t0 + i * step_ns
        # Advance bracket
        while src_idx < n_src - 2 and src_ts[src_idx + 1] < t:
            src_idx += 1
        if src_idx >= n_src - 1:
            out.append(src_mag[-1])
        else:
            dt = src_ts[src_idx + 1] - src_ts[src_idx]
            if dt == 0:
                out.append(src_mag[src_idx])
            else:
                alpha = (t - src_ts[src_idx]) / dt
                out.append(src_mag[src_idx] + alpha * (src_mag[src_idx + 1] - src_mag[src_idx]))

    return out


def _cross_correlate(ref: list[float], sig: list[float]) -> tuple[int, float]:
    """Find the lag (in samples) that maximises normalised cross-correlation.

    Returns (lag_samples, peak_ratio) where:
      lag_samples > 0 means sig is delayed relative to ref.
      peak_ratio ∈ [0, 1] — ratio of the peak to the zero-lag correlation.

    Pure Python — no scipy dependency for this helper.
    """
    if not ref or not sig:
        return 0, 0.0

    # Subtract mean (de-trend)
    ref_mean = sum(ref) / len(ref)
    sig_mean = sum(sig) / len(sig)
    r = [x - ref_mean for x in ref]
    s = [x - sig_mean for x in sig]

    n_ref = len(r)
    n_sig = len(s)
    # Search lags in range [-n_ref//4, +n_ref//4] to keep O(n) manageable
    max_lag = min(n_ref // 4, n_sig // 4, int(_MAX_SAFE_LAG_S * 500))  # 500 ~ 200Hz * 2.5s

    best_lag = 0
    best_val = float("-inf")
    zero_lag_val = 0.0

    for lag in range(-max_lag, max_lag + 1):
        total = 0.0
        count = 0
        for i in range(n_ref):
            j = i - lag
            if 0 <= j < n_sig:
                total += r[i] * s[j]
                count += 1
        val = total / count if count else 0.0
        if lag == 0:
            zero_lag_val = abs(val) if abs(val) > 1e-12 else 1e-12
        if val > best_val:
            best_val = val
            best_lag = lag

    peak_ratio = best_val / zero_lag_val if zero_lag_val else 0.0
    return best_lag, min(1.0, max(0.0, peak_ratio))
