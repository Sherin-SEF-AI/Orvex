"""
core/extractor_insta360.py — Insta360 X4 INSV telemetry + frame extraction.

Produces the same EuRoC-compatible output tree as extractor_gopro:
  <output_dir>/
    cam0/data/<timestamp_ns>.jpg   (equirectangular frames)
    imu0/data.csv                  (EuRoC header)
    gps.csv                        (if GPS present)

IMU is extracted from the INSV binary via exiftool -b -GyroData / -AcclData.
GPS is extracted via exiftool JSON tags (only present if GPS remote was active).

All timestamps are int nanoseconds.  Raises on missing/unreadable files.
"""
from __future__ import annotations

import re
import struct
import tempfile
from pathlib import Path

from loguru import logger

from core.models import (
    DeviceType,
    ExtractionConfig,
    ExtractedSession,
    GPSSample,
    IMUSample,
)
from core.utils import exiftool, exiftool_binary_tag, ffmpeg_extract_frames, ffprobe, file_size_mb

# EuRoC headers (same constants as extractor_gopro — not imported to keep modules independent)
_EUROC_IMU_HEADER = (
    "#timestamp [ns],"
    "w_RS_S_x [rad s^-1],"
    "w_RS_S_y [rad s^-1],"
    "w_RS_S_z [rad s^-1],"
    "a_RS_S_x [m s^-2],"
    "a_RS_S_y [m s^-2],"
    "a_RS_S_z [m s^-2]"
)
_EUROC_GPS_HEADER = (
    "#timestamp [ns],latitude [deg],longitude [deg],"
    "altitude_m [m],speed_mps [m s^-1],fix_type"
)

# Insta360 native IMU rate (X4 records at ~200 Hz)
_INSTA360_IMU_RATE_HZ = 200.0
# Insta360 native video FPS options (for timestamp spacing fallback)
_INSTA360_DEFAULT_FPS = 30.0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_insta360(
    insv_path: str | Path,
    config: ExtractionConfig,
    output_dir: str | Path,
) -> ExtractedSession:
    """Extract IMU, GPS and frames from an Insta360 X4 INSV file.

    Args:
        insv_path:  Path to the .insv file (or .mp4 export).
        config:     Extraction configuration.
        output_dir: Root of the EuRoC output tree.

    Returns:
        ExtractedSession with all extracted data references and stats.
    """
    insv_path = Path(insv_path)
    output_dir = Path(output_dir)

    if not insv_path.exists():
        raise FileNotFoundError(
            f"Insta360 file not found: '{insv_path}'. "
            "Ensure the file is accessible before running extraction."
        )

    logger.info("extract_insta360: {} → {}", insv_path.name, output_dir)

    cam0_dir = output_dir / "cam0" / "data"
    imu0_dir = output_dir / "imu0"
    cam0_dir.mkdir(parents=True, exist_ok=True)
    imu0_dir.mkdir(parents=True, exist_ok=True)

    # --- Video duration ---
    probe = ffprobe(insv_path)
    total_duration = float(probe.get("format", {}).get("duration", 0))
    if total_duration <= 0:
        raise RuntimeError(
            f"Could not determine duration for '{insv_path}'. "
            "Ensure the file is a valid Insta360 X4 INSV or MP4."
        )

    # --- IMU extraction ---
    imu_samples = _extract_imu(insv_path, total_duration)

    # --- GPS extraction ---
    gps_samples = _extract_gps(insv_path)

    # --- Frame extraction ---
    frame_paths, frame_timestamps_ns = _extract_frames(
        insv_path, cam0_dir, config.frame_fps, config.frame_quality
    )

    # --- Write EuRoC output ---
    imu_csv_path = imu0_dir / "data.csv"
    _write_euroc_imu(imu_samples, imu_csv_path)

    if gps_samples:
        gps_csv_path = output_dir / "gps.csv"
        _write_gps_csv(gps_samples, gps_csv_path)
    else:
        logger.warning(
            "No GPS data extracted from '{}' — gps.csv not written. "
            "GPS is only available if the Insta360 GPS remote or app (screen-on) "
            "was active during recording.",
            insv_path.name,
        )

    stats = {
        "imu_count": len(imu_samples),
        "gps_count": len(gps_samples),
        "frame_count": len(frame_paths),
        "imu_rate_hz": round(len(imu_samples) / total_duration, 2) if total_duration else 0.0,
        "gps_rate_hz": round(len(gps_samples) / total_duration, 2) if total_duration else 0.0,
        "output_dir": str(output_dir),
    }

    logger.info(
        "extract_insta360 complete: {} IMU, {} GPS, {} frames, {:.1f}s",
        len(imu_samples), len(gps_samples), len(frame_paths), total_duration,
    )

    return ExtractedSession(
        session_id=config.session_id,
        device_type=DeviceType.INSTA360,
        imu_samples=imu_samples,
        gps_samples=gps_samples,
        frame_paths=frame_paths,
        frame_timestamps_ns=frame_timestamps_ns,
        duration_seconds=total_duration,
        stats=stats,
    )


# ---------------------------------------------------------------------------
# IMU extraction
# ---------------------------------------------------------------------------

def _extract_imu(insv_path: Path, duration_s: float) -> list[IMUSample]:
    """Extract gyro and accel from INSV binary tags via exiftool.

    Insta360 stores IMU data in proprietary binary metadata tags:
      - GyroData: packed little-endian floats [gx, gy, gz] per sample
      - AcclData: packed little-endian floats [ax, ay, az] per sample

    If the binary tags are absent (e.g. MP4 export), falls back to an empty
    list with an explicit warning — never returns silent dummy data.
    """
    gyro_raw: bytes = b""
    accl_raw: bytes = b""

    try:
        gyro_raw = exiftool_binary_tag(insv_path, "GyroData")
    except Exception as exc:
        logger.warning(
            "GyroData tag not found in '{}': {}. "
            "Re-check file is native INSV format.",
            insv_path.name, exc,
        )

    try:
        accl_raw = exiftool_binary_tag(insv_path, "AcclData")
    except Exception as exc:
        logger.warning(
            "AcclData tag not found in '{}': {}. "
            "Re-check file is native INSV format.",
            insv_path.name, exc,
        )

    if not gyro_raw and not accl_raw:
        logger.warning(
            "No IMU binary data found in '{}'. "
            "If this is an MP4 export, use the original INSV file for telemetry.",
            insv_path.name,
        )
        return []

    gyro_samples = _unpack_float3_stream(gyro_raw)
    accl_samples = _unpack_float3_stream(accl_raw)

    n = min(len(gyro_samples), len(accl_samples))
    if n == 0:
        logger.warning("Could not unpack IMU samples from '{}'.", insv_path.name)
        return []

    # Align counts — use the shorter stream
    gyro_samples = gyro_samples[:n]
    accl_samples = accl_samples[:n]

    # Reconstruct timestamps: assume uniform rate across recording duration
    # (Insta360 does not embed per-sample timestamps in the binary tags)
    step_ns = int(duration_s * 1e9 / n)
    imu_samples: list[IMUSample] = []
    for i, (gyro, accl) in enumerate(zip(gyro_samples, accl_samples)):
        ts_ns = i * step_ns
        imu_samples.append(IMUSample(
            timestamp_ns=ts_ns,
            accel_x=float(accl[0]),
            accel_y=float(accl[1]),
            accel_z=float(accl[2]),
            gyro_x=float(gyro[0]),
            gyro_y=float(gyro[1]),
            gyro_z=float(gyro[2]),
        ))

    logger.debug(
        "Insta360 IMU: {} samples, step={}ns (~{:.0f} Hz)",
        n, step_ns, 1e9 / step_ns if step_ns else 0,
    )
    return imu_samples


def _unpack_float3_stream(raw: bytes) -> list[tuple[float, float, float]]:
    """Unpack a stream of little-endian float32 triplets."""
    if len(raw) < 12:
        return []
    n_samples = len(raw) // 12  # 3 floats × 4 bytes = 12 bytes per sample
    usable = n_samples * 12
    fmt = f"<{n_samples * 3}f"
    try:
        flat = struct.unpack_from(fmt, raw, 0)
    except struct.error:
        return []
    return [(flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2]) for i in range(n_samples)]


# ---------------------------------------------------------------------------
# GPS extraction
# ---------------------------------------------------------------------------

def _extract_gps(insv_path: Path) -> list[GPSSample]:
    """Extract GPS from exiftool JSON metadata.

    Returns empty list (with warning) if no GPS tags are present.
    Does not raise — GPS absence is expected when no GPS remote was used.
    """
    try:
        meta = exiftool(insv_path)
    except Exception as exc:
        logger.warning("exiftool failed on '{}': {}", insv_path.name, exc)
        return []

    gps_samples: list[GPSSample] = []

    # Try Insta360-specific GPS tags first, then standard EXIF GPS
    lat = meta.get("GPSLatitude") or meta.get("LocationLatitude")
    lon = meta.get("GPSLongitude") or meta.get("LocationLongitude")
    alt = meta.get("GPSAltitude") or meta.get("LocationAltitude") or 0.0
    speed = meta.get("GPSSpeed") or 0.0

    if lat is None or lon is None:
        return []

    # exiftool may return "10 deg 5' 30.00\" N" or a float string
    lat_f = _parse_gps_coord(str(lat))
    lon_f = _parse_gps_coord(str(lon))
    if lat_f is None or lon_f is None:
        logger.warning(
            "GPS coordinates in '{}' could not be parsed: lat={}, lon={}",
            insv_path.name, lat, lon,
        )
        return []

    try:
        alt_f = float(str(alt).split()[0])
    except (ValueError, IndexError):
        alt_f = 0.0

    try:
        speed_f = float(str(speed).split()[0])
    except (ValueError, IndexError):
        speed_f = 0.0

    # Single GPS fix from metadata (EXIF GPS is a snapshot, not a stream)
    gps_samples.append(GPSSample(
        timestamp_ns=0,
        latitude=lat_f,
        longitude=lon_f,
        altitude_m=alt_f,
        speed_mps=speed_f,
        fix_type=3,
    ))

    logger.debug(
        "Insta360 GPS (EXIF snapshot): lat={:.6f}, lon={:.6f}",
        lat_f, lon_f,
    )
    return gps_samples


def _parse_gps_coord(value: str) -> float | None:
    """Parse a GPS coordinate in 'deg min sec' or decimal float form."""
    # Already a float string
    try:
        return float(value)
    except ValueError:
        pass

    # "10 deg 5' 30.00\" N" or "10 deg 5' 30.00\" S"
    m = re.match(
        r"(\d+)\s+deg\s+([\d.]+)'\s+([\d.]+)\"\s*([NSEW])?",
        value.strip(),
        re.IGNORECASE,
    )
    if m:
        deg = float(m.group(1))
        minutes = float(m.group(2))
        seconds = float(m.group(3))
        direction = (m.group(4) or "N").upper()
        decimal = deg + minutes / 60.0 + seconds / 3600.0
        if direction in ("S", "W"):
            decimal = -decimal
        return decimal

    # "10.12345 N"
    m2 = re.match(r"([\d.]+)\s*([NSEW])", value.strip(), re.IGNORECASE)
    if m2:
        decimal = float(m2.group(1))
        direction = m2.group(2).upper()
        if direction in ("S", "W"):
            decimal = -decimal
        return decimal

    return None


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def _extract_frames(
    insv_path: Path,
    cam0_dir: Path,
    target_fps: float,
    quality: int,
) -> tuple[list[str], list[int]]:
    """Extract equirectangular frames from an INSV file via ffmpeg.

    Returns (frame_paths, frame_timestamps_ns).
    """
    tmp_pattern = cam0_dir / f"_tmp_{insv_path.stem}_%06d.jpg"
    q = max(2, min(31, int((100 - quality) / 100 * 29) + 2))

    ffmpeg_extract_frames(insv_path, tmp_pattern, fps=target_fps, quality=q)

    written = sorted(cam0_dir.glob(f"_tmp_{insv_path.stem}_*.jpg"))
    step_ns = int(1e9 / target_fps)

    frame_paths: list[str] = []
    frame_timestamps_ns: list[int] = []

    for idx, tmp in enumerate(written):
        ts_ns = idx * step_ns
        dest = cam0_dir / f"{ts_ns}.jpg"
        tmp.rename(dest)
        frame_paths.append(str(dest))
        frame_timestamps_ns.append(ts_ns)

    logger.debug(
        "Insta360 frames: {} extracted at {} fps", len(frame_paths), target_fps
    )
    return frame_paths, frame_timestamps_ns


# ---------------------------------------------------------------------------
# EuRoC output writers (duplicated from extractor_gopro for module independence)
# ---------------------------------------------------------------------------

def _write_euroc_imu(samples: list[IMUSample], path: Path) -> None:
    with open(path, "w", newline="") as f:
        f.write(_EUROC_IMU_HEADER + "\n")
        for s in samples:
            f.write(
                f"{s.timestamp_ns},"
                f"{s.gyro_x:.10f},{s.gyro_y:.10f},{s.gyro_z:.10f},"
                f"{s.accel_x:.10f},{s.accel_y:.10f},{s.accel_z:.10f}\n"
            )
    logger.debug("Wrote EuRoC IMU CSV: {} ({} rows)", path, len(samples))


def _write_gps_csv(samples: list[GPSSample], path: Path) -> None:
    with open(path, "w", newline="") as f:
        f.write(_EUROC_GPS_HEADER + "\n")
        for s in samples:
            f.write(
                f"{s.timestamp_ns},"
                f"{s.latitude:.10f},{s.longitude:.10f},"
                f"{s.altitude_m:.4f},{s.speed_mps:.4f},{s.fix_type}\n"
            )
    logger.debug("Wrote GPS CSV: {} ({} rows)", path, len(samples))
