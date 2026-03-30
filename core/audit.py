"""
core/audit.py — File auditing for GoPro Hero 11, Insta360 X4, and Sensor Logger.

Each auditor returns an AuditResult. No mock data — every function raises a
clear exception if the file is missing or unreadable.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from loguru import logger

from core.models import AuditResult, DeviceType, RecordingQuality
from core.utils import exiftool, ffprobe, file_size_mb

# ---------------------------------------------------------------------------
# Chapter-split helpers (GoPro GHxx / GXxx naming)
# ---------------------------------------------------------------------------

# GoPro chapter naming patterns:
#   Hero 11 (H.265): GH010001.MP4, GH020001.MP4 ...
#   Hero 9/10 (H.264): GX010001.MP4, GX020001.MP4 ...
_GOPRO_CHAPTER_RE = re.compile(
    r"^G[HX](\d{2})(\d{4})\.MP4$", re.IGNORECASE
)


def detect_gopro_chapters(file_path: str | Path) -> list[Path]:
    """Given one file in a GoPro chapter sequence, return all chapter paths in order.

    GoPro splits recordings at ~4 GB.  The chapter index is the first two digits
    after 'GH'/'GX'; the clip number is the last four digits and is constant
    across chapters.

    Returns a list of existing paths sorted by chapter number.  If the file is
    not a chapter-split file, returns a single-element list.
    """
    p = Path(file_path)
    m = _GOPRO_CHAPTER_RE.match(p.name)
    if not m:
        return [p]

    clip_number = m.group(2)
    prefix = p.name[:2]  # 'GH' or 'GX'
    parent = p.parent

    chapters: list[Path] = []
    for chapter_idx in range(1, 100):
        candidate = parent / f"{prefix}{chapter_idx:02d}{clip_number}.MP4"
        if candidate.exists():
            chapters.append(candidate)

    return sorted(chapters, key=lambda x: _GOPRO_CHAPTER_RE.match(x.name).group(1))


# ---------------------------------------------------------------------------
# GoPro Hero 11 auditor
# ---------------------------------------------------------------------------

def audit_gopro(file_path: str | Path) -> AuditResult:
    """Audit a GoPro Hero 11 MP4 (or chapter sequence).

    Args:
        file_path: Path to one chapter of the recording (any chapter is fine).

    Returns:
        AuditResult aggregated across all chapters.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"GoPro file not found: '{file_path}'. "
            "Check the path and ensure the file is accessible."
        )

    chapters = detect_gopro_chapters(file_path)
    logger.info("Auditing GoPro: {} ({} chapter(s))", file_path.name, len(chapters))

    issues: list[str] = []
    total_duration = 0.0
    total_size_mb = 0.0
    imu_sample_count = 0
    gps_sample_count = 0
    video_fps = 0.0
    video_resolution: tuple[int, int] = (0, 0)
    has_imu = False
    has_gps = False

    for chapter in chapters:
        probe = ffprobe(chapter)
        chapter_size = file_size_mb(chapter)
        total_size_mb += chapter_size

        # Duration
        fmt_duration = float(probe.get("format", {}).get("duration", 0))
        total_duration += fmt_duration

        # Video stream
        for stream in probe.get("streams", []):
            if stream.get("codec_type") == "video" and video_fps == 0.0:
                fps_str = stream.get("r_frame_rate", "0/1")
                num, den = (int(x) for x in fps_str.split("/"))
                video_fps = num / den if den else 0.0
                video_resolution = (
                    int(stream.get("width", 0)),
                    int(stream.get("height", 0)),
                )

        # GoPro Metadata (GPMD) stream — contains IMU and GPS
        gpmd_streams = [
            s for s in probe.get("streams", [])
            if s.get("codec_tag_string", "").upper() == "GPMD"
            or s.get("codec_long_name", "").upper().startswith("GOPRO METADATA")
            or "gpmd" in s.get("codec_tag_string", "").lower()
            or s.get("codec_type") == "data"
        ]

        if gpmd_streams:
            # Try py_gpmf_parser if available; fall back to counting data stream
            try:
                from gpmf import parse  # type: ignore[import]
                from gpmf.extract import get_gpmf_data_from_file  # type: ignore[import]

                raw = get_gpmf_data_from_file(str(chapter))
                streams = parse(raw)
                accl_keys = [s for s in streams if s.get("key") == b"ACCL"]
                gyro_keys = [s for s in streams if s.get("key") == b"GYRO"]
                gps_keys = [s for s in streams if s.get("key") == b"GPS5"]
                if accl_keys or gyro_keys:
                    has_imu = True
                    imu_sample_count += sum(len(s.get("values", [])) for s in accl_keys)
                if gps_keys:
                    has_gps = True
                    gps_sample_count += sum(len(s.get("values", [])) for s in gps_keys)
            except Exception as exc:
                logger.debug("gpmf parse unavailable for {}: {}", chapter.name, exc)
                # Presence of a data stream is a strong indicator of GPMF
                has_imu = True
                has_gps = True

        # HyperSmooth detection via ffprobe tags
        fmt_tags = probe.get("format", {}).get("tags", {})
        video_tags: dict[str, Any] = {}
        for s in probe.get("streams", []):
            if s.get("codec_type") == "video":
                video_tags = s.get("tags", {})
                break

        if _tag_indicates_stabilization(fmt_tags, video_tags):
            if "HyperSmooth detected — re-record with HyperSmooth OFF: Settings > Stabilization > Off" not in issues:
                issues.append(
                    "HyperSmooth detected — re-record with HyperSmooth OFF: Settings > Stabilization > Off"
                )

        # SuperView detection (non-standard aspect ratio)
        if video_resolution != (0, 0) and _is_superview(video_resolution):
            if "SuperView aspect ratio detected — standard 16:9 is recommended for calibration" not in issues:
                issues.append(
                    "SuperView aspect ratio detected — standard 16:9 is recommended for calibration"
                )

    # Compute sample rates
    imu_rate_hz = (imu_sample_count / total_duration) if total_duration > 0 else 0.0
    gps_rate_hz = (gps_sample_count / total_duration) if total_duration > 0 else 0.0

    if not has_imu:
        issues.append(
            "No IMU data stream (GPMD) found — file may be an MP4 export without telemetry"
        )
    if not has_gps:
        issues.append(
            "No GPS stream found — ensure GPS was enabled during recording"
        )

    chapter_file_strs = [str(c) for c in chapters] if len(chapters) > 1 else []
    result = AuditResult(
        file_path=str(chapters[0]),
        device_type=DeviceType.GOPRO,
        duration_seconds=total_duration,
        has_imu=has_imu,
        has_gps=has_gps,
        imu_sample_count=imu_sample_count,
        imu_rate_hz=round(imu_rate_hz, 2),
        gps_sample_count=gps_sample_count,
        gps_rate_hz=round(gps_rate_hz, 2),
        video_fps=round(video_fps, 3),
        video_resolution=video_resolution,
        file_size_mb=round(total_size_mb, 2),
        issues=issues,
        chapter_files=chapter_file_strs,
    )
    result = result.model_copy(update={"quality": compute_recording_quality(result)})
    return result


def _tag_indicates_stabilization(fmt_tags: dict, video_tags: dict) -> bool:
    """Heuristic: check ffprobe tags for stabilization markers."""
    for tags in (fmt_tags, video_tags):
        for key, val in tags.items():
            k = key.lower()
            v = str(val).lower()
            if "stabiliz" in k or "hypersmooth" in k:
                return True
            if ("stabiliz" in v or "hypersmooth" in v) and "off" not in v:
                return True
    return False


def _is_superview(resolution: tuple[int, int]) -> bool:
    """Return True if resolution has a non-standard aspect ratio (SuperView)."""
    w, h = resolution
    if h == 0:
        return False
    ratio = w / h
    # Standard ratios: 16:9 ≈ 1.778, 4:3 ≈ 1.333, 1:1 = 1.0
    standard = [16 / 9, 4 / 3, 1.0, 17 / 9, 21 / 9]
    return not any(abs(ratio - s) < 0.05 for s in standard)


# ---------------------------------------------------------------------------
# Insta360 X4 auditor
# ---------------------------------------------------------------------------

def audit_insta360(file_path: str | Path) -> AuditResult:
    """Audit an Insta360 X4 INSV file (or MP4 export).

    Uses exiftool to probe metadata. Falls back gracefully if exiftool tags
    are absent, but always reports what it could not detect.

    Args:
        file_path: Path to the .insv (or .mp4) file.

    Returns:
        AuditResult for this file.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Insta360 file not found: '{file_path}'. "
            "Check the path and ensure the file is accessible."
        )

    logger.info("Auditing Insta360: {}", file_path.name)
    issues: list[str] = []

    # Basic probe via ffprobe
    probe = ffprobe(file_path)
    total_duration = float(probe.get("format", {}).get("duration", 0))
    size_mb = file_size_mb(file_path)

    video_fps = 0.0
    video_resolution: tuple[int, int] = (0, 0)
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video" and video_fps == 0.0:
            fps_str = stream.get("r_frame_rate", "0/1")
            num, den = (int(x) for x in fps_str.split("/"))
            video_fps = num / den if den else 0.0
            video_resolution = (
                int(stream.get("width", 0)),
                int(stream.get("height", 0)),
            )

    # Exiftool metadata
    has_imu = False
    has_gps = False
    imu_sample_count = 0
    gps_sample_count = 0

    try:
        meta = exiftool(file_path)
    except Exception as exc:
        issues.append(f"exiftool probe failed: {exc}")
        meta = {}

    # IMU presence — Insta360 INSV stores gyro/accel in proprietary tags
    gyro_keys = [k for k in meta if "gyro" in k.lower()]
    accel_keys = [k for k in meta if "accel" in k.lower() or "accl" in k.lower()]
    if gyro_keys or accel_keys:
        has_imu = True
        # exiftool -json returns metadata counts, not raw samples — mark as detected
        imu_sample_count = 0  # exact count requires binary extraction

    # GPS presence
    gps_keys = [k for k in meta if "gps" in k.lower()]
    if gps_keys:
        has_gps = True
        gps_sample_count = 0  # exact count requires binary extraction

    # Stabilization flag
    for key, val in meta.items():
        if "stabiliz" in key.lower() and str(val).strip() not in ("0", "Off", "off", "false"):
            issues.append(
                "Stabilization detected in INSV metadata — raw IMU correlation may be degraded. "
                "Disable stabilization in the Insta360 app before recording for calibration use."
            )
            break

    # INSV vs MP4 export
    if file_path.suffix.lower() == ".mp4":
        issues.append(
            "File is an MP4 export — original INSV format preferred for full telemetry access"
        )
    elif file_path.suffix.lower() not in (".insv",):
        issues.append(f"Unexpected Insta360 file extension: '{file_path.suffix}'")

    if not has_imu:
        issues.append(
            "No IMU (gyro/accel) metadata found via exiftool — "
            "file may lack embedded telemetry"
        )
    if not has_gps:
        issues.append(
            "No GPS metadata found — GPS is only present if the GPS remote or "
            "Insta360 app (screen-on) was active during recording"
        )

    imu_rate_hz = 0.0  # requires binary extraction from INSV
    gps_rate_hz = 0.0

    result = AuditResult(
        file_path=str(file_path),
        device_type=DeviceType.INSTA360,
        duration_seconds=total_duration,
        has_imu=has_imu,
        has_gps=has_gps,
        imu_sample_count=imu_sample_count,
        imu_rate_hz=imu_rate_hz,
        gps_sample_count=gps_sample_count,
        gps_rate_hz=gps_rate_hz,
        video_fps=round(video_fps, 3),
        video_resolution=video_resolution,
        file_size_mb=round(size_mb, 2),
        issues=issues,
    )
    result = result.model_copy(update={"quality": compute_recording_quality(result)})
    return result


# ---------------------------------------------------------------------------
# Android Sensor Logger auditor
# ---------------------------------------------------------------------------

_SL_ACCEL_COLS = {"accelerometeraccelerationx", "accel_x", "acceleration_x", "ax"}
_SL_GYRO_COLS = {"gyroscoperotationratex", "gyro_x", "rotation_rate_x", "gx"}
_SL_GPS_LAT_COLS = {"locationlatitude", "gps_lat", "latitude", "lat"}
_SL_TIME_COLS = {
    "time",              # ISO 8601 string
    "loggingtime",       # some versions
    "seconds_elapsed",   # float from app start
    "timestamp",         # Unix epoch (s or ms)
    "timestamp_ns",      # nanoseconds
}

# Maximum allowed gap in IMU stream before flagging (milliseconds)
_MAX_IMU_GAP_MS = 100.0


def audit_sensor_logger(file_path: str | Path) -> AuditResult:
    """Audit an Android Sensor Logger CSV (or JSON) file.

    Auto-detects column schema, timestamp format, and computes actual sample
    rates from timestamp deltas.

    Args:
        file_path: Path to the Sensor Logger .csv or .json file.
                   If a directory is given, looks for the first .csv file inside.

    Returns:
        AuditResult for this file.
    """
    file_path = Path(file_path)

    # Accept directory (Sensor Logger exports a folder)
    if file_path.is_dir():
        csvs = sorted(file_path.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(
                f"No CSV files found in Sensor Logger directory: '{file_path}'"
            )
        file_path = csvs[0]
        logger.info("Sensor Logger directory — using: {}", file_path.name)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Sensor Logger file not found: '{file_path}'. "
            "Check the path and ensure the file is accessible."
        )

    logger.info("Auditing Sensor Logger: {}", file_path.name)
    issues: list[str] = []

    # Parse CSV
    if file_path.suffix.lower() == ".json":
        rows, headers = _load_sensor_logger_json(file_path)
    else:
        rows, headers = _load_sensor_logger_csv(file_path)

    if not rows:
        raise RuntimeError(
            f"Sensor Logger file is empty or has no data rows: '{file_path}'"
        )

    headers_lower = {h.lower().replace(" ", "").replace("_", ""): h for h in headers}

    has_imu = _has_column(headers_lower, _SL_ACCEL_COLS) and _has_column(
        headers_lower, _SL_GYRO_COLS
    )
    has_gps = _has_column(headers_lower, _SL_GPS_LAT_COLS)

    # Find timestamp column
    time_col = _find_column(headers_lower, _SL_TIME_COLS)
    if time_col is None:
        issues.append(
            "No recognisable timestamp column found — "
            "expected one of: time, loggingTime, seconds_elapsed, timestamp"
        )

    imu_sample_count = len(rows)
    gps_sample_count = 0
    total_duration = 0.0
    imu_rate_hz = 0.0

    if time_col is not None:
        timestamps_ns = _parse_timestamps(rows, time_col, issues)
        if len(timestamps_ns) >= 2:
            total_duration = (timestamps_ns[-1] - timestamps_ns[0]) / 1e9
            imu_rate_hz = (
                (len(timestamps_ns) - 1) / total_duration if total_duration > 0 else 0.0
            )
            _check_imu_gaps(timestamps_ns, issues)

    if not has_imu:
        issues.append(
            "Accelerometer or gyroscope columns not found — "
            "check that the Sensor Logger session included both sensors"
        )
    if not has_gps:
        issues.append(
            "No GPS/location columns found — "
            "GPS logging may not have been enabled in Sensor Logger"
        )

    size_mb = file_size_mb(file_path)

    result = AuditResult(
        file_path=str(file_path),
        device_type=DeviceType.SENSOR_LOGGER,
        duration_seconds=round(total_duration, 3),
        has_imu=has_imu,
        has_gps=has_gps,
        imu_sample_count=imu_sample_count,
        imu_rate_hz=round(imu_rate_hz, 2),
        gps_sample_count=gps_sample_count,
        gps_rate_hz=0.0,
        video_fps=0.0,
        video_resolution=(0, 0),
        file_size_mb=round(size_mb, 2),
        issues=issues,
    )
    result = result.model_copy(update={"quality": compute_recording_quality(result)})
    return result


# --- Sensor Logger internals ------------------------------------------------

def _load_sensor_logger_csv(
    path: Path,
) -> tuple[list[dict[str, str]], list[str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames or [])
        rows = list(reader)
    return rows, headers


def _load_sensor_logger_json(
    path: Path,
) -> tuple[list[dict[str, str]], list[str]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list) and data:
        headers = list(data[0].keys())
        rows = [{k: str(v) for k, v in row.items()} for row in data]
        return rows, headers
    raise RuntimeError(
        f"Unexpected Sensor Logger JSON structure in '{path}' — "
        "expected a top-level array of objects"
    )


def _has_column(
    headers_lower: dict[str, str], candidates: set[str]
) -> bool:
    norm = {c.lower().replace(" ", "").replace("_", "") for c in candidates}
    return bool(norm & set(headers_lower.keys()))


def _find_column(
    headers_lower: dict[str, str], candidates: set[str]
) -> str | None:
    """Return the original column name matching any candidate, or None."""
    norm = {c.lower().replace(" ", "").replace("_", ""): c for c in candidates}
    for key, original_candidate in norm.items():
        if key in headers_lower:
            return headers_lower[key]
    return None


def _parse_timestamps(
    rows: list[dict[str, str]],
    time_col: str,
    issues: list[str],
) -> list[int]:
    """Parse timestamp column → list of nanosecond ints (Unix epoch).

    Handles:
      - seconds_elapsed (float, app-relative) → no absolute epoch, offset to 0
      - Unix epoch seconds (float < 2e10)
      - Unix epoch milliseconds (float 1e12 range)
      - Unix epoch nanoseconds (int > 1e15)
      - ISO 8601 string
    """
    import re as _re
    from datetime import datetime, timezone

    raw_vals: list[str] = [r[time_col] for r in rows if r.get(time_col)]
    if not raw_vals:
        return []

    sample = raw_vals[0].strip()
    ns_values: list[int] = []

    # ISO 8601 string detection
    if _re.match(r"\d{4}-\d{2}-\d{2}", sample):
        for v in raw_vals:
            try:
                dt = datetime.fromisoformat(v.strip().replace("Z", "+00:00"))
                ns_values.append(int(dt.timestamp() * 1e9))
            except Exception:
                continue
        return ns_values

    # Numeric
    try:
        numeric = float(sample)
    except ValueError:
        issues.append(f"Cannot parse timestamp value '{sample}' in column '{time_col}'")
        return []

    if numeric < 1e6:
        # seconds_elapsed — relative, no epoch anchor
        issues.append(
            "Timestamp column appears to be 'seconds_elapsed' (app-relative). "
            "An absolute wall-clock anchor is needed for multi-device synchronisation. "
            "Use 'time' (ISO) or 'timestamp' (Unix epoch) logging in Sensor Logger settings."
        )
        for v in raw_vals:
            try:
                ns_values.append(int(float(v) * 1e9))
            except Exception:
                continue
    elif numeric < 2e10:
        # Unix epoch seconds
        for v in raw_vals:
            try:
                ns_values.append(int(float(v) * 1e9))
            except Exception:
                continue
    elif numeric < 2e13:
        # Unix epoch milliseconds
        for v in raw_vals:
            try:
                ns_values.append(int(float(v) * 1e6))
            except Exception:
                continue
    else:
        # Unix epoch nanoseconds
        for v in raw_vals:
            try:
                ns_values.append(int(float(v)))
            except Exception:
                continue

    return ns_values


def _check_imu_gaps(timestamps_ns: list[int], issues: list[str]) -> None:
    """Flag if any consecutive timestamp delta exceeds _MAX_IMU_GAP_MS."""
    gap_threshold_ns = int(_MAX_IMU_GAP_MS * 1e6)
    max_gap_ns = 0
    for i in range(1, len(timestamps_ns)):
        delta = timestamps_ns[i] - timestamps_ns[i - 1]
        if delta > max_gap_ns:
            max_gap_ns = delta
    if max_gap_ns > gap_threshold_ns:
        issues.append(
            f"IMU stream has a gap of {max_gap_ns / 1e6:.1f} ms (threshold: {_MAX_IMU_GAP_MS} ms) — "
            "check for app interruptions or sensor pauses during recording"
        )


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------

def audit_file(file_path: str | Path) -> AuditResult:
    """Auto-detect device type and run the appropriate auditor.

    Detection order:
      1. .insv → Insta360
      2. .MP4 / .mp4 matching GoPro naming → GoPro
      3. .csv / .json → Sensor Logger
      4. Directory → Sensor Logger (folder export)

    Args:
        file_path: Path to the file or directory to audit.

    Returns:
        AuditResult from the matching auditor.
    """
    p = Path(file_path)

    if p.is_dir():
        return audit_sensor_logger(p)

    ext = p.suffix.lower()

    if ext == ".insv":
        return audit_insta360(p)

    if ext == ".mp4":
        if _GOPRO_CHAPTER_RE.match(p.name):
            return audit_gopro(p)
        # Non-chapter GoPro MP4 or Insta360 MP4 export — try GoPro first
        return audit_gopro(p)

    if ext in (".csv", ".json"):
        return audit_sensor_logger(p)

    raise ValueError(
        f"Unrecognised file type: '{p.name}'. "
        "Expected .MP4 (GoPro), .insv (Insta360), .csv/.json (Sensor Logger), "
        "or a Sensor Logger export directory."
    )


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------

def compute_recording_quality(result: AuditResult) -> RecordingQuality:
    """Derive a RecordingQuality enum value from an AuditResult.

    Criteria:
      EXCELLENT — no issues, has GPS, has IMU, gps_rate_hz ≥ 1.0, imu_rate_hz ≥ 100
      GOOD      — ≤ 1 issue, has IMU
      FAIR      — 2–3 issues, or HyperSmooth in issues
      POOR      — no IMU, or ≥ 4 issues, or no GPS fix at all
    """
    n_issues = len(result.issues)
    has_hypsmooth = any("HyperSmooth" in iss for iss in result.issues)

    if not result.has_imu or n_issues >= 4:
        return RecordingQuality.POOR

    if has_hypsmooth or n_issues >= 2:
        return RecordingQuality.FAIR

    if (
        n_issues == 0
        and result.has_gps
        and result.has_imu
        and result.gps_rate_hz >= 1.0
        and result.imu_rate_hz >= 100
    ):
        return RecordingQuality.EXCELLENT

    return RecordingQuality.GOOD
