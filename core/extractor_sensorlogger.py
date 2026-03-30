"""
core/extractor_sensorlogger.py — Android Sensor Logger CSV/JSON extraction.

Parses Sensor Logger output, normalises timestamps to nanosecond Unix epoch,
and writes EuRoC-compatible IMU CSV + GPS CSV.

Output tree:
  <output_dir>/
    imu0/data.csv    (EuRoC header, accel + gyro merged)
    gps.csv          (if GPS columns present)

Timestamp handling:
  - seconds_elapsed  (float, app-relative)  → anchored to 0 ns; warns user
  - time / loggingTime (ISO 8601 string)    → converted to Unix epoch ns
  - timestamp (Unix epoch seconds, float)   → multiplied by 1e9
  - timestamp (Unix epoch ms, float ~1e12)  → multiplied by 1e6
  - timestamp_ns (int)                      → used as-is
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from loguru import logger

from core.models import (
    DeviceType,
    ExtractionConfig,
    ExtractedSession,
    GPSSample,
    IMUSample,
)
from core.utils import file_size_mb

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

# ---------------------------------------------------------------------------
# Column-name normalisation map
# Keys are lowercased, stripped of spaces and underscores.
# Values are our canonical field names.
# ---------------------------------------------------------------------------
_ACCEL_X_ALIASES = {"accelerometeraccelerationx", "accelx", "ax", "accelerationx", "accx"}
_ACCEL_Y_ALIASES = {"accelerometeracceleration_y", "accelerometeraccelerationy", "accely", "ay", "accelerationy", "accy"}
_ACCEL_Z_ALIASES = {"accelerometeraccelerationz", "accelz", "az", "accelerationz", "accz"}
_GYRO_X_ALIASES  = {"gyroscoperotatex", "gyroscopeRotationRateX".lower(), "gyrox", "gx", "rotationratex", "gyrorotationratex"}
_GYRO_Y_ALIASES  = {"gyroscoperotatey", "gyroscopeRotationRateY".lower(), "gyroy", "gy", "rotationratey", "gyrorotationratey"}
_GYRO_Z_ALIASES  = {"gyroscoperotatez", "gyroscopeRotationRateZ".lower(), "gyroz", "gz", "rotationratez", "gyrorotationratez"}
_GPS_LAT_ALIASES = {"locationlatitude", "gpslat", "latitude", "lat"}
_GPS_LON_ALIASES = {"locationlongitude", "gpslon", "longitude", "lon"}
_GPS_ALT_ALIASES = {"locationaltitude", "gpsalt", "altitude", "alt"}
_GPS_SPD_ALIASES = {"locationspeed", "gpsspeed", "speed", "speedmps"}
_TIME_ALIASES    = {
    "time", "loggingtime", "seconds_elapsed", "secondselapsed",
    "timestamp", "timestamp_ns", "timestampns",
}


def _norm(s: str) -> str:
    """Lowercase and strip spaces/underscores for column matching."""
    return s.lower().replace(" ", "").replace("_", "")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_sensor_logger(
    file_path: str | Path,
    config: ExtractionConfig,
    output_dir: str | Path,
) -> ExtractedSession:
    """Parse a Sensor Logger CSV/JSON and write EuRoC IMU + GPS output.

    Args:
        file_path:  Path to a .csv / .json file, or a directory containing them.
        config:     Extraction configuration (session_id used for output).
        output_dir: Root of the EuRoC output tree.

    Returns:
        ExtractedSession with all extracted data references and stats.
    """
    file_path = Path(file_path)
    output_dir = Path(output_dir)

    # Accept directory — use first CSV found
    if file_path.is_dir():
        csvs = sorted(file_path.glob("*.csv"))
        jsons = sorted(file_path.glob("*.json"))
        candidates = csvs + jsons
        if not candidates:
            raise FileNotFoundError(
                f"No CSV or JSON files found in '{file_path}'. "
                "Ensure this is a Sensor Logger export directory."
            )
        file_path = candidates[0]
        logger.info("Sensor Logger directory — using: {}", file_path.name)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Sensor Logger file not found: '{file_path}'. "
            "Check the path and ensure the file is accessible."
        )

    logger.info("extract_sensor_logger: {} → {}", file_path.name, output_dir)

    # --- Parse rows ---
    rows, headers = _load_file(file_path)
    if not rows:
        raise RuntimeError(
            f"Sensor Logger file is empty or has no data rows: '{file_path}'"
        )

    col_map = _build_column_map(headers)
    _validate_required_columns(col_map, file_path)

    # --- Parse timestamps ---
    time_col = col_map.get("time")
    timestamps_ns, time_warnings = _parse_timestamps(rows, time_col)
    for w in time_warnings:
        logger.warning("{}", w)

    if len(timestamps_ns) != len(rows):
        raise RuntimeError(
            f"Timestamp parse produced {len(timestamps_ns)} values for "
            f"{len(rows)} rows in '{file_path}'. Check timestamp column format."
        )

    total_duration = (
        (timestamps_ns[-1] - timestamps_ns[0]) / 1e9
        if len(timestamps_ns) >= 2 else 0.0
    )

    # --- Parse IMU ---
    imu_samples = _parse_imu(rows, timestamps_ns, col_map)

    # --- Parse GPS ---
    gps_samples = _parse_gps(rows, timestamps_ns, col_map)

    # --- Write output ---
    imu0_dir = output_dir / "imu0"
    imu0_dir.mkdir(parents=True, exist_ok=True)
    imu_csv_path = imu0_dir / "data.csv"
    _write_euroc_imu(imu_samples, imu_csv_path)

    if gps_samples:
        gps_csv_path = output_dir / "gps.csv"
        _write_gps_csv(gps_samples, gps_csv_path)
    else:
        logger.warning(
            "No GPS columns found in '{}' — gps.csv not written.",
            file_path.name,
        )

    stats = {
        "imu_count": len(imu_samples),
        "gps_count": len(gps_samples),
        "frame_count": 0,  # Sensor Logger has no video
        "imu_rate_hz": round(len(imu_samples) / total_duration, 2) if total_duration else 0.0,
        "gps_rate_hz": round(len(gps_samples) / total_duration, 2) if total_duration else 0.0,
        "output_dir": str(output_dir),
    }

    logger.info(
        "extract_sensor_logger complete: {} IMU, {} GPS, {:.1f}s",
        len(imu_samples), len(gps_samples), total_duration,
    )

    return ExtractedSession(
        session_id=config.session_id,
        device_type=DeviceType.SENSOR_LOGGER,
        imu_samples=imu_samples,
        gps_samples=gps_samples,
        frame_paths=[],
        frame_timestamps_ns=[],
        duration_seconds=total_duration,
        stats=stats,
    )


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def _load_file(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if path.suffix.lower() == ".json":
        return _load_json(path)
    return _load_csv(path)


def _load_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames or [])
        rows = list(reader)
    return rows, headers


def _load_json(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise RuntimeError(
            f"Unexpected Sensor Logger JSON structure in '{path}' — "
            "expected a top-level array of objects."
        )
    headers = list(data[0].keys())
    rows = [{k: str(v) for k, v in row.items()} for row in data]
    return rows, headers


# ---------------------------------------------------------------------------
# Column mapping
# ---------------------------------------------------------------------------

def _build_column_map(headers: list[str]) -> dict[str, str]:
    """Map canonical field names → original column names."""
    norm_to_orig: dict[str, str] = {_norm(h): h for h in headers}

    def find(aliases: set[str]) -> str | None:
        for alias in aliases:
            if alias in norm_to_orig:
                return norm_to_orig[alias]
        return None

    return {
        "accel_x": find(_ACCEL_X_ALIASES),
        "accel_y": find(_ACCEL_Y_ALIASES),
        "accel_z": find(_ACCEL_Z_ALIASES),
        "gyro_x":  find(_GYRO_X_ALIASES),
        "gyro_y":  find(_GYRO_Y_ALIASES),
        "gyro_z":  find(_GYRO_Z_ALIASES),
        "gps_lat": find(_GPS_LAT_ALIASES),
        "gps_lon": find(_GPS_LON_ALIASES),
        "gps_alt": find(_GPS_ALT_ALIASES),
        "gps_spd": find(_GPS_SPD_ALIASES),
        "time":    find(_TIME_ALIASES),
    }


def _validate_required_columns(col_map: dict[str, str | None], path: Path) -> None:
    required = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z", "time"]
    missing = [k for k in required if col_map.get(k) is None]
    if missing:
        raise RuntimeError(
            f"Required columns not found in '{path.name}': {missing}. "
            "Check that the Sensor Logger session included accelerometer, gyroscope, "
            "and a timestamp column."
        )


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------

def _parse_timestamps(
    rows: list[dict[str, str]],
    time_col: str | None,
) -> tuple[list[int], list[str]]:
    """Return (timestamps_ns, warnings)."""
    warnings: list[str] = []

    if time_col is None:
        warnings.append("No timestamp column found — using row index as timestamp.")
        return [i * 5_000_000 for i in range(len(rows))], warnings  # 200 Hz nominal

    raw_vals = [r.get(time_col, "").strip() for r in rows]
    if not raw_vals:
        return [], warnings

    sample = raw_vals[0]

    # ISO 8601 detection
    if re.match(r"\d{4}-\d{2}-\d{2}", sample):
        ts, w = _parse_iso_timestamps(raw_vals)
        warnings.extend(w)
        return ts, warnings

    # Numeric
    try:
        numeric = float(sample)
    except ValueError:
        warnings.append(
            f"Cannot parse timestamp '{sample}' in column '{time_col}'. "
            "Falling back to row-index timestamps."
        )
        return [i * 5_000_000 for i in range(len(rows))], warnings

    if numeric < 1e6:
        # seconds_elapsed — app-relative, no epoch anchor
        warnings.append(
            f"Timestamp column '{time_col}' appears to be 'seconds_elapsed' (app-relative, "
            f"first value={numeric:.3f}). Timestamps will start from 0 ns. "
            "An absolute wall-clock anchor is needed for multi-device synchronisation. "
            "Use 'time' (ISO 8601) or 'timestamp' (Unix epoch) in Sensor Logger settings."
        )
        return [int(float(v) * 1e9) for v in raw_vals if _safe_float(v) is not None], warnings

    if numeric < 2e10:
        # Unix epoch seconds
        return [int(float(v) * 1e9) for v in raw_vals if _safe_float(v) is not None], warnings

    if numeric < 2e13:
        # Unix epoch milliseconds
        return [int(float(v) * 1e6) for v in raw_vals if _safe_float(v) is not None], warnings

    # Unix epoch nanoseconds
    return [int(float(v)) for v in raw_vals if _safe_float(v) is not None], warnings


def _parse_iso_timestamps(raw_vals: list[str]) -> tuple[list[int], list[str]]:
    from datetime import datetime, timezone
    warnings: list[str] = []
    result: list[int] = []
    for v in raw_vals:
        try:
            dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
            result.append(int(dt.timestamp() * 1e9))
        except Exception:
            warnings.append(f"Could not parse ISO timestamp '{v}' — row skipped.")
    return result, warnings


def _safe_float(s: str) -> float | None:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# IMU / GPS parsing
# ---------------------------------------------------------------------------

def _parse_imu(
    rows: list[dict[str, str]],
    timestamps_ns: list[int],
    col_map: dict[str, str | None],
) -> list[IMUSample]:
    samples: list[IMUSample] = []
    ax_col = col_map["accel_x"]
    ay_col = col_map["accel_y"]
    az_col = col_map["accel_z"]
    gx_col = col_map["gyro_x"]
    gy_col = col_map["gyro_y"]
    gz_col = col_map["gyro_z"]

    for ts, row in zip(timestamps_ns, rows):
        try:
            samples.append(IMUSample(
                timestamp_ns=ts,
                accel_x=float(row[ax_col]),
                accel_y=float(row[ay_col]),
                accel_z=float(row[az_col]),
                gyro_x=float(row[gx_col]),
                gyro_y=float(row[gy_col]),
                gyro_z=float(row[gz_col]),
            ))
        except (ValueError, TypeError, KeyError):
            continue  # skip malformed rows

    return samples


def _parse_gps(
    rows: list[dict[str, str]],
    timestamps_ns: list[int],
    col_map: dict[str, str | None],
) -> list[GPSSample]:
    lat_col = col_map.get("gps_lat")
    lon_col = col_map.get("gps_lon")
    if lat_col is None or lon_col is None:
        return []

    alt_col = col_map.get("gps_alt")
    spd_col = col_map.get("gps_spd")
    samples: list[GPSSample] = []

    for ts, row in zip(timestamps_ns, rows):
        try:
            lat = float(row[lat_col])
            lon = float(row[lon_col])
        except (ValueError, TypeError, KeyError):
            continue
        if lat == 0.0 and lon == 0.0:
            continue  # skip zero-fix rows

        try:
            alt = float(row[alt_col]) if alt_col and row.get(alt_col) else 0.0
        except (ValueError, TypeError):
            alt = 0.0
        try:
            spd = float(row[spd_col]) if spd_col and row.get(spd_col) else 0.0
        except (ValueError, TypeError):
            spd = 0.0

        samples.append(GPSSample(
            timestamp_ns=ts,
            latitude=lat,
            longitude=lon,
            altitude_m=alt,
            speed_mps=spd,
            fix_type=3,
        ))

    return samples


# ---------------------------------------------------------------------------
# EuRoC writers
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
