"""
core/insv_telemetry.py — GPS and IMU telemetry extraction from Insta360 X4 INSV files.

Provides:
  find_insv_pairs()           Scan a folder and group INSV files into front/back pairs.
  check_insv_telemetry()      Report what telemetry is present in an _00_ file.
  extract_gps_from_insv()     Pull GPS track via exiftool GPX template.
  extract_imu_from_insv()     Pull gyro + accel via exiftool (or raw trailer fallback).
  parse_insv_trailer_raw()    Low-level binary trailer parser.
  normalize_insv_timestamps() Convert any INSV timestamp variant to nanoseconds UTC epoch.
"""

from __future__ import annotations

import calendar
import json
import os
import re
import struct
import subprocess
import tempfile
import time as _time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from core.models import GPSSample, IMUSample, INSVPair
from core.utils import exiftool, exiftool_binary_tag, ffprobe_duration


# ---------------------------------------------------------------------------
# Module-level GPX template (exiftool print format)
# ---------------------------------------------------------------------------

GPX_TEMPLATE = """\
#------------------------------------------------------------------------------
# File:         gpx.fmt
# Description:  ExifTool print format for GPX track
# Usage:        exiftool -p gpx.fmt FILE [...] > out.gpx
# Requires:     ExifTool version 10.49 or later
#------------------------------------------------------------------------------
#[HEAD]<?xml version="1.0" encoding="utf-8"?>
#[HEAD]<gpx version="1.0"
#[HEAD] creator="ExifTool $ExifToolVersion"
#[HEAD] xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
#[HEAD] xmlns="http://www.topografix.com/GPX/1/0"
#[HEAD] xsi:schemaLocation="http://www.topografix.com/GPX/1/0
#[HEAD]   http://www.topografix.com/GPX/1/0/gpx.xsd">
#[HEAD]<trk><trkseg>
#[IF]  $GPSLatitude
#[BODY]<trkpt lat="$GPSLatitude#" lon="$GPSLongitude#">
#[BODY]  <ele>$GPSAltitude#</ele>
#[BODY]  <time>${DateTimeOriginal;DateFmt("%Y-%m-%dT%H:%M:%SZ")}</time>
#[BODY]</trkpt>
#[TAIL]</trkseg></trk>
#[TAIL]</gpx>
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_iso_to_ns(time_str: str) -> int:
    """Parse an ISO 8601 UTC timestamp string to nanoseconds Unix epoch.

    Handles the "2025-03-01T09:15:32Z" format produced by the GPX template.
    """
    dt = datetime.strptime(time_str.strip(), "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    return int(dt.timestamp() * 1e9)


def _parse_dto_to_epoch(dto_str: str) -> float:
    """Parse a DateTimeOriginal string to Unix epoch float (UTC seconds).

    Tries the three most common formats exiftool produces for Insta360 files.
    Returns 0.0 on any parse failure.
    """
    if not dto_str:
        return 0.0
    formats = [
        "%Y:%m:%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(dto_str.strip(), fmt).replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            continue
    logger.warning("Could not parse DateTimeOriginal '{}' with any known format", dto_str)
    return 0.0


def _unpack_float3_stream(data: bytes) -> list[tuple[float, float, float]]:
    """Unpack a flat little-endian stream of (float32, float32, float32) triplets.

    Stops cleanly if fewer than 12 bytes remain at the end.
    """
    results: list[tuple[float, float, float]] = []
    for offset in range(0, len(data) - 11, 12):
        try:
            x, y, z = struct.unpack_from("<fff", data, offset)
            results.append((x, y, z))
        except struct.error:
            break
    return results


def _parse_triplets_from_string(value: str) -> list[tuple[float, float, float]]:
    """Split a whitespace-separated string of floats into (x, y, z) triplets."""
    tokens = value.split()
    triplets: list[tuple[float, float, float]] = []
    for i in range(0, len(tokens) - 2, 3):
        try:
            triplets.append((float(tokens[i]), float(tokens[i + 1]), float(tokens[i + 2])))
        except ValueError:
            continue
    return triplets


def _parse_triplets_from_list(values: list) -> list[tuple[float, float, float]]:
    """Parse a list where each element is either a 'x y z' string or a numeric triplet."""
    triplets: list[tuple[float, float, float]] = []
    for item in values:
        if isinstance(item, str):
            parts = item.split()
            if len(parts) >= 3:
                try:
                    triplets.append((float(parts[0]), float(parts[1]), float(parts[2])))
                except ValueError:
                    continue
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            try:
                triplets.append((float(item[0]), float(item[1]), float(item[2])))
            except (ValueError, TypeError):
                continue
    return triplets


def _extract_triplets(value: Any) -> list[tuple[float, float, float]]:
    """Route any exiftool tag value to the appropriate triplet parser."""
    if isinstance(value, bytes):
        return _unpack_float3_stream(value)
    if isinstance(value, str):
        return _parse_triplets_from_string(value)
    if isinstance(value, list):
        return _parse_triplets_from_list(value)
    return []


def _count_samples_from_value(value: Any) -> int:
    """Estimate number of telemetry samples from an exiftool tag value."""
    if value is None:
        return 0
    if isinstance(value, list):
        return len(value)
    if isinstance(value, str):
        tokens = value.split()
        return max(1, len(tokens) // 3)
    if isinstance(value, bytes):
        return max(1, len(value) // 12)
    return 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_insv_pairs(folder_path: str) -> list[INSVPair]:
    """Scan *folder_path* for Insta360 INSV file pairs and return grouped results.

    Insta360 X4 writes two simultaneous files for each recording:
      VID_<date>_<time>_00_<idx>.insv  — back lens + all telemetry
      VID_<date>_<time>_10_<idx>.insv  — front lens (video only)

    Files are matched on (datetime_str, index_str). Missing a front file is
    recorded as an issue but does not prevent telemetry extraction.

    Args:
        folder_path: Directory to scan (non-recursive).

    Returns:
        Sorted list of INSVPair objects (ascending by base_name).
    """
    folder = Path(folder_path)
    pattern = re.compile(
        r"VID_(\d{8}_\d{6})_(\d{2})_(\d{3})\.insv$",
        re.IGNORECASE,
    )

    # Collect all .insv / .INSV files
    insv_files: list[Path] = []
    for p in folder.glob("*.insv"):
        insv_files.append(p)
    for p in folder.glob("*.INSV"):
        if p not in insv_files:
            insv_files.append(p)

    # Build dict keyed by (datetime_str, index_str)
    groups: dict[tuple[str, str], dict[str, str | None]] = {}
    for f in insv_files:
        m = pattern.match(f.name)
        if not m:
            logger.debug("Skipping unrecognised INSV filename: {}", f.name)
            continue
        datetime_str, lens_id, index_str = m.group(1), m.group(2), m.group(3)
        key = (datetime_str, index_str)
        if key not in groups:
            groups[key] = {"back_path": None, "front_path": None}
        if lens_id == "00":
            groups[key]["back_path"] = str(f)
        elif lens_id == "10":
            groups[key]["front_path"] = str(f)
        else:
            logger.debug("Unknown lens_id '{}' in {}", lens_id, f.name)

    pairs: list[INSVPair] = []
    for (datetime_str, index_str), paths in groups.items():
        back_path: str | None = paths.get("back_path")
        front_path: str | None = paths.get("front_path")

        if back_path is None:
            # No back (_00_) file — cannot extract telemetry, skip
            logger.warning(
                "INSV group ({}, {}) has no _00_ file — skipping", datetime_str, index_str
            )
            continue

        base_name = f"VID_{datetime_str}_{index_str}"

        # Duration
        duration_seconds = 0.0
        try:
            duration_seconds = ffprobe_duration(back_path)
        except Exception as exc:
            logger.warning("ffprobe_duration failed for '{}': {}", back_path, exc)

        # File size
        try:
            file_size_mb = os.path.getsize(back_path) / (1024 * 1024)
        except OSError as exc:
            logger.warning("os.path.getsize failed for '{}': {}", back_path, exc)
            file_size_mb = 0.0

        # Telemetry presence
        telem = check_insv_telemetry(back_path)

        # Issues
        issues: list[str] = []
        if front_path is None:
            issues.append(
                "Missing front lens file (_10_) — stitching not possible"
            )

        pair = INSVPair(
            back_path=back_path,
            front_path=front_path,
            base_name=base_name,
            duration_seconds=duration_seconds,
            file_size_mb=file_size_mb,
            has_gps=telem.get("has_gps", False),
            has_imu=telem.get("has_gyro", False) or telem.get("has_accel", False),
            issues=issues,
        )
        pairs.append(pair)

    pairs.sort(key=lambda p: p.base_name)
    return pairs


def check_insv_telemetry(insv_00_path: str) -> dict:
    """Inspect an Insta360 _00_ INSV file for available telemetry streams.

    Calls exiftool and examines tag names for GPS, gyroscope, and accelerometer
    data.  Never raises — returns a safe default dict on any exception.

    Args:
        insv_00_path: Path to the _00_ INSV file.

    Returns:
        Dict with keys: has_gps, has_gyro, has_accel, gps_sample_count,
        gyro_sample_count, accel_sample_count, firmware_version, camera_model,
        serial_number.
    """
    _default: dict = {
        "has_gps": False,
        "has_gyro": False,
        "has_accel": False,
        "gps_sample_count": 0,
        "gyro_sample_count": 0,
        "accel_sample_count": 0,
        "firmware_version": None,
        "camera_model": None,
        "serial_number": None,
    }
    try:
        meta = exiftool(insv_00_path)
    except Exception as exc:
        logger.warning("exiftool failed on '{}': {}", insv_00_path, exc)
        return _default

    try:
        has_gps = False
        has_gyro = False
        has_accel = False
        gps_count = 0
        gyro_count = 0
        accel_count = 0

        for key, value in meta.items():
            key_lower = key.lower()

            # GPS detection
            if "gps" in key_lower:
                if value and value not in ("", "0", None):
                    # Specifically validate lat/lon are present and non-zero
                    if key in ("GPSLatitude", "GPSLongitude"):
                        try:
                            if float(str(value).split()[0]) != 0.0:
                                has_gps = True
                        except (ValueError, IndexError):
                            has_gps = True
                    else:
                        has_gps = True
                    if key in ("GPSLatitude", "GPSLongitude", "GPSTrack"):
                        gps_count = max(gps_count, _count_samples_from_value(value))

            # Gyro detection
            if re.search(r"gyro|angular", key_lower):
                if value:
                    has_gyro = True
                    gyro_count = max(gyro_count, _count_samples_from_value(value))

            # Accel detection
            if re.search(r"accel|linear", key_lower):
                if value:
                    has_accel = True
                    accel_count = max(accel_count, _count_samples_from_value(value))

        # Device metadata
        firmware_version: str | None = None
        for k in ("FirmwareVersion", "Software"):
            if k in meta and meta[k]:
                firmware_version = str(meta[k])
                break

        camera_model: str | None = None
        for k in ("Model", "CameraModelName"):
            if k in meta and meta[k]:
                camera_model = str(meta[k])
                break

        serial_number: str | None = None
        for k in ("SerialNumber", "InternalSerialNumber"):
            if k in meta and meta[k]:
                serial_number = str(meta[k])
                break

        logger.debug(
            "check_insv_telemetry '{}': gps={} gyro={} accel={}",
            Path(insv_00_path).name,
            has_gps,
            has_gyro,
            has_accel,
        )

        return {
            "has_gps": has_gps,
            "has_gyro": has_gyro,
            "has_accel": has_accel,
            "gps_sample_count": gps_count,
            "gyro_sample_count": gyro_count,
            "accel_sample_count": accel_count,
            "firmware_version": firmware_version,
            "camera_model": camera_model,
            "serial_number": serial_number,
        }

    except Exception as exc:
        logger.warning(
            "check_insv_telemetry failed for '{}': {}", insv_00_path, exc
        )
        return _default


def extract_gps_from_insv(
    insv_00_path: str,
    output_gpx_path: str | None = None,
) -> list[GPSSample]:
    """Extract GPS track from an Insta360 _00_ INSV file via the exiftool GPX template.

    Writes GPX_TEMPLATE to a temporary file, invokes exiftool with the -p flag,
    then parses the resulting GPX XML.

    If no GPS data is present (GPS remote not connected, or app not active during
    recording), logs an explanatory message and returns an empty list.

    Args:
        insv_00_path:   Path to the _00_ INSV file.
        output_gpx_path: Optional path to save the raw GPX output.

    Returns:
        List of GPSSample objects (may be empty).
    """
    fmt_path: str | None = None
    try:
        # Write the GPX format template to a named temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".fmt", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(GPX_TEMPLATE)
            fmt_path = tmp.name

        logger.debug(
            "extract_gps_from_insv: running exiftool with GPX template on '{}'",
            Path(insv_00_path).name,
        )

        result = subprocess.run(
            [
                "exiftool",
                "-m",
                "-ee",
                "-api", "largefilesupport",
                "-p", fmt_path,
                insv_00_path,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        stdout = result.stdout or ""

        if result.returncode != 0 or not stdout.strip() or "<trkpt" not in stdout:
            logger.warning(
                "No GPS in {}. GPS requires GPS remote or Insta360 app with "
                "screen-on during recording.",
                Path(insv_00_path).name,
            )
            return []

        # Optionally save GPX output
        if output_gpx_path:
            try:
                Path(output_gpx_path).write_text(stdout, encoding="utf-8")
                logger.debug("GPX saved to '{}'", output_gpx_path)
            except OSError as exc:
                logger.warning("Could not write GPX file '{}': {}", output_gpx_path, exc)

        # Parse the GPX XML
        ns = {"gpx": "http://www.topografix.com/GPX/1/0"}
        try:
            root = ET.fromstring(stdout)
        except ET.ParseError as exc:
            logger.warning(
                "GPX XML parse error for '{}': {}", Path(insv_00_path).name, exc
            )
            return []

        samples: list[GPSSample] = []
        for trkpt in root.findall(".//gpx:trkpt", ns):
            try:
                lat = float(trkpt.get("lat", "0"))
                lon = float(trkpt.get("lon", "0"))

                ele_elem = trkpt.find("gpx:ele", ns)
                ele = float(ele_elem.text or "0.0") if ele_elem is not None else 0.0

                time_elem = trkpt.find("gpx:time", ns)
                if time_elem is None or not time_elem.text:
                    logger.debug("trkpt missing <time> element — skipping")
                    continue
                ts_ns = _parse_iso_to_ns(time_elem.text)

                samples.append(
                    GPSSample(
                        timestamp_ns=ts_ns,
                        latitude=lat,
                        longitude=lon,
                        altitude_m=ele,
                        speed_mps=0.0,
                        fix_type=3,
                    )
                )
            except (ValueError, TypeError) as exc:
                logger.debug("Skipping malformed trkpt: {}", exc)
                continue

        logger.debug(
            "extract_gps_from_insv '{}': {} GPS samples extracted",
            Path(insv_00_path).name,
            len(samples),
        )
        return samples

    except subprocess.TimeoutExpired:
        logger.warning(
            "exiftool timed out (120 s) reading GPS from '{}'",
            Path(insv_00_path).name,
        )
        return []
    except Exception as exc:
        logger.warning(
            "extract_gps_from_insv failed for '{}': {}", insv_00_path, exc
        )
        return []
    finally:
        if fmt_path:
            try:
                os.unlink(fmt_path)
            except OSError:
                pass


def extract_imu_from_insv(
    insv_00_path: str,
) -> tuple[list[IMUSample], dict]:
    """Extract gyroscope and accelerometer data from an Insta360 _00_ INSV file.

    Primary path: exiftool JSON tags (GyroData / AccelData or similar keys).
    Fallback: raw INSV binary trailer (parse_insv_trailer_raw → section 0x0300).

    Timestamps are synthesised as evenly-spaced values anchored to the
    DateTimeOriginal UTC epoch and then normalised via normalize_insv_timestamps().

    Args:
        insv_00_path: Path to the _00_ INSV file.

    Returns:
        Tuple of (list[IMUSample], metadata_dict).  The metadata dict always
        contains keys: method, gyro_rate_hz, accel_rate_hz, axes_convention,
        sample_count, warnings.
    """
    warnings_list: list[str] = []

    _failure_meta: dict = {
        "method": "none",
        "gyro_rate_hz": 0.0,
        "accel_rate_hz": 0.0,
        "axes_convention": "X=right,Y=up,Z=forward (Insta360 native)",
        "sample_count": 0,
        "warnings": [
            "No IMU data found",
        ],
        "error": "No IMU data found",
        "suggestion": (
            "Ensure exiftool >= 12.0 is installed and the file is a valid "
            "Insta360 X4 _00_ file"
        ),
    }

    # ------------------------------------------------------------------
    # Step 1: run exiftool and collect metadata
    # ------------------------------------------------------------------
    try:
        meta = exiftool(insv_00_path)
    except Exception as exc:
        logger.warning("exiftool failed on '{}': {}", insv_00_path, exc)
        meta = {}

    # ------------------------------------------------------------------
    # Step 2: locate gyro and accel tag values
    # ------------------------------------------------------------------
    gyro_value: Any = None
    accel_value: Any = None

    # Prefer exact well-known tags, then fall back to regex search
    for key in ("GyroData",):
        if key in meta and meta[key]:
            gyro_value = meta[key]
            break
    if gyro_value is None:
        for key, value in meta.items():
            if re.search(r"gyro|angular", key, re.I) and value:
                gyro_value = value
                break

    for key in ("AccelData",):
        if key in meta and meta[key]:
            accel_value = meta[key]
            break
    if accel_value is None:
        for key, value in meta.items():
            if re.search(r"accel|linear", key, re.I) and value:
                accel_value = value
                break

    # ------------------------------------------------------------------
    # Step 3: get video start time and duration
    # ------------------------------------------------------------------
    dto_str: str = str(meta.get("DateTimeOriginal", ""))
    video_start_utc: float = _parse_dto_to_epoch(dto_str)

    duration_seconds = 0.0
    try:
        duration_seconds = ffprobe_duration(insv_00_path)
    except Exception as exc:
        logger.warning(
            "ffprobe_duration failed for '{}': {}", insv_00_path, exc
        )

    # ------------------------------------------------------------------
    # Step 4: parse triplets
    # ------------------------------------------------------------------
    gyro_triplets: list[tuple[float, float, float]] = []
    accel_triplets: list[tuple[float, float, float]] = []

    if gyro_value is not None:
        gyro_triplets = _extract_triplets(gyro_value)
        logger.debug(
            "Parsed {} gyro samples from exiftool tag", len(gyro_triplets)
        )

    if accel_value is not None:
        accel_triplets = _extract_triplets(accel_value)
        logger.debug(
            "Parsed {} accel samples from exiftool tag", len(accel_triplets)
        )

    # ------------------------------------------------------------------
    # Step 5: fallback to raw trailer if primary path yielded nothing
    # ------------------------------------------------------------------
    method = "exiftool"
    if len(gyro_triplets) == 0:
        logger.debug(
            "Primary IMU extraction yielded 0 samples — trying raw trailer for '{}'",
            Path(insv_00_path).name,
        )
        try:
            trailer = parse_insv_trailer_raw(insv_00_path)
            imu_bytes = trailer.get("0x300") or trailer.get("0x0300")
            if imu_bytes and len(imu_bytes) >= 12:
                # Interleaved layout assumption: gyro then accel per sample (6 floats)
                all_floats: list[tuple[float, float, float]] = _unpack_float3_stream(imu_bytes)
                # Try treating even-indexed triplets as gyro, odd as accel
                gyro_triplets = all_floats[0::2]
                accel_triplets = all_floats[1::2]
                if gyro_triplets:
                    method = "trailer"
                    logger.debug(
                        "IMU extracted via raw trailer parsing ({} gyro, {} accel samples)",
                        len(gyro_triplets),
                        len(accel_triplets),
                    )
                else:
                    logger.warning(
                        "Raw trailer for '{}' contained no decodable IMU samples",
                        Path(insv_00_path).name,
                    )
            else:
                logger.warning(
                    "No IMU section (0x0300) found in INSV trailer of '{}'",
                    Path(insv_00_path).name,
                )
        except Exception as exc:
            logger.warning(
                "parse_insv_trailer_raw failed for '{}': {}", insv_00_path, exc
            )

    # ------------------------------------------------------------------
    # Step 6: if still empty, return failure
    # ------------------------------------------------------------------
    if len(gyro_triplets) == 0:
        return ([], _failure_meta)

    # ------------------------------------------------------------------
    # Step 7: build evenly-spaced timestamps
    # ------------------------------------------------------------------
    n_gyro = len(gyro_triplets)
    n_accel = len(accel_triplets)

    gyro_rate_hz = (n_gyro / duration_seconds) if duration_seconds > 0 else 0.0
    accel_rate_hz = (n_accel / duration_seconds) if duration_seconds > 0 else 0.0

    # Build elapsed-second timestamps so normalize_insv_timestamps() can anchor them
    if duration_seconds > 0 and n_gyro > 1:
        gyro_ts_elapsed = [
            (i / (n_gyro - 1)) * duration_seconds for i in range(n_gyro)
        ]
    else:
        step = 1.0 / gyro_rate_hz if gyro_rate_hz > 0 else 0.005
        gyro_ts_elapsed = [i * step for i in range(n_gyro)]

    if n_accel > 0:
        if duration_seconds > 0 and n_accel > 1:
            accel_ts_elapsed = [
                (i / (n_accel - 1)) * duration_seconds for i in range(n_accel)
            ]
        else:
            step = 1.0 / accel_rate_hz if accel_rate_hz > 0 else 0.005
            accel_ts_elapsed = [i * step for i in range(n_accel)]
    else:
        accel_ts_elapsed = []

    # Convert elapsed seconds to absolute nanosecond UTC epoch
    def _elapsed_to_ns(elapsed_s: float) -> int:
        return int((video_start_utc + elapsed_s) * 1e9)

    gyro_ts_ns = [_elapsed_to_ns(t) for t in gyro_ts_elapsed]
    accel_ts_ns = [_elapsed_to_ns(t) for t in accel_ts_elapsed]

    # ------------------------------------------------------------------
    # Step 8: zip into IMUSamples
    # ------------------------------------------------------------------
    if n_accel != n_gyro:
        logger.warning(
            "Gyro ({}) and accel ({}) sample counts differ for '{}' — "
            "truncating to min length",
            n_gyro,
            n_accel,
            Path(insv_00_path).name,
        )
        warnings_list.append(
            f"Gyro ({n_gyro}) and accel ({n_accel}) sample counts differ; "
            "truncated to min length"
        )

    n_samples = min(
        n_gyro,
        n_accel if n_accel > 0 else n_gyro,
    )

    imu_samples: list[IMUSample] = []
    for i in range(n_samples):
        gx, gy, gz = gyro_triplets[i]
        if n_accel > 0 and i < n_accel:
            ax, ay, az = accel_triplets[i]
        else:
            ax, ay, az = 0.0, 0.0, 0.0

        imu_samples.append(
            IMUSample(
                timestamp_ns=gyro_ts_ns[i],
                accel_x=ax,
                accel_y=ay,
                accel_z=az,
                gyro_x=gx,
                gyro_y=gy,
                gyro_z=gz,
            )
        )

    logger.debug(
        "extract_imu_from_insv '{}': {} samples via '{}' (gyro_rate={:.1f} Hz)",
        Path(insv_00_path).name,
        len(imu_samples),
        method,
        gyro_rate_hz,
    )

    metadata: dict = {
        "method": method,
        "gyro_rate_hz": round(gyro_rate_hz, 2),
        "accel_rate_hz": round(accel_rate_hz, 2),
        "axes_convention": "X=right,Y=up,Z=forward (Insta360 native)",
        "sample_count": len(imu_samples),
        "warnings": warnings_list,
    }
    return (imu_samples, metadata)


def parse_insv_trailer_raw(insv_path: str) -> dict[str, bytes]:
    """Low-level parser for the binary metadata trailer appended to INSV files.

    The Insta360 X4 embeds proprietary telemetry in a trailer block at the end
    of the _00_ file, terminated by a 32-byte magic string.

    Known section IDs:
      0x0101  MakerNotes
      0x0200  GPS
      0x0300  IMU (gyro + accel, interleaved float32 triplets)
      0x0400  Exposure

    Args:
        insv_path: Path to the INSV file (typically the _00_ file).

    Returns:
        Dict mapping hex section ID string (e.g. "0x300") to raw bytes.
        Returns empty dict if the magic is absent or on any parse error.
    """
    MAGIC = "8db42d694ccc418790edff439fe026bf"
    MAGIC_LEN = 32
    ENTRY_SIZE = 6  # uint16 section_id + uint32 data_offset

    collected: dict[str, bytes] = {}

    try:
        with open(insv_path, "rb") as f:
            # Check magic at end of file
            try:
                f.seek(-MAGIC_LEN, 2)
            except OSError:
                logger.warning(
                    "INSV file '{}' is too small to contain a trailer",
                    Path(insv_path).name,
                )
                return {}

            magic_bytes = f.read(MAGIC_LEN)
            magic_str = magic_bytes.decode("ascii", errors="ignore").rstrip("\x00")

            if magic_str != MAGIC:
                logger.warning(
                    "INSV trailer magic not found in '{}' (got '{}...')",
                    Path(insv_path).name,
                    magic_str[:16],
                )
                return {}

            # Determine total file size to calculate absolute positions
            f.seek(0, 2)
            file_size = f.tell()

            # The entry table is immediately before the magic.
            # Walk backward reading (section_id: uint16, data_offset: uint32) entries.
            # We don't know the entry count, so we stop when we reach an obviously
            # invalid offset (> file_size or == 0) or when consecutive entries
            # overlap with the data regions.
            trailer_start = file_size - MAGIC_LEN  # first byte of magic
            cursor = trailer_start - ENTRY_SIZE

            seen_ids: set[int] = set()
            entries: list[tuple[int, int]] = []  # (section_id, data_offset)

            while cursor >= 0:
                try:
                    f.seek(cursor)
                    raw = f.read(ENTRY_SIZE)
                    if len(raw) < ENTRY_SIZE:
                        break
                    section_id, data_offset = struct.unpack("<HI", raw)
                except struct.error:
                    logger.warning(
                        "struct.unpack failed reading INSV trailer entry at offset {}",
                        cursor,
                    )
                    break

                # Sanity-check: data_offset must be within the file and non-zero
                if data_offset == 0 or data_offset >= trailer_start:
                    break

                # Avoid duplicate section IDs (stop if we've looped)
                if section_id in seen_ids:
                    break
                seen_ids.add(section_id)
                entries.append((section_id, data_offset))
                cursor -= ENTRY_SIZE

            if not entries:
                logger.warning(
                    "No valid entries found in INSV trailer of '{}'",
                    Path(insv_path).name,
                )
                return {}

            # Sort entries by data_offset ascending to compute each section's length
            entries.sort(key=lambda e: e[1])

            for idx, (section_id, data_offset) in enumerate(entries):
                # Section length = next entry's offset - this entry's offset,
                # or distance to the start of the entry table for the last entry.
                entry_table_start = trailer_start - (len(entries) * ENTRY_SIZE)
                if idx + 1 < len(entries):
                    next_offset = entries[idx + 1][1]
                else:
                    next_offset = entry_table_start

                section_len = next_offset - data_offset
                if section_len <= 0:
                    logger.debug(
                        "INSV section 0x{:04x} has non-positive length {}, skipping",
                        section_id,
                        section_len,
                    )
                    continue

                try:
                    f.seek(data_offset)
                    section_data = f.read(section_len)
                    hex_key = hex(section_id)
                    collected[hex_key] = section_data
                    logger.debug(
                        "INSV trailer section {}: {} bytes at offset {}",
                        hex_key,
                        len(section_data),
                        data_offset,
                    )
                except OSError as exc:
                    logger.warning(
                        "Failed to read INSV section 0x{:04x}: {}", section_id, exc
                    )
                    continue

    except OSError as exc:
        logger.warning(
            "Could not open INSV file '{}' for trailer parsing: {}", insv_path, exc
        )
        return {}

    return collected


def normalize_insv_timestamps(
    samples: list,
    video_start_time_utc: float,
    timestamp_field: str,
) -> list:
    """Normalise INSV telemetry timestamps to nanoseconds Unix epoch in-place.

    Detects the timestamp representation by inspecting the first sample value:
      - elapsed seconds (< 7200)  → anchored to video_start_time_utc
      - epoch milliseconds (> 1e12) → converted to nanoseconds
      - epoch seconds (> 1e9)     → converted to nanoseconds
      - zero / unknown            → evenly spaced from video_start_time_utc

    Works with both Pydantic model instances (attribute access) and plain dicts
    (key access).

    Args:
        samples:              List of objects or dicts with a timestamp field.
        video_start_time_utc: Unix epoch float (seconds UTC) of the recording start.
        timestamp_field:      Name of the timestamp attribute or dict key.

    Returns:
        The same list with timestamps replaced by nanosecond int values.
        Returns the list unmodified (with a warning) if it is empty.
    """
    if not samples:
        return samples

    def _get(sample: Any) -> Any:
        if isinstance(sample, dict):
            return sample.get(timestamp_field, 0)
        return getattr(sample, timestamp_field, None)

    def _set(sample: Any, value: int) -> None:
        if isinstance(sample, dict):
            sample[timestamp_field] = value
        else:
            object.__setattr__(sample, timestamp_field, value)

    first_val = _get(samples[0])
    if first_val is None:
        first_val = 0

    try:
        first_val = float(first_val)
    except (ValueError, TypeError):
        logger.warning(
            "normalize_insv_timestamps: cannot interpret first timestamp value '{}', "
            "generating evenly spaced timestamps",
            first_val,
        )
        first_val = 0.0

    n = len(samples)

    if first_val == 0.0:
        # Unknown — generate evenly spaced from video start
        logger.debug(
            "normalize_insv_timestamps: zero first timestamp — generating "
            "{} evenly spaced timestamps from {:.3f}",
            n,
            video_start_time_utc,
        )
        # We don't know duration here; use a nominal 1 Hz spacing as a placeholder.
        # Callers that know duration should set timestamps before calling this function.
        for i, sample in enumerate(samples):
            _set(sample, int((video_start_time_utc + i) * 1e9))

    elif first_val < 7200:
        # Elapsed seconds from recording start
        logger.debug(
            "normalize_insv_timestamps: detected elapsed-seconds format "
            "(first_val={:.4f}), anchoring to UTC {:.3f}",
            first_val,
            video_start_time_utc,
        )
        for sample in samples:
            elapsed = float(_get(sample) or 0.0)
            _set(sample, int((video_start_time_utc + elapsed) * 1e9))

    elif first_val > 1e12:
        # Epoch milliseconds
        logger.debug(
            "normalize_insv_timestamps: detected epoch-milliseconds format "
            "(first_val={:.0f})",
            first_val,
        )
        for sample in samples:
            ms = float(_get(sample) or 0.0)
            _set(sample, int(ms * 1e6))

    elif first_val > 1e9:
        # Epoch seconds
        logger.debug(
            "normalize_insv_timestamps: detected epoch-seconds format "
            "(first_val={:.3f})",
            first_val,
        )
        for sample in samples:
            sec = float(_get(sample) or 0.0)
            _set(sample, int(sec * 1e9))

    else:
        # Fallback — treat as elapsed seconds (covers 0 < v < 7200 already handled,
        # but guards against e.g. sub-second positive values)
        logger.warning(
            "normalize_insv_timestamps: ambiguous first timestamp value {:.4f} — "
            "treating as elapsed seconds",
            first_val,
        )
        for sample in samples:
            elapsed = float(_get(sample) or 0.0)
            _set(sample, int((video_start_time_utc + elapsed) * 1e9))

    return samples
