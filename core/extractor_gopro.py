"""
core/extractor_gopro.py — GoPro Hero 11 GPMF telemetry + frame extraction.

Produces EuRoC-compatible output:
  <output_dir>/
    cam0/data/<timestamp_ns>.jpg
    imu0/data.csv          (EuRoC header)
    gps.csv

All timestamps are int nanoseconds. No mock data — raises on missing files.
"""
from __future__ import annotations

import csv
import shutil
from collections.abc import Callable
from pathlib import Path

from loguru import logger

from core.audit import detect_gopro_chapters
from core.models import (
    DeviceType,
    ExtractionConfig,
    ExtractedSession,
    FrameMetadata,
    GPSSample,
    IMUSample,
)
from core.utils import ffmpeg_extract_frames, ffprobe

# EuRoC IMU CSV header — must match exactly for ORBSLAM3 / VINS-Mono
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
# Public entry point
# ---------------------------------------------------------------------------

def extract_gopro(
    mp4_path: str | Path,
    config: ExtractionConfig,
    output_dir: str | Path,
    progress_callback: Callable[[int, str], None] | None = None,
) -> ExtractedSession:
    """Extract IMU, GPS, and frames from a GoPro Hero 11 recording.

    Handles chapter-split files automatically (GH01/GH02/...).

    Args:
        mp4_path:          Path to any chapter of the recording.
        config:            Extraction configuration (fps, format, etc.).
        output_dir:        Root of the EuRoC output tree.
        progress_callback: Optional callable(pct: int, message: str) called
                           at each major stage so the UI can show live progress.

    Returns:
        ExtractedSession with all extracted data references and stats.
    """
    def _progress(pct: int, msg: str) -> None:
        logger.debug("extract_gopro [{}%] {}", pct, msg)
        if progress_callback:
            progress_callback(pct, msg)

    mp4_path = Path(mp4_path)
    output_dir = Path(output_dir)

    if not mp4_path.exists():
        raise FileNotFoundError(
            f"GoPro file not found: '{mp4_path}'. "
            "Ensure the file is accessible before running extraction."
        )

    _progress(5, "Detecting chapters…")
    chapters = detect_gopro_chapters(mp4_path)
    if not chapters:
        chapters = [mp4_path]

    logger.info(
        "extract_gopro: {} ({} chapter(s)) → {}",
        mp4_path.name, len(chapters), output_dir,
    )

    # Prepare output directories (EuRoC layout)
    cam0_dir = output_dir / "cam0" / "data"
    imu0_dir = output_dir / "imu0"
    cam0_dir.mkdir(parents=True, exist_ok=True)
    imu0_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Extract GPMF telemetry across all chapters ------------------
    all_imu: list[IMUSample] = []
    all_gps: list[GPSSample] = []
    total_duration = 0.0
    chapter_offsets: list[int] = []

    n_ch = len(chapters)
    for ch_idx, chapter in enumerate(chapters):
        pct = 10 + int(ch_idx / n_ch * 30)
        _progress(pct, f"Extracting telemetry: {chapter.name} ({ch_idx+1}/{n_ch})…")
        offset_ns = int(total_duration * 1e9)
        chapter_offsets.append(offset_ns)
        dur = float(ffprobe(chapter).get("format", {}).get("duration", 0))
        imu, gps = _extract_gpmf(chapter, offset_ns)
        all_imu.extend(imu)
        all_gps.extend(gps)
        total_duration += dur
        logger.debug(
            "  chapter {}: {:.1f}s, {} IMU, {} GPS",
            chapter.name, dur, len(imu), len(gps),
        )

    # --- Step 2: Interpolate gyro onto accel timestamps (if requested) -------
    if config.imu_interpolation and all_imu:
        _progress(42, "Interpolating gyro onto accel timestamps…")
        all_imu = _interpolate_gyro(all_imu)

    # --- Step 3: Extract video frames -----------------------------------------
    _progress(45, "Extracting video frames…")
    frame_paths: list[str] = []
    frame_timestamps_ns: list[int] = []
    all_frame_meta: list[FrameMetadata] = []

    for i, chapter in enumerate(chapters):
        ch_pct_start = 45 + int(i / n_ch * 40)
        ch_pct_end = 45 + int((i + 1) / n_ch * 40)
        _progress(ch_pct_start, f"Extracting frames: {chapter.name} ({i+1}/{n_ch})…")
        offset_ns = chapter_offsets[i]
        native_fps = _get_chapter_fps(chapter)

        f_paths, f_ts, f_meta = _extract_frames(
            chapter,
            cam0_dir,
            config.frame_fps,
            config.frame_quality,
            offset_ns,
            native_fps,
            config.blur_threshold,
            config.dedup_threshold,
            progress_callback=progress_callback,
            pct_start=ch_pct_start,
            pct_end=ch_pct_end,
        )
        frame_paths.extend(f_paths)
        frame_timestamps_ns.extend(f_ts)
        all_frame_meta.extend(f_meta)

    # Sort frames by timestamp (chapters are already ordered but be explicit)
    paired = sorted(zip(frame_timestamps_ns, frame_paths, all_frame_meta))
    frame_timestamps_ns = [t for t, _, _ in paired]
    frame_paths = [p for _, p, _ in paired]
    all_frame_meta = [m for _, _, m in paired]

    # --- Step 4: Write EuRoC output files -------------------------------------
    _progress(92, "Writing EuRoC CSV files…")
    imu_csv_path = imu0_dir / "data.csv"
    gps_csv_path = output_dir / "gps.csv"

    _write_euroc_imu(all_imu, imu_csv_path)
    _write_gps_csv(all_gps, gps_csv_path)

    # --- Step 5: Compute motion profile ---------------------------------------
    _progress(95, "Computing motion profile…")
    motion_profile = _compute_motion_profile(all_imu)

    # --- Step 6: Build and return ExtractedSession ----------------------------
    _progress(97, "Building session record…")
    blurry_count = sum(1 for m in all_frame_meta if m.is_blurry)
    dedup_count = sum(1 for m in all_frame_meta if m.is_duplicate)
    stats = {
        "chapters": len(chapters),
        "imu_count": len(all_imu),
        "gps_count": len(all_gps),
        "frame_count": len(frame_paths),
        "imu_rate_hz": round(len(all_imu) / total_duration, 2) if total_duration else 0.0,
        "gps_rate_hz": round(len(all_gps) / total_duration, 2) if total_duration else 0.0,
        "output_dir": str(output_dir),
        "blurry_frames": blurry_count,
        "duplicate_frames": dedup_count,
        "motion_profile": motion_profile,
    }

    logger.info(
        "extract_gopro complete: {} IMU, {} GPS, {} frames ({} blurry, {} dup), {:.1f}s",
        len(all_imu), len(all_gps), len(frame_paths),
        blurry_count, dedup_count, total_duration,
    )

    return ExtractedSession(
        session_id=config.session_id,
        device_type=DeviceType.GOPRO,
        imu_samples=all_imu,
        gps_samples=all_gps,
        frame_paths=frame_paths,
        frame_timestamps_ns=frame_timestamps_ns,
        duration_seconds=total_duration,
        stats=stats,
        frame_metadata=all_frame_meta,
    )


# ---------------------------------------------------------------------------
# GPMF extraction
# ---------------------------------------------------------------------------

def _extract_gpmf(
    chapter: Path, offset_ns: int
) -> tuple[list[IMUSample], list[GPSSample]]:
    """Extract ACCL, GYRO, GPS5 from a single chapter via py_gpmf_parser.

    Uses GoProTelemetryExtractor which returns (values_array, timestamps_seconds).
    Timestamps from the library are in seconds since recording start; we convert
    to int nanoseconds and add offset_ns for continuity across chapters.
    """
    imu_samples: list[IMUSample] = []
    gps_samples: list[GPSSample] = []

    try:
        from py_gpmf_parser import GoProTelemetryExtractor  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "py-gpmf-parser is required for GoPro extraction. "
            "Install it with: pip install py-gpmf-parser"
        ) from exc

    extractor = GoProTelemetryExtractor(str(chapter))
    try:
        extractor.open_source()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to open GPMF source '{chapter}': {exc}. "
            "Ensure the file is a valid GoPro Hero 11 MP4 with telemetry."
        ) from exc

    try:
        # --- ACCL (accelerometer) -------------------------------------------
        try:
            accl_vals, accl_ts_s = extractor.extract_data("ACCL")
        except Exception:
            accl_vals, accl_ts_s = [], []

        # --- GYRO (gyroscope) -----------------------------------------------
        try:
            gyro_vals, gyro_ts_s = extractor.extract_data("GYRO")
        except Exception:
            gyro_vals, gyro_ts_s = [], []

        if len(accl_vals) == 0:
            logger.warning("No ACCL stream in '{}' — IMU data unavailable", chapter.name)
            accl_ts_ns: list[int] = []
            gyro_interp: list[tuple[float, float, float]] = []
        else:
            # Convert timestamps: seconds → nanoseconds + chapter offset
            accl_ts_ns = [int(t * 1e9) + offset_ns for t in accl_ts_s]

            if len(gyro_vals) > 0:
                gyro_ts_ns = [int(t * 1e9) + offset_ns for t in gyro_ts_s]
                gyro_xyz = [(float(g[0]), float(g[1]), float(g[2])) for g in gyro_vals]
                gyro_interp = _interp_xyz(gyro_ts_ns, gyro_xyz, accl_ts_ns)
            else:
                logger.warning("No GYRO stream in '{}' — gyro set to zero", chapter.name)
                gyro_interp = [(0.0, 0.0, 0.0)] * len(accl_vals)

            for ts, accl, gyro in zip(accl_ts_ns, accl_vals, gyro_interp):
                imu_samples.append(IMUSample(
                    timestamp_ns=ts,
                    accel_x=float(accl[0]),
                    accel_y=float(accl[1]),
                    accel_z=float(accl[2]),
                    gyro_x=float(gyro[0]),
                    gyro_y=float(gyro[1]),
                    gyro_z=float(gyro[2]),
                ))

        # --- GPS5 -----------------------------------------------------------
        try:
            gps_vals, gps_ts_s = extractor.extract_data("GPS5")
        except Exception:
            gps_vals, gps_ts_s = [], []

        for ts_s, g in zip(gps_ts_s, gps_vals):
            # GPS5 columns: [lat, lon, alt, speed_2d, speed_3d]
            fix = 3 if len(g) >= 3 else 0
            gps_samples.append(GPSSample(
                timestamp_ns=int(ts_s * 1e9) + offset_ns,
                latitude=float(g[0]),
                longitude=float(g[1]),
                altitude_m=float(g[2]) if len(g) > 2 else 0.0,
                speed_mps=float(g[3]) if len(g) > 3 else 0.0,
                fix_type=fix,
            ))

    finally:
        try:
            extractor.close_source()
        except Exception:
            pass

    return imu_samples, gps_samples


# ---------------------------------------------------------------------------
# Gyro interpolation
# ---------------------------------------------------------------------------

def _interpolate_gyro(samples: list[IMUSample]) -> list[IMUSample]:
    """No-op pass-through: gyro is already interpolated onto ACCL timestamps
    during GPMF extraction in _extract_gpmf. This function is preserved as a
    hook for future resampling (e.g. upsampling to 400 Hz)."""
    return samples


def _interp_xyz(
    src_ts: list[int],
    src_vals: list,
    dst_ts: list[int],
) -> list[tuple[float, float, float]]:
    """Linear interpolation of xyz values from src timestamps to dst timestamps."""
    if not src_ts or not src_vals:
        return [(0.0, 0.0, 0.0)] * len(dst_ts)

    result: list[tuple[float, float, float]] = []
    n = len(src_ts)

    for t in dst_ts:
        if t <= src_ts[0]:
            v = src_vals[0]
            result.append((float(v[0]), float(v[1]), float(v[2])))
            continue
        if t >= src_ts[-1]:
            v = src_vals[-1]
            result.append((float(v[0]), float(v[1]), float(v[2])))
            continue
        # Binary search for bracket
        lo, hi = 0, n - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if src_ts[mid] <= t:
                lo = mid
            else:
                hi = mid
        alpha = (t - src_ts[lo]) / (src_ts[hi] - src_ts[lo])
        vlo = src_vals[lo]
        vhi = src_vals[hi]
        result.append((
            float(vlo[0]) + alpha * (float(vhi[0]) - float(vlo[0])),
            float(vlo[1]) + alpha * (float(vhi[1]) - float(vlo[1])),
            float(vlo[2]) + alpha * (float(vhi[2]) - float(vlo[2])),
        ))

    return result


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def _get_chapter_fps(chapter: Path) -> float:
    probe = ffprobe(chapter)
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video":
            fps_str = stream.get("r_frame_rate", "60/1")
            num, den = (int(x) for x in fps_str.split("/"))
            return num / den if den else 60.0
    return 60.0


def _extract_frames(
    chapter: Path,
    cam0_dir: Path,
    target_fps: float,
    quality: int,
    offset_ns: int,
    native_fps: float,
    blur_threshold: float = 100.0,
    dedup_threshold: float = 0.98,
    progress_callback: Callable[[int, str], None] | None = None,
    pct_start: int = 45,
    pct_end: int = 90,
) -> tuple[list[str], list[int], list[FrameMetadata]]:
    """Extract frames from one chapter, naming them by timestamp_ns.

    Returns (list_of_paths, list_of_timestamp_ns, list_of_FrameMetadata).
    """
    # ffmpeg writes to a temp pattern; we rename to timestamp_ns afterward
    tmp_pattern = cam0_dir / f"_tmp_{chapter.stem}_%06d.jpg"

    # quality → ffmpeg -q:v scale (2=best, 31=worst)
    q = max(2, min(31, int((100 - quality) / 100 * 29) + 2))

    # Get video duration for progress tracking
    chapter_dur = float(ffprobe(chapter).get("format", {}).get("duration", 0))

    def _ffmpeg_progress(pct: float) -> None:
        if progress_callback:
            scaled = pct_start + int(pct / 100 * (pct_end - pct_start))
            progress_callback(scaled, f"Extracting frames: {chapter.name} ({int(pct)}%)")

    ffmpeg_extract_frames(
        chapter,
        tmp_pattern,
        fps=target_fps,
        quality=q,
        progress_callback=_ffmpeg_progress,
        total_duration=chapter_dur,
    )

    # Gather written frames and rename to timestamp_ns
    written = sorted(cam0_dir.glob(f"_tmp_{chapter.stem}_*.jpg"))
    step_ns = int(1e9 / target_fps)  # nominal step between frames

    frame_paths: list[str] = []
    frame_timestamps_ns: list[int] = []
    frame_metadata: list[FrameMetadata] = []

    try:
        import cv2  # type: ignore[import]
        have_cv2 = True
    except ImportError:
        have_cv2 = False
        logger.warning("opencv-python not available — blur/dedup scoring skipped")

    prev_gray = None
    prev_score: float = -1.0

    n_written = len(written)
    for idx, tmp in enumerate(written):
        ts_ns = offset_ns + idx * step_ns
        dest = cam0_dir / f"{ts_ns}.jpg"
        tmp.rename(dest)
        frame_paths.append(str(dest))
        frame_timestamps_ns.append(ts_ns)

        # Emit progress during blur/dedup scoring (pct_end → pct_end+5)
        if progress_callback and n_written > 0 and idx % max(1, n_written // 20) == 0:
            score_pct = pct_end + int(idx / n_written * 5)
            progress_callback(score_pct, f"Scoring frames: {idx+1}/{n_written}")

        blur_score = 0.0
        is_blurry = False
        is_duplicate = False

        if have_cv2:
            img = cv2.imread(str(dest))
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                is_blurry = blur_score < blur_threshold

                # SSIM-based dedup against previous frame
                if prev_gray is not None and prev_gray.shape == gray.shape:
                    try:
                        from skimage.metrics import structural_similarity  # type: ignore[import]
                        ssim_val = structural_similarity(prev_gray, gray)
                        is_duplicate = ssim_val > dedup_threshold
                    except ImportError:
                        # Fallback: normalized cross-correlation approximation
                        import numpy as np
                        curr_f = gray.astype(float)
                        prev_f = prev_gray.astype(float)
                        norm = np.linalg.norm(curr_f) * np.linalg.norm(prev_f)
                        if norm > 0:
                            ncc = float(np.sum(curr_f * prev_f) / norm)
                            is_duplicate = ncc > dedup_threshold
                prev_gray = gray

        frame_metadata.append(FrameMetadata(
            frame_path=str(dest),
            timestamp_ns=ts_ns,
            blur_score=round(blur_score, 2),
            is_blurry=is_blurry,
            is_duplicate=is_duplicate,
        ))

    logger.debug(
        "  {} frames from {} at {}fps",
        len(frame_paths), chapter.name, target_fps,
    )
    return frame_paths, frame_timestamps_ns, frame_metadata


# ---------------------------------------------------------------------------
# Motion profile
# ---------------------------------------------------------------------------

def _compute_motion_profile(
    samples: list[IMUSample],
    window_s: float = 1.0,
) -> list[tuple[int, str]]:
    """Classify 1-second windows of IMU data into motion categories.

    Returns list of (timestamp_ns, label) where label is one of:
      "stationary", "high_motion", "normal"
    """
    import math

    if not samples:
        return []

    profile: list[tuple[int, str]] = []
    t_start_ns = samples[0].timestamp_ns
    window_ns = int(window_s * 1e9)
    window_end_ns = t_start_ns + window_ns
    bucket: list[float] = []

    def _classify(magnitudes: list[float]) -> str:
        if not magnitudes:
            return "normal"
        mean_mag = sum(magnitudes) / len(magnitudes)
        variance = sum((m - mean_mag) ** 2 for m in magnitudes) / len(magnitudes)
        std_mag = math.sqrt(variance)
        if mean_mag > 15.0:
            return "high_motion"
        if std_mag < 0.15:
            return "stationary"
        return "normal"

    for s in samples:
        mag = math.sqrt(s.accel_x**2 + s.accel_y**2 + s.accel_z**2)
        if s.timestamp_ns < window_end_ns:
            bucket.append(mag)
        else:
            profile.append((t_start_ns, _classify(bucket)))
            t_start_ns = window_end_ns
            window_end_ns = t_start_ns + window_ns
            bucket = [mag]

    if bucket:
        profile.append((t_start_ns, _classify(bucket)))

    return profile


# ---------------------------------------------------------------------------
# EuRoC output writers
# ---------------------------------------------------------------------------

def _write_euroc_imu(samples: list[IMUSample], path: Path) -> None:
    """Write IMU data in EuRoC CSV format."""
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
    """Write GPS data as a simple CSV alongside the EuRoC tree."""
    with open(path, "w", newline="") as f:
        f.write(_EUROC_GPS_HEADER + "\n")
        for s in samples:
            f.write(
                f"{s.timestamp_ns},"
                f"{s.latitude:.10f},{s.longitude:.10f},"
                f"{s.altitude_m:.4f},{s.speed_mps:.4f},{s.fix_type}\n"
            )
    logger.debug("Wrote GPS CSV: {} ({} rows)", path, len(samples))
