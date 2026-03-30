"""
core/insta360_processor.py — Insta360 X4 full processing pipeline.

Handles dual fisheye stitching → equirectangular → 4 perspective views →
frame extraction → EuRoC multi-camera dataset assembly.

All ffmpeg calls use subprocess.Popen for streaming progress.
Lanczos interpolation is mandatory throughout.
All timestamps are nanosecond integers.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable

import yaml
from loguru import logger

from core.models import (
    Insta360ExtractionResult,
    Insta360ProcessingConfig,
    INSVPair,
    PerspectiveView,
)
from core.utils import ffprobe, ffprobe_duration, ffmpeg_run


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PERSPECTIVE_VIEWS = {
    "front": {"yaw": 0,   "pitch": 0, "roll": 0, "description": "Forward-facing"},
    "right": {"yaw": 90,  "pitch": 0, "roll": 0, "description": "Right side view"},
    "rear":  {"yaw": 180, "pitch": 0, "roll": 0, "description": "Rear-facing view"},
    "left":  {"yaw": -90, "pitch": 0, "roll": 0, "description": "Left side view"},
}

_EUROC_IMU_HEADER = (
    "#timestamp [ns],"
    "w_RS_S_x [rad s^-1],"
    "w_RS_S_y [rad s^-1],"
    "w_RS_S_z [rad s^-1],"
    "a_RS_S_x [m s^-2],"
    "a_RS_S_y [m s^-2],"
    "a_RS_S_z [m s^-2]"
)
_EUROC_GPS_HEADER = "timestamp_ns,latitude,longitude,altitude_m,speed_mps,fix_type"


# ---------------------------------------------------------------------------
# verify_ffmpeg_v360_support
# ---------------------------------------------------------------------------

def verify_ffmpeg_v360_support() -> dict:
    """Check ffmpeg capabilities required for dual fisheye stitching.

    Returns a dict with version info and feature flags for v360 support.
    """
    try:
        ver_proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-version"],
            capture_output=True, text=True,
        )
        version_str = "unknown"
        m = re.search(r"(\d+\.\d+)", ver_proc.stdout.splitlines()[0] if ver_proc.stdout else "")
        if m:
            version_str = m.group(1)

        filters_proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"],
            capture_output=True, text=True,
        )
        v360_available = "v360" in filters_proc.stdout

        h_proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-h", "filter=v360"],
            capture_output=True, text=True,
        )
        dfisheye_supported = "dfisheye" in h_proc.stdout
        lanczos_supported = "lanczos" in h_proc.stdout

        minimum_met = False
        if version_str != "unknown":
            major, minor = map(int, version_str.split(".")[:2])
            minimum_met = major > 4 or (major == 4 and minor >= 3)

        recommendation = (
            "OK" if minimum_met
            else (
                f"Upgrade ffmpeg to 4.3+ (current: {version_str}). "
                "sudo apt install ffmpeg"
            )
        )

        return {
            "v360_available": v360_available,
            "dfisheye_supported": dfisheye_supported,
            "lanczos_supported": lanczos_supported,
            "ffmpeg_version": version_str,
            "minimum_met": minimum_met,
            "recommendation": recommendation,
        }

    except Exception as exc:
        logger.warning("verify_ffmpeg_v360_support failed: {}", exc)
        return {
            "v360_available": False,
            "dfisheye_supported": False,
            "lanczos_supported": False,
            "ffmpeg_version": "unknown",
            "minimum_met": False,
            "recommendation": (
                "ffmpeg not found or not functional. "
                "Install with: sudo apt install ffmpeg"
            ),
        }


# ---------------------------------------------------------------------------
# validate_insv_pair
# ---------------------------------------------------------------------------

def validate_insv_pair(insv_pair: INSVPair, output_dir: str) -> list[str]:
    """Pre-flight validation for an INSV pair before processing.

    Returns a list of issue strings. Empty list means no problems detected.
    """
    issues: list[str] = []

    # Back file must exist
    if not Path(insv_pair.back_path).exists():
        issues.append(f"Back lens file not found: {insv_pair.back_path}")

    # Front file handling
    if insv_pair.front_path is None:
        issues.append(
            "Front lens file (_10_) missing — stitching requires both files"
        )
    elif not Path(insv_pair.front_path).exists():
        issues.append(f"Front lens file not found: {insv_pair.front_path}")

    # Duration mismatch check when both files exist
    if (
        insv_pair.front_path is not None
        and Path(insv_pair.back_path).exists()
        and Path(insv_pair.front_path).exists()
    ):
        try:
            back_dur = ffprobe_duration(insv_pair.back_path)
            front_dur = ffprobe_duration(insv_pair.front_path)
            if abs(back_dur - front_dur) > 1.0:
                issues.append(
                    f"Duration mismatch: back={back_dur:.1f}s, front={front_dur:.1f}s "
                    "— files may be from different recordings"
                )
        except Exception as exc:
            issues.append(f"Could not determine file durations: {exc}")

    # Disk space check
    try:
        required_gb = insv_pair.file_size_mb / 1024 * 8
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        free_gb = shutil.disk_usage(output_dir).free / (1024 ** 3)
        if free_gb < required_gb:
            issues.append(
                f"Insufficient disk space: need {required_gb:.1f}GB free, "
                f"have {free_gb:.1f}GB"
            )
    except Exception as exc:
        issues.append(f"Disk space check failed: {exc}")

    # ffmpeg version check
    v360_info = verify_ffmpeg_v360_support()
    if not v360_info["minimum_met"]:
        issues.append(
            f"ffmpeg {v360_info['ffmpeg_version']} detected — upgrade to 4.3+ "
            "for dual fisheye support: sudo apt install ffmpeg"
        )

    return issues


# ---------------------------------------------------------------------------
# stitch_dual_fisheye_to_equirectangular
# ---------------------------------------------------------------------------

def stitch_dual_fisheye_to_equirectangular(
    front_path: str,
    back_path: str,
    output_path: str,
    output_width: int = 7680,
    output_height: int = 3840,
    quality_crf: int = 10,
    fisheye_fov: float = 210.0,
    progress_callback: Callable[[str, int], None] | None = None,
) -> str:
    """Stitch paired INSV files into a single equirectangular MP4.

    Uses ffmpeg v360=dfisheye with Lanczos interpolation.
    Streams stderr for real-time progress reporting.

    Returns the output_path on success.
    Raises RuntimeError on ffmpeg failure.
    """
    total_s = ffprobe_duration(back_path)

    filter_complex = (
        f"[0:v][1:v]hstack=inputs=2[stacked];"
        f"[stacked]v360=dfisheye:e"
        f":ih_fov={fisheye_fov}:iv_fov={fisheye_fov}"
        f":w={output_width}:h={output_height}"
        f":interp=lanczos:reset_rot=1[equirect]"
    )

    cmd = [
        "ffmpeg", "-hide_banner", "-y",
        "-i", front_path,
        "-i", back_path,
        "-filter_complex", filter_complex,
        "-map", "[equirect]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-crf", str(quality_crf),
        "-preset", "slow",
        "-pix_fmt", "yuv420p",
        "-metadata:s:v", "spherical=true",
        "-metadata:s:v", "stitched=true",
        "-metadata:s:v", "projection_type=equirectangular",
        "-movflags", "+faststart",
        output_path,
    ]

    logger.info("Stitching dual fisheye → {}", output_path)
    logger.debug("stitch cmd: {}", " ".join(cmd))

    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True, bufsize=1)

    for line in proc.stderr:  # type: ignore[union-attr]
        line = line.rstrip()
        m = re.search(r"time=(\d+):(\d+):([\d.]+)", line)
        if m and total_s > 0:
            hours = int(m.group(1))
            minutes = int(m.group(2))
            seconds = float(m.group(3))
            elapsed_s = hours * 3600 + minutes * 60 + seconds
            pct = min(99, int(elapsed_s / total_s * 100))
            if progress_callback is not None:
                progress_callback("Stitching dual fisheye → equirectangular", pct)
        logger.trace("ffmpeg stitch | {}", line)

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg stitching failed (code {proc.returncode}). "
            "Check the log for details. Common causes: insufficient disk space, "
            "unsupported ffmpeg version."
        )

    logger.info("Stitching complete → {}", output_path)
    return output_path


# ---------------------------------------------------------------------------
# equirect_to_perspective_video
# ---------------------------------------------------------------------------

def equirect_to_perspective_video(
    equirect_path: str,
    output_dir: str,
    views: list[str] | None = None,
    output_width: int = 2160,
    output_height: int = 2160,
    h_fov: float = 110.0,
    v_fov: float = 110.0,
    quality_crf: int = 15,
    use_gpu: bool = False,
    progress_callback: Callable[[str, int], None] | None = None,
) -> dict[str, str]:
    """Reproject equirectangular video into 4 perspective view videos.

    All views are processed in parallel using a ThreadPoolExecutor.
    Uses Lanczos interpolation exclusively.
    Returns a dict mapping view name to output video path.
    """
    if views is None:
        views = list(PERSPECTIVE_VIEWS.keys())

    has_gpu = (
        use_gpu
        and subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
    )

    total_s = ffprobe_duration(equirect_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_view_progress: dict[str, int] = {v: 0 for v in views}
    progress_lock = threading.Lock()

    def _aggregate_progress() -> int:
        with progress_lock:
            return sum(per_view_progress.values()) // max(len(per_view_progress), 1)

    def _process_view(view_name: str) -> str:
        view_cfg = PERSPECTIVE_VIEWS[view_name]
        yaw = view_cfg["yaw"]
        pitch = view_cfg["pitch"]
        roll = view_cfg["roll"]
        out_path = str(out_dir / f"{view_name}.mp4")

        vf_filter = (
            f"v360=e:flat"
            f":h_fov={h_fov}:v_fov={v_fov}"
            f":yaw={yaw}:pitch={pitch}:roll={roll}"
            f":w={output_width}:h={output_height}"
            f":interp=lanczos"
        )

        if has_gpu:
            codec_args = [
                "-c:v", "h264_nvenc",
                "-cq", str(quality_crf),
                "-preset", "medium",
            ]
        else:
            codec_args = [
                "-c:v", "libx264",
                "-crf", str(quality_crf),
                "-preset", "slow",
            ]

        cmd = [
            "ffmpeg", "-hide_banner", "-y",
            "-i", equirect_path,
            "-vf", vf_filter,
        ] + codec_args + [
            "-pix_fmt", "yuv420p",
            out_path,
        ]

        logger.info("Perspective split → {} ({})", view_name, out_path)
        logger.debug("perspective cmd [{}]: {}", view_name, " ".join(cmd))

        proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True, bufsize=1)

        for line in proc.stderr:  # type: ignore[union-attr]
            line = line.rstrip()
            m = re.search(r"time=(\d+):(\d+):([\d.]+)", line)
            if m and total_s > 0:
                hours = int(m.group(1))
                minutes = int(m.group(2))
                seconds = float(m.group(3))
                elapsed_s = hours * 3600 + minutes * 60 + seconds
                pct = min(99, int(elapsed_s / total_s * 100))
                with progress_lock:
                    per_view_progress[view_name] = pct
                if progress_callback is not None:
                    progress_callback(
                        "Splitting to 4 perspective views",
                        _aggregate_progress(),
                    )
            logger.trace("ffmpeg perspective [{}] | {}", view_name, line)

        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(
                f"Perspective view '{view_name}' encoding failed "
                f"(ffmpeg exit code {proc.returncode}). "
                "Check log for details."
            )

        with progress_lock:
            per_view_progress[view_name] = 100

        return out_path

    errors: list[str] = []
    result_paths: dict[str, str] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_view = {executor.submit(_process_view, v): v for v in views}
        for future in concurrent.futures.as_completed(future_to_view):
            view_name = future_to_view[future]
            try:
                result_paths[view_name] = future.result()
            except Exception as exc:
                errors.append(f"{view_name}: {exc}")
                logger.error("Perspective view '{}' failed: {}", view_name, exc)

    if errors:
        raise RuntimeError(
            f"Perspective view encoding failed for {len(errors)} view(s): "
            + "; ".join(errors)
        )

    return {
        "front": str(out_dir / "front.mp4"),
        "right": str(out_dir / "right.mp4"),
        "rear":  str(out_dir / "rear.mp4"),
        "left":  str(out_dir / "left.mp4"),
    }


# ---------------------------------------------------------------------------
# extract_frames_multicam
# ---------------------------------------------------------------------------

def extract_frames_multicam(
    perspective_videos: dict[str, str],
    output_dir: str,
    fps: float = 5.0,
    format: str = "jpg",
    quality: int = 95,
    video_start_time_utc: float | None = None,
    progress_callback: Callable[[str, int], None] | None = None,
) -> dict[str, list[str]]:
    """Extract synchronized frames from all 4 perspective view videos.

    Frames are named by nanosecond timestamp and organized into cam_<view>/data/.
    A timestamps.csv is written per camera directory.
    Returns a dict mapping view name to sorted list of frame paths.
    """
    # Quality → qscale mapping (JPEG, 2=best, 31=worst)
    _q_map = {95: 2, 90: 3, 85: 5, 75: 10, 65: 15}
    qscale = _q_map.get(quality)
    if qscale is None:
        # Linear interpolation between anchor points, clamped to [2, 31]
        if quality >= 95:
            qscale = 2
        elif quality <= 65:
            qscale = 15
        else:
            # Interpolate between (95,2) and (65,15)
            qscale = int(2 + (95 - quality) / (95 - 65) * (15 - 2))
        qscale = max(2, min(31, qscale))

    out_root = Path(output_dir)

    def _extract_one_view(view_name: str, video_path: str) -> tuple[str, list[str]]:
        cam_dir = out_root / f"cam_{view_name}" / "data"
        cam_dir.mkdir(parents=True, exist_ok=True)

        pattern = str(cam_dir / f"%019d.{format}")
        # Use sequential numeric naming first (ffmpeg default), then rename
        tmp_pattern = str(cam_dir / f"_tmp_%06d.{format}")

        from core.utils import ffmpeg_extract_frames
        ffmpeg_extract_frames(video_path, tmp_pattern, fps=fps, quality=qscale)

        extracted = sorted(cam_dir.glob(f"_tmp_*.{format}"))
        return view_name, [str(p) for p in extracted]

    errors: list[str] = []
    tmp_results: dict[str, list[str]] = {}

    total_views = len(perspective_videos)
    completed = [0]
    completed_lock = threading.Lock()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_view = {
            executor.submit(_extract_one_view, v, p): v
            for v, p in perspective_videos.items()
        }
        for future in concurrent.futures.as_completed(future_to_view):
            view_name = future_to_view[future]
            try:
                vname, tmp_paths = future.result()
                tmp_results[vname] = tmp_paths
                with completed_lock:
                    completed[0] += 1
                    pct = int(completed[0] / total_views * 80)
                if progress_callback is not None:
                    progress_callback("Extracting frames", pct)
            except Exception as exc:
                errors.append(f"{view_name}: {exc}")
                logger.error("Frame extraction failed for '{}': {}", view_name, exc)

    if errors:
        raise RuntimeError(
            f"Frame extraction failed for {len(errors)} view(s): "
            + "; ".join(errors)
        )

    # Determine video_start_time_utc if not provided
    if video_start_time_utc is None:
        front_video = perspective_videos.get("front") or next(iter(perspective_videos.values()))
        try:
            probe_data = ffprobe(front_video)
            creation_time_str = (
                probe_data.get("format", {})
                .get("tags", {})
                .get("creation_time")
            )
            if creation_time_str:
                from datetime import datetime, timezone
                dt = datetime.fromisoformat(
                    creation_time_str.rstrip("Z").replace("Z", "+00:00")
                )
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                video_start_time_utc = dt.timestamp()
            else:
                video_start_time_utc = 0.0
                logger.warning(
                    "No creation_time tag found in video metadata — "
                    "frame timestamps will start from epoch 0. "
                    "Provide video_start_time_utc for accurate timestamps."
                )
        except Exception as exc:
            video_start_time_utc = 0.0
            logger.warning(
                "Could not determine video start time: {} — "
                "timestamps will start from epoch 0.", exc
            )

    # Rename files with nanosecond timestamps and write timestamps.csv
    final_results: dict[str, list[str]] = {}

    for view_name, tmp_paths in tmp_results.items():
        cam_dir = out_root / f"cam_{view_name}" / "data"
        renamed: list[str] = []
        ts_rows: list[tuple[int, int]] = []  # (frame_index_1based, timestamp_ns)

        for idx_zero, tmp_path in enumerate(tmp_paths):
            n = idx_zero + 1  # 1-based, matching ffmpeg sequential output
            ts_ns = int((video_start_time_utc + (n - 1) / fps) * 1e9)
            new_name = cam_dir / f"{ts_ns:019d}.{format}"
            Path(tmp_path).rename(new_name)
            renamed.append(str(new_name))
            ts_rows.append((n, ts_ns))

        # Write timestamps.csv
        ts_csv_path = cam_dir.parent / "timestamps.csv"
        with open(ts_csv_path, "w", newline="") as f:
            f.write("frame_index,timestamp_ns\n")
            for frame_idx, ts_ns in ts_rows:
                f.write(f"{frame_idx},{ts_ns}\n")

        final_results[view_name] = sorted(renamed)

    # Cross-view timestamp consistency check
    if "front" in final_results and "right" in final_results:
        front_ts = [
            int(Path(p).stem) for p in final_results["front"]
        ]
        right_ts = [
            int(Path(p).stem) for p in final_results["right"]
        ]
        for ft, rt in zip(front_ts, right_ts):
            if abs(ft - rt) > int(1e6):  # > 1ms
                logger.warning(
                    "Cross-view timestamp mismatch detected: "
                    "front={}, right={}, diff={}ns",
                    ft, rt, abs(ft - rt),
                )
                break

    if progress_callback is not None:
        progress_callback("Extracting frames", 100)

    return final_results


# ---------------------------------------------------------------------------
# build_multicam_euroc_dataset
# ---------------------------------------------------------------------------

def build_multicam_euroc_dataset(
    frame_paths: dict[str, list[str]],
    imu_samples: list,
    gps_samples: list,
    output_dir: str,
    config: Insta360ProcessingConfig,
    calibration: dict | None = None,
) -> str:
    """Assemble EuRoC multi-camera dataset from extracted frames + telemetry.

    Creates the standard EuRoC directory tree with cam0..cam3, imu0, gps,
    sensor.yaml per camera, and a top-level manifest.json.

    Returns the output_dir path as a string.
    """
    view_to_cam = {
        "front": "cam0",
        "right": "cam1",
        "rear":  "cam2",
        "left":  "cam3",
    }
    view_to_yaw = {
        "front": 0,
        "right": 90,
        "rear":  180,
        "left":  -90,
    }

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    total_frames = 0

    for view_name, paths in frame_paths.items():
        cam_id = view_to_cam.get(view_name, view_name)
        yaw = view_to_yaw.get(view_name, 0)
        cam_data_dir = out_root / cam_id / "data"
        cam_data_dir.mkdir(parents=True, exist_ok=True)

        # Frames may already be inside cam_<view>/data; if not, symlink/copy
        final_paths: list[str] = []
        for src in paths:
            src_path = Path(src)
            dest_path = cam_data_dir / src_path.name
            if not dest_path.exists():
                try:
                    dest_path.symlink_to(src_path.resolve())
                except (OSError, NotImplementedError):
                    shutil.copy2(str(src_path), str(dest_path))
            final_paths.append(str(dest_path))

        # Write timestamps.csv in cam_id directory
        ts_csv = out_root / cam_id / "timestamps.csv"
        with open(ts_csv, "w", newline="") as f:
            f.write("frame_index,timestamp_ns\n")
            for idx, fp in enumerate(sorted(final_paths), start=1):
                ts_ns_str = Path(fp).stem
                try:
                    ts_ns = int(ts_ns_str)
                except ValueError:
                    ts_ns = int((idx - 1) / config.frame_fps * 1e9)
                f.write(f"{idx},{ts_ns}\n")

        total_frames += len(final_paths)

        # Determine intrinsics for sensor.yaml
        if calibration and view_name in calibration:
            cal = calibration[view_name]
            intrinsics_val = [
                cal.get("fx", 0.0),
                cal.get("fy", 0.0),
                cal.get("cx", config.perspective_width / 2.0),
                cal.get("cy", config.perspective_height / 2.0),
            ]
        elif calibration and "fx" in calibration:
            intrinsics_val = [
                calibration.get("fx", 0.0),
                calibration.get("fy", 0.0),
                calibration.get("cx", config.perspective_width / 2.0),
                calibration.get("cy", config.perspective_height / 2.0),
            ]
        else:
            intrinsics_val = "uncalibrated"

        sensor_data = {
            "sensor_type": "camera",
            "comment": f"Insta360 X4 perspective reframe - {view_name}",
            "T_BS": {
                "rows": 4,
                "cols": 4,
                "data": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            },
            "rate_hz": config.frame_fps,
            "resolution": [config.perspective_width, config.perspective_height],
            "camera_model": "pinhole",
            "intrinsics": intrinsics_val,
            "distortion_model": "none",
            "distortion_coefficients": [0, 0, 0, 0],
            "fov_horizontal_deg": config.h_fov,
            "fov_vertical_deg": config.v_fov,
            "view_direction": view_name,
            "yaw_offset_deg": yaw,
        }

        sensor_yaml_path = out_root / cam_id / "sensor.yaml"
        with open(sensor_yaml_path, "w") as f:
            yaml.dump(sensor_data, f, default_flow_style=False, sort_keys=False)

    # Write imu0/data.csv in EuRoC format
    imu0_dir = out_root / "imu0"
    imu0_dir.mkdir(parents=True, exist_ok=True)
    imu_csv_path = imu0_dir / "data.csv"
    with open(imu_csv_path, "w", newline="") as f:
        f.write(_EUROC_IMU_HEADER + "\n")
        for s in imu_samples:
            f.write(
                f"{s.timestamp_ns},"
                f"{s.gyro_x:.10f},{s.gyro_y:.10f},{s.gyro_z:.10f},"
                f"{s.accel_x:.10f},{s.accel_y:.10f},{s.accel_z:.10f}\n"
            )

    # Write gps/data.csv
    gps_dir = out_root / "gps"
    gps_dir.mkdir(parents=True, exist_ok=True)
    gps_csv_path = gps_dir / "data.csv"
    with open(gps_csv_path, "w", newline="") as f:
        f.write(_EUROC_GPS_HEADER + "\n")
        for s in gps_samples:
            f.write(
                f"{s.timestamp_ns},"
                f"{s.latitude:.10f},{s.longitude:.10f},"
                f"{s.altitude_m:.4f},{s.speed_mps:.4f},{s.fix_type}\n"
            )

    # Compute rates for manifest
    duration_s = total_frames / config.frame_fps if total_frames > 0 else 0.0
    imu_rate_hz = (
        len(imu_samples) / duration_s if duration_s > 0 and imu_samples else 0.0
    )
    gps_rate_hz = (
        len(gps_samples) / duration_s if duration_s > 0 and gps_samples else 0.0
    )

    cameras_meta = {
        view_to_cam[v]: {"view": v, "yaw_deg": view_to_yaw.get(v, 0)}
        for v in frame_paths
        if v in view_to_cam
    }

    manifest = {
        "camera_setup": "insta360_x4_quad_perspective",
        "num_cameras": len(frame_paths),
        "cameras": cameras_meta,
        "frame_fps": config.frame_fps,
        "total_frames": total_frames,
        "has_imu": len(imu_samples) > 0,
        "has_gps": len(gps_samples) > 0,
        "imu_rate_hz": round(imu_rate_hz, 2),
        "gps_rate_hz": round(gps_rate_hz, 2),
        "h_fov_deg": config.h_fov,
        "v_fov_deg": config.v_fov,
        "calibration_available": calibration is not None,
    }

    manifest_path = out_root / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        "EuRoC dataset assembled → {} ({} cams, {} frames, {} IMU, {} GPS)",
        out_root,
        len(frame_paths),
        total_frames,
        len(imu_samples),
        len(gps_samples),
    )

    return str(out_root)


# ---------------------------------------------------------------------------
# run_full_insta360_pipeline
# ---------------------------------------------------------------------------

def run_full_insta360_pipeline(
    insv_pair: INSVPair,
    config: Insta360ProcessingConfig,
    output_dir: str,
    session_id: str,
    progress_callback: Callable[[str, int], None] | None = None,
) -> Insta360ExtractionResult:
    """Orchestrate the full Insta360 X4 processing pipeline.

    Stages:
        0–10%  : Telemetry extraction (GPS + IMU)
        10–40% : Dual fisheye stitching to equirectangular
        40–60% : Equirectangular → 4 perspective videos
        60–80% : Frame extraction
        80–85% : Blur scoring
        85–100%: EuRoC dataset assembly

    Returns Insta360ExtractionResult with all per-view metadata and stats.
    """
    # Deferred import to avoid circular dependency at module load time
    from core.insv_telemetry import extract_gps_from_insv, extract_imu_from_insv

    start_time = time.monotonic()
    issues: list[str] = []
    warnings: list[str] = []

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    def _cb(msg: str, pct: int) -> None:
        if progress_callback is not None:
            progress_callback(msg, pct)

    # ------------------------------------------------------------------
    # Stage 1: Telemetry (0–10%)
    # ------------------------------------------------------------------
    _cb("Extracting telemetry", 0)

    gps_samples = []
    try:
        gps_samples = extract_gps_from_insv(insv_pair.back_path)
        logger.info("GPS extraction: {} samples", len(gps_samples))
    except Exception as exc:
        msg = (
            f"GPS extraction failed for '{Path(insv_pair.back_path).name}': {exc}. "
            "GPS data will be absent from the dataset. This is expected if no "
            "Insta360 GPS remote was used."
        )
        warnings.append(msg)
        logger.warning(msg)

    imu_samples = []
    try:
        imu_samples = extract_imu_from_insv(insv_pair.back_path)
        logger.info("IMU extraction: {} samples", len(imu_samples))
    except Exception as exc:
        msg = (
            f"IMU extraction failed for '{Path(insv_pair.back_path).name}': {exc}. "
            "Ensure the file is a native INSV file, not an MP4 export."
        )
        issues.append(msg)
        logger.error(msg)

    _cb("Telemetry extracted", 10)

    # ------------------------------------------------------------------
    # Stage 2: Stitching (10–40%)
    # ------------------------------------------------------------------
    equirect_path: str | None = None

    if insv_pair.front_path is not None and Path(insv_pair.front_path).exists():
        equirect_out = str(out_root / f"{insv_pair.base_name}_equirect.mp4")

        def _stitch_cb(msg: str, pct: int) -> None:
            # Map internal 0–99 to 10–39
            mapped = 10 + int(pct * 29 / 100)
            _cb(msg, mapped)

        try:
            equirect_path = stitch_dual_fisheye_to_equirectangular(
                front_path=insv_pair.front_path,
                back_path=insv_pair.back_path,
                output_path=equirect_out,
                output_width=config.output_width,
                output_height=config.output_height,
                quality_crf=config.stitch_crf,
                fisheye_fov=config.fisheye_fov,
                progress_callback=_stitch_cb,
            )
        except Exception as exc:
            issues.append(
                f"Stitching failed: {exc}. "
                "Check ffmpeg version (4.3+ required) and available disk space."
            )
            logger.error("Stitching failed: {}", exc)
    else:
        w = (
            "Front lens file (_10_) not available — stitching skipped. "
            "Only back lens (equirectangular single-lens) will be used."
        )
        warnings.append(w)
        logger.warning(w)
        equirect_path = insv_pair.back_path  # fall back to single file

    # Optional horizon leveling
    if (
        equirect_path is not None
        and (config.pitch_correction_deg != 0.0 or config.roll_correction_deg != 0.0)
    ):
        leveled_path = str(out_root / f"{insv_pair.base_name}_leveled.mp4")
        try:
            equirect_path = apply_horizon_leveling(
                equirect_path=equirect_path,
                output_path=leveled_path,
                pitch_correction_deg=config.pitch_correction_deg,
                roll_correction_deg=config.roll_correction_deg,
            )
            logger.info("Horizon leveling applied → {}", leveled_path)
        except Exception as exc:
            warnings.append(f"Horizon leveling failed (skipped): {exc}")
            logger.warning("Horizon leveling failed: {}", exc)

    _cb("Stitching complete", 40)

    # ------------------------------------------------------------------
    # Stage 3: Perspective split (40–60%)
    # ------------------------------------------------------------------
    perspective_video_paths: dict[str, str] = {}
    perspective_dir = str(out_root / "perspective_videos")

    if equirect_path is None:
        issues.append(
            "No equirectangular video available for perspective splitting. "
            "Stitching must succeed before perspective views can be generated."
        )
    else:
        def _persp_cb(msg: str, pct: int) -> None:
            mapped = 40 + int(pct * 20 / 100)
            _cb(msg, mapped)

        try:
            perspective_video_paths = equirect_to_perspective_video(
                equirect_path=equirect_path,
                output_dir=perspective_dir,
                views=config.views,
                output_width=config.perspective_width,
                output_height=config.perspective_height,
                h_fov=config.h_fov,
                v_fov=config.v_fov,
                quality_crf=config.perspective_crf,
                use_gpu=config.use_gpu,
                progress_callback=_persp_cb,
            )
        except Exception as exc:
            issues.append(f"Perspective split failed: {exc}")
            logger.error("Perspective split failed: {}", exc)

    _cb("Perspective split complete", 60)

    # ------------------------------------------------------------------
    # Stage 4: Frame extraction (60–80%)
    # ------------------------------------------------------------------
    frame_paths: dict[str, list[str]] = {}
    frames_dir = str(out_root / "frames")

    if perspective_video_paths:
        def _frames_cb(msg: str, pct: int) -> None:
            mapped = 60 + int(pct * 20 / 100)
            _cb(msg, mapped)

        try:
            frame_paths = extract_frames_multicam(
                perspective_videos=perspective_video_paths,
                output_dir=frames_dir,
                fps=config.frame_fps,
                format=config.frame_format,
                quality=config.frame_quality,
                video_start_time_utc=None,
                progress_callback=_frames_cb,
            )
        except Exception as exc:
            issues.append(f"Frame extraction failed: {exc}")
            logger.error("Frame extraction failed: {}", exc)
    else:
        warnings.append(
            "No perspective videos available for frame extraction. "
            "Perspective split must succeed first."
        )

    _cb("Frame extraction complete", 80)

    # ------------------------------------------------------------------
    # Stage 5: Blur scoring (80–85%)
    # ------------------------------------------------------------------
    _cb("Computing blur scores", 80)

    try:
        import cv2
        cv2_available = True
    except ImportError:
        cv2_available = False
        warnings.append(
            "OpenCV not available — blur scoring skipped. "
            "Install opencv-python to enable blur detection."
        )

    perspective_view_results: dict[str, PerspectiveView] = {}

    for view_name in config.views:
        view_cfg = PERSPECTIVE_VIEWS.get(view_name, {"yaw": 0, "pitch": 0, "roll": 0})
        frames_for_view = frame_paths.get(view_name, [])
        video_path_for_view = perspective_video_paths.get(view_name)
        cam_dir = (
            str(Path(frames_dir) / f"cam_{view_name}")
            if frames_for_view
            else None
        )

        mean_blur = 0.0
        if cv2_available and frames_for_view:
            # Sample 10 frames evenly
            n_frames = len(frames_for_view)
            indices = [
                int(i * (n_frames - 1) / max(9, 1))
                for i in range(min(10, n_frames))
            ]
            blur_scores: list[float] = []
            for idx in indices:
                fp = frames_for_view[idx]
                try:
                    img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        score = float(cv2.Laplacian(img, cv2.CV_64F).var())
                        blur_scores.append(score)
                except Exception as exc:
                    logger.debug("Blur score failed for {}: {}", fp, exc)
            mean_blur = sum(blur_scores) / len(blur_scores) if blur_scores else 0.0

        perspective_view_results[view_name] = PerspectiveView(
            view_name=view_name,
            yaw_deg=float(view_cfg["yaw"]),
            pitch_deg=float(view_cfg["pitch"]),
            roll_deg=float(view_cfg["roll"]),
            video_path=video_path_for_view,
            frame_dir=cam_dir,
            frame_count=len(frames_for_view),
            mean_blur_score=mean_blur,
            is_usable=mean_blur > 50.0 or not cv2_available,
        )

    _cb("Blur scoring complete", 85)

    # ------------------------------------------------------------------
    # Stage 6: EuRoC dataset assembly (85–100%)
    # ------------------------------------------------------------------
    _cb("Assembling EuRoC dataset", 85)

    dataset_dir = str(out_root / "dataset")
    manifest_path_str = str(Path(dataset_dir) / "manifest.json")

    if frame_paths:
        try:
            build_multicam_euroc_dataset(
                frame_paths=frame_paths,
                imu_samples=imu_samples,
                gps_samples=gps_samples,
                output_dir=dataset_dir,
                config=config,
                calibration=None,
            )
        except Exception as exc:
            issues.append(f"Dataset assembly failed: {exc}")
            logger.error("Dataset assembly failed: {}", exc)
    else:
        warnings.append(
            "No frames available — dataset assembly skipped. "
            "Frame extraction must succeed before dataset can be built."
        )

    # ------------------------------------------------------------------
    # Cleanup per config flags
    # ------------------------------------------------------------------
    if not config.keep_equirect_video and equirect_path is not None:
        equirect_p = Path(equirect_path)
        if equirect_p.exists() and equirect_p != Path(insv_pair.back_path):
            try:
                equirect_p.unlink()
                logger.debug("Deleted equirect video: {}", equirect_path)
            except Exception as exc:
                logger.warning("Could not delete equirect video: {}", exc)

    if not config.keep_perspective_videos and perspective_video_paths:
        for vpath in perspective_video_paths.values():
            try:
                vp = Path(vpath)
                if vp.exists():
                    vp.unlink()
            except Exception as exc:
                logger.warning("Could not delete perspective video {}: {}", vpath, exc)

    # ------------------------------------------------------------------
    # Integrity check: all 4 cam dirs must have same frame count
    # ------------------------------------------------------------------
    frame_counts = {v: len(ps) for v, ps in frame_paths.items()}
    unique_counts = set(frame_counts.values())
    if len(unique_counts) > 1:
        warnings.append(
            f"Frame count mismatch across views: {frame_counts}. "
            "Dataset integrity may be compromised — re-run extraction."
        )
        logger.warning("Frame count mismatch: {}", frame_counts)

    # ------------------------------------------------------------------
    # Build result
    # ------------------------------------------------------------------
    end_time = time.monotonic()
    processing_time_minutes = (end_time - start_time) / 60.0

    try:
        disk_used = shutil.disk_usage(output_dir).used / (1024 ** 3)
    except Exception:
        disk_used = 0.0

    imu_rate_hz = (
        len(imu_samples) / insv_pair.duration_seconds
        if insv_pair.duration_seconds > 0
        else 0.0
    )
    gps_rate_hz = (
        len(gps_samples) / insv_pair.duration_seconds
        if insv_pair.duration_seconds > 0
        else 0.0
    )

    total_frames_per_view = (
        max(frame_counts.values()) if frame_counts else 0
    )

    _cb("Pipeline complete", 100)

    logger.info(
        "Insta360 pipeline complete: {:.1f}min, {:.2f}GB disk, "
        "{} issues, {} warnings",
        processing_time_minutes,
        disk_used,
        len(issues),
        len(warnings),
    )

    return Insta360ExtractionResult(
        session_id=session_id,
        insv_pair=insv_pair,
        equirect_path=equirect_path if config.keep_equirect_video else None,
        perspective_views=perspective_view_results,
        imu_samples=len(imu_samples),
        imu_rate_hz=round(imu_rate_hz, 2),
        gps_samples=len(gps_samples),
        gps_rate_hz=round(gps_rate_hz, 2),
        total_frames_per_view=total_frames_per_view,
        dataset_dir=dataset_dir,
        manifest_path=manifest_path_str,
        processing_time_minutes=round(processing_time_minutes, 2),
        disk_usage_gb=round(disk_used, 3),
        issues=issues,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# get_equirect_preview_frame
# ---------------------------------------------------------------------------

def get_equirect_preview_frame(
    equirect_path: str,
    timestamp_seconds: float = 5.0,
    output_path: str | None = None,
) -> str:
    """Extract a single preview frame from an equirectangular video.

    Returns the path to the extracted JPEG frame.
    """
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        output_path = tmp.name
        tmp.close()

    ffmpeg_run([
        "-ss", str(timestamp_seconds),
        "-i", equirect_path,
        "-frames:v", "1",
        "-q:v", "2",
        output_path,
    ])

    return output_path


# ---------------------------------------------------------------------------
# apply_horizon_leveling
# ---------------------------------------------------------------------------

def apply_horizon_leveling(
    equirect_path: str,
    output_path: str,
    pitch_correction_deg: float = 0.0,
    roll_correction_deg: float = 0.0,
) -> str:
    """Apply pitch/roll horizon correction to an equirectangular video.

    Uses ffmpeg v360=e:e with Lanczos interpolation.
    Returns the output_path on success.
    """
    ffmpeg_run([
        "-i", equirect_path,
        "-vf", (
            f"v360=e:e"
            f":pitch={pitch_correction_deg}"
            f":roll={roll_correction_deg}"
            f":yaw=0"
            f":interp=lanczos"
        ),
        "-c:v", "libx264",
        "-crf", "10",
        "-preset", "slow",
        "-pix_fmt", "yuv420p",
        output_path,
    ])
    return output_path


# ---------------------------------------------------------------------------
# generate_aframe_viewer_html
# ---------------------------------------------------------------------------

def generate_aframe_viewer_html(equirect_frame_path: str) -> str:
    """Generate a self-contained A-Frame HTML viewer for a 360° equirectangular frame.

    Returns the HTML string — does not write a file.
    """
    frame_url = Path(equirect_frame_path).as_uri()
    return f'''<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>360° Preview</title>
  <script src="https://aframe.io/releases/1.4.0/aframe.min.js"></script>
  <style>body {{ margin: 0; background: #1a1a2e; }}</style>
</head>
<body>
  <a-scene embedded style="width:100%;height:100vh" vr-mode-ui="enabled:false">
    <a-sky src="{frame_url}" rotation="0 -90 0"></a-sky>
    <a-entity camera look-controls="reverseMouseDrag:true" position="0 0 0"></a-entity>
  </a-scene>
</body>
</html>'''


# ---------------------------------------------------------------------------
# estimate_disk_usage
# ---------------------------------------------------------------------------

def estimate_disk_usage(
    source_size_gb: float,
    config: Insta360ProcessingConfig,
    duration_seconds: float,
) -> dict:
    """Estimate disk usage for all pipeline outputs.

    Uses empirical multipliers for Insta360 X4 8K footage:
      - Equirect 8K @ CRF10: ~4x source
      - Each perspective view @ CRF15: ~0.8x source
      - Frames at 5fps JPEG Q95: ~0.5 GB/hr/view

    Returns a dict with per-component estimates, totals, and an optional warning.
    """
    equirect_gb = source_size_gb * 4.0
    perspectives_gb = source_size_gb * 0.8 * len(config.views)
    hours = duration_seconds / 3600
    frames_gb = 0.5 * hours * len(config.views) * (config.frame_fps / 5.0)

    total_if_keep_all = equirect_gb + perspectives_gb + frames_gb
    total_with_config = (
        (equirect_gb if config.keep_equirect_video else 0)
        + (perspectives_gb if config.keep_perspective_videos else 0)
        + frames_gb
    )

    warning = None
    if total_with_config > 50:
        warning = (
            f"Requires ~{total_with_config:.0f}GB free disk space — "
            "ensure sufficient storage before processing"
        )

    return {
        "equirect_gb": round(equirect_gb, 1),
        "perspectives_gb": round(perspectives_gb, 1),
        "frames_gb": round(frames_gb, 1),
        "total_if_keep_all_gb": round(total_if_keep_all, 1),
        "total_with_config_gb": round(total_with_config, 1),
        "warning": warning,
    }
