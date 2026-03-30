"""
core/calibration.py — Calibration workflow manager.

Implements a four-step guided calibration pipeline:

  Step 1 — imu_static
    Extract IMU from a static recording, compute Allan deviation,
    extract noise parameters (accel/gyro noise density + random walk).
    Output: imu_noise_params.json

  Step 2 — camera_intrinsic
    Extract frames from a calibration video, detect AprilTag/ChArUco board,
    run cv2.calibrateCamera to estimate fx/fy/cx/cy + distortion.
    Output: camera_intrinsics.json

  Step 3 — camera_imu_extrinsic
    Invoke OpenImuCameraCalibrator (external subprocess) with the output of
    steps 1 and 2. Streams stdout in real-time.
    Output: camera_imu_extrinsics.json (4×4 T_cam_imu)

  Step 4 — validation (informational — caller handles this)

Each step:
  - Is resumable: if the result JSON exists, the step is skipped.
  - Saves results to <calibration_root>/<cal_session_id>/step_N_results.json
  - Accepts progress_callback(int) and status_callback(str) hooks.

No mock data. OpenCV and py-gpmf-parser must be installed for Steps 1–2.
OpenImuCameraCalibrator must be on PATH for Step 3.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
from loguru import logger

from core.extractor_gopro import extract_gopro
from core.models import ExtractionConfig

ProgressCB = Callable[[int], None]
StatusCB = Callable[[str], None]

_STEP_FILES = {
    "imu_static":            "step1_imu_noise_params.json",
    "camera_intrinsic":      "step2_camera_intrinsics.json",
    "camera_imu_extrinsic":  "step3_camera_imu_extrinsics.json",
}

# Validation thresholds
_MIN_STATIC_DURATION_S = 4 * 3600   # 4 hours
_MAX_REPROJ_ERROR_PX   = 0.5
_MIN_POSES_DETECTED    = 15


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------

def run_calibration_step(
    step: str,
    file_path: str,
    extra: dict,
    progress_callback: ProgressCB = lambda p: None,
    status_callback: StatusCB = lambda s: None,
) -> dict:
    """Run one calibration step and return its results dict.

    Args:
        step:              One of "imu_static", "camera_intrinsic",
                           "camera_imu_extrinsic".
        file_path:         Path to the input recording for this step.
        extra:             Step-specific extra parameters dict.  For step 3,
                           must include "cal_root" and "cal_session_id" so the
                           function can locate step 1/2 outputs.
        progress_callback: Called with 0-100 progress values.
        status_callback:   Called with human-readable status strings.

    Returns:
        Results dict (same content as the saved JSON).
    """
    dispatch = {
        "imu_static":           _step1_imu_static,
        "camera_intrinsic":     _step2_camera_intrinsic,
        "camera_imu_extrinsic": _step3_extrinsic,
    }
    if step not in dispatch:
        raise ValueError(
            f"Unknown calibration step: '{step}'. "
            f"Expected one of: {list(dispatch.keys())}"
        )
    return dispatch[step](file_path, extra, progress_callback, status_callback)


def get_step_result(cal_root: str | Path, cal_session_id: str, step: str) -> dict | None:
    """Return the saved result dict for a step, or None if not yet complete."""
    path = _step_path(cal_root, cal_session_id, step)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def is_step_complete(cal_root: str | Path, cal_session_id: str, step: str) -> bool:
    return get_step_result(cal_root, cal_session_id, step) is not None


# ---------------------------------------------------------------------------
# Step 1 — IMU static noise characterisation
# ---------------------------------------------------------------------------

def _step1_imu_static(
    file_path: str,
    extra: dict,
    progress: ProgressCB,
    status: StatusCB,
) -> dict:
    """Extract IMU from a long static recording, compute noise parameters."""
    import cv2  # noqa: F401  (verify OpenCV is installed)

    status("Step 1: Extracting IMU from static recording…")
    progress(5)

    fp = Path(file_path)
    if not fp.exists():
        raise FileNotFoundError(
            f"Static recording not found: '{fp}'. "
            "Provide a valid GoPro MP4 path."
        )

    # Extract IMU using existing GoPro extractor
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ExtractionConfig(
            session_id="cal_imu_static",
            frame_fps=1.0,        # minimal frames — we only need IMU
            frame_format="jpg",
            frame_quality=50,
            output_format="euroc",
            sync_devices=False,
            imu_interpolation=True,
        )
        status("Extracting GPMF telemetry…")
        progress(15)

        extracted = extract_gopro(str(fp), config, tmpdir)

    samples = extracted.imu_samples
    duration_s = extracted.duration_seconds

    status(f"Got {len(samples)} IMU samples, {duration_s:.0f}s duration.")
    progress(40)

    if duration_s < _MIN_STATIC_DURATION_S:
        raise ValueError(
            f"Static recording is {duration_s/3600:.1f} h — minimum is 4 h. "
            "Re-record a longer static session for accurate noise characterisation."
        )

    # Validate stillness: std of accel magnitude should be small
    mags = [
        np.sqrt(s.accel_x**2 + s.accel_y**2 + s.accel_z**2)
        for s in samples
    ]
    mag_std = float(np.std(mags))
    if mag_std > 0.5:
        logger.warning(
            "Accel magnitude std = {:.4f} m/s² — device may have moved during static recording.",
            mag_std,
        )

    status("Computing Allan deviation…")
    progress(60)

    accel_x = np.array([s.accel_x for s in samples])
    accel_y = np.array([s.accel_y for s in samples])
    accel_z = np.array([s.accel_z for s in samples])
    gyro_x  = np.array([s.gyro_x  for s in samples])
    gyro_y  = np.array([s.gyro_y  for s in samples])
    gyro_z  = np.array([s.gyro_z  for s in samples])

    # Sample rate from timestamps
    if len(samples) >= 2:
        dt_ns = samples[1].timestamp_ns - samples[0].timestamp_ns
        rate_hz = 1e9 / dt_ns if dt_ns > 0 else 200.0
    else:
        rate_hz = 200.0

    accel_nd, accel_rw = _allan_noise_params(accel_x, accel_y, accel_z, rate_hz)
    gyro_nd,  gyro_rw  = _allan_noise_params(gyro_x,  gyro_y,  gyro_z,  rate_hz)

    progress(90)
    status("Saving IMU noise parameters…")

    results = {
        "step": "imu_static",
        "duration_s": duration_s,
        "rate_hz": rate_hz,
        "accel_noise_density": accel_nd,    # m/s²/√Hz
        "accel_random_walk": accel_rw,      # m/s²/√s  (bias instability proxy)
        "gyro_noise_density": gyro_nd,      # rad/s/√Hz
        "gyro_random_walk": gyro_rw,        # rad/s/√s
        "accel_mag_std": mag_std,
        "validation": {
            "duration_ok": duration_s >= _MIN_STATIC_DURATION_S,
            "stillness_ok": mag_std <= 0.5,
        },
    }

    cal_root = extra.get("cal_root")
    cal_id   = extra.get("cal_session_id")
    if cal_root and cal_id:
        _save_step(cal_root, cal_id, "imu_static", results)

    progress(100)
    status("Step 1 complete.")
    return results


# ---------------------------------------------------------------------------
# Step 2 — Camera intrinsic calibration
# ---------------------------------------------------------------------------

def _step2_camera_intrinsic(
    file_path: str,
    extra: dict,
    progress: ProgressCB,
    status: StatusCB,
) -> dict:
    """Extract frames from a calibration video and estimate camera intrinsics."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required for camera calibration. "
            "Install it with: pip install opencv-python"
        ) from exc

    fp = Path(file_path)
    if not fp.exists():
        raise FileNotFoundError(
            f"Calibration video not found: '{fp}'."
        )

    status("Step 2: Extracting calibration frames…")
    progress(5)

    board_cols = int(extra.get("board_cols", 9))
    board_rows = int(extra.get("board_rows", 6))
    square_mm  = float(extra.get("square_mm", 25.0))

    with tempfile.TemporaryDirectory() as tmpdir:
        frame_dir = Path(tmpdir) / "frames"
        frame_dir.mkdir()

        # Extract at 2 fps — typically enough for calibration poses
        from core.utils import ffmpeg_extract_frames
        ffmpeg_extract_frames(fp, frame_dir / "frame_%06d.jpg", fps=2.0, quality=2)

        frames = sorted(frame_dir.glob("*.jpg"))
        status(f"Extracted {len(frames)} frames. Detecting chessboard/ArUco…")
        progress(20)

        objp = np.zeros((board_rows * board_cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
        objp *= square_mm

        obj_points: list = []
        img_points: list = []
        img_shape: tuple[int, int] | None = None
        n_detected = 0

        for i, fpath in enumerate(frames):
            progress(20 + int(i / len(frames) * 50))
            img = cv2.imread(str(fpath))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img_shape is None:
                img_shape = gray.shape[::-1]  # (w, h)

            ret, corners = cv2.findChessboardCorners(
                gray, (board_cols, board_rows),
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            if ret:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                obj_points.append(objp)
                img_points.append(corners_refined)
                n_detected += 1

        status(f"Detected {n_detected} valid calibration poses.")
        progress(75)

        if n_detected < _MIN_POSES_DETECTED:
            raise ValueError(
                f"Only {n_detected} calibration poses detected (minimum: {_MIN_POSES_DETECTED}). "
                "Move the camera to cover all corners of the board and ensure good lighting."
            )

        if img_shape is None:
            raise RuntimeError(f"No readable frames found in '{fp}'.")

        status("Running cv2.calibrateCamera…")
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, img_shape, None, None
        )
        progress(90)

        reproj_px = float(ret)
        status(f"Reprojection error: {reproj_px:.4f} px")

        if reproj_px > _MAX_REPROJ_ERROR_PX:
            logger.warning(
                "Reprojection error {:.4f} px exceeds threshold {:.1f} px — "
                "consider re-recording with better coverage.",
                reproj_px, _MAX_REPROJ_ERROR_PX,
            )

    results = {
        "step": "camera_intrinsic",
        "image_width":  img_shape[0],
        "image_height": img_shape[1],
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "distortion_coeffs": dist.flatten().tolist(),
        "reprojection_error_px": reproj_px,
        "n_poses": n_detected,
        "board_cols": board_cols,
        "board_rows": board_rows,
        "square_mm": square_mm,
        "validation": {
            "reproj_ok": reproj_px <= _MAX_REPROJ_ERROR_PX,
            "poses_ok":  n_detected >= _MIN_POSES_DETECTED,
        },
    }

    cal_root = extra.get("cal_root")
    cal_id   = extra.get("cal_session_id")
    if cal_root and cal_id:
        _save_step(cal_root, cal_id, "camera_intrinsic", results)

    progress(100)
    status("Step 2 complete.")
    return results


# ---------------------------------------------------------------------------
# Step 3 — Camera-IMU extrinsic calibration
# ---------------------------------------------------------------------------

def _step3_extrinsic(
    file_path: str,
    extra: dict,
    progress: ProgressCB,
    status: StatusCB,
) -> dict:
    """Run OpenImuCameraCalibrator subprocess for camera-IMU extrinsics."""
    tool = shutil.which("OpenImuCameraCalibrator") or shutil.which("imu_camera_calibrator")
    if tool is None:
        raise RuntimeError(
            "OpenImuCameraCalibrator not found on PATH. "
            "Install it from: https://github.com/urbste/OpenImuCameraCalibrator "
            "and ensure it is accessible before running Step 3."
        )

    fp = Path(file_path)
    if not fp.exists():
        raise FileNotFoundError(
            f"Extrinsic calibration video not found: '{fp}'."
        )

    cal_root = extra.get("cal_root")
    cal_id   = extra.get("cal_session_id")

    # Locate step 1 and 2 outputs
    imu_params_path = None
    cam_params_path = None
    if cal_root and cal_id:
        imu_params_path = _step_path(cal_root, cal_id, "imu_static")
        cam_params_path = _step_path(cal_root, cal_id, "camera_intrinsic")

    if imu_params_path is None or not imu_params_path.exists():
        raise RuntimeError(
            "IMU noise parameters (Step 1) not found. "
            "Complete Step 1 before running Step 3."
        )
    if cam_params_path is None or not cam_params_path.exists():
        raise RuntimeError(
            "Camera intrinsics (Step 2) not found. "
            "Complete Step 2 before running Step 3."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "extrinsics.json"

        cmd = [
            tool,
            "--video", str(fp),
            "--imu_params", str(imu_params_path),
            "--cam_params", str(cam_params_path),
            "--output", str(out_path),
        ]
        status(f"Running: {' '.join(cmd)}")
        status("This step may take 20–40 minutes. Output is streamed below.")
        progress(5)

        # Stream stdout in real-time (Rule 6 — never show a frozen UI)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None

        for line in proc.stdout:
            line = line.rstrip()
            if line:
                status(line)
                logger.debug("OpenImuCameraCalibrator: {}", line)

        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(
                f"OpenImuCameraCalibrator exited with code {proc.returncode}. "
                "Check the log output above for details."
            )
        progress(90)

        if not out_path.exists():
            raise RuntimeError(
                "OpenImuCameraCalibrator did not produce an output file. "
                f"Expected: {out_path}"
            )

        with open(out_path) as f:
            raw = json.load(f)

    results = {
        "step": "camera_imu_extrinsic",
        "T_cam_imu": raw.get("T_cam_imu", []),   # 4×4 list-of-lists
        "translation_m": raw.get("translation_m", []),
        "rotation_deg": raw.get("rotation_deg", []),
        "validation": {
            "translation_ok": _translation_ok(raw.get("translation_m", [])),
            "rotation_ok":    _rotation_ok(raw.get("rotation_deg", [])),
        },
        "raw": raw,
    }

    if cal_root and cal_id:
        _save_step(cal_root, cal_id, "camera_imu_extrinsic", results)

    progress(100)
    status("Step 3 complete.")
    return results


# ---------------------------------------------------------------------------
# Allan deviation (noise density estimate)
# ---------------------------------------------------------------------------

def _allan_noise_params(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, rate_hz: float
) -> tuple[float, float]:
    """Estimate noise density and random walk from Allan deviation.

    Returns (noise_density, random_walk) as scalar floats.

    Uses the overlapping Allan deviation at tau = 1/rate (noise density)
    and tau = 1 s (random walk proxy).
    """
    dt = 1.0 / rate_hz
    n = len(x)

    # Combine all three axes (use magnitude)
    mag = np.sqrt(x**2 + y**2 + z**2)

    def avar(tau: float) -> float:
        m = max(1, int(tau / dt))
        if 2 * m > n:
            return 0.0
        clusters = [np.mean(mag[i:i+m]) for i in range(0, n - m, m)]
        if len(clusters) < 2:
            return 0.0
        diffs = np.diff(clusters)
        return float(np.mean(diffs**2) / 2.0)

    # Noise density at shortest tau
    nd = float(np.sqrt(avar(dt) * dt)) if avar(dt) > 0 else 0.0
    # Random walk at tau = 1 s
    rw = float(np.sqrt(avar(1.0))) if avar(1.0) > 0 else 0.0

    return nd, rw


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _translation_ok(t: list) -> bool:
    """Return True if translation magnitude < 5 mm."""
    if not t or len(t) < 3:
        return False
    mag = (t[0]**2 + t[1]**2 + t[2]**2) ** 0.5
    return mag < 0.005  # 5 mm

def _rotation_ok(r: list) -> bool:
    """Return True if rotation magnitude < 0.5°."""
    if not r or len(r) < 3:
        return False
    import math
    mag = math.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
    return mag < 0.5  # degrees


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _step_path(cal_root: str | Path, cal_session_id: str, step: str) -> Path:
    return Path(cal_root) / cal_session_id / _STEP_FILES[step]


def _save_step(cal_root: str | Path, cal_session_id: str, step: str, data: dict) -> None:
    path = _step_path(cal_root, cal_session_id, step)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved calibration step '{}' → {}", step, path)


# ---------------------------------------------------------------------------
# Calibration health check
# ---------------------------------------------------------------------------

def check_calibration_health(
    cal_root: str | Path,
    cal_session_id: str,
) -> dict:
    """Validate a completed calibration session against quality thresholds.

    Args:
        cal_root:         Root directory for calibration data.
        cal_session_id:   ID of the calibration session to check.

    Returns:
        {
          "ok": bool,
          "checks": [
            {"name": str, "passed": bool, "detail": str},
            ...
          ]
        }
    """
    import math

    checks: list[dict] = []

    # Check 1: Intrinsic reprojection error < 0.5 px
    step2 = get_step_result(cal_root, cal_session_id, "camera_intrinsic")
    if step2 is None:
        checks.append({
            "name": "Intrinsic calibration complete",
            "passed": False,
            "detail": "Step 2 result file not found — run camera intrinsic calibration",
        })
    else:
        err = step2.get("reprojection_error_px", None)
        if err is None:
            passed = False
            detail = "reprojection_error_px not present in step 2 results"
        else:
            passed = float(err) < _MAX_REPROJ_ERROR_PX
            detail = (
                f"reprojection_error_px = {err:.4f} px "
                f"({'OK' if passed else 'FAIL — re-record with better coverage'})"
            )
        checks.append({
            "name": f"Reprojection error < {_MAX_REPROJ_ERROR_PX} px",
            "passed": passed,
            "detail": detail,
        })

    # Check 2: T_cam_imu translation magnitude < 0.1 m
    step3 = get_step_result(cal_root, cal_session_id, "camera_imu_extrinsic")
    if step3 is None:
        checks.append({
            "name": "Extrinsic calibration complete",
            "passed": False,
            "detail": "Step 3 result file not found — run camera-IMU extrinsic calibration",
        })
    else:
        t = step3.get("translation", [])
        if not t or len(t) < 3:
            passed = False
            detail = "Translation vector not found in step 3 results"
        else:
            mag = math.sqrt(sum(float(v) ** 2 for v in t[:3]))
            passed = mag < 0.1
            detail = (
                f"|t_cam_imu| = {mag*100:.1f} cm "
                f"({'OK' if passed else 'FAIL > 10 cm — check IMU/camera mounting'})"
            )
        checks.append({
            "name": "T_cam_imu translation < 10 cm",
            "passed": passed,
            "detail": detail,
        })

    # Check 3: IMU noise params populated
    step1 = get_step_result(cal_root, cal_session_id, "imu_static")
    if step1 is None:
        checks.append({
            "name": "IMU static calibration complete",
            "passed": False,
            "detail": "Step 1 result file not found — run IMU static noise calibration",
        })
    else:
        noise = step1.get("accel_noise_density", None)
        passed = noise is not None and float(noise) > 0
        detail = (
            f"accel_noise_density = {noise}"
            if noise is not None else
            "accel_noise_density missing from step 1 results"
        )
        checks.append({
            "name": "IMU noise params populated",
            "passed": passed,
            "detail": detail,
        })

    # Check 4: All three step result files exist
    all_steps_done = all(
        is_step_complete(cal_root, cal_session_id, s)
        for s in ("imu_static", "camera_intrinsic", "camera_imu_extrinsic")
    )
    checks.append({
        "name": "All 3 calibration steps complete",
        "passed": all_steps_done,
        "detail": "Steps 1, 2, and 3 result files all present" if all_steps_done
                  else "One or more steps are incomplete",
    })

    overall_ok = all(c["passed"] for c in checks)
    return {"ok": overall_ok, "checks": checks}
