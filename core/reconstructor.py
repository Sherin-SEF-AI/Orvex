"""
core/reconstructor.py — Structure-from-Motion via COLMAP.

Runs COLMAP feature_extractor → sequential_matcher → mapper as subprocesses,
then loads the sparse reconstruction via pycolmap and exports a colored
point cloud via open3d.

No UI imports — pure Python business logic.

Optional dependencies:
    pip install pycolmap open3d
System dependency: colmap binary on PATH
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Callable

from loguru import logger

from core.models import ColmapResult, TrajectoryPoint

ProgressCB = Callable[[int], None]
StatusCB = Callable[[str], None]


# ---------------------------------------------------------------------------
# Installation check
# ---------------------------------------------------------------------------

def check_colmap_installation() -> bool:
    """Return True if the 'colmap' binary is on PATH."""
    return shutil.which("colmap") is not None


# ---------------------------------------------------------------------------
# Full COLMAP SfM pipeline
# ---------------------------------------------------------------------------

def run_colmap_sfm(
    image_dir: str,
    output_dir: str,
    camera_model: str = "OPENCV_FISHEYE",
    use_gpu: bool = True,
    max_image_size: int = 1600,
    progress_callback: ProgressCB | None = None,
    status_callback: StatusCB | None = None,
) -> ColmapResult:
    """Run the full COLMAP sparse reconstruction pipeline.

    Steps executed:
        1. feature_extractor
        2. sequential_matcher (or exhaustive_matcher if < 100 images)
        3. mapper

    Then loads the result via pycolmap (if installed).

    Args:
        image_dir:     Directory containing the input JPEG/PNG frames.
        output_dir:    Directory for COLMAP database and sparse output.
        camera_model:  COLMAP camera model string.
        use_gpu:       Use GPU for feature extraction if available.
        max_image_size: Maximum image dimension (images rescaled if larger).
        progress_callback: 0-100.
        status_callback:   Human-readable status strings.

    Returns:
        ColmapResult with registered images, points, camera poses.

    Raises:
        FileNotFoundError: if colmap binary is not found.
        RuntimeError: if any COLMAP step fails.
    """
    if not check_colmap_installation():
        raise FileNotFoundError(
            "colmap binary not found on PATH. "
            "Install COLMAP: https://colmap.github.io/install.html "
            "and ensure it is accessible from your terminal."
        )

    img_dir = Path(image_dir)
    if not img_dir.exists():
        raise FileNotFoundError(
            f"Image directory not found: {image_dir}. "
            "Run frame extraction first."
        )

    image_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    n_images = len(image_files)
    if n_images == 0:
        raise RuntimeError(
            f"No JPEG or PNG images found in {image_dir}. "
            "Verify that frame extraction completed successfully."
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    db_path = out / "database.db"
    sparse_dir = out / "sparse"
    sparse_dir.mkdir(exist_ok=True)

    session_id = img_dir.name

    # ── Step 1: Feature extraction ──────────────────────────────────────────
    if status_callback:
        status_callback(f"Extracting features from {n_images} images…")
    _run_colmap_step(
        [
            "colmap", "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(img_dir),
            "--ImageReader.camera_model", camera_model,
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.use_gpu", "1" if use_gpu else "0",
            "--SiftExtraction.max_image_size", str(max_image_size),
        ],
        step_name="feature_extractor",
        status_callback=status_callback,
    )
    if progress_callback:
        progress_callback(30)

    # ── Step 2: Matching ─────────────────────────────────────────────────────
    # Use sequential for video (default); exhaustive only for tiny sets
    use_exhaustive = n_images < 100
    matcher = "exhaustive_matcher" if use_exhaustive else "sequential_matcher"
    matcher_label = "exhaustive" if use_exhaustive else "sequential"

    if status_callback:
        status_callback(
            f"Running {matcher_label} matching "
            f"({'< 100 frames' if use_exhaustive else 'video sequence'})…"
        )
    _run_colmap_step(
        ["colmap", matcher, "--database_path", str(db_path)],
        step_name=matcher,
        status_callback=status_callback,
    )
    if progress_callback:
        progress_callback(60)

    # ── Step 3: Mapping ──────────────────────────────────────────────────────
    if status_callback:
        status_callback("Running COLMAP mapper (this may take several minutes)…")
    _run_colmap_step(
        [
            "colmap", "mapper",
            "--database_path", str(db_path),
            "--image_path", str(img_dir),
            "--output_path", str(sparse_dir),
        ],
        step_name="mapper",
        status_callback=status_callback,
    )
    if progress_callback:
        progress_callback(90)

    # ── Load result ──────────────────────────────────────────────────────────
    if status_callback:
        status_callback("Loading reconstruction…")
    result = load_colmap_reconstruction(str(sparse_dir))
    result = ColmapResult(
        **{**result.model_dump(), "session_id": session_id, "num_images_total": n_images}
    )

    # Coverage warning
    coverage = result.num_images_registered / result.num_images_total if result.num_images_total else 0
    if coverage < 0.8:
        logger.warning(
            "reconstructor: only {:.1f}% of images registered. "
            "Consider using fewer frames, better lighting, or more overlap.",
            coverage * 100,
        )

    if progress_callback:
        progress_callback(100)

    logger.info(
        "reconstructor: {} / {} images registered, {} 3D points",
        result.num_images_registered,
        result.num_images_total,
        result.num_points3d,
    )
    return result


def _clean_snap_env(env: dict) -> dict:
    """Remove snap-injected paths from LD_LIBRARY_PATH.

    When the app is launched from a snap (e.g. VSCode snap), the snap runtime
    prepends /snap/core*/current/lib paths to LD_LIBRARY_PATH.  COLMAP then
    loads that snap's libpthread.so.0 (Ubuntu 20.04 / core20 vintage) which
    is missing GLIBC_PRIVATE symbols present in the system glibc, causing:
        symbol lookup error: __libc_pthread_init, version GLIBC_PRIVATE
    Stripping snap paths lets the linker fall back to the system libs.
    """
    ldpath = env.get("LD_LIBRARY_PATH", "")
    if ldpath:
        cleaned = ":".join(
            p for p in ldpath.split(":")
            if not p.startswith("/snap/")
        )
        if cleaned:
            env["LD_LIBRARY_PATH"] = cleaned
        else:
            env.pop("LD_LIBRARY_PATH", None)
    # Also unset SNAP* vars so child doesn't think it's inside a snap
    for key in list(env):
        if key.startswith("SNAP"):
            env.pop(key)
    return env


def _run_colmap_step(
    cmd: list[str],
    step_name: str,
    status_callback: StatusCB | None = None,
) -> None:
    """Run one COLMAP step as subprocess, stream stdout, raise on failure."""
    import os
    env = _clean_snap_env(os.environ.copy())
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    try:
        for line in proc.stdout:  # type: ignore[union-attr]
            line = line.rstrip()
            if status_callback and line:
                status_callback(f"[{step_name}] {line}")
        proc.wait()
    except Exception as exc:
        proc.kill()
        raise RuntimeError(
            f"COLMAP {step_name} subprocess error: {exc}"
        ) from exc

    if proc.returncode != 0:
        raise RuntimeError(
            f"COLMAP {step_name} failed with exit code {proc.returncode}. "
            "Common causes: insufficient features (need texture/contrast in images), "
            "GPU not available (set use_gpu=False), or corrupted database."
        )


# ---------------------------------------------------------------------------
# Load reconstruction
# ---------------------------------------------------------------------------

def load_colmap_reconstruction(sparse_dir: str) -> ColmapResult:
    """Load a COLMAP sparse reconstruction from disk via pycolmap.

    Args:
        sparse_dir: Path to the COLMAP sparse/ directory.
                    Must contain a subdirectory (e.g., "0") with cameras.bin,
                    images.bin, points3D.bin.

    Returns:
        ColmapResult with points, colors, camera poses.

    Raises:
        ImportError: if pycolmap is not installed.
        FileNotFoundError: if reconstruction directory does not exist.
    """
    try:
        import pycolmap
    except ImportError as exc:
        raise ImportError(
            "pycolmap is not installed. "
            "Install with: pip install pycolmap"
        ) from exc

    sparse_path = Path(sparse_dir)
    # Find the first subdirectory (e.g., "0")
    sub_dirs = [d for d in sparse_path.iterdir() if d.is_dir()]
    if not sub_dirs:
        raise FileNotFoundError(
            f"No reconstruction found in {sparse_dir}. "
            "COLMAP mapper may have failed to register any images. "
            "Try with more overlapping frames or better-textured images."
        )

    recon_dir = sorted(sub_dirs)[0]  # use reconstruction "0"
    recon = pycolmap.Reconstruction(str(recon_dir))

    # Extract 3D points and colors
    points: list[list[float]] = []
    colors: list[list[int]] = []
    for pid, pt3d in recon.points3D.items():
        points.append([float(pt3d.xyz[0]), float(pt3d.xyz[1]), float(pt3d.xyz[2])])
        colors.append([int(pt3d.color[0]), int(pt3d.color[1]), int(pt3d.color[2])])

    # Extract camera poses (image positions in world frame)
    camera_poses: list[TrajectoryPoint] = []
    for img_id, image in sorted(recon.images.items()):
        # COLMAP stores R, t such that p_world = R^T * (p_cam - t)
        # Camera center in world: c = -R^T @ t
        try:
            rot = image.rotmat()  # 3x3 numpy
            t = image.tvec        # 3-vector
            import numpy as np
            center = -rot.T @ t
            # Convert rotation matrix to quaternion
            qw, qx, qy, qz = _rotmat_to_quat(rot)
            camera_poses.append(
                TrajectoryPoint(
                    timestamp_ns=0,  # COLMAP doesn't preserve timestamps
                    tx=float(center[0]),
                    ty=float(center[1]),
                    tz=float(center[2]),
                    qx=qx, qy=qy, qz=qz, qw=qw,
                )
            )
        except Exception:
            continue

    # Reprojection error
    mean_repr_err = 0.0
    try:
        errors = [pt.error for pt in recon.points3D.values() if hasattr(pt, "error")]
        if errors:
            import numpy as np
            mean_repr_err = float(np.mean(errors))
    except Exception:
        pass

    return ColmapResult(
        session_id="",
        num_images_total=len(recon.images),
        num_images_registered=len(recon.images),
        num_points3d=len(recon.points3D),
        mean_reprojection_error=mean_repr_err,
        points=points,
        colors=colors,
        camera_poses=camera_poses,
        ply_path=None,
    )


def _rotmat_to_quat(R: "np.ndarray") -> tuple[float, float, float, float]:  # noqa: F821
    """Convert 3x3 rotation matrix to (qw, qx, qy, qz)."""
    import numpy as np
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / (trace + 1.0) ** 0.5
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * (1.0 + R[0, 0] - R[1, 1] - R[2, 2]) ** 0.5
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * (1.0 + R[1, 1] - R[0, 0] - R[2, 2]) ** 0.5
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * (1.0 + R[2, 2] - R[0, 0] - R[1, 1]) ** 0.5
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return float(qw), float(qx), float(qy), float(qz)


# ---------------------------------------------------------------------------
# PLY export
# ---------------------------------------------------------------------------

def export_point_cloud_ply(
    result: ColmapResult,
    output_path: str,
) -> str:
    """Export the COLMAP point cloud as a colored .ply file via open3d.

    Args:
        result:      ColmapResult with points and colors.
        output_path: Path to write the .ply file.

    Returns:
        Absolute path to the written .ply file.

    Raises:
        ImportError: if open3d is not installed.
        ValueError: if result has no points.
    """
    try:
        import open3d as o3d
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "open3d is not installed. "
            "Install with: pip install open3d"
        ) from exc

    if not result.points:
        raise ValueError(
            "ColmapResult has no 3D points to export. "
            "Ensure COLMAP reconstruction completed successfully."
        )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(result.points, dtype=float))
    if result.colors:
        colors_normalized = np.array(result.colors, dtype=float) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors_normalized)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out), pcd)
    logger.info("reconstructor: PLY written to {}", out)
    return str(out.resolve())


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_reconstruction_stats(result: ColmapResult) -> dict:
    """Compute summary statistics for a ColmapResult.

    Returns dict with:
        num_cameras, num_registered_images, num_points3d,
        mean_reprojection_error, coverage_percent, bounding_box_meters
    """
    coverage = (
        result.num_images_registered / result.num_images_total * 100.0
        if result.num_images_total > 0
        else 0.0
    )

    bbox: dict = {}
    if result.points:
        import numpy as np
        pts = np.array(result.points)
        bbox = {
            "x_min": float(pts[:, 0].min()),
            "x_max": float(pts[:, 0].max()),
            "y_min": float(pts[:, 1].min()),
            "y_max": float(pts[:, 1].max()),
            "z_min": float(pts[:, 2].min()),
            "z_max": float(pts[:, 2].max()),
        }

    return {
        "num_cameras": 1,  # single_camera=1 in feature extraction
        "num_registered_images": result.num_images_registered,
        "num_points3d": result.num_points3d,
        "mean_reprojection_error": result.mean_reprojection_error,
        "coverage_percent": round(coverage, 1),
        "bounding_box_meters": bbox,
    }
