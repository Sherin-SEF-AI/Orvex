"""
core/dataset_builder.py — Dataset assembly from extracted sessions.

Supports three output formats:
  - EuRoC   : cam0/data/*.jpg + imu0/data.csv + gps.csv + dataset.yaml
  - ROS bag  : rosbag2 sqlite3 format (requires ros2 / rosbag2_py)
  - HDF5     : single .h5 file (requires h5py)

Only EuRoC is fully implemented without optional deps. ROS bag and HDF5
raise clear errors if the required packages are absent.

Output structure (EuRoC):
  <output_dir>/
    dataset_manifest.json
    <session_id>/
      cam0/
        data/
          <timestamp_ns>.jpg
        sensor.yaml
      imu0/
        data.csv
        sensor.yaml
      gps.csv
      session.yaml
"""
from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from loguru import logger

from core.models import Session
from core.session_manager import SessionManager

# EuRoC IMU CSV header (must match exactly)
_IMU_HEADER = (
    "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
    "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]"
)


# ---------------------------------------------------------------------------
# Public result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DatasetBuildResult:
    output_dir: Path
    session_ids: list[str]
    total_frames: int
    total_imu_samples: int
    format: str
    manifest_path: Path
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_dataset(
    session_ids: list[str],
    session_manager: SessionManager,
    output_format: str,          # "euroc", "rosbag2", "hdf5"
    output_dir: Path,
    progress_callback: Callable[[int], None] | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> DatasetBuildResult:
    """Assemble a dataset from one or more extracted sessions.

    Args:
        session_ids: Session IDs to include.
        session_manager: Session CRUD interface.
        output_format: "euroc", "rosbag2", or "hdf5".
        output_dir: Root directory for output.
        progress_callback: Called with 0-100 int as work progresses.
        status_callback: Called with status strings.

    Returns:
        DatasetBuildResult with stats and manifest path.
    """
    if not session_ids:
        raise ValueError("No session IDs provided.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _prog(p: int) -> None:
        if progress_callback:
            progress_callback(p)

    def _stat(msg: str) -> None:
        logger.info(msg)
        if status_callback:
            status_callback(msg)

    _stat(f"Building {output_format.upper()} dataset for {len(session_ids)} session(s)…")

    if output_format == "euroc":
        return _build_euroc(session_ids, session_manager, output_dir, _prog, _stat)
    elif output_format == "rosbag2":
        return _build_rosbag2(session_ids, session_manager, output_dir, _prog, _stat)
    elif output_format == "hdf5":
        return _build_hdf5(session_ids, session_manager, output_dir, _prog, _stat)
    else:
        raise ValueError(f"Unknown output_format: {output_format!r}. "
                         "Use 'euroc', 'rosbag2', or 'hdf5'.")


# ---------------------------------------------------------------------------
# EuRoC builder
# ---------------------------------------------------------------------------

def _build_euroc(
    session_ids: list[str],
    sm: SessionManager,
    output_dir: Path,
    prog: Callable[[int], None],
    stat: Callable[[str], None],
) -> DatasetBuildResult:
    total_frames = 0
    total_imu = 0
    warnings: list[str] = []
    manifest: dict = {
        "format": "euroc",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sessions": [],
    }

    n = len(session_ids)
    for si, session_id in enumerate(session_ids):
        base_pct = si * 90 // n
        end_pct  = (si + 1) * 90 // n

        session = sm.get_session(session_id)
        stat(f"[{si+1}/{n}] Processing session: {session.name}")

        sess_out = output_dir / session_id
        sess_out.mkdir(parents=True, exist_ok=True)

        # Locate extraction source
        source_dir = _find_extraction_dir(sm, session_id)
        if source_dir is None:
            w = f"Session {session.name} has no extracted data — skipping."
            warnings.append(w)
            stat(f"  ⚠ {w}")
            manifest["sessions"].append({
                "id": session_id,
                "name": session.name,
                "status": "skipped",
                "reason": "no extraction output found",
            })
            prog(end_pct)
            continue

        # Copy cam0 frames
        prog(base_pct + (end_pct - base_pct) // 4)
        frame_count = _copy_cam0(source_dir, sess_out, stat)
        total_frames += frame_count

        # Copy / rewrite imu0
        prog(base_pct + (end_pct - base_pct) // 2)
        imu_count = _copy_imu0(source_dir, sess_out, stat)
        total_imu += imu_count

        # Copy gps.csv if present
        gps_src = source_dir / "gps.csv"
        if gps_src.exists():
            shutil.copy2(gps_src, sess_out / "gps.csv")

        # Write per-session sensor YAML stubs
        _write_cam_sensor_yaml(sess_out, session)
        _write_imu_sensor_yaml(sess_out)

        # Write session YAML
        _write_session_yaml(sess_out, session)

        manifest["sessions"].append({
            "id": session_id,
            "name": session.name,
            "environment": session.environment,
            "location": session.location,
            "frames": frame_count,
            "imu_samples": imu_count,
            "status": "ok",
        })
        prog(end_pct)
        stat(f"  ✓ {frame_count} frames, {imu_count} IMU samples")

    # Write manifest
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Integrity check
    stat("Running integrity check…")
    _integrity_check(manifest, warnings)

    prog(100)
    stat("Dataset build complete.")

    return DatasetBuildResult(
        output_dir=output_dir,
        session_ids=session_ids,
        total_frames=total_frames,
        total_imu_samples=total_imu,
        format="euroc",
        manifest_path=manifest_path,
        warnings=warnings,
    )


def _find_extraction_dir(sm: SessionManager, session_id: str) -> Path | None:
    """Return the extraction output dir for a session, or None."""
    folder = sm.session_folder(session_id)
    for candidate in [
        folder / "extraction_gopro",
        folder / "extraction_insta360",
        folder / "extraction_sensorlogger",
        folder / "extraction",
    ]:
        if candidate.exists():
            return candidate
    return None


def _copy_cam0(source_dir: Path, sess_out: Path, stat: Callable) -> int:
    """Copy cam0/data/*.jpg to sess_out/cam0/data/. Returns frame count."""
    src_cam = source_dir / "cam0" / "data"
    dst_cam = sess_out / "cam0" / "data"

    if not src_cam.exists():
        stat("  ℹ No cam0/data found — dataset will have no frames for this session.")
        return 0

    dst_cam.mkdir(parents=True, exist_ok=True)
    frames = sorted(src_cam.glob("*.jpg"))
    for jp in frames:
        dst = dst_cam / jp.name
        if not dst.exists():
            shutil.copy2(jp, dst)
    return len(frames)


def _copy_imu0(source_dir: Path, sess_out: Path, stat: Callable) -> int:
    """Copy imu0/data.csv to sess_out/imu0/data.csv. Returns sample count."""
    src_imu = source_dir / "imu0" / "data.csv"
    dst_imu_dir = sess_out / "imu0"
    dst_imu_dir.mkdir(exist_ok=True)
    dst_imu = dst_imu_dir / "data.csv"

    if not src_imu.exists():
        stat("  ℹ No imu0/data.csv found.")
        # Write empty CSV with correct header
        dst_imu.write_text(_IMU_HEADER + "\n")
        return 0

    # Copy and count lines (rewrite to normalise header if needed)
    rows = _read_euroc_imu_rows(src_imu)
    _write_euroc_imu_rows(dst_imu, rows)
    return len(rows)


def _read_euroc_imu_rows(path: Path) -> list[list[str]]:
    rows = []
    with open(path, newline="") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(line.split(","))
    return rows


def _write_euroc_imu_rows(path: Path, rows: list[list[str]]) -> None:
    with open(path, "w", newline="") as f:
        f.write(_IMU_HEADER + "\n")
        writer = csv.writer(f)
        writer.writerows(rows)


def _write_cam_sensor_yaml(sess_out: Path, session: Session) -> None:
    cam0_dir = sess_out / "cam0"
    cam0_dir.mkdir(exist_ok=True)
    yaml_path = cam0_dir / "sensor.yaml"

    # Pull resolution from first audit result if available
    res_w, res_h = 3840, 2160  # Hero 11 default
    if session.audit_results:
        r = session.audit_results[0]
        res_w, res_h = r.video_resolution

    yaml_path.write_text(
        f"sensor_type: camera\n"
        f"comment: {session.name}\n"
        f"T_BS:\n"
        f"  rows: 4\n"
        f"  cols: 4\n"
        f"  data: [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]\n"
        f"rate_hz: 5\n"
        f"resolution: [{res_w}, {res_h}]\n"
        f"camera_model: pinhole\n"
        f"intrinsics: [0, 0, 0, 0]  # Fill after calibration\n"
        f"distortion_model: radial-tangential\n"
        f"distortion_coefficients: [0, 0, 0, 0]  # Fill after calibration\n"
    )


def _write_imu_sensor_yaml(sess_out: Path) -> None:
    imu0_dir = sess_out / "imu0"
    imu0_dir.mkdir(exist_ok=True)
    yaml_path = imu0_dir / "sensor.yaml"
    yaml_path.write_text(
        "sensor_type: imu\n"
        "comment: GoPro Hero 11 / Insta360 X4 IMU\n"
        "T_BS:\n"
        "  rows: 4\n"
        "  cols: 4\n"
        "  data: [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]\n"
        "rate_hz: 200\n"
        "gyroscope_noise_density: 0.0      # Fill from IMU calibration (Step 1)\n"
        "gyroscope_random_walk:   0.0\n"
        "accelerometer_noise_density: 0.0\n"
        "accelerometer_random_walk:   0.0\n"
    )


def _write_session_yaml(sess_out: Path, session: Session) -> None:
    (sess_out / "session.yaml").write_text(
        f"id: {session.id}\n"
        f"name: {session.name}\n"
        f"environment: {session.environment}\n"
        f"location: {session.location}\n"
        f"created_at: {session.created_at.isoformat()}\n"
        f"extraction_status: {session.extraction_status}\n"
        f"notes: |\n"
        + "".join(f"  {line}\n" for line in session.notes.splitlines())
    )


def _integrity_check(manifest: dict, warnings: list[str]) -> None:
    for s in manifest.get("sessions", []):
        if s.get("status") != "ok":
            continue
        n_frames = s.get("frames", 0)
        n_imu = s.get("imu_samples", 0)
        if n_frames == 0:
            warnings.append(
                f"Session {s['name']}: no frames in dataset. "
                "Run extraction to generate frames."
            )
        if n_imu == 0:
            warnings.append(
                f"Session {s['name']}: no IMU samples in dataset. "
                "Run extraction to generate IMU data."
            )
        if n_frames > 0 and n_imu > 0:
            ratio = n_imu / n_frames
            if ratio < 10:
                warnings.append(
                    f"Session {s['name']}: low IMU/frame ratio ({ratio:.1f}x). "
                    "Expected ~40x for 5fps + 200Hz."
                )


# ---------------------------------------------------------------------------
# ROS bag 2 builder (requires rosbag2_py)
# ---------------------------------------------------------------------------

def _build_rosbag2(
    session_ids: list[str],
    sm: SessionManager,
    output_dir: Path,
    prog: Callable[[int], None],
    stat: Callable[[str], None],
) -> DatasetBuildResult:
    try:
        import rosbag2_py  # type: ignore  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "rosbag2_py is not installed. "
            "Install ROS 2 and source its setup.bash, or choose 'euroc' format."
        )

    # Build EuRoC first, then convert to rosbag2
    stat("Building EuRoC intermediate…")
    euroc_dir = output_dir / "_euroc_tmp"
    euroc_result = _build_euroc(session_ids, sm, euroc_dir, prog, stat)

    stat("Converting to rosbag2…")
    bag_path = output_dir / "dataset.bag"
    _euroc_to_rosbag2(euroc_dir, bag_path, stat)

    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps({
        "format": "rosbag2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bag_path": str(bag_path),
        "sessions": session_ids,
        "total_frames": euroc_result.total_frames,
        "total_imu_samples": euroc_result.total_imu_samples,
    }, indent=2))

    prog(100)
    return DatasetBuildResult(
        output_dir=output_dir,
        session_ids=session_ids,
        total_frames=euroc_result.total_frames,
        total_imu_samples=euroc_result.total_imu_samples,
        format="rosbag2",
        manifest_path=manifest_path,
        warnings=euroc_result.warnings,
    )


def _euroc_to_rosbag2(euroc_dir: Path, bag_path: Path, stat: Callable) -> None:
    """Convert EuRoC directory tree to a rosbag2 SQLite bag."""
    import rosbag2_py  # type: ignore
    from rclpy.serialization import serialize_message  # type: ignore
    from sensor_msgs.msg import Image, Imu  # type: ignore
    import cv2

    storage_opts = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="sqlite3")
    conv_opts = rosbag2_py.ConverterOptions("", "")
    writer = rosbag2_py.SequentialWriter()
    writer.open(storage_opts, conv_opts)

    writer.create_topic(rosbag2_py.TopicMetadata(
        id=0, name="/cam0/image_raw", type="sensor_msgs/msg/Image",
        serialization_format="cdr",
    ))
    writer.create_topic(rosbag2_py.TopicMetadata(
        id=1, name="/imu0", type="sensor_msgs/msg/Imu",
        serialization_format="cdr",
    ))

    for sess_dir in sorted(euroc_dir.iterdir()):
        if not sess_dir.is_dir():
            continue

        # Images
        cam_data = sess_dir / "cam0" / "data"
        if cam_data.exists():
            for jp in sorted(cam_data.glob("*.jpg")):
                ts_ns = int(jp.stem)
                img_bgr = cv2.imread(str(jp))
                if img_bgr is None:
                    continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                h, w = img_rgb.shape[:2]
                msg = Image()
                msg.header.stamp.sec = ts_ns // 1_000_000_000
                msg.header.stamp.nanosec = ts_ns % 1_000_000_000
                msg.height = h
                msg.width = w
                msg.encoding = "rgb8"
                msg.data = img_rgb.tobytes()
                writer.write("/cam0/image_raw", serialize_message(msg), ts_ns)

        # IMU
        imu_csv = sess_dir / "imu0" / "data.csv"
        if imu_csv.exists():
            for row in _read_euroc_imu_rows(imu_csv):
                if len(row) < 7:
                    continue
                ts_ns = int(row[0])
                msg = Imu()
                msg.header.stamp.sec = ts_ns // 1_000_000_000
                msg.header.stamp.nanosec = ts_ns % 1_000_000_000
                msg.angular_velocity.x = float(row[1])
                msg.angular_velocity.y = float(row[2])
                msg.angular_velocity.z = float(row[3])
                msg.linear_acceleration.x = float(row[4])
                msg.linear_acceleration.y = float(row[5])
                msg.linear_acceleration.z = float(row[6])
                writer.write("/imu0", serialize_message(msg), ts_ns)

    stat(f"rosbag2 written to {bag_path}")


# ---------------------------------------------------------------------------
# HDF5 builder (requires h5py)
# ---------------------------------------------------------------------------

def _build_hdf5(
    session_ids: list[str],
    sm: SessionManager,
    output_dir: Path,
    prog: Callable[[int], None],
    stat: Callable[[str], None],
) -> DatasetBuildResult:
    try:
        import h5py  # type: ignore  # noqa: F401
        import numpy as np  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "h5py is not installed. "
            "Run: pip install h5py — or choose 'euroc' format."
        )

    import h5py
    import numpy as np
    import cv2

    h5_path = output_dir / "dataset.h5"
    total_frames = 0
    total_imu = 0
    warnings: list[str] = []

    with h5py.File(h5_path, "w") as hf:
        hf.attrs["format"] = "roverdatakit_hdf5_v1"
        hf.attrs["created_at"] = datetime.now(timezone.utc).isoformat()

        n = len(session_ids)
        for si, session_id in enumerate(session_ids):
            base_pct = si * 90 // n
            end_pct  = (si + 1) * 90 // n

            session = sm.get_session(session_id)
            stat(f"[{si+1}/{n}] Writing session {session.name} to HDF5…")

            source_dir = _find_extraction_dir(sm, session_id)
            if source_dir is None:
                w = f"Session {session.name} has no extracted data — skipping."
                warnings.append(w)
                prog(end_pct)
                continue

            grp = hf.create_group(session_id)
            grp.attrs["name"] = session.name
            grp.attrs["environment"] = session.environment
            grp.attrs["location"] = session.location

            # Frames → variable-length byte dataset
            cam_data = source_dir / "cam0" / "data"
            if cam_data.exists():
                frames = sorted(cam_data.glob("*.jpg"))
                ts_arr = np.array([int(jp.stem) for jp in frames], dtype=np.int64)
                grp.create_dataset("cam0/timestamps_ns", data=ts_arr)
                vlen_bytes = h5py.special_dtype(vlen=np.dtype("uint8"))
                dset = grp.create_dataset(
                    "cam0/images", shape=(len(frames),), dtype=vlen_bytes
                )
                for fi, jp in enumerate(frames):
                    raw = np.frombuffer(jp.read_bytes(), dtype=np.uint8)
                    dset[fi] = raw
                total_frames += len(frames)
                prog(base_pct + (end_pct - base_pct) // 2)

            # IMU
            imu_csv = source_dir / "imu0" / "data.csv"
            if imu_csv.exists():
                rows = _read_euroc_imu_rows(imu_csv)
                if rows:
                    arr = np.array([[float(v) for v in r[:7]] for r in rows if len(r) >= 7])
                    grp.create_dataset("imu0/data", data=arr)
                    grp["imu0/data"].attrs["columns"] = (
                        "timestamp_ns,gx,gy,gz,ax,ay,az"
                    )
                    total_imu += len(arr)

            prog(end_pct)
            stat(f"  ✓ session written")

    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps({
        "format": "hdf5",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "h5_path": str(h5_path),
        "sessions": session_ids,
        "total_frames": total_frames,
        "total_imu_samples": total_imu,
    }, indent=2))

    prog(100)
    stat("HDF5 dataset build complete.")
    return DatasetBuildResult(
        output_dir=output_dir,
        session_ids=session_ids,
        total_frames=total_frames,
        total_imu_samples=total_imu,
        format="hdf5",
        manifest_path=manifest_path,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Dataset integrity verification
# ---------------------------------------------------------------------------

def verify_dataset_integrity(output_dir: Path) -> dict:
    """Verify that a built EuRoC dataset has consistent frame count and IMU rows.

    Checks:
      1. Counts .jpg files across all cam0/data/ subdirectories.
      2. Counts non-header rows in all imu0/data.csv files.
      3. Reports any consecutive IMU timestamp gaps > 200 ms.

    Args:
        output_dir: Root of the dataset (contains session subdirs or direct EuRoC layout).

    Returns:
        {
          "ok": bool,
          "frame_count": int,
          "imu_count": int,
          "gaps": list[int],   # gap durations in ms where gap > 200 ms
          "warnings": list[str]
        }
    """
    output_dir = Path(output_dir)
    frame_count = 0
    imu_count = 0
    gaps: list[int] = []
    warnings_out: list[str] = []
    last_ts_ns: int | None = None

    # Search for cam0/data and imu0/data.csv recursively (up to depth 3)
    for cam_dir in list(output_dir.glob("*/cam0/data")) + list(output_dir.glob("cam0/data")):
        frame_count += len(list(cam_dir.glob("*.jpg")))

    for imu_csv in list(output_dir.glob("*/imu0/data.csv")) + list(output_dir.glob("imu0/data.csv")):
        try:
            with open(imu_csv, newline="") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    imu_count += 1
                    parts = line.split(",")
                    if parts:
                        try:
                            ts_ns = int(parts[0])
                            if last_ts_ns is not None:
                                delta_ms = (ts_ns - last_ts_ns) / 1e6
                                if delta_ms > 200:
                                    gaps.append(int(delta_ms))
                            last_ts_ns = ts_ns
                        except (ValueError, IndexError):
                            pass
        except Exception as exc:
            warnings_out.append(f"Could not read {imu_csv}: {exc}")

    if frame_count == 0:
        warnings_out.append("No frames found — dataset may be empty or wrong directory")
    if imu_count == 0:
        warnings_out.append("No IMU rows found — imu0/data.csv missing or empty")
    if gaps:
        warnings_out.append(f"{len(gaps)} IMU gap(s) > 200 ms detected")

    ok = frame_count > 0 and imu_count > 0 and not gaps
    logger.info(
        "verify_dataset_integrity: {} frames, {} IMU, {} gaps",
        frame_count, imu_count, len(gaps),
    )
    return {
        "ok": ok,
        "frame_count": frame_count,
        "imu_count": imu_count,
        "gaps": gaps,
        "warnings": warnings_out,
    }


# ---------------------------------------------------------------------------
# CVAT export
# ---------------------------------------------------------------------------

def export_cvat_format(
    frame_paths: list[str],
    frame_timestamps_ns: list[int],
    output_dir: Path,
) -> Path:
    """Export frames in CVAT-compatible format.

    Creates:
      <output_dir>/
        images/
          <timestamp_ns>.jpg   (symlink or copy)
        images.xml             (CVAT task manifest)

    Args:
        frame_paths:          List of absolute paths to frame .jpg files.
        frame_timestamps_ns:  Corresponding timestamps in nanoseconds.
        output_dir:           Root directory for the CVAT task.

    Returns:
        Path to output_dir.
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Copy frames (symlinks can break on zip/transfer)
    copied_names: list[str] = []
    for src, ts_ns in zip(frame_paths, frame_timestamps_ns):
        src_path = Path(src)
        dest = images_dir / f"{ts_ns}.jpg"
        if not dest.exists():
            try:
                shutil.copy2(src_path, dest)
            except Exception as exc:
                logger.warning("export_cvat: could not copy {}: {}", src_path.name, exc)
                continue
        copied_names.append(dest.name)

    # Write images.xml CVAT manifest
    xml_path = output_dir / "images.xml"
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        f.write('<annotations>\n')
        f.write('  <version>1.1</version>\n')
        for i, name in enumerate(copied_names):
            f.write(
                f'  <image id="{i}" name="{name}" width="0" height="0">\n'
                f'  </image>\n'
            )
        f.write('</annotations>\n')

    logger.info(
        "export_cvat_format: {} frames → {}",
        len(copied_names), output_dir,
    )
    return output_dir
