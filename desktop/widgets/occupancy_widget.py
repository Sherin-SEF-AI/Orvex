"""
desktop/widgets/occupancy_widget.py — BEV Occupancy Grid widget.

Runs the occupancy pipeline on depth results and detection annotations
from a session, shows each frame's bird's-eye-view grid, and allows
exporting a video flythrough of the occupancy sequence.

Prerequisites (checked at runtime, not import time):
  - Depth estimation must have been run (session_folder/depth/)
  - Auto-label must have been run (session_folder/labels/ or autolabel/)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from desktop.theme import (
    ACCENT, BG, BORDER, HI, MUTED, PANEL, SUCCESS, TEXT, WARNING,
)


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class _OccupancyWorker(QThread):
    """Locate depth results + annotations for a session and run the pipeline.

    Signals:
        status(str)      — human-readable status line
        progress(int)    — 0-100
        result(object)   — list[OccupancyFrame]
        error(str)       — actionable error message
    """

    status   = pyqtSignal(str)
    progress = pyqtSignal(int)
    result   = pyqtSignal(object)
    error    = pyqtSignal(str)

    def __init__(
        self,
        session_id: str,
        sm,
        config,  # OccupancyConfig
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._session_id = session_id
        self._sm = sm
        self._config = config

    # ------------------------------------------------------------------

    def run(self) -> None:
        from core.models import (
            CalibrationResult,
            DepthResult,
            Detection,
            FrameAnnotation,
            OccupancyConfig,
        )
        from core.occupancy import run_occupancy_pipeline

        try:
            session_folder = self._sm.session_folder(self._session_id)

            # ── 1. Collect depth results ───────────────────────────────────
            self.status.emit("Scanning for depth results…")
            depth_dir = session_folder / "depth"
            if not depth_dir.exists():
                self.error.emit(
                    f"No depth results found at '{depth_dir}'. "
                    "Run Depth Estimation first."
                )
                return

            depth_results = _collect_depth_results(depth_dir)
            if not depth_results:
                self.error.emit(
                    f"Depth directory '{depth_dir}' contains no valid depth files. "
                    "Re-run Depth Estimation."
                )
                return

            self.status.emit(
                f"Found {len(depth_results)} depth result(s)."
            )
            self.progress.emit(10)

            # ── 2. Collect annotations ─────────────────────────────────────
            self.status.emit("Scanning for annotations…")
            annotations = _collect_annotations(session_folder)
            if not annotations:
                self.status.emit(
                    "No annotations found — occupancy will be geometry-only "
                    "(no detection overlay)."
                )

            self.progress.emit(20)

            # ── 3. Load calibration ────────────────────────────────────────
            fx, fy, cx, cy = _load_calibration(self._sm, self._session_id)
            if fx == 0.0:
                self.status.emit(
                    "Camera calibration not available — "
                    "using default pinhole estimate."
                )

            self.progress.emit(25)

            # ── 4. Run pipeline ────────────────────────────────────────────
            output_dir = str(session_folder / "occupancy")
            self.status.emit(
                f"Running occupancy pipeline on {len(depth_results)} frame(s)…"
            )

            def _progress_cb(pct: int) -> None:
                # Pipeline runs 0-100; we remap to 25-95 overall
                mapped = 25 + int(pct * 0.70)
                self.progress.emit(mapped)

            occ_frames = run_occupancy_pipeline(
                depth_results=depth_results,
                annotations=annotations,
                fx=fx, fy=fy, cx=cx, cy=cy,
                config=self._config,
                output_dir=output_dir,
                progress_callback=_progress_cb,
            )

            self.progress.emit(100)
            self.status.emit(
                f"Occupancy pipeline complete — {len(occ_frames)} frame(s)."
            )
            self.result.emit(occ_frames)

        except Exception as exc:
            self.error.emit(
                f"Occupancy pipeline failed: {exc}\n"
                "Check that depth estimation and auto-label completed successfully."
            )


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class OccupancyWidget(QWidget):
    """Bird's-Eye-View occupancy grid tab widget."""

    def __init__(self, session_manager, parent=None) -> None:
        super().__init__(parent)
        self._sm = session_manager
        self._session_id: str = ""
        self._worker: _OccupancyWorker | None = None
        self._occ_frames: list = []          # list[OccupancyFrame]
        self._frame_index: int = 0
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ── Left config panel ─────────────────────────────────────────
        cfg_w = QWidget()
        cfg_w.setMinimumWidth(260)
        cfg_w.setMaximumWidth(380)
        cfg_layout = QVBoxLayout(cfg_w)
        cfg_layout.setContentsMargins(8, 8, 8, 8)

        # Prereq warning
        prereq = QLabel(
            "Run Depth estimation and Auto-Label first."
        )
        prereq.setWordWrap(True)
        prereq.setStyleSheet(
            f"color: {WARNING}; font-size: 11px; padding: 4px;"
        )
        cfg_layout.addWidget(prereq)

        # Grid Settings group
        grid_group = QGroupBox("Grid Settings")
        gg = QVBoxLayout(grid_group)

        gg.addWidget(QLabel("Resolution (m/cell):"))
        self._res_spin = QDoubleSpinBox()
        self._res_spin.setRange(0.05, 0.5)
        self._res_spin.setSingleStep(0.05)
        self._res_spin.setValue(0.1)
        self._res_spin.setToolTip(
            "Grid cell size in metres. Smaller = finer detail, higher memory."
        )
        gg.addWidget(self._res_spin)

        gg.addWidget(QLabel("Max depth (m):"))
        self._maxdepth_spin = QDoubleSpinBox()
        self._maxdepth_spin.setRange(5.0, 50.0)
        self._maxdepth_spin.setSingleStep(1.0)
        self._maxdepth_spin.setValue(30.0)
        gg.addWidget(self._maxdepth_spin)

        gg.addWidget(QLabel("Camera height (m):"))
        self._camheight_spin = QDoubleSpinBox()
        self._camheight_spin.setRange(0.3, 3.0)
        self._camheight_spin.setSingleStep(0.1)
        self._camheight_spin.setValue(1.0)
        self._camheight_spin.setToolTip(
            "Height of camera above ground plane in metres."
        )
        gg.addWidget(self._camheight_spin)

        cfg_layout.addWidget(grid_group)

        # Fusion group
        fusion_group = QGroupBox("Temporal Fusion")
        fg = QVBoxLayout(fusion_group)

        fg.addWidget(QLabel("Window size (frames):"))
        self._window_spin = QSpinBox()
        self._window_spin.setRange(1, 20)
        self._window_spin.setValue(5)
        self._window_spin.setToolTip(
            "Number of recent frames to fuse. Larger = smoother, less reactive."
        )
        fg.addWidget(self._window_spin)

        fg.addWidget(QLabel("Decay factor:"))
        self._decay_spin = QDoubleSpinBox()
        self._decay_spin.setRange(0.5, 1.0)
        self._decay_spin.setSingleStep(0.01)
        self._decay_spin.setValue(0.95)
        self._decay_spin.setDecimals(2)
        self._decay_spin.setToolTip(
            "Weight multiplier per step back in time (1.0 = no decay)."
        )
        fg.addWidget(self._decay_spin)

        cfg_layout.addWidget(fusion_group)

        # Run button
        self._run_btn = QPushButton("Generate Occupancy Grid")
        self._run_btn.setObjectName("DangerBtn")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._run)
        cfg_layout.addWidget(self._run_btn)

        # Export video button
        self._video_btn = QPushButton("Export Occupancy Video")
        self._video_btn.setEnabled(False)
        self._video_btn.clicked.connect(self._export_video)
        cfg_layout.addWidget(self._video_btn)

        # Status label
        self._status_label = QLabel("Select a session to begin.")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        cfg_layout.addWidget(self._status_label)

        cfg_layout.addStretch()
        splitter.addWidget(cfg_w)

        # ── Right panel ───────────────────────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 8, 8, 8)

        # BEV visualization label
        bev_group = QGroupBox("Bird's-Eye-View Occupancy Grid")
        bl = QVBoxLayout(bev_group)
        self._bev_label = QLabel("No occupancy grid generated yet.")
        self._bev_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._bev_label.setMinimumSize(400, 400)
        self._bev_label.setStyleSheet(
            f"border: 1px solid {BORDER}; "
            f"background: {PANEL}; "
            f"color: {MUTED};"
        )
        bl.addWidget(self._bev_label)
        rl.addWidget(bev_group, stretch=3)

        # Frame scrubber
        scrubber_row = QHBoxLayout()
        scrubber_row.addWidget(QLabel("Frame:"))
        self._scrubber = QSlider(Qt.Orientation.Horizontal)
        self._scrubber.setRange(0, 0)
        self._scrubber.valueChanged.connect(self._on_scrub)
        scrubber_row.addWidget(self._scrubber, stretch=1)
        self._frame_label = QLabel("0 / 0")
        self._frame_label.setMinimumWidth(60)
        scrubber_row.addWidget(self._frame_label)
        rl.addLayout(scrubber_row)

        # Stats group
        stats_group = QGroupBox("Frame Statistics")
        sg = QVBoxLayout(stats_group)
        self._stats_table = QTableWidget(3, 2)
        self._stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self._stats_table.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._stats_table.setMaximumHeight(100)
        self._stats_table.horizontalHeader().setStretchLastSection(True)
        self._stats_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        _set_stats_row(self._stats_table, 0, "Occupied cells", "—")
        _set_stats_row(self._stats_table, 1, "Free cells",     "—")
        _set_stats_row(self._stats_table, 2, "Occupancy %",    "—")
        sg.addWidget(self._stats_table)
        rl.addWidget(stats_group)

        # Log
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(80)
        self._log.setStyleSheet(
            f"background: #0d1020; color: {MUTED}; font-family: monospace; font-size: 11px;"
        )
        rl.addWidget(self._log)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([280, 720])

    # ------------------------------------------------------------------
    # Public slots
    # ------------------------------------------------------------------

    @pyqtSlot(str)
    def on_session_changed(self, session_id: str) -> None:
        """Called by main window when user selects a different session."""
        self._session_id = session_id
        self._occ_frames.clear()
        self._scrubber.setRange(0, 0)
        self._frame_label.setText("0 / 0")
        self._bev_label.setText("No occupancy grid generated yet.")
        self._bev_label.setPixmap(QPixmap())   # clear any old image
        _reset_stats_table(self._stats_table)
        self._video_btn.setEnabled(False)

        if not session_id:
            self._run_btn.setEnabled(False)
            self._status_label.setText("Select a session to begin.")
            return

        # Check if depth results exist
        session_folder = self._sm.session_folder(session_id)
        depth_dir = session_folder / "depth"
        has_depth = depth_dir.exists() and any(
            depth_dir.glob("*.npy")
        ) or (
            depth_dir.exists() and any(depth_dir.glob("*.png"))
        )

        if not has_depth:
            self._run_btn.setEnabled(False)
            self._status_label.setText(
                "No depth results found for this session. "
                "Run Depth Estimation first."
            )
        else:
            self._run_btn.setEnabled(True)
            depth_count = len(list(depth_dir.glob("*.npy"))) + \
                          len(list(depth_dir.glob("*.png")))
            self._status_label.setText(
                f"Session ready — {depth_count} depth file(s) detected."
            )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def _run(self) -> None:
        if not self._session_id:
            return
        if self._worker and self._worker.isRunning():
            self._log.append("[warn] Pipeline is already running.")
            return

        from core.models import OccupancyConfig

        config = OccupancyConfig(
            grid_resolution_m=self._res_spin.value(),
            grid_width_m=20.0,
            grid_height_m=30.0,
            camera_height_m=self._camheight_spin.value(),
            max_depth_m=self._maxdepth_spin.value(),
            temporal_fusion_window=self._window_spin.value(),
            decay_factor=self._decay_spin.value(),
        )

        self._worker = _OccupancyWorker(
            session_id=self._session_id,
            sm=self._sm,
            config=config,
        )
        self._worker.status.connect(self._on_status)
        self._worker.progress.connect(self._on_progress)
        self._worker.result.connect(self._on_result)
        self._worker.error.connect(self._on_error)

        self._run_btn.setEnabled(False)
        self._video_btn.setEnabled(False)
        self._log.clear()
        self._log.append("Starting occupancy pipeline…")
        self._worker.start()

    # ------------------------------------------------------------------
    # Worker signal handlers
    # ------------------------------------------------------------------

    @pyqtSlot(str)
    def _on_status(self, msg: str) -> None:
        self._status_label.setText(msg)
        self._log.append(msg)

    @pyqtSlot(int)
    def _on_progress(self, pct: int) -> None:
        self._log.append(f"[{pct:3d}%]")
        # Scroll log to bottom
        sb = self._log.verticalScrollBar()
        if sb:
            sb.setValue(sb.maximum())

    @pyqtSlot(object)
    def _on_result(self, frames: Any) -> None:
        self._occ_frames = frames or []
        self._run_btn.setEnabled(True)
        n = len(self._occ_frames)
        if n == 0:
            self._status_label.setText(
                "Pipeline complete but no frames were generated. "
                "Check that depth and label files are valid."
            )
            return

        self._scrubber.setRange(0, n - 1)
        self._scrubber.setValue(0)
        self._on_scrub(0)
        self._video_btn.setEnabled(True)
        self._status_label.setText(
            f"Done — {n} occupancy frame(s) generated."
        )

    @pyqtSlot(str)
    def _on_error(self, msg: str) -> None:
        self._run_btn.setEnabled(True)
        self._log.append(f"[ERROR] {msg}")
        QMessageBox.critical(self, "Occupancy Pipeline Error", msg)

    # ------------------------------------------------------------------
    # Scrubber
    # ------------------------------------------------------------------

    @pyqtSlot(int)
    def _on_scrub(self, idx: int) -> None:
        if not self._occ_frames or idx >= len(self._occ_frames):
            return
        self._frame_index = idx
        frame = self._occ_frames[idx]
        n = len(self._occ_frames)
        self._frame_label.setText(f"{idx + 1} / {n}")

        # Load and display BEV visualization
        _load_pixmap_into(self._bev_label, frame.visualization_path)

        # Update stats table
        _set_stats_row(
            self._stats_table, 0,
            "Occupied cells", str(frame.occupied_cells),
        )
        _set_stats_row(
            self._stats_table, 1,
            "Free cells", str(frame.free_cells),
        )
        _set_stats_row(
            self._stats_table, 2,
            "Occupancy %", f"{frame.occupancy_percent:.1f}%",
        )

    # ------------------------------------------------------------------
    # Export video
    # ------------------------------------------------------------------

    def _export_video(self) -> None:
        if not self._occ_frames:
            return

        viz_paths = [
            f.visualization_path
            for f in self._occ_frames
            if f.visualization_path and Path(f.visualization_path).exists()
        ]
        if not viz_paths:
            QMessageBox.warning(
                self, "No visualizations",
                "Visualization PNGs not found — re-run the pipeline.",
            )
            return

        out_dir = Path(viz_paths[0]).parent.parent
        out_path = str(out_dir / "occupancy_video.mp4")

        try:
            _generate_occupancy_video(viz_paths, out_path)
            self._status_label.setText(f"Video saved: {out_path}")
            self._log.append(f"[ok] Occupancy video: {out_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))


# ---------------------------------------------------------------------------
# Helper: export video via ffmpeg subprocess
# ---------------------------------------------------------------------------

def _generate_occupancy_video(
    viz_paths: list[str],
    output_path: str,
    fps: float = 10.0,
) -> None:
    """Write visualization frames to an MP4 using ffmpeg.

    Writes a text file listing frame paths, then calls ffmpeg concat demuxer.

    Args:
        viz_paths:   Ordered list of visualization PNG paths.
        output_path: Destination MP4 path.
        fps:         Output video frame rate.

    Raises:
        RuntimeError: if ffmpeg is not available or exits non-zero.
    """
    import shutil
    import subprocess
    import tempfile

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install ffmpeg to export videos."
        )

    tmp_list = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    )
    duration = 1.0 / fps
    for p in viz_paths:
        tmp_list.write(f"file '{p}'\n")
        tmp_list.write(f"duration {duration:.4f}\n")
    # ffmpeg concat demuxer needs last file repeated
    tmp_list.write(f"file '{viz_paths[-1]}'\n")
    tmp_list.flush()
    tmp_list.close()

    cmd = [
        ffmpeg, "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", tmp_list.name,
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        os.unlink(tmp_list.name)
    except OSError:
        pass

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {result.returncode}):\n{result.stderr[-500:]}"
        )


# ---------------------------------------------------------------------------
# Helper: collect DepthResult objects from depth directory
# ---------------------------------------------------------------------------

def _collect_depth_results(depth_dir: Path) -> list:
    """Scan a depth directory and build DepthResult objects.

    Looks for .npy files (preferred) and .png files as fallback.
    Matches frame paths by pairing depth stems with sibling frame images.

    Returns list[DepthResult].
    """
    from core.models import DepthResult

    results: list[DepthResult] = []

    # Prefer .npy files; also collect any .png that are NOT color maps
    npy_files = sorted(depth_dir.glob("*.npy"))
    png_files  = sorted(depth_dir.glob("*.png"))

    # If .npy files exist, use them as primary depth source
    if npy_files:
        for npy in npy_files:
            stem = npy.stem
            # Try to locate the corresponding color PNG
            color_png = depth_dir / f"{stem}_color.png"
            color_path = str(color_png) if color_png.exists() else None
            # Try to locate the matching frame
            frame_path = _find_frame_for_stem(depth_dir, stem)
            results.append(DepthResult(
                frame_path=frame_path,
                depth_raw_path=str(npy),
                depth_color_path=color_path,
                min_depth=0.0,
                max_depth=30.0,
                mean_depth=5.0,
                inference_time_ms=0.0,
            ))
    else:
        # Fall back to .png depth files
        for png in png_files:
            if "_color" in png.stem:
                continue
            stem = png.stem
            frame_path = _find_frame_for_stem(depth_dir, stem)
            results.append(DepthResult(
                frame_path=frame_path,
                depth_raw_path=str(png),
                depth_color_path=None,
                min_depth=0.0,
                max_depth=30.0,
                mean_depth=5.0,
                inference_time_ms=0.0,
            ))

    return results


def _find_frame_for_stem(depth_dir: Path, stem: str) -> str:
    """Attempt to find the source frame for a depth file stem.

    Checks parent directories for matching .jpg/.png files.
    Returns stem string if nothing is found (not None — models require str).
    """
    # Most common layout: session_folder/depth/xxx.npy, frames at ../cam0/data/
    session_folder = depth_dir.parent
    for candidate_dir in [
        session_folder / "cam0" / "data",
        session_folder / "extraction_gopro" / "cam0" / "data",
        session_folder,
    ]:
        for ext in (".jpg", ".png", ".jpeg"):
            p = candidate_dir / f"{stem}{ext}"
            if p.exists():
                return str(p)

    # Walk the whole session folder for a matching stem
    for ext in (".jpg", ".png", ".jpeg"):
        matches = list(session_folder.rglob(f"{stem}{ext}"))
        if matches:
            return str(matches[0])

    return stem   # best effort — depth_to_point_cloud does not need the frame


# ---------------------------------------------------------------------------
# Helper: collect FrameAnnotation objects from label directories
# ---------------------------------------------------------------------------

def _collect_annotations(session_folder: Path) -> list:
    """Scan label directories for YOLO .txt files and build FrameAnnotation list.

    Search order:
      1. session_folder/autolabel/labels/
      2. session_folder/labels/
      3. session_folder/autolabel/

    Returns list[FrameAnnotation] (may be empty — that is not an error).
    """
    from core.models import Detection, FrameAnnotation

    label_dirs = [
        session_folder / "autolabel" / "labels",
        session_folder / "labels",
        session_folder / "autolabel",
    ]

    label_dir: Path | None = None
    for d in label_dirs:
        if d.exists() and any(d.glob("*.txt")):
            label_dir = d
            break

    if label_dir is None:
        return []

    annotations: list[FrameAnnotation] = []
    for txt_file in sorted(label_dir.glob("*.txt")):
        frame_stem = txt_file.stem
        frame_path = _find_frame_for_stem(session_folder, frame_stem)

        detections: list[Detection] = []
        try:
            with open(txt_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    class_id = int(parts[0])
                    cx_n = float(parts[1])
                    cy_n = float(parts[2])
                    w_n  = float(parts[3])
                    h_n  = float(parts[4])
                    conf = float(parts[5]) if len(parts) > 5 else 1.0
                    # Normalize: we don't know image size here, use 1.0 as proxy
                    # The actual bbox in pixels will be estimated later with depth
                    detections.append(Detection(
                        class_id=class_id,
                        class_name=_class_id_to_name(class_id),
                        confidence=conf,
                        bbox_xyxy=[
                            cx_n - w_n / 2,
                            cy_n - h_n / 2,
                            cx_n + w_n / 2,
                            cy_n + h_n / 2,
                        ],
                        bbox_xywhn=[cx_n, cy_n, w_n, h_n],
                    ))
        except Exception:
            continue

        annotations.append(FrameAnnotation(
            frame_path=frame_path,
            detections=detections,
            inference_time_ms=0.0,
            model_version="unknown",
        ))

    return annotations


# ---------------------------------------------------------------------------
# Helper: load calibration from session metadata
# ---------------------------------------------------------------------------

def _load_calibration(sm, session_id: str) -> tuple[float, float, float, float]:
    """Try to load camera intrinsics from session calibration JSON.

    Returns (fx, fy, cx, cy) as floats.
    Returns (0.0, 0.0, 0.0, 0.0) if no calibration available (not an error —
    depth_to_point_cloud will use auto-estimate).
    """
    session_folder = sm.session_folder(session_id)

    # Look for calibration files
    search_paths = [
        session_folder / "calibration" / "camera_intrinsics.json",
        session_folder / "camera_intrinsics.json",
        session_folder.parent.parent / "calibration" / "camera_intrinsics.json",
    ]

    for p in search_paths:
        if p.exists():
            try:
                with open(p, encoding="utf-8") as f:
                    cal = json.load(f)
                fx = float(cal.get("fx", 0.0))
                fy = float(cal.get("fy", 0.0))
                cx = float(cal.get("cx", 0.0))
                cy = float(cal.get("cy", 0.0))
                if fx > 0.0 and fy > 0.0:
                    return fx, fy, cx, cy
            except Exception:
                continue

    # Try session metadata for calibration session results
    try:
        session = sm.get_session(session_id)
        # Session TOML may have embedded calibration — check notes or audit
        # (No standard field for this; fall through to 0.0 defaults)
    except Exception:
        pass

    return 0.0, 0.0, 0.0, 0.0


# ---------------------------------------------------------------------------
# COCO class name lookup (top-80)
# ---------------------------------------------------------------------------
_COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def _class_id_to_name(class_id: int) -> str:
    if 0 <= class_id < len(_COCO_CLASSES):
        return _COCO_CLASSES[class_id]
    return f"class_{class_id}"


# ---------------------------------------------------------------------------
# Qt helper utilities
# ---------------------------------------------------------------------------

def _load_pixmap_into(label: QLabel, path: str) -> None:
    """Load an image from path into a QLabel, scaled to fit."""
    if not path or not Path(path).exists():
        label.setText("Visualization not available.")
        return
    pix = QPixmap(path)
    if pix.isNull():
        label.setText(f"Could not load image: {Path(path).name}")
        return
    label.setPixmap(
        pix.scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    )


def _set_stats_row(table: QTableWidget, row: int, metric: str, value: str) -> None:
    """Set both cells in a stats table row."""
    table.setItem(row, 0, QTableWidgetItem(metric))
    table.setItem(row, 1, QTableWidgetItem(value))


def _reset_stats_table(table: QTableWidget) -> None:
    """Reset all stats rows to '—'."""
    labels = ["Occupied cells", "Free cells", "Occupancy %"]
    for i, lbl in enumerate(labels):
        _set_stats_row(table, i, lbl, "—")
