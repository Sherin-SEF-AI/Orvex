"""
desktop/widgets/lane_widget.py — Lane detection widget for RoverDataKit.

Provides a complete UI for running the lane detection pipeline on extracted
session frames.  Supports the classical CV pipeline (always available) and an
optional UFLD model loaded from a user-specified weights file.

Layout:
  Left panel  — model/settings config + run/export controls.
  Right panel — original/overlay side-by-side, departure meter, scrubber,
                method badge, stats table, log.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import (
    QColor,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from desktop.theme import (
    ACCENT,
    BG,
    BORDER,
    CARD,
    HI,
    MUTED,
    PANEL,
    SUCCESS,
    TEXT,
    WARNING,
    badge_style,
)


# ─────────────────────────────────────────────────────────────────────────────
# Departure metre — custom QWidget
# ─────────────────────────────────────────────────────────────────────────────

class _DepartureMeter(QWidget):
    """A horizontal bar visualising lateral offset.

    Zones (from centre outward, each side):
      ■ Green   centre 30 % of half-width (|offset| < 0.15)
      ■ Yellow  next   25 % (0.15 ≤ |offset| < 0.40)
      ■ Red     outer  10 % (|offset| ≥ 0.40)

    A white vertical needle tracks ``lateral_offset`` in [−1, +1].
    """

    _BAR_H = 30
    _BAR_W = 300

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._offset: float = 0.0          # signed, −1 … +1
        self._status: str = "no_lane"
        self.setFixedHeight(self._BAR_H + 20)
        self.setMinimumWidth(self._BAR_W + 40)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )

    def set_offset(self, offset: float, status: str) -> None:
        self._offset = max(-1.0, min(1.0, offset))
        self._status = status
        self.update()

    # ------------------------------------------------------------------
    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        bar_top = 8
        bar_h = self._BAR_H
        bar_left = 20
        bar_right = w - 20
        bar_w = bar_right - bar_left
        mid_x = bar_left + bar_w // 2

        # ── Zone rectangles ────────────────────────────────────────────
        # The bar is divided into five zones (mirrored):
        #   |  red  | yellow | GREEN | yellow |  red  |
        # proportions of half-bar: 10% red, 25% yellow, 30% green (per side)
        # Total half: 10+25+30 = 65 % — remaining 35 % is outer red.
        # Expressed as fraction of half_bar:
        #   green  : |offset| < 0.15  → 0..0.15  mapped to 0..green_px
        #   yellow : 0.15 to 0.40
        #   red    : 0.40 to 1.00

        half_w = bar_w // 2

        green_right  = mid_x + int(half_w * 0.15)
        green_left   = mid_x - int(half_w * 0.15)

        yellow_right = mid_x + int(half_w * 0.40)
        yellow_left  = mid_x - int(half_w * 0.40)

        # Draw zones
        # Outer red
        p.fillRect(bar_left, bar_top, bar_w, bar_h, QColor("#8b0000"))
        # Yellow inner
        p.fillRect(yellow_left, bar_top, yellow_right - yellow_left, bar_h, QColor("#b8860b"))
        # Green centre
        p.fillRect(green_left, bar_top, green_right - green_left, bar_h, QColor("#1a6b2a"))

        # ── Border ─────────────────────────────────────────────────────
        p.setPen(QPen(QColor(BORDER), 1))
        p.drawRect(bar_left, bar_top, bar_w, bar_h)

        # ── Centre tick ────────────────────────────────────────────────
        p.setPen(QPen(QColor(TEXT), 1, Qt.PenStyle.DashLine))
        p.drawLine(mid_x, bar_top, mid_x, bar_top + bar_h)

        # ── Needle ─────────────────────────────────────────────────────
        needle_x = int(mid_x + self._offset * half_w)
        needle_x = max(bar_left + 2, min(bar_right - 2, needle_x))

        if self._status == "centered":
            needle_color = QColor("#27ae60")
        elif self._status in ("drifting_left", "drifting_right"):
            needle_color = QColor("#f39c12")
        else:
            needle_color = QColor("#e74c3c")

        pen = QPen(needle_color, 3)
        p.setPen(pen)
        p.drawLine(needle_x, bar_top - 4, needle_x, bar_top + bar_h + 4)

        # ── Label ──────────────────────────────────────────────────────
        p.setPen(QPen(QColor(MUTED), 1))
        p.drawText(bar_left, bar_top + bar_h + 16, "L")
        p.drawText(bar_right - 8, bar_top + bar_h + 16, "R")

        p.end()


# ─────────────────────────────────────────────────────────────────────────────
# Background worker
# ─────────────────────────────────────────────────────────────────────────────

class _LaneWorker(QThread):
    """Runs the lane pipeline in a background thread.

    Signals:
        progress(int):      0–100 percent.
        status(str):        Human-readable status message.
        result(object):     Tuple of (frames_list, report_dict).
        error(str):         Actionable error description.
    """

    progress = pyqtSignal(int)
    status   = pyqtSignal(str)
    result   = pyqtSignal(object)
    error    = pyqtSignal(str)

    def __init__(
        self,
        session_id: str,
        sm: Any,
        model_path: str | None,
        roi_top_percent: float,
        camera_height_m: float,
        camera_pitch_deg: float,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._session_id = session_id
        self._sm = sm
        self._model_path = model_path
        self._roi_top = roi_top_percent
        self._cam_h = camera_height_m
        self._cam_pitch = camera_pitch_deg

    # ------------------------------------------------------------------
    def run(self) -> None:
        try:
            from core.lane_detector import (
                LaneConfig,
                check_lane_dependencies,
                generate_lane_departure_report,
                load_lane_model,
                run_lane_pipeline,
            )
            from core.models import LaneConfig as _LaneConfigModel  # noqa: F401
        except ImportError as exc:
            self.error.emit(
                f"Failed to import lane_detector: {exc}. "
                "Ensure core/lane_detector.py is present and dependencies are installed."
            )
            return

        # ── Dependency check ──────────────────────────────────────────
        deps = check_lane_dependencies()
        if not deps["cv2"] or not deps["numpy"]:
            self.error.emit(
                "OpenCV (cv2) and NumPy are required for lane detection. "
                "Install them with: pip install opencv-python>=4.10.0 numpy>=1.26.0"
            )
            return

        # ── Locate frames ─────────────────────────────────────────────
        session_folder = self._sm.session_folder(self._session_id)
        frame_paths = _find_frames(session_folder)

        if not frame_paths:
            self.error.emit(
                f"No frame images found in session folder '{session_folder}'. "
                "Run the extraction step first to generate frames in "
                "cam0/data/ or frames/ subdirectory."
            )
            return

        self.status.emit(f"Found {len(frame_paths)} frames.")

        # ── Load model ────────────────────────────────────────────────
        model = load_lane_model(self._model_path)
        if model is not None:
            self.status.emit("UFLD model loaded. Running neural inference...")
        else:
            self.status.emit("Using classical CV pipeline (no UFLD model).")

        # ── Build config ──────────────────────────────────────────────
        from core.models import LaneConfig as LaneConfigModel
        config = LaneConfigModel(
            use_ufld=model is not None,
            ufld_conf_threshold=0.5,
            classical_fallback=True,
            roi_top_percent=self._roi_top / 100.0,
            camera_height_m=self._cam_h,
            camera_pitch_deg=self._cam_pitch,
        )

        output_dir = str(session_folder / "lane_detection")

        # ── Run pipeline ──────────────────────────────────────────────
        def _cb(pct: int) -> None:
            self.progress.emit(pct)
            self.status.emit(f"Processing frames... {pct}%")

        try:
            frames = run_lane_pipeline(
                frame_paths=frame_paths,
                model=model,
                config=config,
                output_dir=output_dir,
                progress_callback=_cb,
            )
        except RuntimeError as exc:
            self.error.emit(str(exc))
            return
        except Exception as exc:
            self.error.emit(
                f"Unexpected error during lane pipeline: {exc}. "
                "Check the log for details."
            )
            return

        report = generate_lane_departure_report(frames)
        self.status.emit(
            f"Complete — {report['frames_with_lanes']} / {len(frames)} frames "
            f"with lane markings detected."
        )
        self.result.emit((frames, report))


def _find_frames(session_folder: Path) -> list[str]:
    """Search for JPEG/PNG frame images inside a session folder.

    Checks, in order:
    1. ``<session>/cam0/data/``
    2. ``<session>/frames/``
    3. ``<session>/`` itself (shallow, no recursion)

    Returns a sorted list of absolute paths.
    """
    candidates = [
        session_folder / "cam0" / "data",
        session_folder / "frames",
        session_folder,
    ]
    for search_dir in candidates:
        if not search_dir.is_dir():
            continue
        exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        found: list[Path] = []
        for ext in exts:
            if search_dir == session_folder:
                # Only shallow at session root to avoid recursing into overlays etc.
                found.extend(search_dir.glob(ext))
            else:
                found.extend(search_dir.glob(ext))
        if found:
            return sorted(str(p) for p in found)
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Main widget
# ─────────────────────────────────────────────────────────────────────────────

class LaneWidget(QWidget):
    """Full lane detection UI tab for RoverDataKit desktop."""

    def __init__(self, session_manager: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sm = session_manager
        self._session_id: str = ""
        self._worker: _LaneWorker | None = None
        self._lane_frames: list[Any] = []  # list[LaneFrame]
        self._report: dict[str, Any] = {}
        self._frame_index: int = 0
        self._model_path: str | None = None

        self._build_ui()

    # ─────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        outer_split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(outer_split)

        outer_split.addWidget(self._build_left_panel())
        outer_split.addWidget(self._build_right_panel())
        outer_split.setStretchFactor(0, 0)
        outer_split.setStretchFactor(1, 1)
        outer_split.setSizes([300, 800])

    # ── Left config panel ──────────────────────────────────────────────────────

    def _build_left_panel(self) -> QWidget:
        w = QWidget()
        w.setMinimumWidth(260)
        w.setMaximumWidth(380)
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Model group
        model_group = QGroupBox("Model")
        mg = QVBoxLayout(model_group)
        self._model_info_label = QLabel("Classical CV (always available)")
        self._model_info_label.setWordWrap(True)
        self._model_info_label.setStyleSheet(f"color: {SUCCESS}; font-size: 11px;")
        mg.addWidget(self._model_info_label)

        self._model_path_label = QLabel("No UFLD weights loaded.")
        self._model_path_label.setWordWrap(True)
        self._model_path_label.setStyleSheet(f"color: {MUTED}; font-size: 10px;")
        mg.addWidget(self._model_path_label)

        self._browse_model_btn = QPushButton("Browse UFLD Weights (.pth)")
        self._browse_model_btn.clicked.connect(self._browse_model)
        mg.addWidget(self._browse_model_btn)

        self._clear_model_btn = QPushButton("Clear — use Classical Only")
        self._clear_model_btn.setEnabled(False)
        self._clear_model_btn.clicked.connect(self._clear_model)
        mg.addWidget(self._clear_model_btn)

        layout.addWidget(model_group)

        # Settings group
        settings_group = QGroupBox("Settings")
        sg = QVBoxLayout(settings_group)

        sg.addWidget(QLabel("ROI top (% from top of image):"))
        roi_row = QHBoxLayout()
        self._roi_slider = QSlider(Qt.Orientation.Horizontal)
        self._roi_slider.setRange(30, 80)
        self._roi_slider.setValue(55)
        self._roi_slider.setTickInterval(5)
        self._roi_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._roi_val_label = QLabel("55 %")
        self._roi_val_label.setStyleSheet(f"color: {TEXT}; min-width: 32px;")
        self._roi_slider.valueChanged.connect(
            lambda v: self._roi_val_label.setText(f"{v} %")
        )
        roi_row.addWidget(self._roi_slider, stretch=1)
        roi_row.addWidget(self._roi_val_label)
        sg.addLayout(roi_row)

        sg.addWidget(QLabel("Camera height (m):"))
        self._cam_height_spin = QDoubleSpinBox()
        self._cam_height_spin.setRange(0.3, 3.0)
        self._cam_height_spin.setValue(1.0)
        self._cam_height_spin.setSingleStep(0.1)
        self._cam_height_spin.setDecimals(2)
        sg.addWidget(self._cam_height_spin)

        sg.addWidget(QLabel("Camera pitch (deg, + = nose down):"))
        self._cam_pitch_spin = QDoubleSpinBox()
        self._cam_pitch_spin.setRange(-30.0, 30.0)
        self._cam_pitch_spin.setValue(0.0)
        self._cam_pitch_spin.setSingleStep(0.5)
        self._cam_pitch_spin.setDecimals(1)
        sg.addWidget(self._cam_pitch_spin)

        layout.addWidget(settings_group)

        # Run button
        self._run_btn = QPushButton("Detect Lanes")
        self._run_btn.setObjectName("DangerBtn")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._run)
        layout.addWidget(self._run_btn)

        # Export video button
        self._export_btn = QPushButton("Export Overlay Video")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export_video)
        layout.addWidget(self._export_btn)

        # Status label
        self._status_label = QLabel("Load a session to begin.")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        layout.addWidget(self._status_label)

        layout.addStretch()
        return w

    # ── Right panel ────────────────────────────────────────────────────────────

    def _build_right_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Side-by-side preview
        preview_split = QSplitter(Qt.Orientation.Horizontal)
        orig_group = QGroupBox("Original Frame")
        ol = QVBoxLayout(orig_group)
        self._orig_label = QLabel("No frame loaded.")
        self._orig_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._orig_label.setMinimumSize(300, 220)
        self._orig_label.setStyleSheet(f"background: {CARD}; color: {MUTED};")
        ol.addWidget(self._orig_label)
        preview_split.addWidget(orig_group)

        overlay_group = QGroupBox("Lane Overlay")
        vl = QVBoxLayout(overlay_group)
        self._overlay_label = QLabel("No overlay generated.")
        self._overlay_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._overlay_label.setMinimumSize(300, 220)
        self._overlay_label.setStyleSheet(f"background: {CARD}; color: {MUTED};")
        vl.addWidget(self._overlay_label)
        preview_split.addWidget(overlay_group)

        preview_split.setStretchFactor(0, 1)
        preview_split.setStretchFactor(1, 1)
        layout.addWidget(preview_split, stretch=3)

        # Departure meter
        meter_group = QGroupBox("Lane Departure Meter")
        meter_layout = QVBoxLayout(meter_group)
        self._departure_meter = _DepartureMeter()
        meter_layout.addWidget(self._departure_meter, alignment=Qt.AlignmentFlag.AlignHCenter)
        self._departure_label = QLabel("No data")
        self._departure_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._departure_label.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        meter_layout.addWidget(self._departure_label)
        layout.addWidget(meter_group)

        # Scrubber row
        scrubber_row = QHBoxLayout()
        scrubber_row.addWidget(QLabel("Frame:"))
        self._scrubber = QSlider(Qt.Orientation.Horizontal)
        self._scrubber.setRange(0, 0)
        self._scrubber.valueChanged.connect(self._on_scrub)
        scrubber_row.addWidget(self._scrubber, stretch=1)
        self._frame_label = QLabel("0 / 0")
        self._frame_label.setMinimumWidth(55)
        scrubber_row.addWidget(self._frame_label)

        # Detection method badge
        self._method_badge = QLabel("NONE")
        self._method_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._method_badge.setStyleSheet(badge_style(MUTED))
        self._method_badge.setMinimumWidth(70)
        scrubber_row.addWidget(self._method_badge)
        layout.addLayout(scrubber_row)

        # Stats table
        stats_group = QGroupBox("Detection Statistics")
        stats_layout = QVBoxLayout(stats_group)
        self._stats_table = QTableWidget(0, 2)
        self._stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self._stats_table.horizontalHeader().setStretchLastSection(True)
        self._stats_table.verticalHeader().setVisible(False)
        self._stats_table.setMaximumHeight(130)
        self._stats_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        stats_layout.addWidget(self._stats_table)
        layout.addWidget(stats_group)

        # Log
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(80)
        self._log.setPlaceholderText("Processing log will appear here...")
        layout.addWidget(self._log)

        return w

    # ─────────────────────────────────────────────────────────────────
    # Public slots
    # ─────────────────────────────────────────────────────────────────

    @pyqtSlot(str)
    def on_session_changed(self, session_id: str) -> None:
        """Called by the main window whenever the active session changes."""
        self._session_id = session_id
        self._run_btn.setEnabled(bool(session_id))
        self._lane_frames = []
        self._report = {}
        self._scrubber.setRange(0, 0)
        self._frame_label.setText("0 / 0")
        self._orig_label.setText("No frame loaded.")
        self._overlay_label.setText("No overlay generated.")
        self._stats_table.setRowCount(0)
        self._departure_meter.set_offset(0.0, "no_lane")
        self._departure_label.setText("No data")
        self._method_badge.setText("NONE")
        self._method_badge.setStyleSheet(badge_style(MUTED))
        self._export_btn.setEnabled(False)
        self._status_label.setText("Session loaded. Click 'Detect Lanes' to start.")

    # ─────────────────────────────────────────────────────────────────
    # Model browsing
    # ─────────────────────────────────────────────────────────────────

    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select UFLD Weights File",
            "",
            "PyTorch Weights (*.pth *.pt);;All Files (*)",
        )
        if not path:
            return
        self._model_path = path
        short = Path(path).name
        self._model_path_label.setText(f"Loaded: {short}")
        self._model_path_label.setStyleSheet(f"color: {SUCCESS}; font-size: 10px;")
        self._model_info_label.setText("UFLD + Classical fallback")
        self._clear_model_btn.setEnabled(True)
        self._log_append(f"UFLD model selected: {path}")

    def _clear_model(self) -> None:
        self._model_path = None
        self._model_path_label.setText("No UFLD weights loaded.")
        self._model_path_label.setStyleSheet(f"color: {MUTED}; font-size: 10px;")
        self._model_info_label.setText("Classical CV (always available)")
        self._model_info_label.setStyleSheet(f"color: {SUCCESS}; font-size: 11px;")
        self._clear_model_btn.setEnabled(False)
        self._log_append("UFLD model cleared — classical pipeline only.")

    # ─────────────────────────────────────────────────────────────────
    # Run
    # ─────────────────────────────────────────────────────────────────

    def _run(self) -> None:
        if not self._session_id:
            return
        if self._worker is not None and self._worker.isRunning():
            return

        self._run_btn.setEnabled(False)
        self._export_btn.setEnabled(False)
        self._log.clear()
        self._stats_table.setRowCount(0)

        self._worker = _LaneWorker(
            session_id=self._session_id,
            sm=self._sm,
            model_path=self._model_path,
            roi_top_percent=float(self._roi_slider.value()),
            camera_height_m=self._cam_height_spin.value(),
            camera_pitch_deg=self._cam_pitch_spin.value(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.status.connect(self._on_status)
        self._worker.result.connect(self._on_result)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    # ─────────────────────────────────────────────────────────────────
    # Worker callbacks
    # ─────────────────────────────────────────────────────────────────

    @pyqtSlot(int)
    def _on_progress(self, pct: int) -> None:
        self._status_label.setText(f"Processing... {pct}%")

    @pyqtSlot(str)
    def _on_status(self, msg: str) -> None:
        self._status_label.setText(msg)
        self._log_append(msg)

    @pyqtSlot(object)
    def _on_result(self, payload: Any) -> None:
        frames, report = payload
        self._lane_frames = frames
        self._report = report

        self._run_btn.setEnabled(True)

        n = len(frames)
        if n > 0:
            self._scrubber.setRange(0, n - 1)
            self._scrubber.setValue(0)
            self._on_scrub(0)

        # Enable export if overlays were written
        has_overlays = any(
            Path(f.overlay_path).exists() for f in frames if hasattr(f, "overlay_path")
        )
        self._export_btn.setEnabled(has_overlays)

        self._populate_stats(report)
        self._status_label.setText(
            f"Done — {report.get('frames_with_lanes', 0)} / {n} frames with lanes. "
            f"Unmarked road: {report.get('unmarked_road_percent', 0.0):.1f}%"
        )

    @pyqtSlot(str)
    def _on_error(self, msg: str) -> None:
        self._run_btn.setEnabled(True)
        self._log_append(f"ERROR: {msg}")
        QMessageBox.critical(self, "Lane Detection Error", msg)

    # ─────────────────────────────────────────────────────────────────
    # Frame scrubber
    # ─────────────────────────────────────────────────────────────────

    @pyqtSlot(int)
    def _on_scrub(self, idx: int) -> None:
        if not self._lane_frames or idx >= len(self._lane_frames):
            return
        self._frame_index = idx
        frame = self._lane_frames[idx]
        n = len(self._lane_frames)
        self._frame_label.setText(f"{idx + 1} / {n}")

        # Original frame
        _load_pixmap_into(self._orig_label, frame.frame_path)

        # Overlay
        _load_pixmap_into(self._overlay_label, frame.overlay_path)

        # Departure metre
        dep = frame.departure
        self._departure_meter.set_offset(dep.lateral_offset_percent, dep.status)
        offset_pct = dep.lateral_offset_percent * 100.0
        side = "left" if offset_pct < 0 else "right"
        self._departure_label.setText(
            f"Status: {dep.status}  |  "
            f"Offset: {abs(offset_pct):.1f}% {side}  |  "
            f"Confidence: {dep.confidence:.2f}"
        )

        # Method badge
        method = frame.detection_method.upper()
        if method == "UFLD":
            badge_color = ACCENT
        elif method == "CLASSICAL":
            badge_color = SUCCESS
        else:
            badge_color = MUTED
        self._method_badge.setText(method)
        self._method_badge.setStyleSheet(badge_style(badge_color))

    # ─────────────────────────────────────────────────────────────────
    # Stats table
    # ─────────────────────────────────────────────────────────────────

    def _populate_stats(self, report: dict[str, Any]) -> None:
        rows = [
            ("Frames with lanes",   str(report.get("frames_with_lanes", 0))),
            ("Frames without lanes", str(report.get("frames_without_lanes", 0))),
            ("Unmarked road %",     f"{report.get('unmarked_road_percent', 0.0):.1f}%"),
            ("Mean lateral offset", f"{report.get('mean_lateral_offset', 0.0):.3f}"),
            ("Max lateral offset",  f"{report.get('max_lateral_offset', 0.0):.3f}"),
        ]

        # Detection method breakdown
        for method, count in report.get("detection_method_distribution", {}).items():
            rows.append((f"Method: {method}", str(count)))

        # Departure distribution
        for status, count in report.get("departure_distribution", {}).items():
            rows.append((f"Departure: {status}", str(count)))

        self._stats_table.setRowCount(len(rows))
        for r, (metric, value) in enumerate(rows):
            m_item = QTableWidgetItem(metric)
            v_item = QTableWidgetItem(value)
            m_item.setFlags(m_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            v_item.setFlags(v_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._stats_table.setItem(r, 0, m_item)
            self._stats_table.setItem(r, 1, v_item)

        self._stats_table.resizeColumnsToContents()

    # ─────────────────────────────────────────────────────────────────
    # Video export
    # ─────────────────────────────────────────────────────────────────

    def _export_video(self) -> None:
        if not self._lane_frames:
            return

        overlay_paths = [
            f.overlay_path
            for f in self._lane_frames
            if hasattr(f, "overlay_path") and Path(f.overlay_path).exists()
        ]
        if not overlay_paths:
            QMessageBox.warning(
                self,
                "No Overlays",
                "No overlay images found.  Run lane detection first.",
            )
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Overlay Video",
            str(Path(overlay_paths[0]).parent.parent / "lane_overlay.mp4"),
            "MPEG-4 Video (*.mp4);;AVI Video (*.avi)",
        )
        if not out_path:
            return

        try:
            _write_video(overlay_paths, out_path)
            self._status_label.setText(f"Overlay video saved: {out_path}")
            self._log_append(f"Video exported to: {out_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Video Export Error", str(exc))

    # ─────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────

    def _log_append(self, msg: str) -> None:
        self._log.append(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_pixmap_into(label: QLabel, path: str) -> None:
    """Load an image file into a QLabel, scaled to fit while keeping aspect ratio."""
    if not path or not Path(path).exists():
        return
    pix = QPixmap(path)
    if pix.isNull():
        return
    label.setPixmap(
        pix.scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    )


def _write_video(frame_paths: list[str], out_path: str, fps: float = 10.0) -> None:
    """Write an overlay video using FFmpeg via subprocess.

    Args:
        frame_paths: Ordered list of absolute image paths.
        out_path:    Destination MP4 path.
        fps:         Playback frame rate.

    Raises:
        RuntimeError: if FFmpeg is not found or the command fails.
    """
    import shutil
    import subprocess
    import tempfile

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError(
            "FFmpeg is not installed or not in PATH. "
            "Install it and make sure 'ffmpeg' is accessible from the terminal."
        )

    # Write a file list for ffmpeg concat demuxer
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        list_path = f.name
        duration = 1.0 / fps
        for fp in frame_paths:
            f.write(f"file '{fp}'\n")
            f.write(f"duration {duration:.6f}\n")

    cmd = [
        ffmpeg_bin,
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_path,
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        out_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    Path(list_path).unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed (exit {result.returncode}).\n"
            f"stderr:\n{result.stderr[-1000:]}"
        )
