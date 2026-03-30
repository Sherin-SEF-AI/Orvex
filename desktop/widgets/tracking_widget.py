"""
desktop/widgets/tracking_widget.py — Multi-object tracking widget.

Runs ByteTrack (core/tracker.py) over pre-annotated frames produced by the
Auto-Label pipeline.  Displays frame-by-frame overlays with bounding-box
trails, playback controls, aggregate statistics, and a trajectory heatmap.

Layout
------
Left config panel (260-380px)  |  Right QSplitter (vertical)
  - Prereq note                |    Top: overlay QLabel
  - Tracker Settings group     |    Playback controls + frame slider
  - Run / Export buttons       |    Bottom: QTabWidget
  - Active tracks label        |      "Statistics" tab — QTableWidget
  - Status label               |      "Heatmap" tab — heatmap QLabel + class picker
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from desktop.theme import (
    ACCENT, BG, BORDER, CARD, HI, MUTED, PANEL, SUCCESS, TEXT, WARNING,
)
from core.tracker import (
    BYTETracker,
    TrackingResult,
    build_tracker,
    compute_tracking_statistics,
    export_tracks_mot_format,
    generate_heatmap_from_tracks,
    run_tracking_session,
)
from core.models import FrameAnnotation, TrackingStats


# ---------------------------------------------------------------------------
# Background worker (inner class pattern)
# ---------------------------------------------------------------------------

class _TrackingWorker(QThread):
    """Run ByteTrack over a session's frames + annotations.

    Signals
    -------
    status(str)          — progress messages
    progress(int)        — 0-100
    result(object)       — tuple[TrackingResult, TrackingStats]
    error(str)           — actionable error message
    """

    status   = pyqtSignal(str)
    progress = pyqtSignal(int)
    result   = pyqtSignal(object)
    error    = pyqtSignal(str)

    def __init__(
        self,
        session_folder: str,
        track_thresh: float,
        track_buffer: int,
        match_thresh: float,
        class_filter: list[str],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._session_folder = Path(session_folder)
        self._track_thresh = track_thresh
        self._track_buffer = track_buffer
        self._match_thresh = match_thresh
        self._class_filter = class_filter   # empty list → no filter

    # ------------------------------------------------------------------

    def run(self) -> None:
        try:
            self._do_run()
        except Exception as exc:
            logger.exception("TrackingWorker: unhandled exception")
            self.error.emit(
                f"Tracking failed: {exc}\n"
                "Check that Auto-Label has been run on this session first."
            )

    def _do_run(self) -> None:
        session_folder = self._session_folder

        # ── 1. Find extracted frames ─────────────────────────────────
        self.status.emit("Scanning session folder for frames …")
        frame_paths = _collect_frame_paths(session_folder)
        if not frame_paths:
            self.error.emit(
                f"No extracted frames found under '{session_folder}'.\n"
                "Run extraction (Extraction tab) before tracking."
            )
            return

        # ── 2. Load annotations ──────────────────────────────────────
        self.status.emit("Loading annotations …")
        annotations = _load_annotations(session_folder, frame_paths)
        if not annotations:
            self.error.emit(
                f"No annotation JSON files found under '{session_folder}'.\n"
                "Run Auto-Label (Auto-Label tab) before tracking."
            )
            return

        # ── 3. Apply class filter ────────────────────────────────────
        active_classes = set(self._class_filter) if self._class_filter else None

        if active_classes:
            filtered: list[FrameAnnotation] = []
            for ann in annotations:
                dets = [d for d in ann.detections if d.class_name in active_classes]
                from core.models import FrameAnnotation as FA
                filtered.append(FA(
                    frame_path=ann.frame_path,
                    detections=dets,
                    inference_time_ms=ann.inference_time_ms,
                    model_version=ann.model_version,
                ))
            annotations = filtered

        # ── 4. Build tracker ─────────────────────────────────────────
        self.status.emit("Initialising BYTETracker …")
        tracker = build_tracker(
            track_thresh=self._track_thresh,
            track_buffer=self._track_buffer,
            match_thresh=self._match_thresh,
        )

        # ── 5. Run tracking ──────────────────────────────────────────
        total = len(frame_paths)
        self.status.emit(f"Tracking {total} frames …")

        def _progress_cb(current: int, total: int) -> None:
            pct = int(current / max(total, 1) * 100)
            self.progress.emit(pct)
            if current % max(total // 20, 1) == 0:
                self.status.emit(f"Tracking frame {current}/{total} …")

        tracking_result = run_tracking_session(
            frame_paths=frame_paths,
            annotations=annotations,
            tracker=tracker,
            output_dir=str(session_folder),
            progress_callback=_progress_cb,
        )

        # ── 6. Compute statistics ────────────────────────────────────
        self.status.emit("Computing statistics …")
        stats = compute_tracking_statistics(tracking_result)

        self.progress.emit(100)
        self.status.emit(
            f"Done — {tracking_result.total_tracks} unique objects tracked."
        )
        self.result.emit((tracking_result, stats))


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class TrackingWidget(QWidget):
    """PyQt6 widget for ByteTrack multi-object tracking."""

    def __init__(self, session_manager, parent=None) -> None:
        super().__init__(parent)
        self._sm = session_manager
        self._session_id: str = ""
        self._worker: _TrackingWorker | None = None

        # Results
        self._tracking_result: TrackingResult | None = None
        self._stats: TrackingStats | None = None
        self._overlay_paths: list[str] = []
        self._frame_index: int = 0

        # Playback timer
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(67)   # ~15 fps
        self._play_timer.timeout.connect(self._advance_frame)
        self._playing: bool = False

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(0)

        h_splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(h_splitter)

        # ── Left config panel ─────────────────────────────────────────
        cfg_w = QWidget()
        cfg_w.setMinimumWidth(260)
        cfg_w.setMaximumWidth(380)
        cfg_layout = QVBoxLayout(cfg_w)
        cfg_layout.setContentsMargins(8, 8, 8, 8)
        cfg_layout.setSpacing(8)

        prereq_lbl = QLabel("Run Auto-Label first.")
        prereq_lbl.setStyleSheet(
            f"color: {WARNING}; font-size: 11px; padding: 4px 2px;"
        )
        prereq_lbl.setWordWrap(True)
        cfg_layout.addWidget(prereq_lbl)

        # Tracker settings group
        trk_group = QGroupBox("Tracker Settings")
        tg = QVBoxLayout(trk_group)
        tg.setSpacing(6)

        tg.addWidget(QLabel("Track threshold:"))
        self._thresh_spin = QDoubleSpinBox()
        self._thresh_spin.setRange(0.1, 0.9)
        self._thresh_spin.setSingleStep(0.05)
        self._thresh_spin.setValue(0.5)
        self._thresh_spin.setDecimals(2)
        self._thresh_spin.setToolTip(
            "Minimum detection confidence to initiate/maintain a track."
        )
        tg.addWidget(self._thresh_spin)

        tg.addWidget(QLabel("Track buffer (frames):"))
        self._buffer_spin = QSpinBox()
        self._buffer_spin.setRange(5, 60)
        self._buffer_spin.setValue(30)
        self._buffer_spin.setToolTip(
            "Number of frames a track is kept alive without a matching detection."
        )
        tg.addWidget(self._buffer_spin)

        tg.addWidget(QLabel("Match threshold (IoU):"))
        self._match_spin = QDoubleSpinBox()
        self._match_spin.setRange(0.5, 1.0)
        self._match_spin.setSingleStep(0.05)
        self._match_spin.setValue(0.8)
        self._match_spin.setDecimals(2)
        self._match_spin.setToolTip(
            "Minimum IoU required to assign a detection to an existing track."
        )
        tg.addWidget(self._match_spin)

        tg.addWidget(QLabel("Class filter (comma-separated):"))
        self._class_filter_edit = QLineEdit()
        self._class_filter_edit.setText("car, truck, person, motorcycle, bicycle")
        self._class_filter_edit.setPlaceholderText("Leave blank for all classes")
        self._class_filter_edit.setToolTip(
            "Only track objects of the listed classes.  Leave blank to track all."
        )
        tg.addWidget(self._class_filter_edit)

        cfg_layout.addWidget(trk_group)

        # Run button
        self._run_btn = QPushButton("Run Tracking")
        self._run_btn.setObjectName("DangerBtn")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._run)
        cfg_layout.addWidget(self._run_btn)

        # Export buttons
        self._export_video_btn = QPushButton("Export Tracking Video")
        self._export_video_btn.setEnabled(False)
        self._export_video_btn.clicked.connect(self._export_video)
        cfg_layout.addWidget(self._export_video_btn)

        self._export_mot_btn = QPushButton("Export MOT CSV")
        self._export_mot_btn.setEnabled(False)
        self._export_mot_btn.clicked.connect(self._export_mot)
        cfg_layout.addWidget(self._export_mot_btn)

        cfg_layout.addSpacing(8)

        self._active_tracks_lbl = QLabel("Active: 0")
        self._active_tracks_lbl.setStyleSheet(
            f"color: {SUCCESS}; font-size: 12px; font-weight: bold;"
        )
        cfg_layout.addWidget(self._active_tracks_lbl)

        self._status_lbl = QLabel("Ready.")
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        cfg_layout.addWidget(self._status_lbl)

        cfg_layout.addStretch()
        h_splitter.addWidget(cfg_w)

        # ── Right panel ───────────────────────────────────────────────
        right_w = QWidget()
        right_layout = QVBoxLayout(right_w)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(4)

        v_splitter = QSplitter(Qt.Orientation.Vertical)
        right_layout.addWidget(v_splitter)

        # Top: overlay image
        overlay_container = QWidget()
        oc_layout = QVBoxLayout(overlay_container)
        oc_layout.setContentsMargins(0, 0, 0, 0)

        self._overlay_lbl = QLabel("No tracking results.\nRun Auto-Label first, then click Run Tracking.")
        self._overlay_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._overlay_lbl.setMinimumSize(400, 300)
        self._overlay_lbl.setStyleSheet(
            f"background: {CARD}; border: 1px solid {BORDER}; "
            f"border-radius: 5px; color: {MUTED};"
        )
        oc_layout.addWidget(self._overlay_lbl)

        # Playback controls
        pb_row = QHBoxLayout()
        pb_row.setSpacing(4)

        self._step_back_btn = QPushButton("\u25c0")   # ◀
        self._step_back_btn.setFixedWidth(32)
        self._step_back_btn.setEnabled(False)
        self._step_back_btn.clicked.connect(self._step_back)
        pb_row.addWidget(self._step_back_btn)

        self._play_btn = QPushButton("\u25b6")   # ▶
        self._play_btn.setFixedWidth(40)
        self._play_btn.setEnabled(False)
        self._play_btn.clicked.connect(self._toggle_play)
        pb_row.addWidget(self._play_btn)

        self._step_fwd_btn = QPushButton("\u25b6\u25b6")   # ▶▶ (step forward)
        self._step_fwd_btn.setFixedWidth(32)
        self._step_fwd_btn.setEnabled(False)
        self._step_fwd_btn.clicked.connect(self._step_forward)
        pb_row.addWidget(self._step_fwd_btn)

        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        self._frame_slider.setRange(0, 0)
        self._frame_slider.setEnabled(False)
        self._frame_slider.valueChanged.connect(self._on_scrub)
        pb_row.addWidget(self._frame_slider, stretch=1)

        self._frame_lbl = QLabel("0 / 0")
        self._frame_lbl.setFixedWidth(70)
        self._frame_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        pb_row.addWidget(self._frame_lbl)

        oc_layout.addLayout(pb_row)
        v_splitter.addWidget(overlay_container)

        # Bottom: tab widget
        self._tab_widget = QTabWidget()

        # Statistics tab
        stats_w = QWidget()
        stats_layout = QVBoxLayout(stats_w)
        stats_layout.setContentsMargins(4, 4, 4, 4)

        self._stats_table = QTableWidget(0, 2)
        self._stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self._stats_table.horizontalHeader().setStretchLastSection(True)
        self._stats_table.verticalHeader().setVisible(False)
        self._stats_table.setAlternatingRowColors(True)
        self._stats_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        stats_layout.addWidget(self._stats_table)

        self._tab_widget.addTab(stats_w, "Statistics")

        # Heatmap tab
        heatmap_w = QWidget()
        hm_layout = QVBoxLayout(heatmap_w)
        hm_layout.setContentsMargins(4, 4, 4, 4)
        hm_layout.setSpacing(4)

        hm_ctrl_row = QHBoxLayout()
        hm_ctrl_row.addWidget(QLabel("Class:"))
        self._heatmap_class_combo = QComboBox()
        self._heatmap_class_combo.addItem("All classes")
        self._heatmap_class_combo.currentTextChanged.connect(self._refresh_heatmap)
        hm_ctrl_row.addWidget(self._heatmap_class_combo)
        hm_ctrl_row.addStretch()
        hm_layout.addLayout(hm_ctrl_row)

        self._heatmap_lbl = QLabel("Heatmap will appear after tracking completes.")
        self._heatmap_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._heatmap_lbl.setMinimumSize(300, 200)
        self._heatmap_lbl.setStyleSheet(
            f"background: {CARD}; border: 1px solid {BORDER}; border-radius: 4px;"
            f" color: {MUTED};"
        )
        hm_layout.addWidget(self._heatmap_lbl, stretch=1)

        self._tab_widget.addTab(heatmap_w, "Heatmap")

        v_splitter.addWidget(self._tab_widget)
        v_splitter.setStretchFactor(0, 3)
        v_splitter.setStretchFactor(1, 1)
        v_splitter.setSizes([450, 200])

        h_splitter.addWidget(right_w)
        h_splitter.setStretchFactor(0, 0)
        h_splitter.setStretchFactor(1, 1)
        h_splitter.setSizes([280, 760])

    # ------------------------------------------------------------------
    # Session change
    # ------------------------------------------------------------------

    @pyqtSlot(str)
    def on_session_changed(self, session_id: str) -> None:
        """Called by the main window whenever the active session changes."""
        self._session_id = session_id
        self._tracking_result = None
        self._stats = None
        self._overlay_paths = []

        if session_id:
            # Check whether annotations already exist
            has_ann = self._has_annotations(session_id)
            self._run_btn.setEnabled(has_ann)
            if not has_ann:
                self._status_lbl.setText(
                    "No annotations found.  Run Auto-Label first."
                )
            else:
                self._status_lbl.setText("Ready — annotations found.")
                # Check for existing tracking overlays
                existing = self._find_existing_overlays(session_id)
                if existing:
                    self._overlay_paths = existing
                    self._setup_playback(existing)
                    self._status_lbl.setText(
                        f"Loaded {len(existing)} existing overlays."
                    )
        else:
            self._run_btn.setEnabled(False)
            self._status_lbl.setText("No session selected.")

        self._export_video_btn.setEnabled(False)
        self._export_mot_btn.setEnabled(False)
        self._active_tracks_lbl.setText("Active: 0")
        _reset_label(self._overlay_lbl, "No tracking results.")
        self._stats_table.setRowCount(0)

    # ------------------------------------------------------------------
    # Run tracking
    # ------------------------------------------------------------------

    def _run(self) -> None:
        if not self._session_id:
            return
        if self._worker and self._worker.isRunning():
            return

        # Stop any ongoing playback
        self._stop_play()

        class_filter = _parse_class_filter(self._class_filter_edit.text())

        session_folder = str(self._sm.session_folder(self._session_id))

        self._worker = _TrackingWorker(
            session_folder=session_folder,
            track_thresh=self._thresh_spin.value(),
            track_buffer=self._buffer_spin.value(),
            match_thresh=self._match_spin.value(),
            class_filter=class_filter,
        )
        self._worker.status.connect(self._on_status)
        self._worker.progress.connect(self._on_progress)
        self._worker.result.connect(self._on_result)
        self._worker.error.connect(self._on_error)

        self._run_btn.setEnabled(False)
        self._export_video_btn.setEnabled(False)
        self._export_mot_btn.setEnabled(False)
        self._status_lbl.setText("Starting …")
        self._worker.start()

    # ------------------------------------------------------------------
    # Worker callbacks
    # ------------------------------------------------------------------

    @pyqtSlot(str)
    def _on_status(self, msg: str) -> None:
        self._status_lbl.setText(msg)

    @pyqtSlot(int)
    def _on_progress(self, pct: int) -> None:
        self._status_lbl.setText(f"Progress: {pct}%")

    @pyqtSlot(object)
    def _on_result(self, payload: Any) -> None:
        tracking_result, stats = payload
        self._tracking_result = tracking_result
        self._stats = stats

        self._run_btn.setEnabled(True)
        self._export_mot_btn.setEnabled(True)

        # Populate stats table
        self._populate_stats_table(stats)

        # Populate heatmap class selector
        classes = sorted(tracking_result.class_track_counts.keys())
        self._heatmap_class_combo.blockSignals(True)
        self._heatmap_class_combo.clear()
        self._heatmap_class_combo.addItem("All classes")
        for cls in classes:
            self._heatmap_class_combo.addItem(cls)
        self._heatmap_class_combo.blockSignals(False)

        # Find overlay images written by run_tracking_session
        session_folder = self._sm.session_folder(self._session_id)
        overlays = sorted(
            (session_folder / "tracking").glob("*_track.jpg"),
            key=lambda p: p.name,
        )
        self._overlay_paths = [str(p) for p in overlays]

        if self._overlay_paths:
            self._setup_playback(self._overlay_paths)
            self._export_video_btn.setEnabled(True)

        # Generate heatmap
        self._refresh_heatmap()

        self._status_lbl.setText(
            f"Done — {tracking_result.total_tracks} unique objects tracked "
            f"across {len(self._overlay_paths)} frames."
        )

    @pyqtSlot(str)
    def _on_error(self, msg: str) -> None:
        self._run_btn.setEnabled(True)
        QMessageBox.critical(self, "Tracking Error", msg)
        self._status_lbl.setText("Error — see dialog.")

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def _setup_playback(self, overlay_paths: list[str]) -> None:
        n = len(overlay_paths)
        self._frame_slider.setEnabled(True)
        self._frame_slider.setRange(0, max(n - 1, 0))
        self._frame_slider.setValue(0)
        self._play_btn.setEnabled(n > 1)
        self._step_back_btn.setEnabled(True)
        self._step_fwd_btn.setEnabled(True)
        self._frame_lbl.setText(f"1 / {n}")
        self._show_frame(0)

    @pyqtSlot(int)
    def _on_scrub(self, idx: int) -> None:
        self._show_frame(idx)

    def _show_frame(self, idx: int) -> None:
        if not self._overlay_paths:
            return
        idx = max(0, min(idx, len(self._overlay_paths) - 1))
        self._frame_index = idx

        path = self._overlay_paths[idx]
        _load_pixmap_into(self._overlay_lbl, path)

        n = len(self._overlay_paths)
        self._frame_lbl.setText(f"{idx + 1} / {n}")

        # Update "Active tracks" label from result if available
        if self._tracking_result:
            atpf = self._tracking_result.active_tracks_per_frame
            if idx < len(atpf):
                self._active_tracks_lbl.setText(f"Active: {atpf[idx]}")

    def _toggle_play(self) -> None:
        if self._playing:
            self._stop_play()
        else:
            self._start_play()

    def _start_play(self) -> None:
        self._playing = True
        self._play_btn.setText("\u23f8")   # ⏸
        self._play_timer.start()

    def _stop_play(self) -> None:
        self._playing = False
        self._play_btn.setText("\u25b6")   # ▶
        self._play_timer.stop()

    def _advance_frame(self) -> None:
        if not self._overlay_paths:
            self._stop_play()
            return
        next_idx = self._frame_index + 1
        if next_idx >= len(self._overlay_paths):
            self._stop_play()
            next_idx = len(self._overlay_paths) - 1
        self._frame_slider.setValue(next_idx)

    def _step_back(self) -> None:
        self._stop_play()
        self._frame_slider.setValue(max(0, self._frame_index - 1))

    def _step_forward(self) -> None:
        self._stop_play()
        n = len(self._overlay_paths)
        self._frame_slider.setValue(min(n - 1, self._frame_index + 1))

    # ------------------------------------------------------------------
    # Heatmap
    # ------------------------------------------------------------------

    def _refresh_heatmap(self, _text: str = "") -> None:
        if not self._tracking_result:
            return

        sel = self._heatmap_class_combo.currentText()
        class_filter = None if sel == "All classes" else [sel]

        # Determine image size from first overlay
        img_size = (1920, 1080)  # fallback
        if self._overlay_paths:
            from pathlib import Path as _P
            import cv2
            try:
                first_img = cv2.imread(self._overlay_paths[0])
                if first_img is not None:
                    h_img, w_img = first_img.shape[:2]
                    img_size = (w_img, h_img)
            except Exception:
                pass

        try:
            rgb = generate_heatmap_from_tracks(
                self._tracking_result.track_histories,
                image_size=img_size,
                class_filter=class_filter,
            )
        except Exception as exc:
            logger.warning("Heatmap generation failed: {}", exc)
            return

        h_hm, w_hm = rgb.shape[:2]
        qimg = QImage(
            rgb.tobytes(),
            w_hm, h_hm,
            w_hm * 3,
            QImage.Format.Format_RGB888,
        )
        pix = QPixmap.fromImage(qimg)
        self._heatmap_lbl.setPixmap(
            pix.scaled(
                self._heatmap_lbl.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    # ------------------------------------------------------------------
    # Statistics table
    # ------------------------------------------------------------------

    def _populate_stats_table(self, stats: TrackingStats) -> None:
        rows = [
            ("Unique objects tracked", str(stats.total_unique_objects)),
            ("Mean track duration (s)", f"{stats.mean_track_duration_seconds:.2f}"),
            ("Max simultaneous tracks", str(stats.max_simultaneous_tracks)),
            ("Mean objects per frame", f"{stats.mean_objects_per_frame:.2f}"),
            ("Static objects (%)", f"{stats.static_objects_percent:.1f}%"),
            ("Track fragmentation rate", f"{stats.track_fragmentation_rate:.3f}"),
            ("High-density frames", str(len(stats.high_density_frames))),
        ]

        for class_name, count in sorted(stats.objects_per_class.items()):
            rows.append((f"  {class_name}", str(count)))

        self._stats_table.setRowCount(len(rows))
        for row_idx, (metric, value) in enumerate(rows):
            metric_item = QTableWidgetItem(metric)
            metric_item.setFlags(metric_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            value_item = QTableWidgetItem(value)
            value_item.setFlags(value_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            value_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self._stats_table.setItem(row_idx, 0, metric_item)
            self._stats_table.setItem(row_idx, 1, value_item)

        self._stats_table.resizeColumnsToContents()

    # ------------------------------------------------------------------
    # Export actions
    # ------------------------------------------------------------------

    def _export_video(self) -> None:
        if not self._overlay_paths:
            QMessageBox.warning(self, "No Overlays", "No tracking overlays to export.")
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save Tracking Video", "", "MP4 Video (*.mp4)"
        )
        if not out_path:
            return

        try:
            _write_video_from_frames(self._overlay_paths, out_path)
            self._status_lbl.setText(f"Video saved: {out_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def _export_mot(self) -> None:
        if not self._tracking_result:
            QMessageBox.warning(self, "No Results", "Run tracking first.")
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save MOT CSV", "", "CSV Files (*.csv)"
        )
        if not out_path:
            return

        try:
            written = export_tracks_mot_format(self._tracking_result, out_path)
            self._status_lbl.setText(f"MOT CSV saved: {written}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _has_annotations(self, session_id: str) -> bool:
        """Return True if any annotation JSON files exist for this session."""
        folder = self._sm.session_folder(session_id)
        return _has_annotation_files(folder)

    def _find_existing_overlays(self, session_id: str) -> list[str]:
        folder = self._sm.session_folder(session_id)
        tracking_dir = folder / "tracking"
        if not tracking_dir.exists():
            return []
        paths = sorted(tracking_dir.glob("*_track.jpg"), key=lambda p: p.name)
        return [str(p) for p in paths]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _parse_class_filter(text: str) -> list[str]:
    """Parse a comma-separated class filter string into a list of stripped names.
    Returns an empty list if the input is blank (means: no filter).
    """
    if not text.strip():
        return []
    return [s.strip() for s in text.split(",") if s.strip()]


def _collect_frame_paths(session_folder: Path) -> list[str]:
    """Find all extracted frame images under the session folder.

    Searches common extraction output patterns:
      <session>/extraction*/cam0/data/*.jpg
      <session>/extraction*/**/*.jpg
      <session>/frames/**/*.jpg
    """
    candidates: list[Path] = []

    for pattern in (
        "extraction*/cam0/data/*.jpg",
        "extraction*/cam0/data/*.png",
        "extraction*/**/*.jpg",
        "frames/**/*.jpg",
        "frames/**/*.png",
    ):
        candidates.extend(session_folder.glob(pattern))

    seen: set[str] = set()
    unique: list[Path] = []
    for p in candidates:
        s = str(p)
        if s not in seen:
            seen.add(s)
            unique.append(p)

    return sorted(str(p) for p in unique)


def _load_annotations(
    session_folder: Path,
    frame_paths: list[str],
) -> list[FrameAnnotation]:
    """Load FrameAnnotation objects from JSON files in the session folder.

    Looks for:
      <session>/labels/**/*.json   (one JSON per frame, YOLO-style)
      <session>/annotations.json   (single bulk file)
    """
    annotations: list[FrameAnnotation] = []

    # Bulk file
    bulk = session_folder / "annotations.json"
    if bulk.exists():
        try:
            with open(bulk, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    annotations.append(FrameAnnotation.model_validate(item))
            logger.debug("Loaded {} annotations from bulk file", len(annotations))
            return annotations
        except Exception as exc:
            logger.warning("Failed to load bulk annotations.json: {}", exc)

    # Per-frame JSON files
    json_files = list(session_folder.glob("labels/**/*.json"))
    json_files += list(session_folder.glob("autolabel/**/*.json"))

    stem_to_frame: dict[str, str] = {Path(fp).stem: fp for fp in frame_paths}

    for jf in json_files:
        try:
            with open(jf, encoding="utf-8") as f:
                item = json.load(f)
            ann = FrameAnnotation.model_validate(item)
            annotations.append(ann)
        except Exception as exc:
            logger.warning("Skipping annotation file '{}': {}", jf, exc)

    logger.debug(
        "Loaded {} per-frame annotation files from '{}'",
        len(annotations), session_folder,
    )
    return annotations


def _has_annotation_files(session_folder: Path) -> bool:
    """Return True if any annotation data is present."""
    if (session_folder / "annotations.json").exists():
        return True
    if any(session_folder.glob("labels/**/*.json")):
        return True
    if any(session_folder.glob("autolabel/**/*.json")):
        return True
    return False


def _load_pixmap_into(label: QLabel, path: str) -> None:
    """Load image at *path* into *label*, scaled to fit while keeping aspect."""
    p = Path(path)
    if not p.exists():
        return
    pix = QPixmap(str(p))
    if pix.isNull():
        return
    label.setPixmap(
        pix.scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    )


def _reset_label(label: QLabel, text: str) -> None:
    label.clear()
    label.setText(text)


def _write_video_from_frames(frame_paths: list[str], out_path: str) -> None:
    """Assemble frame images into an MP4 video using FFmpeg subprocess.

    Raises:
        RuntimeError: if FFmpeg is not available or encoding fails.
    """
    import subprocess
    import shutil
    import tempfile

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "FFmpeg not found on PATH.  Install FFmpeg and ensure it is accessible."
        )

    # Write a temporary file list
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tf:
        list_path = tf.name
        for fp in sorted(frame_paths):
            tf.write(f"file '{fp}'\n")
            tf.write("duration 0.0667\n")   # ~15 fps

    try:
        cmd = [
            ffmpeg, "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-vf", "fps=15",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            out_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg encoding failed (exit {result.returncode}):\n{result.stderr}"
            )
    finally:
        import os
        try:
            os.unlink(list_path)
        except OSError:
            pass
