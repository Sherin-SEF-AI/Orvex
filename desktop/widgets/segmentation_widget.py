"""
desktop/widgets/segmentation_widget.py — Semantic segmentation widget.

Runs SegFormer on extracted rover frames, shows original vs overlay
side-by-side, and displays per-class statistics.
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
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
    ACCENT,
    BG,
    BORDER,
    CARD,
    HI,
    HOVER,
    MUTED,
    PANEL,
    SUCCESS,
    TEXT,
    WARNING,
)


# ---------------------------------------------------------------------------
# Background worker (defined here for locality — no circular dep with workers.py)
# ---------------------------------------------------------------------------

class _SegWorker(QThread):
    """Background worker that loads a SegFormer model and runs batch segmentation.

    Signals:
        status(str):      Human-readable status message.
        progress(int):    0-100 progress percent.
        result(object):   Tuple (list[SegmentationResult], SegmentationStats).
        error(str):       Actionable error message on failure.
    """

    status   = pyqtSignal(str)
    progress = pyqtSignal(int)
    result   = pyqtSignal(object)
    error    = pyqtSignal(str)

    def __init__(
        self,
        session_id: str,
        sm,
        model_name: str,
        batch_size: int,
        device: str,
        overlay_alpha: float,
        mark_invalid: bool,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._session_id = session_id
        self._sm = sm
        self._model_name = model_name
        self._batch_size = batch_size
        self._device = device
        self._overlay_alpha = overlay_alpha
        self._mark_invalid = mark_invalid

    def run(self) -> None:
        from core.segmentation import (
            check_segmentation_dependencies,
            compute_segmentation_statistics,
            load_segmentation_model,
            run_segmentation_batch,
        )

        try:
            # Dependency check — fail fast with actionable message
            deps = check_segmentation_dependencies()
            missing = [k for k, v in deps.items() if not v]
            if missing:
                self.error.emit(
                    f"Missing packages: {', '.join(missing)}. "
                    "Install with: pip install transformers torch opencv-python"
                )
                return

            # Locate frames in the session folder
            session_folder = self._sm.session_folder(self._session_id)
            cam0_dir = session_folder / "cam0" / "data"
            frames_dir = session_folder / "frames"

            if cam0_dir.exists():
                frame_paths = sorted(str(p) for p in cam0_dir.glob("*.jpg"))
                if not frame_paths:
                    frame_paths = sorted(str(p) for p in cam0_dir.glob("*.png"))
            elif frames_dir.exists():
                frame_paths = sorted(str(p) for p in frames_dir.glob("*.jpg"))
                if not frame_paths:
                    frame_paths = sorted(str(p) for p in frames_dir.glob("*.png"))
            else:
                # Fall back: scan the entire session folder recursively
                frame_paths = sorted(str(p) for p in session_folder.rglob("*.jpg"))

            if not frame_paths:
                self.error.emit(
                    "No extracted frames found for this session. "
                    "Run extraction first to generate frames in cam0/data/ or frames/."
                )
                return

            self.status.emit(f"Found {len(frame_paths)} frame(s). Loading model…")
            self.progress.emit(2)

            model, processor, id2label = load_segmentation_model(
                model_name=self._model_name,
                device=self._device,
            )
            self.status.emit("Model loaded. Running inference…")
            self.progress.emit(10)

            output_dir = str(session_folder / "segmentation")

            def _prog_cb(pct: int) -> None:
                # Map inference progress (0-100) onto 10-95 range
                mapped = 10 + int(pct * 0.85)
                self.progress.emit(mapped)
                self.status.emit(
                    f"Segmentation: {pct}% — processing frame batches…"
                )

            results = run_segmentation_batch(
                image_paths=frame_paths,
                model=model,
                processor=processor,
                id2label=id2label,
                batch_size=self._batch_size,
                output_dir=output_dir,
                overlay_alpha=self._overlay_alpha,
                progress_callback=_prog_cb,
            )

            self.status.emit("Computing statistics…")
            self.progress.emit(97)

            stats = compute_segmentation_statistics(results)

            self.progress.emit(100)
            self.status.emit(
                f"Done — {len(results)} frames processed, "
                f"{stats.frames_with_road} with road, "
                f"{len(stats.invalid_frames)} invalid."
            )
            self.result.emit((results, stats))

        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class SegmentationWidget(QWidget):
    """UI for semantic segmentation of extracted rover frames using SegFormer."""

    def __init__(self, session_manager, parent=None) -> None:
        super().__init__(parent)
        self._sm = session_manager
        self._session_id: str = ""
        self._worker: _SegWorker | None = None
        self._results: list = []
        self._stats = None
        self._frame_index: int = 0
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(main_splitter)

        # ── Left config panel ────────────────────────────────────────
        cfg_w = QWidget()
        cfg_w.setMinimumWidth(280)
        cfg_w.setMaximumWidth(400)
        cfg_layout = QVBoxLayout(cfg_w)
        cfg_layout.setContentsMargins(8, 8, 8, 8)
        cfg_layout.setSpacing(8)

        # Model group
        model_group = QGroupBox("Model")
        mg = QVBoxLayout(model_group)
        self._model_combo = QComboBox()
        self._model_combo.addItems([
            "SegFormer-B0 (fast)",
            "SegFormer-B2 (balanced)",
            "SegFormer-B5 (accurate)",
        ])
        self._model_combo.setCurrentIndex(1)
        mg.addWidget(self._model_combo)
        cfg_layout.addWidget(model_group)

        # Settings group
        settings_group = QGroupBox("Settings")
        sg = QVBoxLayout(settings_group)

        sg.addWidget(QLabel("Batch size:"))
        self._batch_spin = QSpinBox()
        self._batch_spin.setRange(1, 16)
        self._batch_spin.setValue(4)
        sg.addWidget(self._batch_spin)

        sg.addWidget(QLabel("Device:"))
        self._device_combo = QComboBox()
        self._device_combo.addItems(["auto", "cpu", "cuda:0"])
        sg.addWidget(self._device_combo)

        sg.addWidget(QLabel("Overlay opacity:"))
        opacity_row = QHBoxLayout()
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(50)
        self._opacity_slider.setTickInterval(10)
        self._opacity_val_label = QLabel("50%")
        self._opacity_slider.valueChanged.connect(
            lambda v: self._opacity_val_label.setText(f"{v}%")
        )
        opacity_row.addWidget(self._opacity_slider, stretch=1)
        opacity_row.addWidget(self._opacity_val_label)
        sg.addLayout(opacity_row)
        cfg_layout.addWidget(settings_group)

        # Output group
        output_group = QGroupBox("Output")
        og = QVBoxLayout(output_group)
        self._mark_invalid_cb = QCheckBox("Mark invalid frames")
        self._mark_invalid_cb.setChecked(True)
        og.addWidget(self._mark_invalid_cb)
        cfg_layout.addWidget(output_group)

        # Run button
        self._run_btn = QPushButton("Run Segmentation")
        self._run_btn.setObjectName("DangerBtn")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._run)
        cfg_layout.addWidget(self._run_btn)

        # Export buttons
        self._export_masks_btn = QPushButton("Export Masks")
        self._export_masks_btn.setEnabled(False)
        self._export_masks_btn.clicked.connect(self._export_masks)
        cfg_layout.addWidget(self._export_masks_btn)

        self._export_overlays_btn = QPushButton("Export Overlays")
        self._export_overlays_btn.setEnabled(False)
        self._export_overlays_btn.clicked.connect(self._export_overlays)
        cfg_layout.addWidget(self._export_overlays_btn)

        # Status label
        self._status_label = QLabel("Ready.")
        self._status_label.setWordWrap(True)
        cfg_layout.addWidget(self._status_label)

        cfg_layout.addStretch()
        main_splitter.addWidget(cfg_w)

        # ── Right panel ──────────────────────────────────────────────
        right_w = QWidget()
        right_layout = QVBoxLayout(right_w)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(6)

        # Side-by-side preview splitter
        preview_splitter = QSplitter(Qt.Orientation.Horizontal)

        orig_group = QGroupBox("Original Frame")
        ol = QVBoxLayout(orig_group)
        self._orig_label = QLabel("No frame loaded.")
        self._orig_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._orig_label.setMinimumSize(300, 250)
        ol.addWidget(self._orig_label)
        preview_splitter.addWidget(orig_group)

        overlay_group = QGroupBox("Segmentation Overlay")
        dl = QVBoxLayout(overlay_group)
        self._overlay_label = QLabel("No segmentation run yet.")
        self._overlay_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._overlay_label.setMinimumSize(300, 250)
        dl.addWidget(self._overlay_label)
        preview_splitter.addWidget(overlay_group)

        right_layout.addWidget(preview_splitter, stretch=3)

        # Frame scrubber
        scrubber_row = QHBoxLayout()
        scrubber_row.addWidget(QLabel("Frame:"))
        self._scrubber = QSlider(Qt.Orientation.Horizontal)
        self._scrubber.setRange(0, 0)
        self._scrubber.valueChanged.connect(self._on_scrub)
        scrubber_row.addWidget(self._scrubber, stretch=1)
        self._frame_label = QLabel("0 / 0")
        scrubber_row.addWidget(self._frame_label)
        right_layout.addLayout(scrubber_row)

        # Statistics table
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)
        self._stats_table = QTableWidget(5, 2)
        self._stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self._stats_table.horizontalHeader().setStretchLastSection(True)
        self._stats_table.verticalHeader().setVisible(False)
        self._stats_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self._stats_table.setAlternatingRowColors(True)
        self._stats_table.setMaximumHeight(160)
        _stat_rows = [
            "road_area_percent",
            "sky_area_percent",
            "valid_frames",
            "invalid_frames",
            "mean_road_coverage",
        ]
        for row, name in enumerate(_stat_rows):
            self._stats_table.setItem(row, 0, QTableWidgetItem(name))
            self._stats_table.setItem(row, 1, QTableWidgetItem("—"))
        stats_layout.addWidget(self._stats_table)
        right_layout.addWidget(stats_group)

        # Log
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(80)
        right_layout.addWidget(self._log)

        main_splitter.addWidget(right_w)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([300, 700])

    # ------------------------------------------------------------------
    # Session slot
    # ------------------------------------------------------------------

    @pyqtSlot(str)
    def on_session_changed(self, session_id: str) -> None:
        """Called by the main window when the active session changes."""
        self._session_id = session_id
        self._run_btn.setEnabled(bool(session_id))
        self._results = []
        self._stats = None
        self._export_masks_btn.setEnabled(False)
        self._export_overlays_btn.setEnabled(False)
        self._orig_label.setText("No frame loaded.")
        self._overlay_label.setText("No segmentation run yet.")
        self._frame_label.setText("0 / 0")
        self._scrubber.setRange(0, 0)
        self._reset_stats_table()
        self._status_label.setText("Ready.")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def _run(self) -> None:
        if not self._session_id:
            return
        if self._worker and self._worker.isRunning():
            return

        model_map = {
            "SegFormer-B0 (fast)":      "nvidia/segformer-b0-finetuned-cityscapes-512-1024",
            "SegFormer-B2 (balanced)":  "nvidia/segformer-b2-finetuned-cityscapes-512-1024",
            "SegFormer-B5 (accurate)":  "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        }
        model_name = model_map.get(
            self._model_combo.currentText(),
            "nvidia/segformer-b2-finetuned-cityscapes-512-1024",
        )

        self._worker = _SegWorker(
            session_id=self._session_id,
            sm=self._sm,
            model_name=model_name,
            batch_size=self._batch_spin.value(),
            device=self._device_combo.currentText(),
            overlay_alpha=self._opacity_slider.value() / 100.0,
            mark_invalid=self._mark_invalid_cb.isChecked(),
        )
        self._worker.status.connect(self._on_status)
        self._worker.progress.connect(self._on_progress)
        self._worker.result.connect(self._on_result)
        self._worker.error.connect(self._on_error)

        self._run_btn.setEnabled(False)
        self._export_masks_btn.setEnabled(False)
        self._export_overlays_btn.setEnabled(False)
        self._log.clear()
        self._worker.start()

    # ------------------------------------------------------------------
    # Worker slots
    # ------------------------------------------------------------------

    @pyqtSlot(str)
    def _on_status(self, msg: str) -> None:
        self._status_label.setText(msg)
        self._log.append(msg)

    @pyqtSlot(int)
    def _on_progress(self, pct: int) -> None:
        # Status label already updated by _on_status; nothing extra needed here.
        pass

    @pyqtSlot(object)
    def _on_result(self, payload: Any) -> None:
        results, stats = payload
        self._results = results
        self._stats = stats
        self._run_btn.setEnabled(True)

        n = len(results)
        if n > 0:
            self._scrubber.setRange(0, n - 1)
            self._scrubber.setValue(0)
            self._on_scrub(0)
            self._export_masks_btn.setEnabled(True)
            self._export_overlays_btn.setEnabled(True)

        self._populate_stats_table(stats, results)
        self._status_label.setText(
            f"Done — {n} frame(s) processed. "
            f"{stats.frames_with_road} with road, "
            f"{len(stats.invalid_frames)} invalid."
        )

    @pyqtSlot(str)
    def _on_error(self, msg: str) -> None:
        self._run_btn.setEnabled(True)
        self._log.append(f"ERROR: {msg}")
        QMessageBox.critical(self, "Segmentation Error", msg)

    # ------------------------------------------------------------------
    # Scrubber
    # ------------------------------------------------------------------

    @pyqtSlot(int)
    def _on_scrub(self, idx: int) -> None:
        if not self._results or idx >= len(self._results):
            return
        self._frame_index = idx
        r = self._results[idx]
        n = len(self._results)
        self._frame_label.setText(f"{idx + 1} / {n}")

        _load_pixmap_into(self._orig_label, r.frame_path)
        _load_pixmap_into(self._overlay_label, r.overlay_path)

        validity = "VALID" if r.is_valid_rover_frame else "INVALID"
        self._status_label.setText(
            f"Frame {idx + 1}: road={r.road_area_percent:.1f}% "
            f"sky={r.sky_area_percent:.1f}% "
            f"infer={r.inference_time_ms:.1f}ms  [{validity}]"
        )

    # ------------------------------------------------------------------
    # Stats table helpers
    # ------------------------------------------------------------------

    def _reset_stats_table(self) -> None:
        for row in range(self._stats_table.rowCount()):
            self._stats_table.setItem(row, 1, QTableWidgetItem("—"))

    def _populate_stats_table(self, stats, results: list) -> None:
        # road_area_percent — mean across all frames
        if results:
            mean_road = stats.mean_road_coverage
            mean_sky = (
                sum(r.sky_area_percent for r in results) / len(results)
            )
        else:
            mean_road = 0.0
            mean_sky = 0.0

        n = len(results)
        valid_count = n - len(stats.invalid_frames)

        row_values = [
            f"{mean_road:.2f}%",
            f"{mean_sky:.2f}%",
            str(valid_count),
            str(len(stats.invalid_frames)),
            f"{stats.mean_road_coverage:.2f}%",
        ]
        for row, val in enumerate(row_values):
            self._stats_table.setItem(row, 1, QTableWidgetItem(val))

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def _export_masks(self) -> None:
        if not self._results:
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Masks as ZIP",
            "segmentation_masks.zip",
            "ZIP files (*.zip)",
        )
        if not out_path:
            return
        try:
            with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for r in self._results:
                    p = Path(r.mask_path)
                    if p.exists():
                        zf.write(str(p), p.name)
            self._status_label.setText(f"Masks exported: {out_path}")
            self._log.append(f"Exported {len(self._results)} mask(s) to {out_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def _export_overlays(self) -> None:
        if not self._results:
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Overlays as ZIP",
            "segmentation_overlays.zip",
            "ZIP files (*.zip)",
        )
        if not out_path:
            return
        try:
            with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for r in self._results:
                    p = Path(r.overlay_path)
                    if p.exists():
                        zf.write(str(p), p.name)
            self._status_label.setText(f"Overlays exported: {out_path}")
            self._log.append(f"Exported {len(self._results)} overlay(s) to {out_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _load_pixmap_into(label: QLabel, path: str) -> None:
    """Load an image from path into a QLabel, scaled to fit while keeping aspect."""
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
