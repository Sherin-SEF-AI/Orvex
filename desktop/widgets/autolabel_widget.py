"""
desktop/widgets/autolabel_widget.py — Auto-labeling widget.

Runs YOLOv8 on extracted frames, shows live inference preview,
exports CVAT XML + YOLO format annotations.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.autolabel import ROVER_CLASSES
from desktop.theme import BG, PANEL, ACCENT, HI, TEXT, MUTED, SUCCESS, WARNING, BORDER, CARD
from desktop.workers import AutoLabelWorker

# Alias for local references
HIGHLIGHT = HI


class AutoLabelWidget(QWidget):
    """UI for YOLOv8 auto-labeling of extracted rover frames."""

    def __init__(self, session_manager, parent=None) -> None:
        super().__init__(parent)
        self._sm = session_manager
        self._session_id: str = ""
        self._worker: AutoLabelWorker | None = None
        self._annotations: list = []
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ── Left config panel ──────────────────────────────────────────
        config_widget = QWidget()
        config_widget.setMinimumWidth(280)
        config_widget.setMaximumWidth(400)
        cfg_layout = QVBoxLayout(config_widget)
        cfg_layout.setContentsMargins(8, 8, 8, 8)

        # Model selector
        model_group = QGroupBox("Model")
        mg_layout = QVBoxLayout(model_group)
        self._model_combo = QComboBox()
        self._model_combo.addItems([
            "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"
        ])
        self._model_combo.setEditable(True)
        self._model_browse_btn = QPushButton("Browse custom .pt…")
        self._model_browse_btn.clicked.connect(self._browse_model)
        mg_layout.addWidget(self._model_combo)
        mg_layout.addWidget(self._model_browse_btn)
        cfg_layout.addWidget(model_group)

        # Thresholds
        thresh_group = QGroupBox("Thresholds")
        tg_layout = QVBoxLayout(thresh_group)
        tg_layout.addWidget(QLabel("Confidence:"))
        self._conf_slider = _make_slider(10, 90, 25)
        self._conf_label = QLabel("0.25")
        self._conf_slider.valueChanged.connect(
            lambda v: self._conf_label.setText(f"{v/100:.2f}")
        )
        tg_layout.addWidget(self._conf_slider)
        tg_layout.addWidget(self._conf_label)
        tg_layout.addWidget(QLabel("IoU threshold:"))
        self._iou_slider = _make_slider(10, 90, 45)
        self._iou_label = QLabel("0.45")
        self._iou_slider.valueChanged.connect(
            lambda v: self._iou_label.setText(f"{v/100:.2f}")
        )
        tg_layout.addWidget(self._iou_slider)
        tg_layout.addWidget(self._iou_label)
        cfg_layout.addWidget(thresh_group)

        # Batch size + device
        hw_group = QGroupBox("Hardware")
        hg_layout = QVBoxLayout(hw_group)
        hg_layout.addWidget(QLabel("Batch size:"))
        self._batch_spin = QSpinBox()
        self._batch_spin.setRange(1, 64)
        self._batch_spin.setValue(16)
        hg_layout.addWidget(self._batch_spin)
        hg_layout.addWidget(QLabel("Device:"))
        self._device_combo = QComboBox()
        self._device_combo.addItems(["auto", "cpu", "cuda:0", "cuda:1"])
        hg_layout.addWidget(self._device_combo)
        cfg_layout.addWidget(hw_group)

        # Export format
        fmt_group = QGroupBox("Export format")
        fg_layout = QVBoxLayout(fmt_group)
        self._fmt_combo = QComboBox()
        self._fmt_combo.addItems(["Both (CVAT + YOLO)", "CVAT XML only", "YOLO only"])
        fg_layout.addWidget(self._fmt_combo)
        cfg_layout.addWidget(fmt_group)

        # Class filter
        cls_group = QGroupBox("Classes")
        cls_scroll = QScrollArea()
        cls_scroll.setWidgetResizable(True)
        cls_inner = QWidget()
        cls_vbox = QVBoxLayout(cls_inner)
        self._class_checks: dict[str, QCheckBox] = {}
        for cls in ROVER_CLASSES:
            cb = QCheckBox(cls)
            cb.setChecked(True)
            self._class_checks[cls] = cb
            cls_vbox.addWidget(cb)
        cls_scroll.setWidget(cls_inner)
        fg_layout2 = QVBoxLayout(cls_group)
        fg_layout2.addWidget(cls_scroll)
        cfg_layout.addWidget(cls_group)

        # Run button
        self._run_btn = QPushButton("Run Auto-Label")
        self._run_btn.setEnabled(False)
        self._run_btn.setObjectName("DangerBtn")
        self._run_btn.clicked.connect(self._run)
        cfg_layout.addWidget(self._run_btn)

        cfg_layout.addStretch()
        splitter.addWidget(config_widget)

        # ── Right panel ────────────────────────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)

        # Live preview
        preview_group = QGroupBox("Live Preview")
        pl = QVBoxLayout(preview_group)
        self._preview_label = QLabel("No inference running.")
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setMinimumHeight(300)
        self._preview_label.setScaledContents(False)
        pl.addWidget(self._preview_label)
        right_layout.addWidget(preview_group, stretch=2)

        # Frame navigation (for browsing annotated results)
        nav_row = QHBoxLayout()
        self._prev_btn = QPushButton("< Prev")
        self._prev_btn.setEnabled(False)
        self._prev_btn.clicked.connect(self._show_prev_frame)
        nav_row.addWidget(self._prev_btn)

        self._frame_counter = QLabel("0 / 0")
        self._frame_counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_row.addWidget(self._frame_counter)

        self._next_btn = QPushButton("Next >")
        self._next_btn.setEnabled(False)
        self._next_btn.clicked.connect(self._show_next_frame)
        nav_row.addWidget(self._next_btn)
        right_layout.addLayout(nav_row)

        # Progress + log
        self._status_label = QLabel("Ready.")
        right_layout.addWidget(self._status_label)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(80)
        right_layout.addWidget(self._log)

        # Stats + export buttons
        btn_row = QHBoxLayout()
        self._cvat_btn = QPushButton("Open in CVAT")
        self._cvat_btn.setEnabled(False)
        self._cvat_btn.clicked.connect(self._open_cvat)
        btn_row.addWidget(self._cvat_btn)

        self._export_btn = QPushButton("Show Output Folder")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._show_output)
        btn_row.addWidget(self._export_btn)
        right_layout.addLayout(btn_row)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([280, 700])
        self._output_dir = ""

    # ------------------------------------------------------------------
    # Session slot
    # ------------------------------------------------------------------

    @pyqtSlot(str)
    def on_session_changed(self, session_id: str) -> None:
        self._session_id = session_id
        has_session = bool(session_id)
        self._run_btn.setEnabled(has_session)
        if has_session:
            self._status_label.setText(f"Session: {session_id}")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLOv8 Weights", "", "Model weights (*.pt)"
        )
        if path:
            self._model_combo.setCurrentText(path)

    def _run(self) -> None:
        if not self._session_id:
            return
        if self._worker and self._worker.isRunning():
            return

        fmt_map = {
            "Both (CVAT + YOLO)": "both",
            "CVAT XML only": "cvat",
            "YOLO only": "yolo",
        }
        export_fmt = fmt_map.get(self._fmt_combo.currentText(), "both")
        ext_dir = str(self._sm.session_folder(self._session_id) / "autolabel")
        self._output_dir = ext_dir

        self._preview_paths: list[str] = []
        self._preview_idx: int = 0

        self._worker = AutoLabelWorker(
            session_id=self._session_id,
            sm=self._sm,
            model_path=self._model_combo.currentText(),
            conf=self._conf_slider.value() / 100.0,
            iou=self._iou_slider.value() / 100.0,
            batch_size=self._batch_spin.value(),
            export_format=export_fmt,
            output_dir=ext_dir,
        )
        self._worker.status.connect(self._on_status)
        self._worker.preview.connect(self._on_preview)
        self._worker.result.connect(self._on_result)
        self._worker.error.connect(self._on_error)
        self._run_btn.setEnabled(False)
        self._preview_label.setText("Running inference…")
        self._worker.start()

    @pyqtSlot(str)
    def _on_status(self, msg: str) -> None:
        self._status_label.setText(msg)
        self._log.append(msg)

    @pyqtSlot(object)
    def _on_preview(self, qimg) -> None:
        """Show live annotated frame preview during inference."""
        from PyQt6.QtGui import QPixmap
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            self._preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._preview_label.setPixmap(scaled)

    @pyqtSlot(object)
    def _on_result(self, payload: Any) -> None:
        self._annotations = payload["annotations"]
        stats = payload["stats"]
        self._preview_paths = payload.get("preview_paths", [])
        self._run_btn.setEnabled(True)
        self._export_btn.setEnabled(True)
        self._cvat_btn.setEnabled(True)
        total = stats.get("total_detections", 0)
        n_frames = len(self._annotations)
        self._status_label.setText(
            f"Done — {n_frames} frames, {total} detections. Output: {self._output_dir}"
        )
        # Enable frame browsing
        if self._preview_paths:
            self._preview_idx = 0
            self._prev_btn.setEnabled(True)
            self._next_btn.setEnabled(True)
            self._show_frame_at(0)

    @pyqtSlot(str)
    def _on_error(self, msg: str) -> None:
        self._run_btn.setEnabled(True)
        QMessageBox.critical(self, "Auto-Label Error", msg)

    # ------------------------------------------------------------------
    # Frame browsing (after completion)
    # ------------------------------------------------------------------

    def _show_frame_at(self, idx: int) -> None:
        """Display an annotated frame by index."""
        if not self._preview_paths or idx < 0 or idx >= len(self._preview_paths):
            return
        self._preview_idx = idx
        path = self._preview_paths[idx]
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self._preview_label.setText(f"Could not load: {path}")
            return
        scaled = pixmap.scaled(
            self._preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._preview_label.setPixmap(scaled)
        self._frame_counter.setText(f"{idx + 1} / {len(self._preview_paths)}")
        self._prev_btn.setEnabled(idx > 0)
        self._next_btn.setEnabled(idx < len(self._preview_paths) - 1)

    def _show_prev_frame(self) -> None:
        self._show_frame_at(self._preview_idx - 1)

    def _show_next_frame(self) -> None:
        self._show_frame_at(self._preview_idx + 1)

    def _open_cvat(self) -> None:
        import subprocess, sys, urllib.request
        url = "http://localhost:8080"
        # Check if CVAT is reachable before opening
        try:
            urllib.request.urlopen(url, timeout=3)
        except Exception:
            QMessageBox.warning(
                self, "CVAT Not Running",
                "CVAT is not reachable at http://localhost:8080.\n\n"
                "Start it with:\n  cd ~/cvat && docker compose up -d\n\n"
                "Login: admin / admin",
            )
            return
        if sys.platform == "linux":
            subprocess.Popen(["xdg-open", url])
        else:
            import webbrowser
            webbrowser.open(url)

    def _show_output(self) -> None:
        if self._output_dir and Path(self._output_dir).exists():
            import subprocess, sys
            if sys.platform == "linux":
                subprocess.Popen(["xdg-open", self._output_dir])

    def get_annotations(self) -> list:
        """Return current annotations for use by other widgets."""
        return self._annotations


def _make_slider(min_val: int, max_val: int, default: int) -> QSlider:
    s = QSlider(Qt.Orientation.Horizontal)
    s.setRange(min_val, max_val)
    s.setValue(default)
    return s
