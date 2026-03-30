"""
desktop/widgets/depth_widget.py — Depth estimation widget.

Runs Depth-Anything-v2 on extracted frames, shows side-by-side
original vs colorized depth, and allows generating a depth video.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from desktop.theme import BG, PANEL, ACCENT, HI, TEXT, MUTED, SUCCESS, WARNING, BORDER, HOVER, CARD
from desktop.workers import DepthWorker

HIGHLIGHT = HI


class DepthWidget(QWidget):
    """UI for monocular depth estimation using Depth-Anything-v2."""

    def __init__(self, session_manager, parent=None) -> None:
        super().__init__(parent)
        self._sm = session_manager
        self._session_id: str = ""
        self._worker: DepthWorker | None = None
        self._depth_results: list = []
        self._frame_index: int = 0
        self._build_ui()

    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ── Left config ──────────────────────────────────────────────
        cfg_w = QWidget()
        cfg_w.setMinimumWidth(260)
        cfg_w.setMaximumWidth(380)
        cfg_layout = QVBoxLayout(cfg_w)
        cfg_layout.setContentsMargins(8, 8, 8, 8)

        model_group = QGroupBox("Model")
        mg = QVBoxLayout(model_group)
        self._model_combo = QComboBox()
        self._model_combo.addItems([
            "small (fast)", "base (balanced)", "large (accurate)"
        ])
        mg.addWidget(self._model_combo)
        cfg_layout.addWidget(model_group)

        hw_group = QGroupBox("Hardware")
        hg = QVBoxLayout(hw_group)
        hg.addWidget(QLabel("Batch size:"))
        self._batch_spin = QSpinBox()
        self._batch_spin.setRange(1, 32)
        self._batch_spin.setValue(8)
        hg.addWidget(self._batch_spin)
        hg.addWidget(QLabel("Device:"))
        self._device_combo = QComboBox()
        self._device_combo.addItems(["auto", "cpu", "cuda:0"])
        hg.addWidget(self._device_combo)
        cfg_layout.addWidget(hw_group)

        out_group = QGroupBox("Output")
        og = QVBoxLayout(out_group)
        self._colorize_cb = QCheckBox("Save colorized depth")
        self._colorize_cb.setChecked(True)
        self._metric_cb = QCheckBox("Estimate metric scale (GPS)")
        og.addWidget(self._colorize_cb)
        og.addWidget(self._metric_cb)
        cfg_layout.addWidget(out_group)

        self._run_btn = QPushButton("Estimate Depth")
        self._run_btn.setEnabled(False)
        self._run_btn.setObjectName("DangerBtn")
        self._run_btn.clicked.connect(self._run)
        cfg_layout.addWidget(self._run_btn)

        self._video_btn = QPushButton("Generate Depth Video")
        self._video_btn.setEnabled(False)
        self._video_btn.clicked.connect(self._generate_video)
        cfg_layout.addWidget(self._video_btn)

        cfg_layout.addStretch()
        splitter.addWidget(cfg_w)

        # ── Right panel ──────────────────────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 8, 8, 8)

        # Side-by-side preview
        preview_row = QHBoxLayout()
        orig_group = QGroupBox("Original Frame")
        ol = QVBoxLayout(orig_group)
        self._orig_label = QLabel("No frame loaded.")
        self._orig_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._orig_label.setMinimumSize(300, 250)
        ol.addWidget(self._orig_label)
        preview_row.addWidget(orig_group)

        depth_group = QGroupBox("Depth Map (Relative)")
        dl = QVBoxLayout(depth_group)
        self._depth_label = QLabel("No depth estimated.")
        self._depth_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._depth_label.setMinimumSize(300, 250)
        dl.addWidget(self._depth_label)
        preview_row.addWidget(depth_group)
        rl.addLayout(preview_row, stretch=3)

        # Scrubber
        scrubber_row = QHBoxLayout()
        scrubber_row.addWidget(QLabel("Frame:"))
        self._scrubber = QSlider(Qt.Orientation.Horizontal)
        self._scrubber.setRange(0, 0)
        self._scrubber.valueChanged.connect(self._on_scrub)
        scrubber_row.addWidget(self._scrubber, stretch=1)
        self._frame_label = QLabel("0 / 0")
        scrubber_row.addWidget(self._frame_label)
        rl.addLayout(scrubber_row)

        # Status + log
        self._status_label = QLabel("Ready.")
        rl.addWidget(self._status_label)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(80)
        rl.addWidget(self._log)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([260, 700])

    # ------------------------------------------------------------------

    @pyqtSlot(str)
    def on_session_changed(self, session_id: str) -> None:
        self._session_id = session_id
        self._run_btn.setEnabled(bool(session_id))

    def _run(self) -> None:
        if not self._session_id or (self._worker and self._worker.isRunning()):
            return
        model_map = {"small (fast)": "small", "base (balanced)": "base", "large (accurate)": "large"}
        model_name = model_map.get(self._model_combo.currentText(), "small")
        self._worker = DepthWorker(
            session_id=self._session_id,
            sm=self._sm,
            model_name=model_name,
            batch_size=self._batch_spin.value(),
            colorize=self._colorize_cb.isChecked(),
            metric_scale=self._metric_cb.isChecked(),
        )
        self._worker.status.connect(self._on_status)
        self._worker.result.connect(self._on_result)
        self._worker.error.connect(self._on_error)
        self._run_btn.setEnabled(False)
        self._worker.start()

    @pyqtSlot(str)
    def _on_status(self, msg: str) -> None:
        self._status_label.setText(msg)
        self._log.append(msg)

    @pyqtSlot(object)
    def _on_result(self, results: Any) -> None:
        self._depth_results = results
        self._run_btn.setEnabled(True)
        n = len(results)
        if n > 0:
            self._scrubber.setRange(0, n - 1)
            self._scrubber.setValue(0)
            self._on_scrub(0)
        self._video_btn.setEnabled(
            any(r.depth_color_path for r in results)
        )
        self._status_label.setText(f"Done — {n} depth maps generated.")

    @pyqtSlot(str)
    def _on_error(self, msg: str) -> None:
        self._run_btn.setEnabled(True)
        QMessageBox.critical(self, "Depth Estimation Error", msg)

    @pyqtSlot(int)
    def _on_scrub(self, idx: int) -> None:
        if not self._depth_results or idx >= len(self._depth_results):
            return
        self._frame_index = idx
        result = self._depth_results[idx]
        n = len(self._depth_results)
        self._frame_label.setText(f"{idx + 1} / {n}")

        # Original frame
        _load_pixmap_into(self._orig_label, result.frame_path)
        # Depth colorized (or raw)
        depth_path = result.depth_color_path or result.depth_raw_path
        if depth_path:
            _load_pixmap_into(self._depth_label, depth_path)
        self._status_label.setText(
            f"Frame {idx + 1}: min={result.min_depth:.3f} "
            f"max={result.max_depth:.3f} mean={result.mean_depth:.3f} "
            f"[Relative depth — NOT metric]"
        )

    def _generate_video(self) -> None:
        if not self._depth_results:
            return
        from core.depth_estimator import generate_depth_video
        color_paths = [r.depth_color_path for r in self._depth_results if r.depth_color_path]
        if not color_paths:
            QMessageBox.warning(self, "No colorized frames", "Run with colorize=True first.")
            return
        out_dir = Path(color_paths[0]).parent.parent
        out_path = str(out_dir / "depth_video.mp4")
        try:
            generate_depth_video(color_paths, out_path)
            self._status_label.setText(f"Depth video saved: {out_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Video Error", str(exc))


def _load_pixmap_into(label: QLabel, path: str) -> None:
    """Load an image from path into a QLabel, scaled to fit."""
    if not Path(path).exists():
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
