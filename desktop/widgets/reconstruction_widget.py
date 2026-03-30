"""
desktop/widgets/reconstruction_widget.py — 3D reconstruction widget.

Runs COLMAP on extracted frames, shows colored point cloud in 3D,
exports PLY file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from desktop.theme import BG, PANEL, ACCENT, HI, TEXT, MUTED, SUCCESS, WARNING, BORDER, HOVER, CARD
from desktop.workers import ReconstructWorker

HIGHLIGHT = HI


class ReconstructionWidget(QWidget):
    """UI for COLMAP SfM 3D reconstruction."""

    def __init__(self, session_manager, parent=None) -> None:
        super().__init__(parent)
        self._sm = session_manager
        self._session_id: str = ""
        self._worker: ReconstructWorker | None = None
        self._result = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ── Left config ────────────────────────────────────────────────
        cfg_w = QWidget()
        cfg_w.setMinimumWidth(260)
        cfg_w.setMaximumWidth(380)
        cfg_layout = QVBoxLayout(cfg_w)
        cfg_layout.setContentsMargins(8, 8, 8, 8)

        # Installation notice
        self._install_label = QLabel("")
        self._install_label.setWordWrap(True)
        self._install_label.setStyleSheet(
            f"color:{WARNING}; font-size:11px; padding:4px; "
            f"background:#2a1a00; border:1px solid {WARNING}; border-radius:3px;"
        )
        self._install_label.hide()
        cfg_layout.addWidget(self._install_label)
        self._check_colmap()

        sample_group = QGroupBox("Frame sampling")
        sg = QVBoxLayout(sample_group)
        sg.addWidget(QLabel("Use every N-th frame:"))
        self._nth_spin = QSpinBox()
        self._nth_spin.setRange(1, 30)
        self._nth_spin.setValue(6)
        self._nth_spin.setToolTip("Default 6 (1 in 6 frames) reduces redundancy in video sequences.")
        sg.addWidget(self._nth_spin)
        cfg_layout.addWidget(sample_group)

        hw_group = QGroupBox("Reconstruction settings")
        hg = QVBoxLayout(hw_group)
        self._gpu_cb = QCheckBox("Use GPU for feature extraction")
        self._gpu_cb.setChecked(True)
        hg.addWidget(self._gpu_cb)
        hg.addWidget(QLabel("Camera model:"))
        self._camera_combo = QComboBox()
        self._camera_combo.addItems([
            "OPENCV_FISHEYE", "OPENCV", "PINHOLE", "RADIAL"
        ])
        hg.addWidget(self._camera_combo)
        cfg_layout.addWidget(hw_group)

        # Matcher info label
        self._matcher_label = QLabel("Matcher: sequential (video sequence)")
        self._matcher_label.setWordWrap(True)
        cfg_layout.addWidget(self._matcher_label)
        self._nth_spin.valueChanged.connect(self._update_matcher_label)

        self._run_btn = QPushButton("Run 3D Reconstruction")
        self._run_btn.setEnabled(False)
        self._run_btn.setObjectName("DangerBtn")
        self._run_btn.clicked.connect(self._run)
        cfg_layout.addWidget(self._run_btn)

        self._ply_btn = QPushButton("Export PLY")
        self._ply_btn.setEnabled(False)
        self._ply_btn.clicked.connect(self._export_ply)
        cfg_layout.addWidget(self._ply_btn)

        self._meshlab_btn = QPushButton("Open in MeshLab")
        self._meshlab_btn.setEnabled(False)
        self._meshlab_btn.clicked.connect(self._open_meshlab)
        cfg_layout.addWidget(self._meshlab_btn)

        cfg_layout.addStretch()
        splitter.addWidget(cfg_w)

        # ── Right panel ────────────────────────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 8, 8, 8)

        self._cloud_widget = _PointCloudPlaceholder()
        rl.addWidget(self._cloud_widget, stretch=3)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(120)
        rl.addWidget(self._log)

        self._coverage_warn = QLabel("")
        rl.addWidget(self._coverage_warn)

        self._stats_table = QTableWidget(0, 2)
        self._stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self._stats_table.horizontalHeader().setStretchLastSection(True)
        self._stats_table.setMaximumHeight(150)
        rl.addWidget(self._stats_table)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([260, 740])

    def _check_colmap(self) -> None:
        from core.reconstructor import check_colmap_installation
        if not check_colmap_installation():
            self._install_label.setText(
                "⚠ COLMAP not found on PATH.\n"
                "Install: https://colmap.github.io/install.html\n"
                "  Ubuntu: sudo apt install colmap\n"
                "  macOS:  brew install colmap"
            )
            self._install_label.show()
            self._colmap_ok = False
        else:
            self._install_label.hide()
            self._colmap_ok = True

    @pyqtSlot(str)
    def on_session_changed(self, session_id: str) -> None:
        self._session_id = session_id
        self._run_btn.setEnabled(bool(session_id) and getattr(self, "_colmap_ok", True))

    def _update_matcher_label(self, n: int) -> None:
        if n < 100:
            self._matcher_label.setText(
                f"Matcher: exhaustive (< 100 frames when using 1-in-{n} sampling)"
            )
        else:
            self._matcher_label.setText("Matcher: sequential (video sequence)")

    def _run(self) -> None:
        if not self._session_id or (self._worker and self._worker.isRunning()):
            return
        self._log.clear()
        self._coverage_warn.setText("")
        self._worker = ReconstructWorker(
            session_id=self._session_id,
            sm=self._sm,
            every_nth=self._nth_spin.value(),
            use_gpu=self._gpu_cb.isChecked(),
            camera_model=self._camera_combo.currentText(),
        )
        self._worker.status.connect(lambda m: self._log.append(m))
        self._worker.result.connect(self._on_result)
        self._worker.error.connect(lambda e: QMessageBox.critical(self, "Reconstruction Error", e))
        self._run_btn.setEnabled(False)
        self._worker.start()

    @pyqtSlot(object)
    def _on_result(self, result: Any) -> None:
        self._result = result
        self._run_btn.setEnabled(True)
        coverage = result.num_images_registered / max(result.num_images_total, 1) * 100
        if coverage < 80:
            self._coverage_warn.setText(
                f"⚠ Only {coverage:.1f}% of images registered. "
                "Try fewer frames or better-lit scenes."
            )
        self._populate_stats(result)
        if result.points:
            self._cloud_widget.set_point_cloud(result.points, result.colors)
        self._ply_btn.setEnabled(bool(result.points))

    def _populate_stats(self, result: Any) -> None:
        from core.reconstructor import compute_reconstruction_stats
        stats = compute_reconstruction_stats(result)
        rows = [
            ("Registered images", f"{stats['num_registered_images']} / {result.num_images_total}"),
            ("Coverage", f"{stats['coverage_percent']:.1f}%"),
            ("3D points", str(stats["num_points3d"])),
            ("Reprojection error", f"{stats['mean_reprojection_error']:.4f} px"),
        ]
        self._stats_table.setRowCount(len(rows))
        for i, (lbl, val) in enumerate(rows):
            self._stats_table.setItem(i, 0, QTableWidgetItem(lbl))
            self._stats_table.setItem(i, 1, QTableWidgetItem(val))

    def _export_ply(self) -> None:
        if not self._result:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Point Cloud", "reconstruction.ply", "PLY files (*.ply)"
        )
        if not path:
            return
        try:
            from core.reconstructor import export_point_cloud_ply
            export_point_cloud_ply(self._result, path)
            self._log.append(f"PLY exported to: {path}")
            self._result = type(self._result)(**{**self._result.model_dump(), "ply_path": path})
            self._meshlab_btn.setEnabled(True)
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def _open_meshlab(self) -> None:
        import subprocess, sys
        if self._result and self._result.ply_path:
            if sys.platform == "linux":
                subprocess.Popen(["meshlab", self._result.ply_path])


class _PointCloudPlaceholder(QWidget):
    """Placeholder for PyQtGraph 3D point cloud view."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._label = QLabel("3D point cloud will appear here after reconstruction.")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._label)
        self._gl_widget = None

    def set_point_cloud(self, points: list, colors: list) -> None:
        try:
            import pyqtgraph.opengl as gl
            import numpy as np

            if self._gl_widget is None:
                self._gl_widget = gl.GLViewWidget()
                self._gl_widget.setBackgroundColor(BG)
                layout = self.layout()
                layout.removeWidget(self._label)
                self._label.hide()
                layout.addWidget(self._gl_widget)

            pts = np.array(points, dtype=float)
            cols = np.array(colors, dtype=float) / 255.0
            # Add alpha channel
            alpha = np.ones((len(pts), 1), dtype=float)
            rgba = np.hstack([cols, alpha])

            scatter = gl.GLScatterPlotItem(pos=pts, color=rgba, size=1.5)
            self._gl_widget.addItem(scatter)
            grid = gl.GLGridItem()
            self._gl_widget.addItem(grid)
        except Exception:
            self._label.setText(
                f"3D view requires pyqtgraph with OpenGL support.\n"
                f"Point cloud: {len(points)} points loaded."
            )
            self._label.show()
