"""
desktop/widgets/slam_widget.py — SLAM validation widget.

Runs ORBSLAM3 on EuRoC extracted sessions, shows 3D trajectory,
loop closure drift metrics.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from desktop.theme import BG, PANEL, ACCENT, HI, TEXT, MUTED, SUCCESS, WARNING, BORDER, HOVER, CARD
from desktop.workers import SLAMWorker

HIGHLIGHT = HI


class SlamWidget(QWidget):
    """UI for ORBSLAM3 validation on EuRoC sessions."""

    def __init__(self, session_manager, parent=None) -> None:
        super().__init__(parent)
        self._sm = session_manager
        self._session_id: str = ""
        self._worker: SLAMWorker | None = None
        self._slam_result = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ── Left config ────────────────────────────────────────────────
        cfg_w = QWidget()
        cfg_w.setMinimumWidth(280)
        cfg_w.setMaximumWidth(400)
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
        self._check_slam_install()

        mode_group = QGroupBox("SLAM Mode")
        mg = QVBoxLayout(mode_group)
        self._mode_combo = QComboBox()
        self._mode_combo.addItems([
            "mono_inertial", "mono", "stereo", "stereo_inertial"
        ])
        mg.addWidget(self._mode_combo)
        cfg_layout.addWidget(mode_group)

        files_group = QGroupBox("Files")
        fg = QVBoxLayout(files_group)

        fg.addWidget(QLabel("Vocabulary (.txt):"))
        voc_row = QHBoxLayout()
        self._voc_edit = QLineEdit()
        self._voc_edit.setPlaceholderText("Path to ORBvoc.txt…")
        voc_btn = QPushButton("…")
        voc_btn.setFixedWidth(32)
        voc_btn.clicked.connect(lambda: self._browse_file(self._voc_edit, "*.txt"))
        voc_row.addWidget(self._voc_edit)
        voc_row.addWidget(voc_btn)
        fg.addLayout(voc_row)

        fg.addWidget(QLabel("Config YAML:"))
        cfg_row = QHBoxLayout()
        self._cfg_edit = QLineEdit()
        self._cfg_edit.setPlaceholderText("Path to orbslam3.yaml…")
        cfg_file_btn = QPushButton("…")
        cfg_file_btn.setFixedWidth(32)
        cfg_file_btn.clicked.connect(lambda: self._browse_file(self._cfg_edit, "*.yaml"))
        cfg_row.addWidget(self._cfg_edit)
        cfg_row.addWidget(cfg_file_btn)
        fg.addLayout(cfg_row)

        self._autogen_btn = QPushButton("Auto-generate config from calibration")
        self._autogen_btn.clicked.connect(self._autogen_config)
        fg.addWidget(self._autogen_btn)
        cfg_layout.addWidget(files_group)

        # Status warning
        self._prereq_label = QLabel("⚠ Run extraction (EuRoC) first.")
        self._prereq_label.setWordWrap(True)
        cfg_layout.addWidget(self._prereq_label)

        self._run_btn = QPushButton("Run SLAM")
        self._run_btn.setEnabled(False)
        self._run_btn.setObjectName("DangerBtn")
        self._run_btn.clicked.connect(self._run)
        cfg_layout.addWidget(self._run_btn)

        cfg_layout.addStretch()
        splitter.addWidget(cfg_w)

        # ── Right panel ────────────────────────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 8, 8, 8)

        right_splitter = QSplitter(Qt.Orientation.Vertical)

        # Live log
        log_group = QGroupBox("SLAM Log")
        lg = QVBoxLayout(log_group)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        lg.addWidget(self._log)
        right_splitter.addWidget(log_group)

        # 3D trajectory placeholder (pyqtgraph GL)
        self._traj_widget = _TrajectoryPlaceholder()
        right_splitter.addWidget(self._traj_widget)

        right_splitter.setSizes([300, 400])
        rl.addWidget(right_splitter)

        # Metrics table
        self._metrics_table = QTableWidget(0, 2)
        self._metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self._metrics_table.horizontalHeader().setStretchLastSection(True)
        self._metrics_table.setMaximumHeight(200)
        rl.addWidget(self._metrics_table)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([280, 720])

    def _check_slam_install(self) -> None:
        from core.slam_validator import check_slam_installation
        info = check_slam_installation()
        if not info.get("orbslam3"):
            msgs = ["⚠ ORBSLAM3 not found on PATH."]
            msgs.append("Build from: https://github.com/UZ-SLAMLab/ORB_SLAM3")
            if not info.get("evo"):
                msgs.append("Optional: pip install evo  (for ATE/RPE metrics)")
            self._install_label.setText("\n".join(msgs))
            self._install_label.show()
            self._slam_ok = False
        else:
            self._install_label.hide()
            self._slam_ok = True

    # ------------------------------------------------------------------

    @pyqtSlot(str)
    def on_session_changed(self, session_id: str) -> None:
        self._session_id = session_id
        if session_id:
            ext_dir = self._sm.session_folder(session_id)
            has_euroc = (ext_dir / "cam0").exists() or any(ext_dir.rglob("cam0"))
            slam_ok = getattr(self, "_slam_ok", True)
            self._run_btn.setEnabled(has_euroc and slam_ok)
            if has_euroc:
                self._prereq_label.setText("✓ EuRoC data found.")
                self._prereq_label.setStyleSheet(f"color:{SUCCESS};")
            else:
                self._prereq_label.setText("⚠ Run extraction (EuRoC) first.")
                self._prereq_label.setStyleSheet(f"color:{WARNING};")

    def _browse_file(self, line_edit: QLineEdit, ext: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select file", "", f"Files ({ext})")
        if path:
            line_edit.setText(path)

    def _autogen_config(self) -> None:
        if not self._session_id:
            return
        try:
            from core.calibration import get_step_result
            from core.slam_validator import generate_orbslam3_config
            calib = get_step_result(self._session_id, "camera_intrinsic")
            if not calib:
                QMessageBox.warning(
                    self, "No Calibration",
                    "Camera intrinsic calibration (Step 2) not found. "
                    "Complete calibration first."
                )
                return
            from core.models import CalibrationResult
            from datetime import datetime
            cal_result = CalibrationResult(
                session_id=self._session_id,
                step="camera_intrinsic",
                completed_at=datetime.now(),
                **{k: v for k, v in calib.items()
                   if k in CalibrationResult.model_fields},
            )
            out_path = str(
                self._sm.session_folder(self._session_id) / "orbslam3_config.yaml"
            )
            generate_orbslam3_config(cal_result, 30.0, 200.0, out_path)
            self._cfg_edit.setText(out_path)
            self._log.append(f"Config auto-generated: {out_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Config Error", str(exc))

    def _run(self) -> None:
        if not self._session_id or (self._worker and self._worker.isRunning()):
            return
        voc = self._voc_edit.text().strip()
        cfg = self._cfg_edit.text().strip()
        if not voc or not cfg:
            QMessageBox.warning(
                self, "Missing Files",
                "Provide both the ORB vocabulary file and ORBSLAM3 YAML config."
            )
            return
        self._log.clear()
        self._worker = SLAMWorker(
            session_id=self._session_id,
            sm=self._sm,
            vocabulary_path=voc,
            config_yaml=cfg,
            mode=self._mode_combo.currentText(),
        )
        self._worker.status.connect(lambda m: self._log.append(m))
        self._worker.result.connect(self._on_result)
        self._worker.error.connect(lambda e: QMessageBox.critical(self, "SLAM Error", e))
        self._run_btn.setEnabled(False)
        self._worker.start()

    @pyqtSlot(object)
    def _on_result(self, result: Any) -> None:
        self._slam_result = result
        self._run_btn.setEnabled(True)
        self._populate_metrics(result)
        if result.trajectory:
            self._traj_widget.set_trajectory(result.trajectory)

    def _populate_metrics(self, result: Any) -> None:
        m = result.metrics
        drift_m = m.loop_closure_drift_m
        drift_pct = m.loop_closure_drift_percent

        rows = [
            ("Status", "✓ Success" if result.success else "✗ Failed"),
            ("Mode", result.mode),
            ("Total distance", f"{m.total_distance_m:.2f} m"),
            ("Duration", f"{m.duration_seconds:.1f} s"),
            ("Avg speed", f"{m.avg_speed_mps:.3f} m/s"),
            ("Loop closure drift",
             f"{drift_m:.3f} m ({drift_pct:.1f}%)" if drift_m is not None else "N/A (not a loop)"),
            ("Tracking lost", str(m.tracking_lost_count)),
            ("Keyframes", str(m.keyframe_count)),
            ("Map points", str(m.map_point_count)),
        ]
        self._metrics_table.setRowCount(len(rows))
        for i, (label, value) in enumerate(rows):
            self._metrics_table.setItem(i, 0, QTableWidgetItem(label))
            item = QTableWidgetItem(value)
            # Color loop closure drift
            if label == "Loop closure drift" and drift_pct is not None:
                if drift_pct < 2.0:
                    item.setForeground(Qt.GlobalColor.green)
                elif drift_pct < 5.0:
                    item.setForeground(Qt.GlobalColor.yellow)
                else:
                    item.setForeground(Qt.GlobalColor.red)
            self._metrics_table.setItem(i, 1, item)


class _TrajectoryPlaceholder(QWidget):
    """Placeholder for PyQtGraph 3D trajectory view."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._label = QLabel("3D trajectory will appear here after SLAM completes.")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._label)
        self._traj_widget = None

    def set_trajectory(self, trajectory: list) -> None:
        """Render 3D trajectory using PyQtGraph GLViewWidget."""
        try:
            import pyqtgraph.opengl as gl
            import numpy as np
            from PyQt6.QtWidgets import QApplication

            if self._traj_widget is None:
                self._traj_widget = gl.GLViewWidget()
                self._traj_widget.setBackgroundColor(BG)
                layout = self.layout()
                layout.removeWidget(self._label)
                self._label.hide()
                layout.addWidget(self._traj_widget)

            pts = np.array([[p.tx, p.ty, p.tz] for p in trajectory])
            # Color by time: blue → red
            n = len(pts)
            colors = np.zeros((n, 4), dtype=float)
            colors[:, 0] = np.linspace(0, 1, n)  # R
            colors[:, 2] = np.linspace(1, 0, n)  # B
            colors[:, 3] = 1.0

            line = gl.GLLinePlotItem(pos=pts, color=colors, width=2, antialias=True)
            self._traj_widget.addItem(line)

            # Grid
            grid = gl.GLGridItem()
            self._traj_widget.addItem(grid)
        except Exception:
            self._label.setText(
                f"3D view requires pyqtgraph with OpenGL support.\n"
                f"Trajectory: {len(trajectory)} poses recorded."
            )
            self._label.show()
