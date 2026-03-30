"""
desktop/widgets/calibration_widget.py — Guided 4-step calibration workflow.

Layout:
  ┌────────────────────────────────────────────────────────────┐
  │  Step indicator: [1 IMU Static] [2 Intrinsic] [3 Extrin]  │
  ├───────────────────────┬────────────────────────────────────┤
  │  Step instructions    │  Results / status for this step    │
  │  + file picker        │                                    │
  │  + [▶ Run Step]       │                                    │
  ├───────────────────────┴────────────────────────────────────┤
  │  Live log                                                  │
  └────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.calibration import check_calibration_health, get_step_result, is_step_complete
from core.session_manager import SessionManager
from desktop.workers import CalibrationWorker
from desktop.widgets.timing_bar import TimingBar

_PANEL    = "#16213e"
_ACCENT   = "#0f3460"
_TEXT     = "#e0e0e0"
_MUTED    = "#888888"
_OK       = "#27ae60"
_WARN     = "#f39c12"
_ERR      = "#e94560"

_STEPS = [
    {
        "key": "imu_static",
        "label": "1  IMU Static",
        "title": "Step 1 — IMU Static Recording",
        "instructions": (
            "Place the GoPro flat on a stable table and record for at least 4 hours "
            "without touching it.\n\n"
            "This recording is used to compute accelerometer and gyroscope noise "
            "parameters (noise density, random walk) via Allan deviation analysis.\n\n"
            "Requirements:\n"
            "  • Duration ≥ 4 hours\n"
            "  • Device completely still throughout\n"
            "  • HyperSmooth OFF\n\n"
            "Select the resulting MP4 file below."
        ),
        "file_filter": "GoPro MP4 (*.MP4 *.mp4)",
    },
    {
        "key": "camera_intrinsic",
        "label": "2  Intrinsic",
        "title": "Step 2 — Camera Intrinsic Calibration",
        "instructions": (
            "Move the GoPro in front of a chessboard or AprilTag calibration board, "
            "covering all four corners and varying the distance and tilt.\n\n"
            "This step estimates fx, fy, cx, cy and lens distortion coefficients "
            "using OpenCV's calibrateCamera.\n\n"
            "Requirements:\n"
            "  • ≥ 15 distinct calibration poses detected\n"
            "  • Reprojection error < 0.5 px\n"
            "  • Board fully visible, good lighting, no motion blur\n\n"
            "Select the calibration video below."
        ),
        "file_filter": "GoPro MP4 (*.MP4 *.mp4)",
    },
    {
        "key": "camera_imu_extrinsic",
        "label": "3  Extrinsic",
        "title": "Step 3 — Camera-IMU Extrinsic Calibration",
        "instructions": (
            "Shake the GoPro aggressively in front of a fixed calibration board, "
            "covering rotations around all three axes.\n\n"
            "This step invokes OpenImuCameraCalibrator (external tool) and may take "
            "20–40 minutes. Output is a 4×4 T_cam_imu transformation matrix.\n\n"
            "Requirements:\n"
            "  • Steps 1 and 2 must be complete\n"
            "  • OpenImuCameraCalibrator must be installed and on PATH\n"
            "  • Board fixed and clearly visible throughout\n\n"
            "Select the extrinsic calibration video below."
        ),
        "file_filter": "GoPro MP4 (*.MP4 *.mp4)",
    },
    {
        "key": "validation",
        "label": "4  Validation",
        "title": "Step 4 — Validation",
        "instructions": (
            "Run a short test sequence through VINS-Mono or ORBSLAM3 to validate "
            "the calibration.\n\n"
            "Review the trajectory plot and check for loop-closure drift.\n\n"
            "This step is informational — no automated action is taken here. "
            "Refer to the VINS-Mono / ORBSLAM3 documentation for test procedure."
        ),
        "file_filter": "",
    },
]


class CalibrationWidget(QWidget):
    def __init__(self, session_manager: SessionManager, parent=None) -> None:
        super().__init__(parent)
        self._sm = session_manager
        self._session_id: str = ""
        self._worker: CalibrationWorker | None = None
        self._current_step_idx: int = 0
        self._cal_root: str = ""
        self._cal_session_id: str = ""
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Session header
        self._session_label = QLabel("No session selected")
        self._session_label.setStyleSheet(
            f"color: {_TEXT}; font-size: 13px; font-weight: bold;"
        )
        root.addWidget(self._session_label)

        # Step indicator bar
        step_bar = QHBoxLayout()
        step_bar.setSpacing(4)
        self._step_btns: list[QPushButton] = []
        for i, step in enumerate(_STEPS):
            btn = QPushButton(step["label"])
            btn.setCheckable(True)
            btn.setFixedHeight(30)
            btn.setStyleSheet(_step_btn_style())
            idx = i
            btn.clicked.connect(lambda checked, x=idx: self._select_step(x))
            self._step_btns.append(btn)
            step_bar.addWidget(btn)
        step_bar.addStretch()

        self._health_btn = QPushButton("Health Check")
        self._health_btn.setFixedHeight(30)
        self._health_btn.setEnabled(False)
        self._health_btn.setStyleSheet(_small_btn_style())
        self._health_btn.clicked.connect(self._run_health_check)
        step_bar.addWidget(self._health_btn)

        root.addLayout(step_bar)

        # Main splitter: instructions+controls | results
        top_split = QSplitter(Qt.Orientation.Horizontal)

        # Left: instructions + file picker + run button
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 4, 0)

        self._step_title = QLabel()
        self._step_title.setFont(QFont("sans-serif", 11, QFont.Weight.Bold))
        self._step_title.setStyleSheet(f"color: {_TEXT};")
        left_layout.addWidget(self._step_title)

        self._instructions = QTextEdit()
        self._instructions.setReadOnly(True)
        self._instructions.setStyleSheet(
            f"background: {_PANEL}; color: {_TEXT};"
            f" border: 1px solid {_ACCENT}; border-radius: 3px; font-size: 11px;"
        )
        self._instructions.setMaximumHeight(200)
        left_layout.addWidget(self._instructions)

        # File picker row
        file_row = QHBoxLayout()
        self._file_label = QLabel("No file selected")
        self._file_label.setStyleSheet(f"color: {_MUTED}; font-size: 11px;")
        self._file_label.setWordWrap(True)
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(80)
        browse_btn.setStyleSheet(_small_btn_style())
        browse_btn.clicked.connect(self._browse_file)
        file_row.addWidget(self._file_label, 1)
        file_row.addWidget(browse_btn)
        left_layout.addLayout(file_row)

        self._run_btn = QPushButton("▶  Run Step")
        self._run_btn.setEnabled(False)
        self._run_btn.setFixedHeight(34)
        self._run_btn.setStyleSheet(_btn_style())
        self._run_btn.clicked.connect(self._run_step)
        left_layout.addWidget(self._run_btn)
        left_layout.addStretch()

        top_split.addWidget(left)

        # Right: results panel
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 0, 0, 0)

        results_hdr = QLabel("Step Results")
        results_hdr.setFont(QFont("sans-serif", 9, QFont.Weight.Bold))
        results_hdr.setStyleSheet(f"color: {_MUTED};")
        right_layout.addWidget(results_hdr)

        self._results_view = QTextEdit()
        self._results_view.setReadOnly(True)
        self._results_view.setFont(QFont("monospace", 9))
        self._results_view.setStyleSheet(
            f"background: {_PANEL}; color: {_TEXT};"
            f" border: 1px solid {_ACCENT}; border-radius: 3px;"
        )
        right_layout.addWidget(self._results_view)

        top_split.addWidget(right)
        top_split.setStretchFactor(0, 1)
        top_split.setStretchFactor(1, 1)

        # Vertical splitter: top | log
        v_split = QSplitter(Qt.Orientation.Vertical)
        v_split.addWidget(top_split)

        log_w = QWidget()
        log_layout = QVBoxLayout(log_w)
        log_layout.setContentsMargins(0, 4, 0, 0)
        log_layout.setSpacing(4)

        self._timing_bar = TimingBar(label="Calibration", parent=self)
        log_layout.addWidget(self._timing_bar)

        log_hdr = QLabel("Calibration Log")
        log_hdr.setFont(QFont("sans-serif", 9, QFont.Weight.Bold))
        log_hdr.setStyleSheet(f"color: {_MUTED};")
        log_layout.addWidget(log_hdr)

        self._log_view = QTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setFont(QFont("monospace", 9))
        self._log_view.setStyleSheet(
            f"background: {_PANEL}; color: {_TEXT};"
            f" border: 1px solid {_ACCENT}; border-radius: 3px;"
        )
        log_layout.addWidget(self._log_view)

        v_split.addWidget(log_w)
        v_split.setStretchFactor(0, 2)
        v_split.setStretchFactor(1, 1)

        root.addWidget(v_split)

        # Placeholder shown when no session
        self._placeholder = QLabel("Select a session to begin calibration.")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet(f"color: {_MUTED}; font-size: 13px;")
        root.addWidget(self._placeholder)

        self._placeholder.setVisible(True)
        v_split.setVisible(False)
        self._v_split = v_split
        for btn in self._step_btns:
            btn.setVisible(False)
        self._step_bar_widget = step_bar

        self._selected_file: str = ""
        self._select_step(0)

    # ------------------------------------------------------------------
    # Session change
    # ------------------------------------------------------------------

    @pyqtSlot(str)
    def on_session_changed(self, session_id: str) -> None:
        self._session_id = session_id
        self._worker = None

        if not session_id:
            self._session_label.setText("No session selected")
            self._placeholder.setVisible(True)
            self._v_split.setVisible(False)
            for btn in self._step_btns:
                btn.setVisible(False)
            return

        try:
            s = self._sm.get_session(session_id)
        except Exception:
            return

        self._session_label.setText(f"{s.name}  [{s.environment} · {s.location}]")

        # Calibration data lives next to the session folder
        session_folder = self._sm.session_folder(session_id)
        self._cal_root = str(session_folder.parent / "calibration")
        self._cal_session_id = session_id

        self._placeholder.setVisible(False)
        self._v_split.setVisible(True)
        for btn in self._step_btns:
            btn.setVisible(True)
        self._health_btn.setEnabled(True)

        self._select_step(self._current_step_idx)

    # ------------------------------------------------------------------
    # Step navigation
    # ------------------------------------------------------------------

    def _select_step(self, idx: int) -> None:
        self._current_step_idx = idx
        step = _STEPS[idx]

        for i, btn in enumerate(self._step_btns):
            btn.setChecked(i == idx)

        self._step_title.setText(step["title"])
        self._instructions.setPlainText(step["instructions"])
        self._selected_file = ""
        self._file_label.setText("No file selected")

        # Disable run for validation step and when no session
        is_validation = step["key"] == "validation"
        self._run_btn.setEnabled(
            not is_validation and bool(self._session_id)
        )
        if is_validation:
            self._run_btn.setText("(Manual step — see instructions)")
        else:
            self._run_btn.setText("▶  Run Step")

        # Show existing results if available
        self._refresh_results(idx)

    def _refresh_results(self, idx: int) -> None:
        step = _STEPS[idx]
        if step["key"] == "validation" or not self._cal_root:
            self._results_view.setPlainText("—")
            return

        result = get_step_result(self._cal_root, self._cal_session_id, step["key"])
        if result is None:
            complete_prev = all(
                is_step_complete(self._cal_root, self._cal_session_id, _STEPS[i]["key"])
                for i in range(idx)
                if _STEPS[i]["key"] != "validation"
            )
            self._results_view.setPlainText(
                "Not yet run." + (
                    "" if complete_prev or idx == 0
                    else "\n\n⚠  Complete previous steps first."
                )
            )
        else:
            self._results_view.setPlainText(_format_result(step["key"], result))

    # ------------------------------------------------------------------
    # File picker
    # ------------------------------------------------------------------

    def _browse_file(self) -> None:
        step = _STEPS[self._current_step_idx]
        f_filter = step.get("file_filter", "All Files (*)")
        path, _ = QFileDialog.getOpenFileName(self, "Select file", "", f_filter)
        if path:
            self._selected_file = path
            self._file_label.setText(Path(path).name)

    # ------------------------------------------------------------------
    # Run step
    # ------------------------------------------------------------------

    def _run_step(self) -> None:
        if not self._session_id:
            return
        if self._worker and self._worker.isRunning():
            return

        step = _STEPS[self._current_step_idx]
        if step["key"] == "validation":
            return

        if not self._selected_file:
            QMessageBox.warning(self, "No File", "Select an input file before running.")
            return

        self._run_btn.setEnabled(False)
        self._log_view.clear()
        self._timing_bar.start()
        self._log(f"Starting {step['title']}…")

        extra = {
            "cal_root": self._cal_root,
            "cal_session_id": self._cal_session_id,
        }

        self._worker = CalibrationWorker(
            self._cal_session_id,
            step["key"],
            self._selected_file,
            extra=extra,
            parent=self,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.status.connect(self._on_status)
        self._worker.result.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.timing.connect(self._timing_bar.on_timing)
        self._worker.progress.connect(self._timing_bar.on_progress)
        self._worker.start()

    @pyqtSlot(int)
    def _on_progress(self, pct: int) -> None:
        mw = self._main_window()
        if mw:
            mw.set_progress(pct)

    @pyqtSlot(str)
    def _on_status(self, msg: str) -> None:
        self._log(msg)
        mw = self._main_window()
        if mw:
            mw.set_status(msg)
            mw.log(msg)

    @pyqtSlot(object)
    def _on_done(self, result: object) -> None:
        self._run_btn.setEnabled(True)
        self._timing_bar.stop(success=True)
        self._refresh_results(self._current_step_idx)
        self._log("Step complete.")
        mw = self._main_window()
        if mw:
            mw.set_progress(100)
        # Auto-advance to next step
        next_idx = self._current_step_idx + 1
        if next_idx < len(_STEPS):
            reply = QMessageBox.question(
                self,
                "Step Complete",
                f"{_STEPS[self._current_step_idx]['title']} is done.\n\n"
                f"Proceed to {_STEPS[next_idx]['title']}?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._select_step(next_idx)

    @pyqtSlot(str)
    def _on_error(self, msg: str) -> None:
        self._run_btn.setEnabled(True)
        self._timing_bar.stop(success=False)
        self._log(f"ERROR: {msg}")
        QMessageBox.critical(self, "Calibration Error", msg)
        mw = self._main_window()
        if mw:
            mw.set_progress(100)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        self._log_view.append(msg)
        sb = self._log_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _run_health_check(self) -> None:
        if not self._cal_root or not self._cal_session_id:
            return
        report = check_calibration_health(self._cal_root, self._cal_session_id)
        dlg = _HealthCheckDialog(report, self)
        dlg.exec()

    def _main_window(self):
        w = self.parent()
        while w is not None:
            if hasattr(w, "log") and hasattr(w, "set_progress"):
                return w
            w = w.parent() if hasattr(w, "parent") else None
        return None


class _HealthCheckDialog(QDialog):
    """Modal dialog showing calibration health check results as a checklist."""

    def __init__(self, report: dict, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Calibration Health Check")
        self.setMinimumWidth(560)
        self.setStyleSheet(
            f"background: {_PANEL}; color: {_TEXT};"
        )
        layout = QVBoxLayout(self)

        overall = report.get("ok", False)
        status_lbl = QLabel("Overall: PASS" if overall else "Overall: FAIL")
        status_lbl.setStyleSheet(
            f"color: {'#27ae60' if overall else '#e94560'};"
            " font-size: 13px; font-weight: bold; padding: 4px;"
        )
        layout.addWidget(status_lbl)

        table = QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Check", "Result", "Detail"])
        table.horizontalHeader().setStretchLastSection(True)
        table.setStyleSheet(
            f"QTableWidget {{ background: {_PANEL}; color: {_TEXT};"
            f" border: 1px solid {_ACCENT}; gridline-color: {_ACCENT}; }}"
            f" QHeaderView::section {{ background: {_ACCENT}; color: {_TEXT};"
            f" padding: 4px; }}"
        )
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setAlternatingRowColors(False)
        table.verticalHeader().setVisible(False)

        checks = report.get("checks", [])
        table.setRowCount(len(checks))
        for row, c in enumerate(checks):
            name_item = QTableWidgetItem(c["name"])
            passed = c["passed"]
            result_item = QTableWidgetItem("✓ PASS" if passed else "✕ FAIL")
            result_item.setForeground(
                QTableWidgetItem().foreground() if True else None  # placeholder
            )
            from PyQt6.QtGui import QColor
            result_item.setForeground(QColor("#27ae60" if passed else "#e94560"))
            detail_item = QTableWidgetItem(c["detail"])
            for item in (name_item, result_item, detail_item):
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 0, name_item)
            table.setItem(row, 1, result_item)
            table.setItem(row, 2, detail_item)

        table.resizeColumnsToContents()
        layout.addWidget(table)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.accept)
        layout.addWidget(bb)


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def _format_result(step_key: str, result: dict) -> str:
    lines: list[str] = []
    v = result.get("validation", {})

    def check(ok: bool) -> str:
        return "✓" if ok else "✗"

    if step_key == "imu_static":
        dur = result.get("duration_s", 0)
        lines += [
            f"Duration:           {dur/3600:.2f} h  {check(v.get('duration_ok', False))}",
            f"Rate:               {result.get('rate_hz', 0):.0f} Hz",
            f"Accel noise density:{result.get('accel_noise_density', 0):.6f} m/s²/√Hz",
            f"Accel random walk:  {result.get('accel_random_walk', 0):.6f} m/s²/√s",
            f"Gyro noise density: {result.get('gyro_noise_density', 0):.6f} rad/s/√Hz",
            f"Gyro random walk:   {result.get('gyro_random_walk', 0):.6f} rad/s/√s",
            f"Stillness:          {check(v.get('stillness_ok', False))}",
        ]

    elif step_key == "camera_intrinsic":
        lines += [
            f"Resolution:         {result.get('image_width')}×{result.get('image_height')}",
            f"fx / fy:            {result.get('fx', 0):.2f} / {result.get('fy', 0):.2f}",
            f"cx / cy:            {result.get('cx', 0):.2f} / {result.get('cy', 0):.2f}",
            f"Reprojection error: {result.get('reprojection_error_px', 0):.4f} px  "
            f"{check(v.get('reproj_ok', False))}",
            f"Poses detected:     {result.get('n_poses', 0)}  {check(v.get('poses_ok', False))}",
        ]
        dist = result.get("distortion_coeffs", [])
        if dist:
            lines.append(f"Distortion:         {[f'{d:.4f}' for d in dist]}")

    elif step_key == "camera_imu_extrinsic":
        t = result.get("translation_m", [])
        r = result.get("rotation_deg", [])
        lines += [
            f"Translation:        {[f'{x:.4f}' for x in t]} m  {check(v.get('translation_ok', False))}",
            f"Rotation:           {[f'{x:.4f}' for x in r]} °  {check(v.get('rotation_ok', False))}",
        ]
        T = result.get("T_cam_imu", [])
        if T:
            lines.append("\nT_cam_imu (4×4):")
            for row in T:
                lines.append("  " + "  ".join(f"{x:8.5f}" for x in row))

    return "\n".join(lines) if lines else str(result)


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def _step_btn_style() -> str:
    return (
        f"QPushButton {{ background: {_ACCENT}; color: {_MUTED}; border: none;"
        f" border-radius: 3px; padding: 4px 10px; font-size: 11px; }}"
        f" QPushButton:checked {{ color: {_TEXT}; background: #1a3a70;"
        f" border-bottom: 2px solid #e94560; }}"
        f" QPushButton:hover {{ color: {_TEXT}; }}"
    )


def _btn_style() -> str:
    return (
        f"QPushButton {{ background: {_ACCENT}; color: {_TEXT}; border: none;"
        f" border-radius: 4px; padding: 4px 14px; font-size: 12px; }}"
        f" QPushButton:hover {{ background: #e94560; }}"
        f" QPushButton:disabled {{ background: #333355; color: {_MUTED}; }}"
    )


def _small_btn_style() -> str:
    return (
        f"QPushButton {{ background: {_ACCENT}; color: {_TEXT}; border: none;"
        f" border-radius: 3px; padding: 3px 8px; font-size: 11px; }}"
        f" QPushButton:hover {{ background: #e94560; }}"
    )
