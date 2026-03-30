"""
desktop/widgets/dataset_widget.py — Dataset builder UI tab.

Layout:
  ┌──────────────────────────────────────────────────────────────┐
  │  Session selector (multi-select list)                       │
  ├──────────────────────────────────────────────────────────────┤
  │  Config: output format | output dir | [Build Dataset]       │
  ├──────────────────────────────────────────────────────────────┤
  │  Progress bar + live log                                     │
  ├──────────────────────────────────────────────────────────────┤
  │  Result summary (frame count, IMU samples, manifest path)   │
  └──────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

from datetime import timezone
from pathlib import Path

import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.dataset_builder import DatasetBuildResult
from core.export_profiles import ExportProfile, list_profiles, save_profile
from core.session_manager import SessionManager
from desktop.theme import (
    BG, PANEL, ACCENT, HI, TEXT, MUTED, SUCCESS, WARNING, BORDER, CARD,
    apply_plot_theme,
)
from desktop.workers import DatasetBuildWorker
from desktop.widgets.timing_bar import TimingBar

# Backwards-compat aliases
_BG     = BG
_PANEL  = PANEL
_ACCENT = ACCENT
_TEXT   = TEXT
_MUTED  = MUTED
_OK     = SUCCESS
_ERR    = HI
_WARN   = WARNING

_STATUS_COLOR = {
    "pending": _MUTED,
    "running": _WARN,
    "done":    _OK,
    "failed":  _ERR,
}


class SessionTimelineWidget(QWidget):
    """PyQtGraph bar chart showing each session's recording span on a timeline.

    X-axis: Unix time (seconds). Y-axis: session index.
    Each bar spans from session.created_at to created_at + total_duration_s.
    Color encodes extraction_status.
    """

    _STATUS_BRUSHES = {
        "pending": pg.mkBrush("#666680"),
        "running": pg.mkBrush("#f39c12"),
        "done":    pg.mkBrush("#27ae60"),
        "failed":  pg.mkBrush("#e94560"),
    }

    def __init__(self, session_manager: SessionManager, parent=None) -> None:
        super().__init__(parent)
        self._sm = session_manager
        self._build_ui()
        self.setMinimumHeight(80)
        self.setMaximumHeight(180)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        lbl = QLabel("Session Timeline")
        layout.addWidget(lbl)

        self._plot = pg.PlotWidget()
        apply_plot_theme(self._plot)
        self._plot.getPlotItem().hideAxis("left")
        self._plot.setLabel("bottom", "Time (Unix s)")
        self._plot.setMouseEnabled(x=True, y=False)
        layout.addWidget(self._plot)

        self._tooltip = pg.TextItem(text="", color=_TEXT, fill=pg.mkBrush(_PANEL))
        self._tooltip.setZValue(100)
        self._plot.addItem(self._tooltip)
        self._tooltip.hide()

        self._proxy = pg.SignalProxy(
            self._plot.scene().sigMouseMoved,
            rateLimit=30,
            slot=self._on_mouse_move,
        )
        self._bars: list[dict] = []   # [{bar, session}]

    def refresh(self) -> None:
        self._plot.clear()
        self._bars.clear()
        self._tooltip = pg.TextItem(text="", color=_TEXT, fill=pg.mkBrush(_PANEL))
        self._tooltip.setZValue(100)
        self._plot.addItem(self._tooltip)
        self._tooltip.hide()

        sessions = self._sm.list_sessions()
        if not sessions:
            return

        for y_idx, s in enumerate(sessions):
            # Determine x_start from created_at
            try:
                if s.created_at.tzinfo is None:
                    x_start = s.created_at.replace(tzinfo=timezone.utc).timestamp()
                else:
                    x_start = s.created_at.timestamp()
            except Exception:
                x_start = 0.0

            # Duration from audit results
            total_dur = sum(r.duration_seconds for r in s.audit_results) if s.audit_results else 60.0
            x_end = x_start + max(total_dur, 1.0)

            brush = self._STATUS_BRUSHES.get(s.extraction_status, pg.mkBrush(_MUTED))
            bar = pg.BarGraphItem(
                x0=[x_start], x1=[x_end],
                y0=[y_idx + 0.1], y1=[y_idx + 0.8],
                brush=brush, pen=pg.mkPen(None),
            )
            self._plot.addItem(bar)
            self._bars.append({"bar": bar, "session": s, "x0": x_start, "x1": x_end, "y": y_idx})

        self._plot.autoRange()

    def _on_mouse_move(self, evt) -> None:
        pos = evt[0]
        vb = self._plot.getPlotItem().vb
        if not self._plot.sceneBoundingRect().contains(pos):
            self._tooltip.hide()
            return
        mouse_point = vb.mapSceneToView(pos)
        mx, my = mouse_point.x(), mouse_point.y()
        for b in self._bars:
            if b["x0"] <= mx <= b["x1"] and b["y"] <= my <= b["y"] + 1:
                s = b["session"]
                dur = sum(r.duration_seconds for r in s.audit_results) if s.audit_results else 0.0
                n_frames = sum(r.imu_sample_count for r in s.audit_results)
                text = (
                    f"{s.name}\n"
                    f"Status: {s.extraction_status}\n"
                    f"Duration: {dur:.1f}s\n"
                    f"IMU samples: {n_frames}"
                )
                self._tooltip.setText(text)
                self._tooltip.setPos(mx, my)
                self._tooltip.show()
                return
        self._tooltip.hide()


class DatasetWidget(QWidget):
    """Dataset builder tab."""

    def __init__(self, session_manager: SessionManager, parent=None) -> None:
        super().__init__(parent)
        self._sm = session_manager
        self._worker: DatasetBuildWorker | None = None
        self._output_dir: Path | None = None
        self._profiles_dir = Path.home() / ".roverdatakit" / "data" / "profiles"
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # --- Session timeline ---
        self._timeline = SessionTimelineWidget(self._sm, self)
        root.addWidget(self._timeline)

        # --- Session selector ---
        sess_group = QGroupBox("Sessions to Include")
        sg_layout = QVBoxLayout(sess_group)

        self._sess_list = QListWidget()
        self._sess_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self._sess_list.setMinimumHeight(120)
        self._sess_list.setMaximumHeight(200)
        sg_layout.addWidget(self._sess_list)

        hint = QLabel("Hold Ctrl / \u2318 to select multiple sessions.")
        sg_layout.addWidget(hint)
        root.addWidget(sess_group)

        # --- Config ---
        cfg_group = QGroupBox("Build Configuration")
        cfg_layout = QHBoxLayout(cfg_group)

        cfg_layout.addWidget(QLabel("Profile:"))
        self._profile_combo = QComboBox()
        self._profile_combo.setMinimumWidth(120)
        self._refresh_profiles()
        self._profile_combo.currentIndexChanged.connect(self._on_profile_changed)
        cfg_layout.addWidget(self._profile_combo)

        save_profile_btn = QPushButton("Save…")
        save_profile_btn.setFixedHeight(26)
        save_profile_btn.setToolTip("Save current configuration as a named profile")
        save_profile_btn.clicked.connect(self._save_current_profile)
        cfg_layout.addWidget(save_profile_btn)

        cfg_layout.addSpacing(12)
        cfg_layout.addWidget(QLabel("Format:"))
        self._format_combo = QComboBox()
        self._format_combo.addItem("EuRoC",     userData="euroc")
        self._format_combo.addItem("ROS bag 2", userData="rosbag2")
        self._format_combo.addItem("HDF5",      userData="hdf5")
        self._format_combo.setMinimumWidth(130)
        cfg_layout.addWidget(self._format_combo)

        cfg_layout.addSpacing(16)
        cfg_layout.addWidget(QLabel("Output dir:"))
        self._outdir_label = QLabel("(not set)")
        self._outdir_label.setMaximumWidth(300)
        cfg_layout.addWidget(self._outdir_label)

        browse_btn = QPushButton("Browse\u2026")
        browse_btn.setFixedHeight(26)
        browse_btn.clicked.connect(self._pick_output_dir)
        cfg_layout.addWidget(browse_btn)

        cfg_layout.addStretch()

        self._build_btn = QPushButton("Build Dataset")
        self._build_btn.setEnabled(False)
        self._build_btn.setFixedHeight(30)
        self._build_btn.setObjectName("PrimaryBtn")
        self._build_btn.clicked.connect(self._run_build)
        cfg_layout.addWidget(self._build_btn)

        root.addWidget(cfg_group)

        # --- Timing bar + progress ---
        self._timing_bar = TimingBar(label="Dataset Build", parent=self)
        root.addWidget(self._timing_bar)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFixedHeight(10)
        self._progress_bar.setVisible(False)
        root.addWidget(self._progress_bar)

        # --- Log + result splitter ---
        log_split = QSplitter(Qt.Orientation.Vertical)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("monospace", 9))
        self._log.setMinimumHeight(100)
        log_split.addWidget(self._log)

        self._result_box = QGroupBox("Last Build Result")
        rb_layout = QVBoxLayout(self._result_box)
        self._result_label = QLabel("No dataset built yet.")
        self._result_label.setFont(QFont("monospace", 9))
        self._result_label.setWordWrap(True)
        rb_layout.addWidget(self._result_label)
        log_split.addWidget(self._result_box)
        log_split.setStretchFactor(0, 3)
        log_split.setStretchFactor(1, 1)

        root.addWidget(log_split, stretch=1)

        self._sess_list.itemSelectionChanged.connect(self._update_build_btn)

    # ------------------------------------------------------------------
    # Session change
    # ------------------------------------------------------------------

    @pyqtSlot(str)
    def on_session_changed(self, session_id: str) -> None:
        self._refresh_session_list()

    def _refresh_session_list(self) -> None:
        self._sess_list.clear()
        self._timeline.refresh()
        for s in self._sm.list_sessions():
            item = QListWidgetItem(f"{s.name}  [{s.environment} \u00b7 {s.location}]")
            item.setData(Qt.ItemDataRole.UserRole, s.id)
            color = _STATUS_COLOR.get(s.extraction_status, _MUTED)
            item.setForeground(QColor(color))
            item.setToolTip(f"Extraction: {s.extraction_status}")
            self._sess_list.addItem(item)
        self._update_build_btn()

    # ------------------------------------------------------------------
    # Profile management
    # ------------------------------------------------------------------

    def _refresh_profiles(self) -> None:
        self._profile_combo.blockSignals(True)
        self._profile_combo.clear()
        self._profile_combo.addItem("(custom)", userData=None)
        for p in list_profiles(self._profiles_dir):
            self._profile_combo.addItem(p.name, userData=p)
        self._profile_combo.blockSignals(False)

    def _on_profile_changed(self, index: int) -> None:
        profile = self._profile_combo.itemData(index)
        if not isinstance(profile, ExportProfile):
            return
        self._apply_profile(profile)

    def _apply_profile(self, profile: ExportProfile) -> None:
        # Set format combo
        for i in range(self._format_combo.count()):
            if self._format_combo.itemData(i) == profile.output_format:
                self._format_combo.setCurrentIndex(i)
                break

    def _save_current_profile(self) -> None:
        name, ok = QInputDialog.getText(
            self, "Save Profile", "Profile name:",
        )
        if not ok or not name.strip():
            return
        fmt = self._format_combo.currentData() or "euroc"
        profile = ExportProfile(
            name=name.strip(),
            output_format=fmt,
        )
        try:
            save_profile(profile, self._profiles_dir)
            self._refresh_profiles()
            QMessageBox.information(self, "Profile Saved", f"Profile '{name}' saved.")
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))

    # ------------------------------------------------------------------
    # Output dir picker
    # ------------------------------------------------------------------

    def _pick_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select output directory", str(Path.home())
        )
        if path:
            self._output_dir = Path(path)
            self._outdir_label.setText(str(self._output_dir))
            self._outdir_label.setStyleSheet(f"color: {_TEXT}; font-size: 11px;")
            self._update_build_btn()

    def _update_build_btn(self) -> None:
        ready = (
            len(self._sess_list.selectedItems()) > 0
            and self._output_dir is not None
            and (self._worker is None or not self._worker.isRunning())
        )
        self._build_btn.setEnabled(ready)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _run_build(self) -> None:
        if self._worker and self._worker.isRunning():
            return

        session_ids = [
            item.data(Qt.ItemDataRole.UserRole)
            for item in self._sess_list.selectedItems()
        ]
        if not session_ids:
            QMessageBox.warning(self, "No Sessions", "Select at least one session.")
            return
        if self._output_dir is None:
            QMessageBox.warning(self, "No Output Dir", "Choose an output directory first.")
            return

        fmt = self._format_combo.currentData()

        self._log.clear()
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)
        self._timing_bar.start()
        self._build_btn.setEnabled(False)
        self._result_label.setText("Building\u2026")

        self._worker = DatasetBuildWorker(
            session_ids=session_ids,
            session_manager=self._sm,
            output_format=fmt,
            output_dir=self._output_dir,
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
        self._progress_bar.setValue(pct)
        mw = self._main_window()
        if mw:
            mw.set_progress(pct)

    @pyqtSlot(str)
    def _on_status(self, msg: str) -> None:
        self._log.append(msg)
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())
        mw = self._main_window()
        if mw:
            mw.log(msg)

    @pyqtSlot(object)
    def _on_done(self, result: object) -> None:
        self._progress_bar.setVisible(False)
        self._timing_bar.stop(success=True)
        self._update_build_btn()

        if not isinstance(result, DatasetBuildResult):
            return

        lines = [
            f"Format:         {result.format.upper()}",
            f"Total frames:   {result.total_frames}",
            f"IMU samples:    {result.total_imu_samples}",
            f"Manifest:       {result.manifest_path}",
            f"Output dir:     {result.output_dir}",
        ]
        if result.warnings:
            lines.append("")
            lines.append(f"Warnings ({len(result.warnings)}):")
            for w in result.warnings:
                lines.append(f"  \u26a0 {w}")

        # Run integrity check
        try:
            from core.dataset_builder import verify_dataset_integrity
            integrity = verify_dataset_integrity(result.output_dir)
            if integrity["ok"]:
                lines.append(f"\nIntegrity:      OK ({integrity['frame_count']} frames, {integrity['imu_count']} IMU rows)")
            else:
                lines.append(f"\nIntegrity:      WARN — {', '.join(integrity['warnings'])}")
        except Exception as exc:
            lines.append(f"\nIntegrity:      check failed: {exc}")

        self._result_label.setStyleSheet(f"color: {_TEXT};")
        self._result_label.setText("\n".join(lines))

        mw = self._main_window()
        if mw:
            mw.set_progress(100)
            mw.set_status(
                f"Dataset built: {result.total_frames} frames, "
                f"{result.total_imu_samples} IMU samples"
            )

    @pyqtSlot(str)
    def _on_error(self, msg: str) -> None:
        self._progress_bar.setVisible(False)
        self._timing_bar.stop(success=False)
        self._update_build_btn()
        self._result_label.setStyleSheet(f"color: {_ERR};")
        self._result_label.setText(f"Build failed:\n{msg}")
        QMessageBox.critical(self, "Build Error", msg)
        mw = self._main_window()
        if mw:
            mw.log(f"ERROR: {msg}")
            mw.set_progress(100)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _main_window(self):
        p = self.parent()
        while p is not None:
            if hasattr(p, "log") and hasattr(p, "set_progress"):
                return p
            p = p.parent()
        return None


