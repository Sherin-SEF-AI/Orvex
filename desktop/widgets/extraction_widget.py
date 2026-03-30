"""
desktop/widgets/extraction_widget.py — Telemetry extraction tab.

Layout:
  ┌──────────────────────────────────────────────────────────┐
  │  Session info + extraction status badge                  │
  ├────────────────────┬─────────────────────────────────────┤
  │  Config panel      │  Output stats grid                  │
  │  - Frame FPS       │  IMU / GPS / Frames / Rate          │
  │  - Frame format    │                                     │
  │  - JPEG quality    │                                     │
  │  - Output format   │                                     │
  │  - Output dir      │                                     │
  │  - Sync / Interp   │                                     │
  │  [▶ Run Extract]   │                                     │
  ├────────────────────┴─────────────────────────────────────┤
  │  Live progress log (scrolling)                           │
  └──────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QSpinBox,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.models import ExtractionConfig
from core.session_manager import SessionManager
from desktop.theme import (
    BG, PANEL, ACCENT, HI, TEXT, MUTED, SUCCESS, WARNING, BORDER, CARD, INPUT,
    card_style,
)
from desktop.workers import ExtractionWorker
from desktop.widgets.timing_bar import TimingBar

# Backwards-compat aliases used throughout this file
_BG        = BG
_PANEL     = PANEL
_ACCENT    = ACCENT
_TEXT      = TEXT
_MUTED     = MUTED
_HIGHLIGHT = HI
_SUCCESS   = SUCCESS
_WARNING   = WARNING

_STATUS_COLORS = {
    "pending": _MUTED,
    "running": _WARNING,
    "done":    _SUCCESS,
    "failed":  _HIGHLIGHT,
}
_STATUS_ICONS = {
    "pending": "○",
    "running": "◉",
    "done":    "●",
    "failed":  "✕",
}


class ExtractionWidget(QWidget):
    def __init__(self, session_manager: SessionManager, parent=None) -> None:
        super().__init__(parent)
        self._sm = session_manager
        self._session_id: str = ""
        self._worker: ExtractionWorker | None = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # ── Session header ──────────────────────────────────────────────
        hdr_widget = QWidget()
        hdr_widget.setObjectName("ExtrHdr")
        hdr_widget.setFixedHeight(40)
        hdr_widget.setStyleSheet(
            f"#ExtrHdr {{ background: {_PANEL}; border-radius: 5px;"
            f" border: 1px solid {_ACCENT}; }}"
        )
        hdr = QHBoxLayout(hdr_widget)
        hdr.setContentsMargins(10, 0, 10, 0)
        hdr.setSpacing(8)

        self._session_label = QLabel("No session selected")
        self._session_label.setStyleSheet(
            f"color: {_TEXT}; font-size: 13px; font-weight: bold; background: transparent;"
        )
        self._status_icon = QLabel("○")
        self._status_icon.setFont(QFont("monospace", 13))
        self._status_icon.setStyleSheet(f"color: {_MUTED}; background: transparent;")
        self._status_badge = QLabel("—")
        self._status_badge.setStyleSheet(
            f"color: {_MUTED}; font-size: 11px; background: transparent;"
        )

        hdr.addWidget(self._session_label, stretch=1)
        hdr.addWidget(self._status_icon)
        hdr.addWidget(self._status_badge)
        root.addWidget(hdr_widget)

        # ── Timing bar ─────────────────────────────────────────────────
        self._timing_bar = TimingBar(label="Extraction", parent=self)
        root.addWidget(self._timing_bar)

        # ── Inline progress bar ────────────────────────────────────────
        self._inline_progress = QProgressBar()
        self._inline_progress.setFixedHeight(10)
        self._inline_progress.setTextVisible(False)
        self._inline_progress.setRange(0, 100)
        self._inline_progress.setVisible(False)
        root.addWidget(self._inline_progress)

        # ── Main splitter: config | stats ──────────────────────────────
        top_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Config group
        config_group = QGroupBox("Extraction Configuration")
        config_layout = QFormLayout(config_group)
        config_layout.setSpacing(10)
        config_layout.setContentsMargins(12, 16, 12, 12)

        _lbl = lambda t: _make_lbl(t)

        self._fps_spin = QDoubleSpinBox()
        self._fps_spin.setRange(0.5, 120.0)
        self._fps_spin.setSingleStep(0.5)
        self._fps_spin.setValue(5.0)
        self._fps_spin.setSuffix(" fps")
        self._fps_spin.setToolTip("Frames per second to extract from video")
        config_layout.addRow(_lbl("Frame FPS:"), self._fps_spin)

        self._format_combo = QComboBox()
        self._format_combo.addItems(["jpg", "png"])
        config_layout.addRow(_lbl("Frame format:"), self._format_combo)

        self._quality_spin = QSpinBox()
        self._quality_spin.setRange(1, 100)
        self._quality_spin.setValue(95)
        self._quality_spin.setSuffix("%")
        self._quality_spin.setToolTip("JPEG quality (1=worst, 100=best)")
        config_layout.addRow(_lbl("JPEG quality:"), self._quality_spin)

        self._output_combo = QComboBox()
        self._output_combo.addItems(["euroc", "custom"])
        self._output_combo.setToolTip("EuRoC = SLAM-compatible format")
        config_layout.addRow(_lbl("Output format:"), self._output_combo)

        # Output directory picker
        out_dir_row = QHBoxLayout()
        out_dir_row.setSpacing(4)
        self._out_dir_edit = QLineEdit()
        self._out_dir_edit.setPlaceholderText("(session folder / extraction_gopro)")
        self._out_dir_edit.setToolTip("Leave blank to use default location inside session folder")
        browse_btn = QToolButton()
        browse_btn.setText("…")
        browse_btn.setFixedWidth(28)
        browse_btn.setFixedHeight(28)
        browse_btn.setToolTip("Browse output directory")
        browse_btn.clicked.connect(self._browse_output_dir)
        out_dir_row.addWidget(self._out_dir_edit)
        out_dir_row.addWidget(browse_btn)
        config_layout.addRow(_lbl("Output dir:"), out_dir_row)

        self._sync_check = QCheckBox("Sync devices")
        self._sync_check.setChecked(True)
        self._sync_check.setToolTip("Align timestamps across multiple devices")
        config_layout.addRow("", self._sync_check)

        self._interp_check = QCheckBox("IMU interpolation")
        self._interp_check.setChecked(True)
        self._interp_check.setToolTip("Interpolate gyro onto accel timestamps")
        config_layout.addRow("", self._interp_check)

        self._run_btn = QPushButton("▶  Run Extraction")
        self._run_btn.setEnabled(False)
        self._run_btn.setFixedHeight(36)
        self._run_btn.setObjectName("DangerBtn")
        self._run_btn.clicked.connect(self._run_extraction)
        config_layout.addRow("", self._run_btn)

        top_splitter.addWidget(config_group)

        # Stats group
        stats_group = QGroupBox("Output Statistics")
        stats_outer = QVBoxLayout(stats_group)
        stats_outer.setContentsMargins(12, 16, 12, 12)
        stats_outer.setSpacing(8)

        # Stats grid: 2×3
        grid = QGridLayout()
        grid.setSpacing(8)
        self._stat_widgets: dict[str, QLabel] = {}
        _stats_defs = [
            ("imu",    "IMU Samples",  "0"),
            ("gps",    "GPS Samples",  "0"),
            ("frames", "Frames",       "0"),
            ("imu_hz", "IMU Rate",     "—"),
            ("gps_hz", "GPS Rate",     "—"),
            ("dur",    "Duration",     "—"),
        ]
        for i, (key, title, default) in enumerate(_stats_defs):
            row, col = divmod(i, 2)
            card = _StatCard(title, default)
            self._stat_widgets[key] = card
            grid.addWidget(card, row, col)

        stats_outer.addLayout(grid)

        self._out_dir_lbl = QLabel("Output:  —")
        self._out_dir_lbl.setFont(QFont("monospace", 9))
        self._out_dir_lbl.setStyleSheet(f"color: {_MUTED};")
        self._out_dir_lbl.setWordWrap(True)
        stats_outer.addWidget(self._out_dir_lbl)
        stats_outer.addStretch()

        top_splitter.addWidget(stats_group)
        top_splitter.setStretchFactor(0, 1)
        top_splitter.setStretchFactor(1, 2)

        # ── Vertical splitter: top | log ───────────────────────────────
        v_splitter = QSplitter(Qt.Orientation.Vertical)
        v_splitter.addWidget(top_splitter)

        log_container = QWidget()
        log_layout = QVBoxLayout(log_container)
        log_layout.setContentsMargins(0, 4, 0, 0)
        log_layout.setSpacing(2)

        log_hdr_row = QHBoxLayout()
        log_hdr_lbl = QLabel("Extraction Log")
        log_hdr_lbl.setFont(QFont("sans-serif", 9, QFont.Weight.Bold))
        log_hdr_lbl.setStyleSheet(f"color: {_MUTED};")
        self._clear_log_btn = QToolButton()
        self._clear_log_btn.setText("✕ Clear")
        self._clear_log_btn.setObjectName("SmallBtn")
        self._clear_log_btn.setFixedHeight(20)
        self._clear_log_btn.clicked.connect(self._log_view_clear)
        log_hdr_row.addWidget(log_hdr_lbl)
        log_hdr_row.addStretch()
        log_hdr_row.addWidget(self._clear_log_btn)
        log_layout.addLayout(log_hdr_row)

        self._log_view = QTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setFont(QFont("monospace", 9))
        log_layout.addWidget(self._log_view)

        v_splitter.addWidget(log_container)
        v_splitter.setStretchFactor(0, 2)
        v_splitter.setStretchFactor(1, 1)

        root.addWidget(v_splitter)

    # ------------------------------------------------------------------
    # Session change
    # ------------------------------------------------------------------

    @pyqtSlot(str)
    def on_session_changed(self, session_id: str) -> None:
        self._session_id = session_id
        self._worker = None
        self._timing_bar.reset()

        if not session_id:
            self._session_label.setText("No session selected")
            self._status_badge.setText("—")
            self._status_icon.setText("○")
            self._run_btn.setEnabled(False)
            self._reset_stats()
            return

        try:
            s = self._sm.get_session(session_id)
        except Exception:
            return

        self._session_label.setText(f"{s.name}")
        color = _STATUS_COLORS.get(s.extraction_status, _MUTED)
        icon  = _STATUS_ICONS.get(s.extraction_status, "○")
        self._status_icon.setText(icon)
        self._status_icon.setStyleSheet(
            f"color: {color}; font-weight: bold; background: transparent;"
        )
        self._status_badge.setText(s.extraction_status)
        self._status_badge.setStyleSheet(
            f"color: {color}; font-size: 11px; background: transparent;"
        )

        has_mp4 = any(f.lower().endswith(".mp4") for f in s.files)
        self._run_btn.setEnabled(has_mp4)

    # ------------------------------------------------------------------
    # Extraction execution
    # ------------------------------------------------------------------

    def _run_extraction(self) -> None:
        if not self._session_id:
            return
        if self._worker and self._worker.isRunning():
            return

        out_dir = self._out_dir_edit.text().strip() or None

        config = ExtractionConfig(
            session_id=self._session_id,
            frame_fps=self._fps_spin.value(),
            frame_format=self._format_combo.currentText(),
            frame_quality=self._quality_spin.value(),
            output_format=self._output_combo.currentText(),
            sync_devices=self._sync_check.isChecked(),
            imu_interpolation=self._interp_check.isChecked(),
        )

        self._run_btn.setEnabled(False)
        self._log_view.clear()
        self._inline_progress.setValue(0)
        self._inline_progress.setVisible(True)
        self._timing_bar.start()
        self._log("Starting extraction…")

        self._worker = ExtractionWorker(self._session_id, config, self._sm, parent=self)
        self._worker.progress.connect(self._on_progress)
        self._worker.status.connect(self._on_status)
        self._worker.result.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.timing.connect(self._timing_bar.on_timing)
        self._worker.progress.connect(self._timing_bar.on_progress)
        self._worker.start()

    @pyqtSlot(int)
    def _on_progress(self, pct: int) -> None:
        self._inline_progress.setValue(pct)
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
    def _on_done(self, extracted: object) -> None:
        self._run_btn.setEnabled(True)
        self._inline_progress.setVisible(False)
        self._timing_bar.stop(success=True)
        stats = getattr(extracted, "stats", {})

        self._stat_widgets["imu"].set_value(str(stats.get("imu_count", "?")))
        self._stat_widgets["gps"].set_value(str(stats.get("gps_count", "?")))
        self._stat_widgets["frames"].set_value(str(stats.get("frame_count", "?")))
        self._stat_widgets["imu_hz"].set_value(f"{stats.get('imu_rate_hz', '?')} Hz")
        self._stat_widgets["gps_hz"].set_value(f"{stats.get('gps_rate_hz', '?')} Hz")
        dur = getattr(extracted, "duration_seconds", None)
        self._stat_widgets["dur"].set_value(f"{dur:.1f} s" if dur else "—")

        out_dir = stats.get("output_dir", "—")
        self._out_dir_lbl.setText(f"Output:  {out_dir}")

        self._status_icon.setText("●")
        self._status_icon.setStyleSheet(f"color: {_SUCCESS}; font-weight: bold; background: transparent;")
        self._status_badge.setText("done")
        self._status_badge.setStyleSheet(f"color: {_SUCCESS}; font-size: 11px; background: transparent;")
        self._log("✓  Extraction complete.")
        mw = self._main_window()
        if mw:
            mw.set_progress(100)

    @pyqtSlot(str)
    def _on_error(self, msg: str) -> None:
        self._run_btn.setEnabled(True)
        self._inline_progress.setVisible(False)
        self._timing_bar.stop(success=False)
        self._log(f"✕  ERROR: {msg}")
        self._status_icon.setText("✕")
        self._status_icon.setStyleSheet(f"color: {_HIGHLIGHT}; font-weight: bold; background: transparent;")
        self._status_badge.setText("failed")
        self._status_badge.setStyleSheet(f"color: {_HIGHLIGHT}; font-size: 11px; background: transparent;")
        QMessageBox.critical(self, "Extraction Error", msg)
        mw = self._main_window()
        if mw:
            mw.set_progress(100)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _browse_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", str(Path.home())
        )
        if path:
            self._out_dir_edit.setText(path)

    def _reset_stats(self) -> None:
        for w in self._stat_widgets.values():
            w.set_value("—")
        self._out_dir_lbl.setText("Output:  —")

    def _log_view_clear(self) -> None:
        self._log_view.clear()

    def _log(self, msg: str) -> None:
        self._log_view.append(msg)
        sb = self._log_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _main_window(self):
        w = self.parent()
        while w is not None:
            if hasattr(w, "log") and hasattr(w, "set_progress"):
                return w
            w = w.parent() if hasattr(w, "parent") else None
        return None


# ---------------------------------------------------------------------------
# Helper widgets
# ---------------------------------------------------------------------------

def _make_lbl(text: str) -> QLabel:
    return QLabel(text)


class _StatCard(QWidget):
    """A small metric card: title + large value."""

    def __init__(self, title: str, value: str = "—", parent=None):
        super().__init__(parent)
        self.setObjectName("StatCard")
        self.setStyleSheet(card_style(ACCENT))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)

        self._title = QLabel(title)
        self._title.setFont(QFont("sans-serif", 9))

        self._value = QLabel(value)
        self._value.setFont(QFont("monospace", 16, QFont.Weight.Bold))

        layout.addWidget(self._title)
        layout.addWidget(self._value)

    def set_value(self, text: str) -> None:
        self._value.setText(text)
