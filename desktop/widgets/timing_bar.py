"""
desktop/widgets/timing_bar.py — Reusable elapsed + ETA timing strip.

Shows:  ⏱ Extraction  Elapsed: 12s  |  ETA: ~45s  |  [████░░░░] 27%

Design notes
------------
- Elapsed time is driven by a 1-second QTimer anchored to wall-clock time
  recorded at start() — it never drifts even if progress signals are sparse.
- ETA is supplied by the worker via on_timing(elapsed_s, eta_s).  It is only
  displayed once at least ETA_MIN_ELAPSED seconds have passed AND the worker
  has reported >0% progress, to avoid a spurious "0s" flash during setup.
- on_timing() updates the ETA label only; it does NOT touch elapsed (the timer
  owns elapsed via wall clock).
"""
from __future__ import annotations

import time

from PyQt6.QtCore import QTimer, pyqtSlot
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QWidget

from desktop.theme import PANEL, ACCENT, TEXT, MUTED, HI as HIGHLIGHT, SUCCESS, WARNING

_PANEL     = PANEL
_ACCENT    = ACCENT
_TEXT      = TEXT
_MUTED     = MUTED
_HIGHLIGHT = HIGHLIGHT
_SUCCESS   = SUCCESS
_WARNING   = WARNING

# Minimum elapsed seconds before ETA is considered meaningful
ETA_MIN_ELAPSED = 2.0


def _fmt(seconds: float) -> str:
    """Format seconds → human-readable (e.g. '3s', '1m 20s', '2h 05m')."""
    if seconds < 0:
        return "--"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


class TimingBar(QWidget):
    """Horizontal strip: label | elapsed | ETA | progress bar | pct%.

    Usage:
        bar = TimingBar("Extraction", parent)
        worker.timing.connect(bar.on_timing)
        worker.progress.connect(bar.on_progress)
        bar.start()    # call immediately before worker.start()
        bar.stop(success=True/False)   # call in on_done / on_error
        bar.reset()    # hide and clear (e.g. on session change)
    """

    def __init__(self, label: str = "", parent=None) -> None:
        super().__init__(parent)
        self._label_text = label
        self._wall_start: float | None = None   # set by start()
        self._last_eta: float = -1.0             # last ETA from worker
        self._eta_received_at: float = 0.0       # wall-clock when last ETA arrived
        self._last_pct: int = 0
        self._timer = QTimer(self)
        self._timer.setInterval(500)             # update every 0.5 s
        self._timer.timeout.connect(self._tick)
        self._build_ui()
        self.setVisible(False)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.setObjectName("TimingBar")
        self.setFixedHeight(28)
        self.setStyleSheet(
            f"#TimingBar {{ background: {_PANEL}; border-radius: 4px;"
            f" border: 1px solid {_ACCENT}; }}"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(10)

        self._icon_lbl = QLabel("⏱")
        self._icon_lbl.setStyleSheet(f"color: {_WARNING}; font-size: 12px;")
        layout.addWidget(self._icon_lbl)

        if self._label_text:
            op_lbl = QLabel(self._label_text)
            op_lbl.setFont(QFont("sans-serif", 9, QFont.Weight.Bold))
            op_lbl.setStyleSheet(f"color: {_MUTED};")
            layout.addWidget(op_lbl)

        def _sep() -> QLabel:
            lbl = QLabel("|")
            lbl.setStyleSheet(f"color: {_ACCENT}; background: transparent;")
            return lbl

        self._elapsed_lbl = QLabel("Elapsed: 0s")
        self._elapsed_lbl.setFont(QFont("monospace", 9))
        self._elapsed_lbl.setMinimumWidth(95)
        layout.addWidget(self._elapsed_lbl)

        layout.addWidget(_sep())

        self._eta_lbl = QLabel("ETA: —")
        self._eta_lbl.setFont(QFont("monospace", 9))
        self._eta_lbl.setStyleSheet(f"color: {_MUTED};")
        self._eta_lbl.setMinimumWidth(110)
        layout.addWidget(self._eta_lbl)

        layout.addWidget(_sep())

        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedHeight(8)
        self._progress_bar.setFixedWidth(120)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setRange(0, 100)
        layout.addWidget(self._progress_bar)

        self._pct_lbl = QLabel("0%")
        self._pct_lbl.setFont(QFont("monospace", 9, QFont.Weight.Bold))
        self._pct_lbl.setMinimumWidth(35)
        layout.addWidget(self._pct_lbl)

        layout.addStretch()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Show the bar and start the elapsed timer."""
        self._wall_start = time.monotonic()
        self._last_eta = -1.0
        self._last_pct = 0
        self._elapsed_lbl.setText("Elapsed: 0s")
        self._eta_lbl.setText("ETA: —")
        self._eta_lbl.setStyleSheet(f"color: {_MUTED};")
        self._pct_lbl.setText("0%")
        self._progress_bar.setValue(0)
        self._icon_lbl.setText("⏱")
        self._icon_lbl.setStyleSheet(f"color: {_WARNING}; font-size: 12px;")
        self.setVisible(True)
        self._timer.start()

    def stop(self, success: bool = True) -> None:
        """Freeze at final elapsed time and show Done / Failed."""
        self._timer.stop()
        color = _SUCCESS if success else _HIGHLIGHT
        icon  = "✓" if success else "✕"
        self._icon_lbl.setText(icon)
        self._icon_lbl.setStyleSheet(f"color: {color}; font-size: 12px;")
        self._eta_lbl.setText("Done" if success else "Failed")
        self._eta_lbl.setStyleSheet(f"color: {color};")
        self._progress_bar.setValue(100 if success else self._last_pct)

    def reset(self) -> None:
        """Hide and clear — call when session changes."""
        self._timer.stop()
        self._wall_start = None
        self.setVisible(False)

    # ------------------------------------------------------------------
    # Slots connected to worker signals
    # ------------------------------------------------------------------

    @pyqtSlot(float, float)
    def on_timing(self, elapsed_s: float, eta_s: float) -> None:
        """Receive (elapsed, eta) from BaseWorker.timing signal.

        Only updates the ETA label — elapsed is owned by the wall-clock timer.
        ETA is shown only after ETA_MIN_ELAPSED seconds to avoid '0s' flash.
        """
        self._last_eta = eta_s
        self._eta_received_at = time.monotonic()
        elapsed = (
            time.monotonic() - self._wall_start
            if self._wall_start is not None
            else elapsed_s
        )
        if elapsed >= ETA_MIN_ELAPSED and eta_s > 0:
            self._eta_lbl.setText(f"ETA: ~{_fmt(eta_s)}")
            self._eta_lbl.setStyleSheet(f"color: {_WARNING};")
        elif elapsed < ETA_MIN_ELAPSED:
            self._eta_lbl.setText("ETA: —")
            self._eta_lbl.setStyleSheet(f"color: {_MUTED};")

    @pyqtSlot(int)
    def on_progress(self, pct: int) -> None:
        """Receive progress 0-100 from worker."""
        self._last_pct = pct
        self._progress_bar.setValue(pct)
        self._pct_lbl.setText(f"{pct}%")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        """Fire every 500 ms — update elapsed label from wall clock."""
        if self._wall_start is None:
            return
        elapsed = time.monotonic() - self._wall_start
        self._elapsed_lbl.setText(f"Elapsed: {_fmt(elapsed)}")

        # Countdown ETA between progress updates
        if elapsed >= ETA_MIN_ELAPSED and self._last_eta > 0:
            since = time.monotonic() - self._eta_received_at
            adjusted = max(0.0, self._last_eta - since)
            self._eta_lbl.setText(f"ETA: ~{_fmt(adjusted)}")
            self._eta_lbl.setStyleSheet(f"color: {_WARNING};")
