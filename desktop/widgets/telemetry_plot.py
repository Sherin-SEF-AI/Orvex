"""
desktop/widgets/telemetry_plot.py — Real-time IMU/GPS telemetry plots.

Layout:
  ┌──────────────────────────────────────────────────────────┐
  │  [Session label]   [Device selector]   [Load]           │
  ├──────────────────────────────────────────────────────────┤
  │  Tab: IMU | GPS Map                                     │
  │                                                         │
  │  IMU tab:                                               │
  │    Accel X/Y/Z  (PyQtGraph, colour-coded)               │
  │    Gyro  X/Y/Z                                          │
  │    Scrub bar (linked to frame browser timestamp)        │
  │                                                         │
  │  GPS Map tab:                                           │
  │    Folium map rendered in QWebEngineView                │
  └──────────────────────────────────────────────────────────┘

Uses PyQtGraph for IMU performance. GPS map uses folium + QWebEngineView.
Motion profile (stationary/high_motion) shown as shaded bands on accel plot.
"""
from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Any

import pyqtgraph as pg
from PyQt6.QtCore import Qt, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollBar,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.models import ExtractedSession, GPSSample, IMUSample
from core.session_manager import SessionManager

try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView as _QWebEngineView  # type: ignore[import]
    from PyQt6.QtWebEngineCore import QWebEngineSettings as _QWebEngineSettings  # type: ignore[import]
    _WEBENGINE_OK = True
except ImportError:
    _WEBENGINE_OK = False

# Colour palette (matches dark theme)
_COLORS = {
    "ax": "#e74c3c",   # red
    "ay": "#2ecc71",   # green
    "az": "#3498db",   # blue
    "gx": "#e67e22",   # orange
    "gy": "#9b59b6",   # purple
    "gz": "#1abc9c",   # teal
}

_BG    = "#1a1a2e"
_PANEL = "#16213e"
_ACCENT = "#0f3460"
_TEXT  = "#e0e0e0"
_MUTED = "#888888"

# Configure pyqtgraph global defaults once
pg.setConfigOption("background", _BG)
pg.setConfigOption("foreground", _TEXT)
pg.setConfigOption("antialias", True)


class TelemetryPlot(QWidget):
    """IMU/GPS telemetry visualisation panel.

    Emits scrub_changed(float) in seconds when the user moves the scrub bar,
    so the frame browser can sync to the nearest frame.
    """

    scrub_changed = pyqtSignal(float)   # seconds from recording start

    def __init__(self, session_manager: SessionManager, parent=None) -> None:
        super().__init__(parent)
        self._sm = session_manager
        self._session_id: str = ""
        self._imu_samples: list[IMUSample] = []
        self._duration_s: float = 0.0
        self._motion_regions: list[pg.LinearRegionItem] = []
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # --- Toolbar ---
        toolbar = QHBoxLayout()

        self._session_label = QLabel("No session selected")
        self._session_label.setStyleSheet(f"color: {_TEXT}; font-weight: bold;")
        toolbar.addWidget(self._session_label)

        toolbar.addStretch()

        toolbar.addWidget(QLabel("Device:"))
        self._device_combo = QComboBox()
        self._device_combo.setMinimumWidth(160)
        self._device_combo.setStyleSheet(_input_style())
        toolbar.addWidget(self._device_combo)

        self._load_btn = QPushButton("Load")
        self._load_btn.setEnabled(False)
        self._load_btn.setFixedHeight(28)
        self._load_btn.setStyleSheet(_btn_style())
        self._load_btn.clicked.connect(self._load_telemetry)
        toolbar.addWidget(self._load_btn)

        root.addLayout(toolbar)

        # --- Axis toggles ---
        toggle_row = QHBoxLayout()
        toggle_row.addWidget(QLabel("Show:"))
        self._toggles: dict[str, QCheckBox] = {}
        labels = {
            "ax": "Accel X", "ay": "Accel Y", "az": "Accel Z",
            "gx": "Gyro X",  "gy": "Gyro Y",  "gz": "Gyro Z",
        }
        for key, label in labels.items():
            cb = QCheckBox(label)
            cb.setChecked(True)
            cb.setStyleSheet(f"color: {_COLORS[key]}; font-size: 11px;")
            cb.stateChanged.connect(lambda state, k=key: self._toggle_axis(k))
            self._toggles[key] = cb
            toggle_row.addWidget(cb)
        toggle_row.addStretch()
        root.addLayout(toggle_row)

        # --- Tab widget: IMU | GPS Map ---
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(
            f"QTabWidget::pane {{ border: 1px solid {_ACCENT}; }}"
            f" QTabBar::tab {{ background: {_PANEL}; color: {_MUTED};"
            f" padding: 4px 12px; }}"
            f" QTabBar::tab:selected {{ background: {_ACCENT}; color: {_TEXT}; }}"
        )
        root.addWidget(self._tabs, stretch=1)

        # ----- IMU tab -----
        imu_tab = QWidget()
        imu_layout = QVBoxLayout(imu_tab)
        imu_layout.setContentsMargins(0, 4, 0, 0)
        imu_layout.setSpacing(4)

        # Plot splitter
        plot_split = QSplitter(Qt.Orientation.Vertical)

        self._accel_plot = pg.PlotWidget(title="Accelerometer  (m/s²)")
        self._accel_plot.setLabel("left", "m/s²")
        self._accel_plot.addLegend(offset=(10, 10))
        self._accel_plot.showGrid(x=True, y=True, alpha=0.15)
        self._accel_curves: dict[str, pg.PlotDataItem] = {}
        for key in ("ax", "ay", "az"):
            curve = self._accel_plot.plot(
                [], [], pen=pg.mkPen(_COLORS[key], width=1),
                name=labels[key],
            )
            self._accel_curves[key] = curve

        self._gyro_plot = pg.PlotWidget(title="Gyroscope  (rad/s)")
        self._gyro_plot.setLabel("left", "rad/s")
        self._gyro_plot.setXLink(self._accel_plot)
        self._gyro_plot.addLegend(offset=(10, 10))
        self._gyro_plot.showGrid(x=True, y=True, alpha=0.15)
        self._gyro_curves: dict[str, pg.PlotDataItem] = {}
        for key in ("gx", "gy", "gz"):
            curve = self._gyro_plot.plot(
                [], [], pen=pg.mkPen(_COLORS[key], width=1),
                name=labels[key],
            )
            self._gyro_curves[key] = curve

        plot_split.addWidget(self._accel_plot)
        plot_split.addWidget(self._gyro_plot)
        plot_split.setStretchFactor(0, 1)
        plot_split.setStretchFactor(1, 1)
        imu_layout.addWidget(plot_split, stretch=1)

        # Scrub bar
        scrub_row = QHBoxLayout()
        scrub_label = QLabel("Scrub:")
        scrub_label.setStyleSheet(f"color: {_MUTED}; font-size: 11px;")
        scrub_row.addWidget(scrub_label)

        self._scrub_bar = QScrollBar(Qt.Orientation.Horizontal)
        self._scrub_bar.setRange(0, 1000)
        self._scrub_bar.setValue(0)
        self._scrub_bar.setEnabled(False)
        self._scrub_bar.setStyleSheet(_scrollbar_style())
        self._scrub_bar.valueChanged.connect(self._on_scrub)
        scrub_row.addWidget(self._scrub_bar, stretch=1)

        self._scrub_time_label = QLabel("0.00 s")
        self._scrub_time_label.setFixedWidth(60)
        self._scrub_time_label.setStyleSheet(f"color: {_MUTED}; font-size: 11px;")
        scrub_row.addWidget(self._scrub_time_label)
        imu_layout.addLayout(scrub_row)

        # Scrub line on both plots
        pen = pg.mkPen("#e94560", width=1, style=Qt.PenStyle.DashLine)
        self._scrub_line_accel = pg.InfiniteLine(pos=0, angle=90, pen=pen, movable=False)
        self._scrub_line_gyro  = pg.InfiniteLine(pos=0, angle=90, pen=pen, movable=False)
        self._accel_plot.addItem(self._scrub_line_accel)
        self._gyro_plot.addItem(self._scrub_line_gyro)

        # Placeholder label
        self._placeholder = QLabel("Select a session and click Load to display telemetry.")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet(f"color: {_MUTED}; font-size: 13px;")
        imu_layout.addWidget(self._placeholder)

        self._plot_split = plot_split
        plot_split.setVisible(False)

        self._tabs.addTab(imu_tab, "IMU")

        # ----- GPS Map tab -----
        self._gps_map = GPSMapWidget(self)
        self._tabs.addTab(self._gps_map, "GPS Map")

    # ------------------------------------------------------------------
    # Session change
    # ------------------------------------------------------------------

    @pyqtSlot(str)
    def on_session_changed(self, session_id: str) -> None:
        self._session_id = session_id
        self._imu_samples = []
        self._duration_s = 0.0
        self._clear_plots()

        if not session_id:
            self._session_label.setText("No session selected")
            self._device_combo.clear()
            self._load_btn.setEnabled(False)
            self._scrub_bar.setEnabled(False)
            self._show_placeholder("Select a session and click Load to display telemetry.")
            self._gps_map.clear()
            return

        try:
            s = self._sm.get_session(session_id)
        except Exception:
            return

        self._session_label.setText(f"{s.name}  [{s.environment} · {s.location}]")

        self._device_combo.clear()
        if s.audit_results:
            for r in s.audit_results:
                self._device_combo.addItem(
                    f"{r.device_type.value} — {Path(r.file_path).name}",
                    userData=r.file_path,
                )
        else:
            for fp in s.files:
                self._device_combo.addItem(Path(fp).name, userData=fp)

        self._load_btn.setEnabled(self._device_combo.count() > 0)
        self._show_placeholder("Click Load to display telemetry for the selected device.")

        # Reload GPS map
        session_folder = self._sm.session_folder(session_id)
        for candidate in [
            session_folder / "extraction_gopro" / "gps.csv",
            session_folder / "extraction_insta360" / "gps.csv",
            session_folder / "extraction" / "gps.csv",
        ]:
            if candidate.exists():
                self._gps_map.load_from_csv(candidate)
                break
        else:
            self._gps_map.clear()

    # ------------------------------------------------------------------
    # Load telemetry from an extracted session CSV
    # ------------------------------------------------------------------

    def _load_telemetry(self) -> None:
        if not self._session_id:
            return

        session_folder = self._sm.session_folder(self._session_id)
        imu_csv = None
        for candidate in [
            session_folder / "extraction_gopro" / "imu0" / "data.csv",
            session_folder / "extraction_insta360" / "imu0" / "data.csv",
            session_folder / "extraction" / "imu0" / "data.csv",
        ]:
            if candidate.exists():
                imu_csv = candidate
                break

        if imu_csv is None:
            self._show_placeholder(
                "No extracted IMU data found for this session.\n"
                "Run extraction first (Extract tab)."
            )
            return

        samples = _load_euroc_imu(imu_csv)
        if not samples:
            self._show_placeholder(f"IMU CSV is empty:\n{imu_csv}")
            return

        self._imu_samples = samples
        self._duration_s = (
            (samples[-1].timestamp_ns - samples[0].timestamp_ns) / 1e9
            if len(samples) > 1 else 0.0
        )
        self._scrub_bar.setEnabled(True)
        self._plot_data(samples)

    def set_extracted_session(self, extracted: ExtractedSession) -> None:
        """Load motion profile and GPS from an ExtractedSession result."""
        motion_profile = extracted.stats.get("motion_profile", [])
        if motion_profile and self._imu_samples:
            self._apply_motion_profile(motion_profile)

        if extracted.gps_samples:
            self._gps_map.load_from_samples(extracted.gps_samples)

    def _plot_data(self, samples: list[IMUSample]) -> None:
        if not samples:
            return

        t0 = samples[0].timestamp_ns
        ts = [(s.timestamp_ns - t0) / 1e9 for s in samples]

        self._accel_curves["ax"].setData(ts, [s.accel_x for s in samples])
        self._accel_curves["ay"].setData(ts, [s.accel_y for s in samples])
        self._accel_curves["az"].setData(ts, [s.accel_z for s in samples])
        self._gyro_curves["gx"].setData(ts, [s.gyro_x for s in samples])
        self._gyro_curves["gy"].setData(ts, [s.gyro_y for s in samples])
        self._gyro_curves["gz"].setData(ts, [s.gyro_z for s in samples])

        for key in self._toggles:
            self._apply_toggle(key)

        self._placeholder.setVisible(False)
        self._plot_split.setVisible(True)

    def _apply_motion_profile(self, profile: list[tuple[int, str]]) -> None:
        """Draw shaded bands on accel plot for stationary/high_motion windows."""
        for region in self._motion_regions:
            self._accel_plot.removeItem(region)
        self._motion_regions.clear()

        if not self._imu_samples:
            return
        t0 = self._imu_samples[0].timestamp_ns

        for i, (ts_ns, label) in enumerate(profile):
            if label == "normal":
                continue
            t_start = (ts_ns - t0) / 1e9
            # End of window = start of next entry or start + 1s
            if i + 1 < len(profile):
                t_end = (profile[i + 1][0] - t0) / 1e9
            else:
                t_end = t_start + 1.0

            if label == "stationary":
                color = pg.mkColor(0, 77, 204, 20)   # blue tint
            else:  # high_motion
                color = pg.mkColor(230, 26, 26, 20)  # red tint

            region = pg.LinearRegionItem(
                values=(t_start, t_end),
                brush=pg.mkBrush(color),
                pen=pg.mkPen(None),
                movable=False,
            )
            self._accel_plot.addItem(region)
            self._motion_regions.append(region)

    def _clear_plots(self) -> None:
        for c in list(self._accel_curves.values()) + list(self._gyro_curves.values()):
            c.setData([], [])
        for region in self._motion_regions:
            self._accel_plot.removeItem(region)
        self._motion_regions.clear()
        self._scrub_line_accel.setValue(0)
        self._scrub_line_gyro.setValue(0)

    # ------------------------------------------------------------------
    # Axis toggles
    # ------------------------------------------------------------------

    def _toggle_axis(self, key: str) -> None:
        self._apply_toggle(key)

    def _apply_toggle(self, key: str) -> None:
        visible = self._toggles[key].isChecked()
        curves = {**self._accel_curves, **self._gyro_curves}
        if key in curves:
            curves[key].setVisible(visible)

    # ------------------------------------------------------------------
    # Scrub bar
    # ------------------------------------------------------------------

    @pyqtSlot(int)
    def _on_scrub(self, value: int) -> None:
        if self._duration_s <= 0:
            return
        t = value / 1000.0 * self._duration_s
        self._scrub_time_label.setText(f"{t:.2f} s")
        self._scrub_line_accel.setValue(t)
        self._scrub_line_gyro.setValue(t)
        self.scrub_changed.emit(t)

    def set_scrub_position(self, seconds: float) -> None:
        if self._duration_s <= 0:
            return
        val = int(seconds / self._duration_s * 1000)
        self._scrub_bar.blockSignals(True)
        self._scrub_bar.setValue(max(0, min(1000, val)))
        self._scrub_bar.blockSignals(False)
        self._scrub_line_accel.setValue(seconds)
        self._scrub_line_gyro.setValue(seconds)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _show_placeholder(self, msg: str) -> None:
        self._placeholder.setText(msg)
        self._placeholder.setVisible(True)
        self._plot_split.setVisible(False)


# ---------------------------------------------------------------------------
# GPS Map widget
# ---------------------------------------------------------------------------

class GPSMapWidget(QWidget):
    """Renders a GPS track on an interactive folium map via QWebEngineView.

    Falls back to a plain text placeholder if folium or PyQt6-WebEngine
    are not installed.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._tmp_file: str | None = None
        self._has_webengine = _WEBENGINE_OK
        self._view = None
        self._build_ui()
        if self._has_webengine and self._view:
            settings = self._view.settings()
            settings.setAttribute(
                _QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True
            )
            settings.setAttribute(
                _QWebEngineSettings.WebAttribute.JavascriptEnabled, True
            )

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if _WEBENGINE_OK:
            self._view = _QWebEngineView()
            layout.addWidget(self._view)
        else:
            self._label = QLabel(
                "GPS map requires PyQt6-WebEngine.\n"
                "Install: pip install PyQt6-WebEngine"
            )
            self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._label.setStyleSheet(f"color: {_MUTED}; font-size: 12px;")
            layout.addWidget(self._label)

    def clear(self) -> None:
        if self._has_webengine and self._view:
            self._view.setHtml(
                f"<body style='background:{_BG};color:{_MUTED};"
                "font-family:sans-serif;display:flex;align-items:center;"
                "justify-content:center;height:100vh;margin:0;'>"
                "<p>No GPS data available for this session.</p></body>"
            )

    def load_from_csv(self, gps_csv: Path) -> None:
        """Load GPS from a gps.csv file (written by extractor_gopro)."""
        samples: list[GPSSample] = []
        try:
            with open(gps_csv, newline="") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(",")
                    if len(parts) < 3:
                        continue
                    from core.models import GPSSample as _GPS
                    samples.append(_GPS(
                        timestamp_ns=int(parts[0]),
                        latitude=float(parts[1]),
                        longitude=float(parts[2]),
                        altitude_m=float(parts[3]) if len(parts) > 3 else 0.0,
                        speed_mps=float(parts[4]) if len(parts) > 4 else 0.0,
                        fix_type=int(parts[5]) if len(parts) > 5 else 0,
                    ))
        except Exception:
            pass
        self.load_from_samples(samples)

    def load_from_samples(self, samples: list[GPSSample]) -> None:
        if not samples:
            self.clear()
            return

        # Filter out zero-fix samples
        valid = [s for s in samples if s.fix_type > 0 and s.latitude != 0.0]
        if not valid:
            valid = samples

        if not self._has_webengine:
            return

        html = self._build_map_html(valid)
        if not html:
            return

        # Write to a persistent temp file (keep reference so it isn't deleted)
        try:
            if self._tmp_file and os.path.exists(self._tmp_file):
                os.unlink(self._tmp_file)
        except Exception:
            pass

        try:
            fd, path = tempfile.mkstemp(suffix=".html", prefix="roverdatakit_gps_")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(html)
            self._tmp_file = path
            self._view.load(QUrl.fromLocalFile(path))
        except Exception as exc:
            # Fallback: inject HTML directly (works for smaller maps)
            self._view.setHtml(html, QUrl("about:blank"))

    def _build_map_html(self, samples: list[GPSSample]) -> str:
        try:
            import folium  # type: ignore[import]
        except ImportError:
            return (
                "<html><body style='background:#1a1a2e;color:#888;"
                "font-family:sans-serif;display:flex;align-items:center;"
                "justify-content:center;height:100vh;margin:0;'>"
                "<p>folium not installed. Run: pip install folium</p>"
                "</body></html>"
            )

        center_lat = sum(s.latitude for s in samples) / len(samples)
        center_lon = sum(s.longitude for s in samples) / len(samples)

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=15,
            tiles="CartoDB dark_matter",
        )

        coords = [(s.latitude, s.longitude) for s in samples]
        folium.PolyLine(coords, color="#e94560", weight=3, opacity=0.9).add_to(m)

        # Start/end markers
        folium.CircleMarker(
            location=coords[0], radius=8,
            color="#27ae60", fill=True, fill_color="#27ae60",
            tooltip="Start",
        ).add_to(m)
        folium.CircleMarker(
            location=coords[-1], radius=8,
            color="#e94560", fill=True, fill_color="#e94560",
            tooltip="End",
        ).add_to(m)

        # get_root().render() returns a complete standalone HTML page
        return m.get_root().render()


# ---------------------------------------------------------------------------
# EuRoC IMU CSV loader
# ---------------------------------------------------------------------------

def _load_euroc_imu(csv_path: Path) -> list[IMUSample]:
    """Parse a EuRoC imu0/data.csv and return IMUSample list."""
    samples: list[IMUSample] = []
    try:
        with open(csv_path, newline="") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) < 7:
                    continue
                ts, gx, gy, gz, ax, ay, az = (float(p) for p in parts[:7])
                samples.append(IMUSample(
                    timestamp_ns=int(ts),
                    accel_x=ax, accel_y=ay, accel_z=az,
                    gyro_x=gx, gyro_y=gy, gyro_z=gz,
                ))
    except Exception:
        pass
    return samples


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def _btn_style() -> str:
    return (
        f"QPushButton {{ background: {_ACCENT}; color: {_TEXT}; border: none;"
        f" border-radius: 4px; padding: 3px 12px; font-size: 11px; }}"
        f" QPushButton:hover {{ background: #e94560; }}"
        f" QPushButton:disabled {{ background: #333355; color: {_MUTED}; }}"
    )


def _input_style() -> str:
    return (
        f"QComboBox {{ background: #1a1a2e; color: {_TEXT};"
        f" border: 1px solid {_ACCENT}; border-radius: 3px; padding: 2px 4px; }}"
    )


def _scrollbar_style() -> str:
    return (
        f"QScrollBar:horizontal {{ background: {_PANEL}; height: 14px; border: none; }}"
        f" QScrollBar::handle:horizontal {{ background: {_ACCENT};"
        f" border-radius: 4px; min-width: 20px; }}"
        f" QScrollBar::handle:horizontal:hover {{ background: #e94560; }}"
        f" QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal"
        f" {{ width: 0px; }}"
    )
