"""
desktop/widgets/analytics_widget.py — Dataset analytics dashboard.

Shows class distribution, lighting stats, GPS coverage map,
and temporal data completeness.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from desktop.theme import BG, PANEL, ACCENT, HI, TEXT, MUTED, SUCCESS, WARNING, BORDER, CARD
from desktop.workers import AnalyticsWorker

# Alias
HIGHLIGHT = HI


class AnalyticsWidget(QWidget):
    """Dashboard analytics widget for scene diversity and coverage."""

    def __init__(self, session_manager, parent=None) -> None:
        super().__init__(parent)
        self._sm = session_manager
        self._session_id: str = ""
        self._worker: AnalyticsWorker | None = None
        self._annotations: list = []
        self._gps_samples: list = []
        self._last_result: dict = {}
        self._build_ui()

    def set_annotations(self, annotations: list) -> None:
        self._annotations = annotations

    def set_gps_samples(self, gps_samples: list) -> None:
        self._gps_samples = gps_samples

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        # Top bar
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Analytics Dashboard"))
        self._run_btn = QPushButton("Compute Analytics")
        self._run_btn.setEnabled(False)
        self._run_btn.setObjectName("DangerBtn")
        self._run_btn.clicked.connect(self._run)
        top_row.addStretch()
        top_row.addWidget(self._run_btn)

        self._report_btn = QPushButton("Generate HTML Report")
        self._report_btn.setEnabled(False)
        self._report_btn.clicked.connect(self._generate_report)
        top_row.addWidget(self._report_btn)
        root.addLayout(top_row)

        self._status_label = QLabel("Ready.")
        root.addWidget(self._status_label)

        # Tab widget for analytics sections
        self._tabs = QTabWidget()

        # Tab 1: Class distribution
        self._cls_tab = _ClassDistributionTab()
        self._tabs.addTab(self._cls_tab, "Class Distribution")

        # Tab 2: Scene diversity
        self._scene_tab = _SceneDiversityTab()
        self._tabs.addTab(self._scene_tab, "Scene Diversity")

        # Tab 3: GPS coverage map
        self._map_tab = _CoverageMapTab()
        self._tabs.addTab(self._map_tab, "GPS Coverage")

        # Tab 4: Temporal
        self._temporal_tab = _TemporalTab()
        self._tabs.addTab(self._temporal_tab, "Temporal")

        root.addWidget(self._tabs)

    @pyqtSlot(str)
    def on_session_changed(self, session_id: str) -> None:
        self._session_id = session_id
        self._run_btn.setEnabled(bool(session_id))

    def _run(self) -> None:
        if not self._session_id or (self._worker and self._worker.isRunning()):
            return
        self._worker = AnalyticsWorker(
            session_id=self._session_id,
            sm=self._sm,
            annotations=self._annotations,
            gps_samples=self._gps_samples,
        )
        self._worker.status.connect(lambda m: self._status_label.setText(m))
        self._worker.result.connect(self._on_result)
        self._worker.error.connect(lambda e: QMessageBox.critical(self, "Analytics Error", e))
        self._run_btn.setEnabled(False)
        self._worker.start()

    @pyqtSlot(object)
    def _on_result(self, payload: Any) -> None:
        self._last_result = payload
        self._run_btn.setEnabled(True)
        self._report_btn.setEnabled(True)

        diversity = payload.get("diversity")
        coverage = payload.get("coverage")
        cls_dist = payload.get("class_distribution", {})

        if diversity:
            self._scene_tab.update(diversity)
        if coverage:
            self._map_tab.update(coverage)
        if cls_dist:
            self._cls_tab.update(cls_dist)

        self._status_label.setText("Analytics complete.")

    def _generate_report(self) -> None:
        if not self._last_result:
            return
        save_dir = QFileDialog.getExistingDirectory(self, "Save report to…")
        if not save_dir:
            return
        try:
            from core.road_analytics import generate_dataset_report
            path = generate_dataset_report(
                session_ids=[self._session_id],
                annotations=self._annotations,
                gps_samples=self._gps_samples,
                output_dir=save_dir,
            )
            self._status_label.setText(f"Report saved: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Report Error", str(exc))


class _ClassDistributionTab(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._label = QLabel("No data.")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._label)
        self._plot = None

    def update(self, cls_dist: dict) -> None:
        per_class = cls_dist.get("per_class", {})
        if not per_class:
            return
        try:
            import pyqtgraph as pg
            import numpy as np

            if self._plot is None:
                self._plot = pg.PlotWidget(background=BG)
                self._plot.setLabel("left", "Count")
                self._plot.setLabel("bottom", "Class")
                layout = self.layout()
                layout.removeWidget(self._label)
                self._label.hide()
                layout.addWidget(self._plot)

            names = list(per_class.keys())
            counts = [per_class[n]["count"] for n in names]
            x = np.arange(len(names))
            bar = pg.BarGraphItem(x=x, height=counts, width=0.7,
                                  brush=pg.mkBrush(233, 69, 96, 200))
            self._plot.clear()
            self._plot.addItem(bar)
            ax = self._plot.getAxis("bottom")
            ax.setTicks([list(zip(x, names))])
        except Exception:
            self._label.setText(
                "\n".join(
                    f"{n}: {d['count']} ({d['percent']:.1f}%)"
                    for n, d in sorted(per_class.items(),
                                       key=lambda x: x[1]["count"], reverse=True)
                )
            )
            self._label.show()


class _SceneDiversityTab(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._text = QTextEdit()
        self._text.setReadOnly(True)
        layout.addWidget(self._text)

    def update(self, report: Any) -> None:
        ld = report.lighting_distribution
        bs = report.brightness_stats
        st = report.estimated_scene_types
        text = (
            f"Lighting distribution:\n"
            f"  Bright: {ld.get('bright', 0):.1f}%\n"
            f"  Normal: {ld.get('normal', 0):.1f}%\n"
            f"  Dark:   {ld.get('dark', 0):.1f}%\n\n"
            f"Brightness stats:\n"
            f"  Mean: {bs.get('mean', 0):.1f}  Std: {bs.get('std', 0):.1f}\n"
            f"  Min:  {bs.get('min', 0):.1f}  Max: {bs.get('max', 0):.1f}\n\n"
            f"Scene types:\n"
            + "\n".join(f"  {k}: {v:.1f}%" for k, v in st.items())
        )
        self._text.setPlainText(text)


class _CoverageMapTab(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._stats_label = QLabel("No GPS data.")
        layout.addWidget(self._stats_label)
        self._map_view = QWebEngineView()
        layout.addWidget(self._map_view, stretch=1)

    def update(self, report: Any) -> None:
        self._stats_label.setText(
            f"Distance: {report.total_distance_km:.2f} km  |  "
            f"Grid cells: {report.unique_grid_cells}  |  "
            f"Avg speed: {report.avg_speed_mps:.1f} m/s  |  "
            f"Stationary: {report.stationary_time_percent:.1f}%"
        )
        map_path = report.coverage_map_path
        if Path(map_path).exists():
            self._map_view.load(
                __import__("PyQt6.QtCore", fromlist=["QUrl"]).QUrl.fromLocalFile(map_path)
            )


class _TemporalTab(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._text = QTextEdit()
        self._text.setReadOnly(True)
        layout.addWidget(self._text)

    def set_temporal(self, temporal: dict) -> None:
        gaps = temporal.get("frame_gaps", [])
        self._text.setPlainText(
            f"Frame gaps > 0.5s: {len(gaps)}\n"
            f"Total gap time: {temporal.get('total_gap_time_seconds', 0):.2f}s\n"
            f"Data completeness: {temporal.get('data_completeness_percent', 100):.1f}%\n\n"
            + ("\n".join(f"  {s} → {e} ({g:.2f}s)" for s, e, g in gaps[:20]) if gaps else "No gaps detected.")
        )
