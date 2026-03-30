"""
desktop/widgets/active_learning_widget.py — Active learning frame selection.

Scores frames by uncertainty + diversity, selects the best subset to label.
Requires auto-label to have been run first.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
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

from desktop.theme import BG, PANEL, ACCENT, HI, TEXT, MUTED, SUCCESS, WARNING, BORDER, CARD
from desktop.workers import ActiveLearningWorker

# Alias
HIGHLIGHT = HI


class ActiveLearningWidget(QWidget):
    """UI for uncertainty + diversity-based frame selection for labeling."""

    def __init__(self, session_manager, parent=None) -> None:
        super().__init__(parent)
        self._sm = session_manager
        self._session_id: str = ""
        self._worker: ActiveLearningWorker | None = None
        self._annotations: list = []
        self._selected_frames: list[str] = []
        self._build_ui()

    def set_annotations(self, annotations: list) -> None:
        """Called by autolabel_widget after annotation completes."""
        self._annotations = annotations
        has_ann = bool(annotations)
        self._run_btn.setEnabled(has_ann and bool(self._session_id))
        if has_ann:
            self._prereq_label.setText(f"✓ {len(annotations)} annotated frames available.")
            self._prereq_label.setStyleSheet(f"color:#27ae60;")
        else:
            self._prereq_label.setText("Run Auto-Label first to generate confidence scores.")
            self._prereq_label.setStyleSheet(f"color:{WARNING};")

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

        self._prereq_label = QLabel("Run Auto-Label first to generate confidence scores.")
        self._prereq_label.setStyleSheet(f"color:{WARNING}; font-size:11px;")
        self._prereq_label.setWordWrap(True)
        cfg_layout.addWidget(self._prereq_label)

        method_group = QGroupBox("Uncertainty method")
        mg = QVBoxLayout(method_group)
        self._method_combo = QComboBox()
        self._method_combo.addItems(["entropy", "margin", "least_confidence"])
        mg.addWidget(self._method_combo)
        cfg_layout.addWidget(method_group)

        sel_group = QGroupBox("Selection")
        sg = QVBoxLayout(sel_group)
        sg.addWidget(QLabel("Frames to select:"))
        self._n_spin = QSpinBox()
        self._n_spin.setRange(1, 10000)
        self._n_spin.setValue(100)
        sg.addWidget(self._n_spin)
        sg.addWidget(QLabel("Uncertainty weight:"))
        self._unc_slider = _make_slider(0, 100, 60)
        self._unc_val = QLabel("0.60")
        self._unc_slider.valueChanged.connect(
            lambda v: self._unc_val.setText(f"{v/100:.2f}")
        )
        sg.addWidget(self._unc_slider)
        sg.addWidget(self._unc_val)
        sg.addWidget(QLabel("Diversity weight:"))
        self._div_slider = _make_slider(0, 100, 40)
        self._div_val = QLabel("0.40")
        self._div_slider.valueChanged.connect(
            lambda v: self._div_val.setText(f"{v/100:.2f}")
        )
        sg.addWidget(self._div_slider)
        sg.addWidget(self._div_val)
        cfg_layout.addWidget(sel_group)

        self._run_btn = QPushButton("Score Frames")
        self._run_btn.setEnabled(False)
        self._run_btn.setToolTip("Run Auto-Label first to generate confidence scores.")
        self._run_btn.setObjectName("DangerBtn")
        self._run_btn.clicked.connect(self._run)
        cfg_layout.addWidget(self._run_btn)

        self._export_btn = QPushButton("Export selected paths (.txt)")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export_paths)
        cfg_layout.addWidget(self._export_btn)

        cfg_layout.addStretch()
        splitter.addWidget(cfg_w)

        # ── Right panel ────────────────────────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 8, 8, 8)

        # Scatter plot placeholder
        scatter_group = QGroupBox("Frame scores (X=Diversity, Y=Uncertainty)")
        scatter_layout = QVBoxLayout(scatter_group)
        self._scatter_widget = _ScatterPlaceholder()
        scatter_layout.addWidget(self._scatter_widget)
        rl.addWidget(scatter_group, stretch=2)

        # Budget panel
        self._budget_label = QLabel("Select frames to see annotation budget estimate.")
        self._budget_label.setStyleSheet(
            f"background:{PANEL}; color:{TEXT}; padding:8px; border-radius:4px;"
        )
        self._budget_label.setWordWrap(True)
        rl.addWidget(self._budget_label)

        # Thumbnail grid of selected frames
        thumb_group = QGroupBox("Top selected frames (thumbnails)")
        thumb_scroll = QScrollArea()
        thumb_scroll.setWidgetResizable(True)
        self._thumb_container = QWidget()
        self._thumb_grid = QHBoxLayout(self._thumb_container)
        thumb_scroll.setWidget(self._thumb_container)
        tg_layout = QVBoxLayout(thumb_group)
        tg_layout.addWidget(thumb_scroll)
        rl.addWidget(thumb_group, stretch=1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([260, 740])

    @pyqtSlot(str)
    def on_session_changed(self, session_id: str) -> None:
        self._session_id = session_id
        self._run_btn.setEnabled(bool(session_id) and bool(self._annotations))

    def _run(self) -> None:
        if not self._annotations or (self._worker and self._worker.isRunning()):
            return
        self._worker = ActiveLearningWorker(
            session_id=self._session_id,
            sm=self._sm,
            annotations=self._annotations,
            method=self._method_combo.currentText(),
            n_frames=self._n_spin.value(),
            uncertainty_weight=self._unc_slider.value() / 100.0,
            diversity_weight=self._div_slider.value() / 100.0,
        )
        self._worker.result.connect(self._on_result)
        self._worker.error.connect(lambda e: QMessageBox.critical(self, "Scoring Error", e))
        self._run_btn.setEnabled(False)
        self._worker.start()

    @pyqtSlot(object)
    def _on_result(self, payload: Any) -> None:
        self._run_btn.setEnabled(True)
        self._selected_frames = payload["selected_frames"]
        budget = payload["budget"]
        unc_scores = payload["uncertainty_scores"]
        div_scores = payload["diversity_scores"]

        self._budget_label.setText(
            f"Label {budget['selected_frames']} frames  →  "
            f"~{budget['estimated_annotation_hours']}h  →  "
            f"~${budget['estimated_cost_usd']}  "
            f"({budget['coverage_percent']:.1f}% coverage)"
        )
        self._scatter_widget.set_data(unc_scores, div_scores, self._selected_frames)
        self._populate_thumbnails(self._selected_frames[:12])
        self._export_btn.setEnabled(True)

    def _populate_thumbnails(self, paths: list[str]) -> None:
        # Clear existing
        while self._thumb_grid.count():
            item = self._thumb_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for fp in paths:
            lbl = QLabel()
            lbl.setFixedSize(100, 75)
            pix = QPixmap(fp)
            if not pix.isNull():
                lbl.setPixmap(
                    pix.scaled(100, 75,
                               Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
                )
            self._thumb_grid.addWidget(lbl)

    def _export_paths(self) -> None:
        if not self._selected_frames:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Selected Frames", "selected_frames.txt", "Text files (*.txt)"
        )
        if path:
            Path(path).write_text("\n".join(self._selected_frames))


class _ScatterPlaceholder(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._label = QLabel("Scatter plot will appear after scoring.")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setMinimumHeight(180)
        layout.addWidget(self._label)
        self._plot = None

    def set_data(self, unc_scores: list, div_scores: list, selected: list[str]) -> None:
        try:
            import pyqtgraph as pg
            import numpy as np

            if self._plot is None:
                self._plot = pg.PlotWidget(background=BG)
                self._plot.setLabel("left", "Uncertainty")
                self._plot.setLabel("bottom", "Diversity")
                layout = self.layout()
                layout.removeWidget(self._label)
                self._label.hide()
                layout.addWidget(self._plot)

            unc = np.array([s.score for s in unc_scores])
            div = np.array(div_scores)
            selected_set = set(selected)

            # All points
            self._plot.clear()
            scatter = pg.ScatterPlotItem(
                x=div, y=unc, pen=None,
                brush=pg.mkBrush(100, 100, 200, 150), size=4,
            )
            self._plot.addItem(scatter)

            # Selected frames (larger, highlighted)
            sel_idx = [i for i, s in enumerate(unc_scores)
                       if s.frame_path in selected_set]
            if sel_idx:
                sel_scatter = pg.ScatterPlotItem(
                    x=div[sel_idx], y=unc[sel_idx], pen=None,
                    brush=pg.mkBrush(233, 69, 96, 220), size=8,
                )
                self._plot.addItem(sel_scatter)
        except Exception:
            self._label.setText(
                f"Scatter plot requires pyqtgraph.\n"
                f"Selected {len(selected)} / {len(unc_scores)} frames."
            )
            self._label.show()


def _make_slider(min_val: int, max_val: int, default: int) -> QSlider:
    s = QSlider(Qt.Orientation.Horizontal)
    s.setRange(min_val, max_val)
    s.setValue(default)
    return s
