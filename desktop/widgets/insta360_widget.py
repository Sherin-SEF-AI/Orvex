"""
desktop/widgets/insta360_widget.py — Insta360 X4 360° processing widget.

Handles full pipeline: paired INSV files -> dual fisheye stitching ->
equirectangular -> 4 perspective views -> frame extraction -> GPS/IMU -> EuRoC dataset.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QColor, QImage, QPixmap, QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from desktop.theme import (
    BG, PANEL, ACCENT, HI, TEXT, MUTED, SUCCESS, WARNING, BORDER, HOVER, CARD
)
from desktop.workers import Insta360Worker

HIGHLIGHT = HI


# ---------------------------------------------------------------------------
# Pipeline stage status widget
# ---------------------------------------------------------------------------

class _PipelineStageWidget(QWidget):
    """Displays 5 pipeline stages as labeled rows with live status icons."""

    STAGES = ["Telemetry", "Stitching", "Perspective", "Frames", "Dataset"]
    STATUS_PENDING = "⏸ waiting"
    STATUS_RUNNING = "⏳"
    STATUS_DONE    = "✅ done"
    STATUS_FAILED  = "✕ failed"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._labels: dict[str, QLabel] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        for stage in self.STAGES:
            row = QHBoxLayout()
            name_lbl = QLabel(stage)
            name_lbl.setStyleSheet(f"color: {TEXT}; font-size: 11px;")
            name_lbl.setFixedWidth(90)
            status_lbl = QLabel(self.STATUS_PENDING)
            status_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
            row.addWidget(name_lbl)
            row.addWidget(status_lbl)
            row.addStretch()
            self._labels[stage] = status_lbl
            layout.addLayout(row)

    def set_stage_status(self, stage: str, status: str) -> None:
        """Update the status text and color for a named stage."""
        lbl = self._labels.get(stage)
        if lbl is None:
            return
        lbl.setText(status)
        if "✅" in status:
            color = SUCCESS
        elif "⏳" in status:
            color = WARNING
        elif "✕" in status:
            color = HI
        else:
            color = MUTED
        lbl.setStyleSheet(f"color: {color}; font-size: 11px;")

    def reset(self) -> None:
        """Reset all stages back to pending."""
        for lbl in self._labels.values():
            lbl.setText(self.STATUS_PENDING)
            lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class Insta360Widget(QWidget):
    """UI for Insta360 X4 dual fisheye -> 4-perspective-view pipeline."""

    def __init__(self, session_manager, parent=None) -> None:
        super().__init__(parent)
        self._sm = session_manager
        self._worker: Insta360Worker | None = None
        self._insv_pairs: list = []
        self._selected_pair_idx: int = -1
        # 4-view preview state
        self._view_frame_paths: dict[str, list[str]] = {}
        self._preview_index: int = 0
        self._playback_timer = QTimer()
        self._playback_timer.timeout.connect(self._advance_preview)
        # ETA countdown state
        self._eta_value: float = 0.0
        self._eta_received_at: float = 0.0
        self._eta_timer = QTimer()
        self._eta_timer.setInterval(500)
        self._eta_timer.timeout.connect(self._tick_eta)
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Build three panels
        cfg_w = self._build_config_panel()
        tabs = self._build_center_tabs()
        right_w = self._build_right_panel()

        splitter.addWidget(cfg_w)
        splitter.addWidget(tabs)
        splitter.addWidget(right_w)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([300, 700, 260])

        root.addWidget(splitter)

    # ── Left config panel ──────────────────────────────────────────────

    def _build_config_panel(self) -> QWidget:
        # Wrap in scroll area so content doesn't compress
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(260)
        scroll.setMaximumWidth(380)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            f"QScrollArea {{ background: {PANEL}; border: none; }}"
            f"QScrollArea > QWidget > QWidget {{ background: {PANEL}; }}"
        )

        cfg_w = QWidget()
        cfg_layout = QVBoxLayout(cfg_w)
        cfg_layout.setContentsMargins(8, 8, 8, 8)
        cfg_layout.setSpacing(4)

        lbl_style = f"color: {MUTED}; font-size: 10px; margin-top: 2px;"

        # ── Source group ──────────────────────────────────────────────
        src_group = QGroupBox("Source")
        sg = QVBoxLayout(src_group)
        sg.setSpacing(4)
        folder_row = QHBoxLayout()
        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText("Folder with .insv files…")
        self._browse_btn = QPushButton("Browse…")
        self._browse_btn.setFixedWidth(65)
        self._browse_btn.clicked.connect(self._browse_folder)
        folder_row.addWidget(self._folder_edit, stretch=1)
        folder_row.addWidget(self._browse_btn)
        sg.addLayout(folder_row)
        self._pair_list = QListWidget()
        self._pair_list.setMaximumHeight(90)
        self._pair_list.currentRowChanged.connect(self._on_pair_selected)
        sg.addWidget(self._pair_list)
        self._scan_btn = QPushButton("Scan Folder")
        self._scan_btn.clicked.connect(self._scan_folder)
        sg.addWidget(self._scan_btn)
        cfg_layout.addWidget(src_group)

        # ── Stitching group ───────────────────────────────────────────
        stitch_group = QGroupBox("Stitching")
        stg = QVBoxLayout(stitch_group)
        stg.setSpacing(3)
        lbl = QLabel("Resolution")
        lbl.setStyleSheet(lbl_style)
        stg.addWidget(lbl)
        self._res_combo = QComboBox()
        self._res_combo.addItems(["7680×3840 (8K)", "3840×1920 (4K)", "2560×1280 (2.5K)"])
        stg.addWidget(self._res_combo)
        lbl = QLabel("Fisheye FOV")
        lbl.setStyleSheet(lbl_style)
        stg.addWidget(lbl)
        self._fov_spin = QDoubleSpinBox()
        self._fov_spin.setRange(180.0, 220.0)
        self._fov_spin.setValue(210.0)
        self._fov_spin.setSingleStep(1.0)
        self._fov_spin.setSuffix(" °")
        stg.addWidget(self._fov_spin)
        crf_lbl = QLabel("CRF (quality)")
        crf_lbl.setStyleSheet(lbl_style)
        stg.addWidget(crf_lbl)
        crf_row = QHBoxLayout()
        self._stitch_crf_slider = QSlider(Qt.Orientation.Horizontal)
        self._stitch_crf_slider.setRange(0, 30)
        self._stitch_crf_slider.setValue(10)
        self._stitch_crf_label = QLabel("10")
        self._stitch_crf_label.setFixedWidth(24)
        self._stitch_crf_label.setStyleSheet(f"color: {TEXT}; font-size: 11px;")
        self._stitch_crf_slider.valueChanged.connect(
            lambda v: self._stitch_crf_label.setText(str(v))
        )
        crf_row.addWidget(self._stitch_crf_slider, stretch=1)
        crf_row.addWidget(self._stitch_crf_label)
        stg.addLayout(crf_row)
        cfg_layout.addWidget(stitch_group)

        # ── Perspective views group ───────────────────────────────────
        persp_group = QGroupBox("Perspective Views")
        pg = QVBoxLayout(persp_group)
        pg.setSpacing(3)
        views_row = QHBoxLayout()
        self._view_front = QCheckBox("Front")
        self._view_right = QCheckBox("Right")
        self._view_rear  = QCheckBox("Rear")
        self._view_left  = QCheckBox("Left")
        for cb in (self._view_front, self._view_right, self._view_rear, self._view_left):
            cb.setChecked(True)
            cb.stateChanged.connect(self._update_disk_estimate)
            views_row.addWidget(cb)
        pg.addLayout(views_row)
        lbl = QLabel("Resolution")
        lbl.setStyleSheet(lbl_style)
        pg.addWidget(lbl)
        self._persp_res_combo = QComboBox()
        self._persp_res_combo.addItems(["2160×2160", "1440×1440", "1080×1080"])
        pg.addWidget(self._persp_res_combo)
        fov_row = QHBoxLayout()
        fov_row.setSpacing(6)
        h_col = QVBoxLayout()
        lbl = QLabel("H-FOV")
        lbl.setStyleSheet(lbl_style)
        h_col.addWidget(lbl)
        self._h_fov_spin = QDoubleSpinBox()
        self._h_fov_spin.setRange(60.0, 150.0)
        self._h_fov_spin.setValue(110.0)
        self._h_fov_spin.setSingleStep(5.0)
        self._h_fov_spin.setSuffix("°")
        h_col.addWidget(self._h_fov_spin)
        fov_row.addLayout(h_col)
        v_col = QVBoxLayout()
        lbl = QLabel("V-FOV")
        lbl.setStyleSheet(lbl_style)
        v_col.addWidget(lbl)
        self._v_fov_spin = QDoubleSpinBox()
        self._v_fov_spin.setRange(60.0, 150.0)
        self._v_fov_spin.setValue(110.0)
        self._v_fov_spin.setSingleStep(5.0)
        self._v_fov_spin.setSuffix("°")
        v_col.addWidget(self._v_fov_spin)
        fov_row.addLayout(v_col)
        pg.addLayout(fov_row)
        corr_row = QHBoxLayout()
        corr_row.setSpacing(6)
        p_col = QVBoxLayout()
        lbl = QLabel("Pitch")
        lbl.setStyleSheet(lbl_style)
        p_col.addWidget(lbl)
        self._pitch_spin = QDoubleSpinBox()
        self._pitch_spin.setRange(-45.0, 45.0)
        self._pitch_spin.setValue(0.0)
        self._pitch_spin.setSingleStep(0.5)
        self._pitch_spin.setSuffix("°")
        p_col.addWidget(self._pitch_spin)
        corr_row.addLayout(p_col)
        r_col = QVBoxLayout()
        lbl = QLabel("Roll")
        lbl.setStyleSheet(lbl_style)
        r_col.addWidget(lbl)
        self._roll_spin = QDoubleSpinBox()
        self._roll_spin.setRange(-45.0, 45.0)
        self._roll_spin.setValue(0.0)
        self._roll_spin.setSingleStep(0.5)
        self._roll_spin.setSuffix("°")
        r_col.addWidget(self._roll_spin)
        corr_row.addLayout(r_col)
        pg.addLayout(corr_row)
        cfg_layout.addWidget(persp_group)

        # ── Frame extraction group ────────────────────────────────────
        frame_group = QGroupBox("Frames")
        fg = QHBoxLayout(frame_group)
        fg.setSpacing(6)
        fps_col = QVBoxLayout()
        lbl = QLabel("FPS")
        lbl.setStyleSheet(lbl_style)
        fps_col.addWidget(lbl)
        self._fps_spin = QDoubleSpinBox()
        self._fps_spin.setRange(0.5, 30.0)
        self._fps_spin.setValue(5.0)
        self._fps_spin.setSingleStep(0.5)
        self._fps_spin.valueChanged.connect(self._update_disk_estimate)
        fps_col.addWidget(self._fps_spin)
        fg.addLayout(fps_col)
        fmt_col = QVBoxLayout()
        lbl = QLabel("Format")
        lbl.setStyleSheet(lbl_style)
        fmt_col.addWidget(lbl)
        self._fmt_combo = QComboBox()
        self._fmt_combo.addItems(["jpg", "png"])
        fmt_col.addWidget(self._fmt_combo)
        fg.addLayout(fmt_col)
        q_col = QVBoxLayout()
        lbl = QLabel("Quality")
        lbl.setStyleSheet(lbl_style)
        q_col.addWidget(lbl)
        self._quality_spin = QSpinBox()
        self._quality_spin.setRange(50, 100)
        self._quality_spin.setValue(95)
        q_col.addWidget(self._quality_spin)
        fg.addLayout(q_col)
        cfg_layout.addWidget(frame_group)

        # ── Storage group ─────────────────────────────────────────────
        store_group = QGroupBox("Storage")
        stg2 = QVBoxLayout(store_group)
        stg2.setSpacing(3)
        self._keep_equirect_cb = QCheckBox("Keep equirect video")
        self._keep_equirect_cb.setChecked(True)
        self._keep_equirect_cb.stateChanged.connect(self._update_disk_estimate)
        self._keep_persp_cb = QCheckBox("Keep perspective videos")
        self._keep_persp_cb.setChecked(False)
        self._keep_persp_cb.stateChanged.connect(self._update_disk_estimate)
        self._disk_label = QLabel("Est. disk: —")
        self._disk_label.setStyleSheet(f"color: {MUTED}; font-size: 10px;")
        stg2.addWidget(self._keep_equirect_cb)
        stg2.addWidget(self._keep_persp_cb)
        stg2.addWidget(self._disk_label)
        lbl = QLabel("Output folder")
        lbl.setStyleSheet(lbl_style)
        stg2.addWidget(lbl)
        out_row = QHBoxLayout()
        self._output_edit = QLineEdit()
        self._output_edit.setPlaceholderText("Same as INSV folder")
        out_browse_btn = QPushButton("…")
        out_browse_btn.setFixedWidth(28)
        out_browse_btn.clicked.connect(self._browse_output_folder)
        out_row.addWidget(self._output_edit, stretch=1)
        out_row.addWidget(out_browse_btn)
        stg2.addLayout(out_row)
        cfg_layout.addWidget(store_group)

        # ── Run / Cancel ──────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("Run Pipeline")
        self._run_btn.setEnabled(False)
        self._run_btn.setObjectName("DangerBtn")
        self._run_btn.clicked.connect(self._run_pipeline)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel)
        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(self._cancel_btn)
        cfg_layout.addLayout(btn_row)
        cfg_layout.addStretch()

        scroll.setWidget(cfg_w)
        return scroll

    # ── Center tabs ────────────────────────────────────────────────────

    def _build_center_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        tabs.addTab(self._build_preview_tab(), "360° Preview")
        tabs.addTab(self._build_grid_tab(), "4-View Grid")
        tabs.addTab(self._build_telemetry_tab(), "Telemetry")
        tabs.addTab(self._build_log_tab(), "Log")
        return tabs

    def _build_preview_tab(self) -> QWidget:
        preview_tab = QWidget()
        prev_layout = QVBoxLayout(preview_tab)
        prev_layout.setContentsMargins(6, 6, 6, 6)

        try:
            from PyQt6.QtWebEngineWidgets import QWebEngineView
            self._webview = QWebEngineView()
            self._webview.setMinimumHeight(300)
            prev_layout.addWidget(self._webview, stretch=1)
            self._aframe_available = True
        except ImportError:
            self._webview = None
            self._aframe_available = False
            no_web_lbl = QLabel(
                "360° interactive preview requires PyQtWebEngine.\n"
                "pip install PyQt6-WebEngine\n\n"
                "Still frame preview:"
            )
            no_web_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
            no_web_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            prev_layout.addWidget(no_web_lbl)
            self._equirect_preview_label = QLabel("No equirectangular frame loaded.")
            self._equirect_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._equirect_preview_label.setMinimumHeight(300)
            self._equirect_preview_label.setStyleSheet(
                f"background: {CARD}; color: {MUTED};"
            )
            prev_layout.addWidget(self._equirect_preview_label, stretch=1)

        # Scrubber
        scrub_row = QHBoxLayout()
        scrub_row.addWidget(QLabel("Time (s):"))
        self._preview_scrubber = QSlider(Qt.Orientation.Horizontal)
        self._preview_scrubber.setRange(0, 300)
        self._preview_scrubber.setValue(5)
        self._preview_scrubber.valueChanged.connect(self._update_360_preview)
        scrub_row.addWidget(self._preview_scrubber, stretch=1)
        self._preview_time_label = QLabel("5s")
        self._preview_time_label.setStyleSheet(f"color: {MUTED};")
        scrub_row.addWidget(self._preview_time_label)
        prev_layout.addLayout(scrub_row)

        return preview_tab

    def _build_grid_tab(self) -> QWidget:
        grid_tab = QWidget()
        grid_layout = QVBoxLayout(grid_tab)
        grid_layout.setContentsMargins(6, 6, 6, 6)

        # 2x2 grid of labelled view panels
        grid_w = QWidget()
        grid_inner = QVBoxLayout(grid_w)
        grid_inner.setSpacing(4)
        top_row = QHBoxLayout()
        top_row.setSpacing(4)
        bot_row = QHBoxLayout()
        bot_row.setSpacing(4)

        self._view_labels: dict[str, QLabel] = {}
        for view, row_layout in [
            ("front", top_row), ("right", top_row),
            ("rear",  bot_row), ("left",  bot_row),
        ]:
            view_container = QWidget()
            vc_layout = QVBoxLayout(view_container)
            vc_layout.setContentsMargins(2, 2, 2, 2)
            vc_layout.setSpacing(2)
            title_lbl = QLabel(view.upper())
            title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title_lbl.setStyleSheet(
                f"color: {HI}; font-weight: bold; font-size: 10px;"
                f" background: {CARD}; padding: 2px;"
            )
            img_lbl = QLabel("No frames")
            img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_lbl.setMinimumSize(200, 180)
            img_lbl.setStyleSheet(f"background: {BG}; color: {MUTED};")
            vc_layout.addWidget(title_lbl)
            vc_layout.addWidget(img_lbl, stretch=1)
            self._view_labels[view] = img_lbl
            row_layout.addWidget(view_container)

        grid_inner.addLayout(top_row)
        grid_inner.addLayout(bot_row)
        grid_layout.addWidget(grid_w, stretch=1)

        # Playback controls
        ctrl_row = QHBoxLayout()
        self._play_btn = QPushButton("▶ Play")
        self._play_btn.setEnabled(False)
        self._play_btn.clicked.connect(self._toggle_playback)
        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        self._frame_slider.setRange(0, 0)
        self._frame_slider.valueChanged.connect(self._show_frame_at)
        self._frame_label = QLabel("Frame: 0 / 0")
        self._frame_label.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        ctrl_row.addWidget(self._play_btn)
        ctrl_row.addWidget(self._frame_slider, stretch=1)
        ctrl_row.addWidget(self._frame_label)
        grid_layout.addLayout(ctrl_row)

        return grid_tab

    def _build_telemetry_tab(self) -> QWidget:
        telem_tab = QWidget()
        telem_layout = QVBoxLayout(telem_tab)
        telem_layout.setContentsMargins(6, 6, 6, 6)

        self._telem_stats = QLabel("GPS: —   IMU: —")
        self._telem_stats.setStyleSheet(f"color: {MUTED}; font-size: 11px; padding: 4px;")
        telem_layout.addWidget(self._telem_stats)

        # GPS map
        try:
            from PyQt6.QtWebEngineWidgets import QWebEngineView as WEV
            self._gps_webview = WEV()
            self._gps_webview.setMinimumHeight(200)
            telem_layout.addWidget(self._gps_webview, stretch=1)
        except ImportError:
            self._gps_webview = None
            gps_placeholder = QLabel(
                "GPS map requires PyQtWebEngine\npip install PyQt6-WebEngine"
            )
            gps_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            gps_placeholder.setStyleSheet(
                f"color: {MUTED}; background: {CARD}; padding: 8px;"
            )
            telem_layout.addWidget(gps_placeholder, stretch=1)

        # IMU plots via PyQtGraph
        try:
            import pyqtgraph as pg
            pg.setConfigOption("background", BG)
            pg.setConfigOption("foreground", TEXT)
            self._accel_plot = pg.PlotWidget(title="Accelerometer (m/s²)")
            self._gyro_plot  = pg.PlotWidget(title="Gyroscope (rad/s)")
            self._accel_plot.setMinimumHeight(120)
            self._gyro_plot.setMinimumHeight(120)
            telem_layout.addWidget(self._accel_plot)
            telem_layout.addWidget(self._gyro_plot)
            self._imu_plots_available = True
        except ImportError:
            self._imu_plots_available = False
            imu_lbl = QLabel("IMU plots require PyQtGraph\npip install pyqtgraph")
            imu_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            imu_lbl.setStyleSheet(f"color: {MUTED}; background: {CARD}; padding: 8px;")
            telem_layout.addWidget(imu_lbl)

        return telem_tab

    def _build_log_tab(self) -> QWidget:
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        log_layout.setContentsMargins(4, 4, 4, 4)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet(
            f"background: {CARD}; color: {TEXT};"
            f" font-family: monospace; font-size: 11px; border: none;"
        )
        log_layout.addWidget(self._log)
        return log_tab

    # ── Right status panel ─────────────────────────────────────────────

    def _build_right_panel(self) -> QWidget:
        right_w = QWidget()
        right_w.setMinimumWidth(220)
        right_w.setMaximumWidth(320)
        right_layout = QVBoxLayout(right_w)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(6)

        # Pipeline status
        status_group = QGroupBox("Pipeline Status")
        sg = QVBoxLayout(status_group)
        self._pair_name_label = QLabel("No pair selected")
        self._pair_name_label.setStyleSheet(
            f"color: {TEXT}; font-size: 11px; font-weight: bold;"
        )
        sg.addWidget(self._pair_name_label)
        self._stage_widget = _PipelineStageWidget()
        sg.addWidget(self._stage_widget)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._eta_label = QLabel("ETA: —")
        self._eta_label.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        sg.addWidget(self._progress_bar)
        sg.addWidget(self._eta_label)
        right_layout.addWidget(status_group)

        # Results (initially hidden)
        self._results_group = QGroupBox("Results")
        rg = QVBoxLayout(self._results_group)
        self._result_labels: dict[str, QLabel] = {}
        for view in ["front", "right", "rear", "left"]:
            lbl = QLabel(f"{view.upper()}: —")
            lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
            rg.addWidget(lbl)
            self._result_labels[view] = lbl
        sep_lbl = QLabel("─" * 20)
        sep_lbl.setStyleSheet(f"color: {BORDER}; font-size: 11px;")
        rg.addWidget(sep_lbl)
        self._gps_result_lbl = QLabel("GPS: —")
        self._gps_result_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        self._imu_result_lbl = QLabel("IMU: —")
        self._imu_result_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        self._disk_result_lbl = QLabel("Disk: —")
        self._disk_result_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        self._time_result_lbl = QLabel("Time: —")
        self._time_result_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        rg.addWidget(self._gps_result_lbl)
        rg.addWidget(self._imu_result_lbl)
        rg.addWidget(self._disk_result_lbl)
        rg.addWidget(self._time_result_lbl)
        self._open_folder_btn = QPushButton("Open Output Folder")
        self._open_folder_btn.clicked.connect(self._open_output_folder)
        rg.addWidget(self._open_folder_btn)
        self._results_group.setVisible(False)
        right_layout.addWidget(self._results_group)

        right_layout.addStretch()
        return right_w

    # ------------------------------------------------------------------
    # Slots and logic
    # ------------------------------------------------------------------

    def _browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select INSV Folder")
        if folder:
            self._folder_edit.setText(folder)
            if not self._output_edit.text():
                self._output_edit.setText(folder)
            self._scan_folder()

    def _browse_output_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self._output_edit.setText(folder)

    def _scan_folder(self) -> None:
        folder = self._folder_edit.text().strip()
        if not folder or not Path(folder).exists():
            QMessageBox.warning(
                self, "Invalid Folder",
                "Please select a valid folder containing INSV files."
            )
            return
        self._log_line("Scanning folder for INSV pairs…")
        try:
            from core.insv_telemetry import find_insv_pairs
            pairs = find_insv_pairs(folder)
            self._insv_pairs = pairs
            self._pair_list.clear()
            for pair in pairs:
                issues = pair.issues
                icon = "✅" if not issues else "⚠️"
                dur = (
                    f"{pair.duration_seconds / 60:.1f}min"
                    if pair.duration_seconds > 0 else "?"
                )
                gps = "GPS" if pair.has_gps else ""
                imu = "IMU" if pair.has_imu else ""
                tags = " ".join(filter(None, [gps, imu]))
                label = (
                    f"{icon} {pair.base_name} ({dur})"
                    + (f" — {tags}" if tags else "")
                )
                item = QListWidgetItem(label)
                if issues:
                    item.setForeground(QColor(WARNING))
                self._pair_list.addItem(item)
            self._log_line(f"Found {len(pairs)} INSV pair(s).")
            if pairs:
                self._pair_list.setCurrentRow(0)
                self._run_btn.setEnabled(True)
            self._update_disk_estimate()
        except Exception as exc:
            self._log_line(f"ERROR: {exc}")
            QMessageBox.critical(self, "Scan Error", str(exc))

    def _on_pair_selected(self, row: int) -> None:
        if 0 <= row < len(self._insv_pairs):
            self._selected_pair_idx = row
            pair = self._insv_pairs[row]
            self._pair_name_label.setText(pair.base_name)
            self._update_disk_estimate()

    def _update_disk_estimate(self) -> None:
        if not self._insv_pairs or self._selected_pair_idx < 0:
            return
        try:
            from core.insta360_processor import estimate_disk_usage
            config = self._build_config()
            pair = self._insv_pairs[self._selected_pair_idx]
            est = estimate_disk_usage(
                pair.file_size_mb / 1024,
                config,
                pair.duration_seconds,
            )
            color = WARNING if est.get("warning") else SUCCESS
            self._disk_label.setText(
                f"Est. disk: ~{est['total_with_config_gb']:.1f} GB"
            )
            self._disk_label.setStyleSheet(f"color: {color}; font-size: 11px;")
        except Exception:
            pass

    def _build_config(self):
        from core.models import Insta360ProcessingConfig
        res_map = {
            "7680×3840 (8K)":   (7680, 3840),
            "3840×1920 (4K)":   (3840, 1920),
            "2560×1280 (2.5K)": (2560, 1280),
        }
        W, H = res_map.get(self._res_combo.currentText(), (7680, 3840))
        persp_map = {
            "2160×2160": (2160, 2160),
            "1440×1440": (1440, 1440),
            "1080×1080": (1080, 1080),
        }
        pW, pH = persp_map.get(self._persp_res_combo.currentText(), (2160, 2160))
        views = [
            v for v, cb in [
                ("front", self._view_front),
                ("right", self._view_right),
                ("rear",  self._view_rear),
                ("left",  self._view_left),
            ]
            if cb.isChecked()
        ]
        return Insta360ProcessingConfig(
            output_width=W,
            output_height=H,
            stitch_crf=self._stitch_crf_slider.value(),
            fisheye_fov=self._fov_spin.value(),
            perspective_width=pW,
            perspective_height=pH,
            h_fov=self._h_fov_spin.value(),
            v_fov=self._v_fov_spin.value(),
            perspective_crf=15,
            views=views,
            frame_fps=self._fps_spin.value(),
            frame_format=self._fmt_combo.currentText(),
            frame_quality=self._quality_spin.value(),
            pitch_correction_deg=self._pitch_spin.value(),
            roll_correction_deg=self._roll_spin.value(),
            keep_equirect_video=self._keep_equirect_cb.isChecked(),
            keep_perspective_videos=self._keep_persp_cb.isChecked(),
        )

    def _run_pipeline(self) -> None:
        if self._selected_pair_idx < 0:
            return
        pair = self._insv_pairs[self._selected_pair_idx]

        # Pre-flight validation
        try:
            from core.insta360_processor import validate_insv_pair
            output_dir = self._output_edit.text().strip() or str(
                Path(pair.back_path).parent / "insta360_output"
            )
            issues = validate_insv_pair(pair, output_dir)
            if issues:
                msg = "Pre-flight checks failed:\n\n" + "\n".join(
                    f"• {i}" for i in issues
                )
                QMessageBox.warning(self, "Validation Failed", msg)
                return
        except Exception as exc:
            QMessageBox.critical(self, "Validation Error", str(exc))
            return

        config = self._build_config()
        session_id = f"insta360_{pair.base_name}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        self._stage_widget.reset()
        self._progress_bar.setValue(0)
        self._eta_label.setText("ETA: —")
        self._results_group.setVisible(False)
        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._log_line(f"Starting pipeline for {pair.base_name}…")

        self._worker = Insta360Worker(pair, config, output_dir, session_id)
        self._worker.progress.connect(self._on_progress)
        self._worker.status.connect(self._on_status)
        self._worker.result.connect(self._on_result)
        self._worker.error.connect(self._on_error)
        self._worker.timing.connect(self._on_timing)
        self._eta_value = 0.0
        self._eta_timer.start()
        self._worker.start()

    def _cancel(self) -> None:
        self._eta_timer.stop()
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._log_line("Pipeline cancelled.")
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)

    @pyqtSlot(int)
    def _on_progress(self, pct: int) -> None:
        self._progress_bar.setValue(pct)
        stages = _PipelineStageWidget.STAGES
        # Map progress percentage to current stage
        if pct < 10:
            current_stage = "Telemetry"
        elif pct < 40:
            current_stage = "Stitching"
        elif pct < 60:
            current_stage = "Perspective"
        elif pct < 80:
            current_stage = "Frames"
        else:
            current_stage = "Dataset"
        idx_curr = stages.index(current_stage)
        for s in stages:
            idx_s = stages.index(s)
            if idx_s < idx_curr:
                self._stage_widget.set_stage_status(s, _PipelineStageWidget.STATUS_DONE)
            elif idx_s == idx_curr:
                self._stage_widget.set_stage_status(s, f"⏳ {pct}%")

    @pyqtSlot(str)
    def _on_status(self, msg: str) -> None:
        self._log_line(msg)

    @pyqtSlot(float, float)
    def _on_timing(self, elapsed: float, eta: float) -> None:
        self._eta_value = eta
        self._eta_received_at = time.monotonic()
        if eta > 0:
            mins, secs = divmod(int(eta), 60)
            self._eta_label.setText(f"ETA: {mins}m {secs:02d}s")
        else:
            self._eta_label.setText("ETA: —")

    def _tick_eta(self) -> None:
        """Countdown ETA between progress updates."""
        if self._eta_value <= 0:
            return
        since = time.monotonic() - self._eta_received_at
        adjusted = max(0.0, self._eta_value - since)
        mins, secs = divmod(int(adjusted), 60)
        self._eta_label.setText(f"ETA: {mins}m {secs:02d}s")

    @pyqtSlot(object)
    def _on_result(self, result: Any) -> None:
        self._eta_timer.stop()
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._progress_bar.setValue(100)
        for stage in _PipelineStageWidget.STAGES:
            self._stage_widget.set_stage_status(stage, _PipelineStageWidget.STATUS_DONE)
        self._eta_label.setText("ETA: done")
        self._log_line(
            f"Pipeline complete: {result.total_frames_per_view} frames/view, "
            f"output: {result.dataset_dir}"
        )

        # Populate per-view result labels
        for view_name, pv in result.perspective_views.items():
            lbl = self._result_labels.get(view_name)
            if lbl:
                blur_icon = "✅" if pv.mean_blur_score > 50 else "⚠️"
                lbl.setText(
                    f"{view_name.upper()}: {pv.frame_count} frames  "
                    f"blur:{pv.mean_blur_score:.0f} {blur_icon}"
                )
                lbl.setStyleSheet(f"color: {TEXT}; font-size: 11px;")

        gps_icon = "✅" if result.gps_samples > 0 else "—"
        self._gps_result_lbl.setText(
            f"GPS: {result.gps_samples} @ {result.gps_rate_hz:.1f}Hz {gps_icon}"
        )
        imu_icon = "✅" if result.imu_samples > 0 else "—"
        self._imu_result_lbl.setText(
            f"IMU: {result.imu_samples} @ {result.imu_rate_hz:.0f}Hz {imu_icon}"
        )
        self._disk_result_lbl.setText(f"Disk: {result.disk_usage_gb:.1f} GB")
        self._time_result_lbl.setText(
            f"Time: {result.processing_time_minutes:.1f} min"
        )
        self._results_group.setVisible(True)

        # Load frame previews into 4-view grid
        self._load_frame_previews(result)

    @pyqtSlot(str)
    def _on_error(self, msg: str) -> None:
        self._eta_timer.stop()
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._stage_widget.set_stage_status(
            "Dataset", _PipelineStageWidget.STATUS_FAILED
        )
        self._log_line(f"ERROR: {msg}")
        QMessageBox.critical(self, "Pipeline Error", msg)

    def _load_frame_previews(self, result: Any) -> None:
        """Scan result perspective_views for frame dirs and populate the grid."""
        self._view_frame_paths = {}
        for view_name, pv in result.perspective_views.items():
            if pv.frame_dir and Path(pv.frame_dir).exists():
                frames = sorted(Path(pv.frame_dir).glob("*.jpg")) + sorted(
                    Path(pv.frame_dir).glob("*.png")
                )
                if frames:
                    self._view_frame_paths[view_name] = [str(f) for f in frames]
        if self._view_frame_paths:
            max_frames = max(len(v) for v in self._view_frame_paths.values())
            self._frame_slider.setRange(0, max(0, max_frames - 1))
            self._frame_slider.setValue(0)
            self._play_btn.setEnabled(True)
            self._show_frame_at(0)

    def _show_frame_at(self, idx: int) -> None:
        """Load frame at index idx for each view into its QLabel."""
        self._preview_index = idx
        total = max(
            (len(v) for v in self._view_frame_paths.values()), default=0
        )
        self._frame_label.setText(f"Frame: {idx + 1} / {total}")
        for view, paths in self._view_frame_paths.items():
            lbl = self._view_labels.get(view)
            if lbl is None or idx >= len(paths):
                continue
            pix = QPixmap(paths[idx])
            if not pix.isNull():
                scaled = pix.scaled(
                    lbl.width(),
                    lbl.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                lbl.setPixmap(scaled)

    def _toggle_playback(self) -> None:
        if self._playback_timer.isActive():
            self._playback_timer.stop()
            self._play_btn.setText("▶ Play")
        else:
            self._playback_timer.start(67)  # ~15 fps
            self._play_btn.setText("⏸ Pause")

    def _advance_preview(self) -> None:
        if not self._view_frame_paths:
            return
        max_f = max(len(v) for v in self._view_frame_paths.values())
        next_idx = (self._preview_index + 1) % max_f
        self._frame_slider.setValue(next_idx)

    def _update_360_preview(self, value: int) -> None:
        """Update the preview time label; actual frame load happens after pipeline."""
        self._preview_time_label.setText(f"{value}s")

    def _open_output_folder(self) -> None:
        output_dir = self._output_edit.text().strip()
        if output_dir and Path(output_dir).exists():
            import subprocess as sp
            sp.run(["xdg-open", output_dir])

    def _log_line(self, msg: str) -> None:
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        self._log.append(f"[{ts}] {msg}")
