"""
desktop/widgets/continuous_learning_widget.py — Continuous learning orchestration UI.

Top    : config panel (threshold, auto-promote, aug config, training config)
Middle : learning log table
Bottom : live metric plots + run controls
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.continuous_learning import get_learning_log
from core.models import AugmentationConfig, TrainingConfig
from core.session_manager import SessionManager
from desktop.theme import (
    BG, PANEL, ACCENT, HI, TEXT, MUTED, SUCCESS, BORDER, CARD,
    apply_plot_theme,
)
from desktop.workers import ContinuousLearningWorker

# Semantic alias
GREEN = SUCCESS

try:
    import pyqtgraph as pg
    _HAS_PG = True
except ImportError:
    _HAS_PG = False


class ContinuousLearningWidget(QWidget):
    """Continuous learning loop UI: config → run → log → promote."""

    def __init__(self, sm: SessionManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sm = sm
        self._session_id: str | None = None
        self._worker: ContinuousLearningWorker | None = None

        # Plot data
        self._map_x: list[float] = []
        self._map_y: list[float] = []

        self._build_ui()
        self._refresh_log()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ── Top: config ───────────────────────────────────────────────
        config_row = QHBoxLayout()

        # Trigger config
        trig_grp = QGroupBox("Trigger Config")
        tg = QHBoxLayout(trig_grp)
        tg.addWidget(QLabel("Threshold:"))
        self._threshold_spin = QSpinBox()
        self._threshold_spin.setRange(1, 10000)
        self._threshold_spin.setValue(50)
        self._threshold_spin.setFixedWidth(70)
        tg.addWidget(self._threshold_spin)
        tg.addSpacing(12)
        self._auto_promote_chk = QCheckBox("Auto-promote")
        self._auto_promote_chk.setChecked(True)
        tg.addWidget(self._auto_promote_chk)
        tg.addStretch()
        config_row.addWidget(trig_grp)

        # Training config (compact)
        train_grp = QGroupBox("Training Config")
        tng = QHBoxLayout(train_grp)
        tng.addWidget(QLabel("Variant:"))
        self._variant_combo = QComboBox()
        self._variant_combo.addItems(["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"])
        self._variant_combo.setFixedWidth(90)
        tng.addWidget(self._variant_combo)
        tng.addWidget(QLabel("Epochs:"))
        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(1, 500)
        self._epochs_spin.setValue(50)
        self._epochs_spin.setFixedWidth(65)
        tng.addWidget(self._epochs_spin)
        tng.addWidget(QLabel("Batch:"))
        self._batch_spin = QSpinBox()
        self._batch_spin.setRange(1, 128)
        self._batch_spin.setValue(16)
        self._batch_spin.setFixedWidth(55)
        tng.addWidget(self._batch_spin)
        tng.addWidget(QLabel("LR:"))
        self._lr_spin = QDoubleSpinBox()
        self._lr_spin.setRange(1e-5, 0.1)
        self._lr_spin.setDecimals(5)
        self._lr_spin.setValue(0.01)
        self._lr_spin.setFixedWidth(80)
        tng.addWidget(self._lr_spin)
        tng.addStretch()
        config_row.addWidget(train_grp)

        root.addLayout(config_row)

        # Augmentation config (compact row of checkboxes)
        aug_grp = QGroupBox("Augmentation")
        aug_layout = QHBoxLayout(aug_grp)
        self._aug_hflip   = QCheckBox("H-flip");    self._aug_hflip.setChecked(True)
        self._aug_rot     = QCheckBox("Rotate-90");  self._aug_rot.setChecked(True)
        self._aug_bc      = QCheckBox("Bright/Cont"); self._aug_bc.setChecked(True)
        self._aug_noise   = QCheckBox("Noise");       self._aug_noise.setChecked(True)
        self._aug_blur    = QCheckBox("MotionBlur");  self._aug_blur.setChecked(True)
        self._aug_mosaic  = QCheckBox("Mosaic");      self._aug_mosaic.setChecked(True)
        aug_layout.addWidget(QLabel("Multiplier:"))
        self._aug_mult_spin = QSpinBox()
        self._aug_mult_spin.setRange(1, 20)
        self._aug_mult_spin.setValue(3)
        self._aug_mult_spin.setFixedWidth(55)
        aug_layout.addWidget(self._aug_mult_spin)
        aug_layout.addSpacing(8)
        for chk in (self._aug_hflip, self._aug_rot, self._aug_bc,
                    self._aug_noise, self._aug_blur, self._aug_mosaic):
            aug_layout.addWidget(chk)
        aug_layout.addStretch()
        root.addWidget(aug_grp)

        # ── Middle: log table ─────────────────────────────────────────
        log_grp = QGroupBox("Learning History")
        lg = QVBoxLayout(log_grp)
        self._log_table = QTableWidget(0, 7)
        self._log_table.setHorizontalHeaderLabels([
            "Date", "Session", "Trigger", "Corrections",
            "Run ID", "Before mAP50", "After mAP50",
        ])
        self._log_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self._log_table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.Stretch
        )
        self._log_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._log_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._log_table.setMinimumHeight(100)
        lg.addWidget(self._log_table)
        root.addWidget(log_grp)

        # ── Bottom: plot + run controls ───────────────────────────────
        bottom_row = QHBoxLayout()

        # mAP plot
        plot_grp = QGroupBox("mAP50 over Training Cycles")
        pg_layout = QVBoxLayout(plot_grp)
        if _HAS_PG:
            self._plot_widget = pg.PlotWidget()
            self._plot_widget.setLabel("left", "mAP50")
            self._plot_widget.setLabel("bottom", "Cycle")
            apply_plot_theme(self._plot_widget)
            self._map_curve = self._plot_widget.plot(
                pen=pg.mkPen(color=HI, width=2), symbol="o", symbolSize=6
            )
            pg_layout.addWidget(self._plot_widget)
        else:
            pg_layout.addWidget(QLabel("pyqtgraph not installed — plots unavailable"))
        bottom_row.addWidget(plot_grp, stretch=2)

        # Controls
        ctrl_grp = QGroupBox("Run")
        cg = QVBoxLayout(ctrl_grp)
        self._btn_run    = QPushButton("Run Learning Cycle")
        self._btn_run.setObjectName("DangerBtn")
        self._btn_check  = QPushButton("Check trigger now")
        self._btn_refresh_log = QPushButton("Refresh log")
        cg.addWidget(self._btn_run)
        cg.addWidget(self._btn_check)
        cg.addWidget(self._btn_refresh_log)
        cg.addSpacing(8)
        self._status_label = QLabel("Idle")
        self._status_label.setWordWrap(True)
        cg.addWidget(self._status_label)
        self._progress_label = QLabel("")
        cg.addWidget(self._progress_label)
        cg.addStretch()
        bottom_row.addWidget(ctrl_grp, stretch=1)

        root.addLayout(bottom_row)

        # ── Connections ───────────────────────────────────────────────
        self._btn_run.clicked.connect(self._on_run_cycle)
        self._btn_check.clicked.connect(self._on_check_trigger)
        self._btn_refresh_log.clicked.connect(self._refresh_log)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @pyqtSlot(str)
    def on_session_changed(self, session_id: str) -> None:
        self._session_id = session_id

    def show_for_session(self, session_id: str) -> None:
        """Called from AnnotationReviewWidget 'Trigger retraining' button."""
        self._session_id = session_id

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_aug_config(self) -> AugmentationConfig:
        return AugmentationConfig(
            horizontal_flip=self._aug_hflip.isChecked(),
            random_rotate_90=self._aug_rot.isChecked(),
            brightness_contrast=self._aug_bc.isChecked(),
            gaussian_noise=self._aug_noise.isChecked(),
            motion_blur=self._aug_blur.isChecked(),
            mosaic=self._aug_mosaic.isChecked(),
            multiplier=self._aug_mult_spin.value(),
        )

    def _build_train_config(self) -> TrainingConfig:
        return TrainingConfig(
            dataset_dir="",    # filled in by run_learning_cycle
            model_variant=self._variant_combo.currentText(),
            pretrained_weights=f"{self._variant_combo.currentText()}.pt",
            epochs=self._epochs_spin.value(),
            batch_size=self._batch_spin.value(),
            learning_rate=self._lr_spin.value(),
            project_name="rover_cl",
        )

    def _refresh_log(self) -> None:
        entries = get_learning_log()
        self._log_table.setRowCount(0)
        for e in entries:
            row = self._log_table.rowCount()
            self._log_table.insertRow(row)
            date_str = e.triggered_at.strftime("%Y-%m-%d %H:%M") if e.triggered_at else ""
            for col, val in enumerate([
                date_str,
                e.session_id[:8],
                e.trigger_type,
                str(e.corrections_count),
                (e.resulting_run_id or "")[:12],
                "—",
                "—",
            ]):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._log_table.setItem(row, col, item)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_check_trigger(self) -> None:
        if not self._session_id:
            QMessageBox.warning(self, "No session", "Select a session first.")
            return
        from core.continuous_learning import check_and_maybe_trigger
        threshold = self._threshold_spin.value()
        ready = check_and_maybe_trigger(self._session_id, self._sm, threshold)
        if ready:
            QMessageBox.information(
                self, "Threshold met",
                f"Correction threshold ({threshold}) reached!\n"
                "Click 'Run Learning Cycle' to start retraining."
            )
        else:
            from core.annotation_review import get_review_stats
            stats = get_review_stats(self._session_id, self._sm)
            usable = stats["usable_for_training"]
            QMessageBox.information(
                self, "Not yet",
                f"Usable reviews: {usable} / {threshold} required."
            )

    @pyqtSlot()
    def _on_run_cycle(self) -> None:
        if not self._session_id:
            QMessageBox.warning(self, "No session", "Select a session first.")
            return
        if self._worker and self._worker.isRunning():
            return

        self._btn_run.setEnabled(False)
        self._map_x.clear()
        self._map_y.clear()
        if _HAS_PG:
            self._map_curve.setData([], [])

        aug_config   = self._build_aug_config()
        train_config = self._build_train_config()
        auto_promote = self._auto_promote_chk.isChecked()

        self._worker = ContinuousLearningWorker(
            session_id=self._session_id,
            sm=self._sm,
            aug_config=aug_config,
            train_config=train_config,
            auto_promote=auto_promote,
        )
        self._worker.status.connect(self._on_status)
        self._worker.progress.connect(self._on_progress)
        self._worker.epoch_metric.connect(self._on_epoch_metric)
        self._worker.result.connect(self._on_cycle_done)
        self._worker.error.connect(self._on_cycle_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    @pyqtSlot(str)
    def _on_status(self, msg: str) -> None:
        self._status_label.setText(msg)

    @pyqtSlot(int)
    def _on_progress(self, pct: int) -> None:
        self._progress_label.setText(f"Progress: {pct}%")

    @pyqtSlot(object)
    def _on_epoch_metric(self, metric: object) -> None:
        """Receive EpochMetrics; update mAP plot."""
        if not _HAS_PG:
            return
        if hasattr(metric, "epoch") and hasattr(metric, "map50"):
            self._map_x.append(float(metric.epoch))
            self._map_y.append(float(metric.map50))
            self._map_curve.setData(self._map_x, self._map_y)

    @pyqtSlot(object)
    def _on_cycle_done(self, result: object) -> None:
        training = result.get("training") if isinstance(result, dict) else None
        comparison = result.get("comparison") if isinstance(result, dict) else None
        msg = "Learning cycle complete!"
        if comparison:
            msg += (
                f"\nBaseline mAP50: {comparison.baseline_map50:.4f}"
                f"\nCandidate mAP50: {comparison.candidate_map50:.4f}"
                f"\nΔ: {comparison.delta_map50:+.4f}"
            )
            if comparison.improved:
                msg += "\n✓ Candidate promoted to active model."
            else:
                msg += "\n⚠ Candidate NOT promoted (insufficient improvement)."
        QMessageBox.information(self, "Cycle complete", msg)
        self._refresh_log()

    @pyqtSlot(str)
    def _on_cycle_error(self, err: str) -> None:
        self._status_label.setText(f"Error: {err}")
        QMessageBox.critical(self, "Learning cycle failed", err)

    @pyqtSlot()
    def _on_worker_finished(self) -> None:
        self._btn_run.setEnabled(True)
        self._progress_label.setText("")
