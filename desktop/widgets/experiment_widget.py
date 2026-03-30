"""
desktop/widgets/experiment_widget.py — MLflow experiment browser widget.

Displays all training runs fetched from MLflow, supports single-run detail
inspection and multi-run metric comparison.  Session-agnostic: it reads all
runs regardless of which session is active.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from desktop.theme import (
    ACCENT,
    BORDER,
    CARD,
    HI,
    MUTED,
    PANEL,
    SUCCESS,
    TEXT,
    WARNING,
    apply_plot_theme,
    badge_style,
)

# Lazy import — tolerate mlflow not being installed
try:
    from core.experiment_tracker import (
        check_mlflow_installation,
        compare_runs,
        get_all_runs,
        launch_mlflow_ui,
    )
    from core.models import MLflowRun, RunComparison
except ImportError:
    check_mlflow_installation = lambda: False  # type: ignore[assignment]
    get_all_runs = lambda **kw: []  # type: ignore[assignment]
    compare_runs = lambda run_ids: None  # type: ignore[assignment]
    launch_mlflow_ui = lambda port=5000: None  # type: ignore[assignment]
    MLflowRun = None  # type: ignore[assignment, misc]
    RunComparison = None  # type: ignore[assignment, misc]


# ---------------------------------------------------------------------------
# Background refresh worker
# ---------------------------------------------------------------------------

class _RefreshWorker(QThread):
    """Fetch MLflow runs in a background thread so the UI never freezes."""

    finished = pyqtSignal(list)   # list[MLflowRun]
    error    = pyqtSignal(str)

    def __init__(self, experiment_name: str = "rover_detection", parent=None) -> None:
        super().__init__(parent)
        self._experiment_name = experiment_name

    def run(self) -> None:
        try:
            runs = get_all_runs(experiment_name=self._experiment_name)
            self.finished.emit(runs)
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RUN_COLS = [
    "Run Name",
    "Status",
    "mAP50",
    "mAP50-95",
    "Precision",
    "Recall",
    "Epochs",
    "Batch",
    "Model",
    "Duration (min)",
    "Date",
]


def _fmt(val: Any, decimals: int = 4) -> str:
    """Format a numeric or None value for table display."""
    if val is None or val != val:  # None or NaN
        return "—"
    try:
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return str(val)


def _map50_color(value: float | None) -> QColor | None:
    """Return a foreground QColor based on mAP50 value thresholds."""
    if value is None:
        return None
    if value < 0.3:
        return QColor(HI)           # red
    if value < 0.5:
        return QColor(255, 200, 0)  # amber
    return QColor(SUCCESS)          # green


def _status_color(status: str) -> str:
    """Return a theme color hex for a run status string."""
    s = status.upper()
    if s == "FINISHED":
        return SUCCESS
    if s in ("FAILED", "KILLED"):
        return HI
    if s == "RUNNING":
        return WARNING
    return MUTED


# ---------------------------------------------------------------------------
# Detail panel
# ---------------------------------------------------------------------------

class _DetailPanel(QGroupBox):
    """Shows full detail for a single selected MLflow run."""

    deploy_requested = pyqtSignal(str)   # emits run_id

    def __init__(self, parent=None) -> None:
        super().__init__("Run Details", parent)
        self._run: Any = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 14, 10, 10)
        root.setSpacing(8)

        # -- Identity row ------------------------------------------------
        identity_row = QHBoxLayout()
        self._name_lbl  = QLabel("—")
        name_font = QFont()
        name_font.setBold(True)
        name_font.setPointSize(11)
        self._name_lbl.setFont(name_font)
        self._status_badge = QLabel()
        identity_row.addWidget(self._name_lbl)
        identity_row.addStretch()
        identity_row.addWidget(self._status_badge)
        root.addLayout(identity_row)

        self._date_lbl = QLabel("Started: —")
        self._date_lbl.setStyleSheet(f"color: {MUTED}; font-size: 10px;")
        root.addWidget(self._date_lbl)

        # -- Best metrics ------------------------------------------------
        metrics_box = QGroupBox("Best Metrics")
        metrics_box.setFlat(True)
        mg = QHBoxLayout(metrics_box)
        self._map50_lbl     = _MetricBadge("mAP50",     "—")
        self._map5095_lbl   = _MetricBadge("mAP50-95",  "—")
        self._precision_lbl = _MetricBadge("Precision", "—")
        self._recall_lbl    = _MetricBadge("Recall",    "—")
        for w in (self._map50_lbl, self._map5095_lbl,
                  self._precision_lbl, self._recall_lbl):
            mg.addWidget(w)
        mg.addStretch()
        root.addWidget(metrics_box)

        # -- Params table ------------------------------------------------
        root.addWidget(QLabel("Hyperparameters:"))
        self._params_table = QTableWidget(0, 2)
        self._params_table.setHorizontalHeaderLabels(["Param", "Value"])
        self._params_table.horizontalHeader().setStretchLastSection(True)
        self._params_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self._params_table.setAlternatingRowColors(True)
        self._params_table.setMaximumHeight(180)
        root.addWidget(self._params_table)

        # -- Weights path ------------------------------------------------
        weights_row = QHBoxLayout()
        self._weights_lbl = QLabel("Weights: —")
        self._weights_lbl.setWordWrap(True)
        self._weights_lbl.setStyleSheet(f"color: {MUTED}; font-size: 10px;")
        weights_row.addWidget(self._weights_lbl, stretch=1)
        self._open_folder_btn = QPushButton("Open Folder")
        self._open_folder_btn.setEnabled(False)
        self._open_folder_btn.clicked.connect(self._open_folder)
        weights_row.addWidget(self._open_folder_btn)
        root.addLayout(weights_row)

        # -- Action buttons ------------------------------------------------
        action_row = QHBoxLayout()
        self._use_autolabel_btn = QPushButton("Use in Auto-Label")
        self._use_autolabel_btn.setEnabled(False)
        self._use_autolabel_btn.setToolTip(
            "Select model in Auto-Label tab to use it for inference."
        )
        self._deploy_btn = QPushButton("Deploy this model")
        self._deploy_btn.setObjectName("PrimaryBtn")
        self._deploy_btn.setEnabled(False)
        self._deploy_btn.clicked.connect(self._on_deploy)
        action_row.addWidget(self._use_autolabel_btn)
        action_row.addWidget(self._deploy_btn)
        action_row.addStretch()
        root.addLayout(action_row)

        root.addStretch()

    # -- slots -----------------------------------------------------------

    def _open_folder(self) -> None:
        if self._run is None:
            return
        artifact_uri: str = getattr(self._run, "artifact_uri", "") or ""
        # Convert mlflow artifact URI to filesystem path when possible
        folder = artifact_uri.replace("file://", "")
        if not folder or not Path(folder).exists():
            # Fall back to weights path stored in params
            weights = self._run.params.get("best_weights_path", "")
            if weights:
                folder = str(Path(weights).parent)
        if folder and Path(folder).exists():
            try:
                if os.name == "nt":
                    os.startfile(folder)  # type: ignore[attr-defined]
                else:
                    subprocess.Popen(["xdg-open", folder])
            except Exception as exc:
                QMessageBox.warning(
                    self,
                    "Cannot Open Folder",
                    f"Failed to open folder:\n{folder}\n\n{exc}",
                )
        else:
            QMessageBox.information(
                self,
                "Folder Not Found",
                f"Artifact folder not found on disk:\n{artifact_uri}",
            )

    def _on_deploy(self) -> None:
        if self._run is not None:
            self.deploy_requested.emit(self._run.run_id)

    # -- public ---------------------------------------------------------

    def load_run(self, run: Any) -> None:
        """Populate the panel with data from *run* (MLflowRun)."""
        self._run = run

        self._name_lbl.setText(run.run_name or run.run_id)
        self._status_badge.setText(run.status)
        self._status_badge.setStyleSheet(badge_style(_status_color(run.status)))
        self._date_lbl.setText(
            f"Started: {run.start_time.strftime('%Y-%m-%d  %H:%M:%S UTC')}"
        )

        # Metrics
        m = run.metrics
        map50_val = m.get("val/mAP50") or m.get("best/mAP50")
        map5095_val = m.get("val/mAP50-95") or m.get("best/mAP50-95")
        self._map50_lbl.set_value(_fmt(map50_val))
        self._map5095_lbl.set_value(_fmt(map5095_val))
        self._precision_lbl.set_value(_fmt(m.get("val/precision")))
        self._recall_lbl.set_value(_fmt(m.get("val/recall")))

        # Params
        self._params_table.setRowCount(0)
        for key, val in sorted(run.params.items()):
            row = self._params_table.rowCount()
            self._params_table.insertRow(row)
            self._params_table.setItem(row, 0, QTableWidgetItem(str(key)))
            self._params_table.setItem(row, 1, QTableWidgetItem(str(val)))

        # Weights path (may be stored as a param or inferred from artifact_uri)
        weights = run.params.get("best_weights_path", run.artifact_uri or "—")
        self._weights_lbl.setText(f"Weights: {weights}")
        artifact_ok = bool(run.artifact_uri)
        self._open_folder_btn.setEnabled(artifact_ok)
        self._deploy_btn.setEnabled(True)

    def clear(self) -> None:
        self._run = None
        self._name_lbl.setText("—")
        self._status_badge.setText("")
        self._date_lbl.setText("Started: —")
        self._map50_lbl.set_value("—")
        self._map5095_lbl.set_value("—")
        self._precision_lbl.set_value("—")
        self._recall_lbl.set_value("—")
        self._params_table.setRowCount(0)
        self._weights_lbl.setText("Weights: —")
        self._open_folder_btn.setEnabled(False)
        self._deploy_btn.setEnabled(False)


class _MetricBadge(QWidget):
    """Small label-value pair card for a single metric."""

    def __init__(self, label: str, value: str = "—", parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)

        lbl = QLabel(label.upper())
        lbl.setStyleSheet(
            f"color: {MUTED}; font-size: 9px; font-weight: bold; letter-spacing: 0.8px;"
        )
        layout.addWidget(lbl)

        self._value_lbl = QLabel(value)
        font = QFont()
        font.setBold(True)
        font.setPointSize(11)
        self._value_lbl.setFont(font)
        layout.addWidget(self._value_lbl)

        self.setStyleSheet(
            f"background: {CARD}; border: 1px solid {BORDER}; border-radius: 5px;"
        )

    def set_value(self, value: str) -> None:
        self._value_lbl.setText(value)


# ---------------------------------------------------------------------------
# Comparison panel
# ---------------------------------------------------------------------------

class _ComparisonPanel(QGroupBox):
    """Metric comparison table shown when 2+ runs are selected."""

    def __init__(self, parent=None) -> None:
        super().__init__("Run Comparison", parent)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 14, 10, 10)

        self._best_lbl = QLabel("Best run: —")
        self._best_lbl.setStyleSheet(f"color: {SUCCESS}; font-weight: bold;")
        root.addWidget(self._best_lbl)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Metric", "Run A", "Run B", "Better"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        root.addWidget(self._table)

        self._param_diff_lbl = QLabel("Parameter differences: none")
        self._param_diff_lbl.setWordWrap(True)
        self._param_diff_lbl.setStyleSheet(f"color: {MUTED}; font-size: 10px;")
        root.addWidget(self._param_diff_lbl)

    def load_comparison(self, comparison: Any) -> None:
        """Populate the panel from a RunComparison instance."""
        if comparison is None:
            self.clear()
            return

        runs = comparison.runs
        if len(runs) < 2:
            self.clear()
            return

        run_a = runs[0]
        run_b = runs[1]

        self._best_lbl.setText(
            f"Best run: {comparison.best_run_id}  "
            f"(mAP50 = {_fmt(run_a.metrics.get('val/mAP50') if run_a.run_id == comparison.best_run_id else run_b.metrics.get('val/mAP50'))})"
        )

        # Metric table — show a curated subset of key metrics
        key_metrics = [
            "val/mAP50", "val/mAP50-95", "val/precision", "val/recall",
            "best/mAP50", "best/mAP50-95", "training_time_minutes",
        ]
        rows = [
            k for k in key_metrics
            if k in comparison.metric_comparison
        ]
        # Also add any remaining metrics not in the curated list
        for k in sorted(comparison.metric_comparison.keys()):
            if k not in rows:
                rows.append(k)

        self._table.setRowCount(0)
        for key in rows:
            mc = comparison.metric_comparison.get(key, {})
            val_a = mc.get(run_a.run_id)
            val_b = mc.get(run_b.run_id)

            # Determine which run is better (higher is better for most metrics)
            better = ""
            try:
                fa, fb = float(val_a), float(val_b)  # type: ignore[arg-type]
                if fa > fb:
                    better = run_a.run_name or run_a.run_id
                elif fb > fa:
                    better = run_b.run_name or run_b.run_id
                else:
                    better = "equal"
            except (TypeError, ValueError):
                better = "—"

            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(key))
            self._table.setItem(row, 1, QTableWidgetItem(_fmt(val_a)))
            self._table.setItem(row, 2, QTableWidgetItem(_fmt(val_b)))
            item_better = QTableWidgetItem(better)
            item_better.setForeground(QColor(SUCCESS) if better not in ("—", "equal", "") else QColor(MUTED))
            self._table.setItem(row, 3, item_better)

        # Parameter differences
        if comparison.param_differences:
            lines = []
            for param, vals in comparison.param_differences.items():
                val_a_str = str(vals.get(run_a.run_id, "—"))
                val_b_str = str(vals.get(run_b.run_id, "—"))
                lines.append(f"{param}: {val_a_str} vs {val_b_str}")
            self._param_diff_lbl.setText("Differing params:\n" + "  |  ".join(lines))
        else:
            self._param_diff_lbl.setText("No parameter differences between selected runs.")

    def clear(self) -> None:
        self._best_lbl.setText("Best run: —")
        self._table.setRowCount(0)
        self._param_diff_lbl.setText("Parameter differences: none")


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class ExperimentWidget(QWidget):
    """
    MLflow experiment browser.

    Session-agnostic: reads all runs regardless of which session is active.
    ``on_session_changed()`` is a no-op provided for interface uniformity.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._runs: list[Any] = []
        self._refresh_worker: _RefreshWorker | None = None
        self._mlflow_proc: subprocess.Popen | None = None  # type: ignore[type-arg]
        self._build_ui()
        # Initial data load
        self._refresh()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Top bar ────────────────────────────────────────────────────────
        top_bar = QHBoxLayout()
        top_bar.setSpacing(8)

        self._launch_ui_btn = QPushButton("Launch MLflow UI")
        self._launch_ui_btn.setObjectName("PrimaryBtn")
        self._launch_ui_btn.setToolTip("Open the MLflow web UI in your browser (port 5000)")
        self._launch_ui_btn.clicked.connect(self._launch_mlflow_ui)
        top_bar.addWidget(self._launch_ui_btn)

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self._refresh)
        top_bar.addWidget(self._refresh_btn)

        top_bar.addStretch()

        self._mlflow_badge = QLabel()
        self._update_mlflow_badge()
        top_bar.addWidget(self._mlflow_badge)

        root.addLayout(top_bar)

        # ── Splitter: table (left) | detail/comparison (right) ───────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, stretch=1)

        # -- Runs table ---------------------------------------------------
        table_container = QWidget()
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(0, 0, 0, 0)

        self._runs_table = QTableWidget(0, len(_RUN_COLS))
        self._runs_table.setHorizontalHeaderLabels(_RUN_COLS)
        self._runs_table.horizontalHeader().setStretchLastSection(True)
        self._runs_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._runs_table.setAlternatingRowColors(True)
        self._runs_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self._runs_table.setSortingEnabled(True)
        self._runs_table.setMinimumWidth(600)
        self._runs_table.itemSelectionChanged.connect(self._on_selection_changed)
        table_layout.addWidget(self._runs_table)

        self._status_lbl = QLabel("No runs loaded.")
        self._status_lbl.setStyleSheet(f"color: {MUTED}; font-size: 10px;")
        table_layout.addWidget(self._status_lbl)

        splitter.addWidget(table_container)

        # -- Right panel: detail or comparison ----------------------------
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setMinimumWidth(320)

        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(4, 0, 4, 0)
        right_layout.setSpacing(8)

        self._detail_panel = _DetailPanel()
        self._detail_panel.deploy_requested.connect(self._on_deploy)
        right_layout.addWidget(self._detail_panel)

        self._comparison_panel = _ComparisonPanel()
        self._comparison_panel.hide()
        right_layout.addWidget(self._comparison_panel)

        right_layout.addStretch()
        right_scroll.setWidget(right_container)
        splitter.addWidget(right_scroll)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([700, 380])

    # ── helpers ───────────────────────────────────────────────────────────

    def _update_mlflow_badge(self) -> None:
        if check_mlflow_installation():
            self._mlflow_badge.setText("✓ MLflow Ready")
            self._mlflow_badge.setStyleSheet(badge_style(SUCCESS))
        else:
            self._mlflow_badge.setText("⚠ MLflow not installed")
            self._mlflow_badge.setStyleSheet(badge_style(WARNING))

    def _populate_table(self, runs: list[Any]) -> None:
        """Fill the runs table from a list of MLflowRun objects."""
        self._runs = runs
        # Disable sorting while populating to preserve order
        self._runs_table.setSortingEnabled(False)
        self._runs_table.setRowCount(0)

        for run in runs:
            m   = run.metrics
            p   = run.params
            row = self._runs_table.rowCount()
            self._runs_table.insertRow(row)

            run_name = run.run_name or run.run_id
            date_str = run.start_time.strftime("%Y-%m-%d %H:%M")

            map50_raw = m.get("val/mAP50") or m.get("best/mAP50")
            try:
                map50_float: float | None = float(map50_raw)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                map50_float = None

            values = [
                run_name,
                run.status,
                _fmt(map50_float),
                _fmt(m.get("val/mAP50-95") or m.get("best/mAP50-95")),
                _fmt(m.get("val/precision")),
                _fmt(m.get("val/recall")),
                str(p.get("epochs", "—")),
                str(p.get("batch_size", "—")),
                str(p.get("model_variant", "—")),
                _fmt(m.get("training_time_minutes"), decimals=1),
                date_str,
            ]

            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)

                # Color-code status column
                if col == 1:
                    item.setForeground(QColor(_status_color(run.status)))

                # Color-code mAP50 column
                if col == 2:
                    color = _map50_color(map50_float)
                    if color is not None:
                        item.setForeground(color)

                self._runs_table.setItem(row, col, item)

            # Store run_id in the first column's UserRole for retrieval
            self._runs_table.item(row, 0).setData(
                Qt.ItemDataRole.UserRole, run.run_id
            )

        self._runs_table.setSortingEnabled(True)
        self._runs_table.resizeColumnsToContents()

        count = len(runs)
        self._status_lbl.setText(
            f"{count} run{'s' if count != 1 else ''} loaded."
            if count > 0
            else "No runs found. Start a training run to see results here."
        )

    def _selected_run_ids(self) -> list[str]:
        """Return run_ids for all currently selected table rows."""
        selected_rows = {idx.row() for idx in self._runs_table.selectedIndexes()}
        run_ids = []
        for row in sorted(selected_rows):
            item = self._runs_table.item(row, 0)
            if item is not None:
                rid = item.data(Qt.ItemDataRole.UserRole)
                if rid:
                    run_ids.append(rid)
        return run_ids

    def _run_by_id(self, run_id: str) -> Any | None:
        for r in self._runs:
            if r.run_id == run_id:
                return r
        return None

    # ── slots ─────────────────────────────────────────────────────────────

    @pyqtSlot()
    def _refresh(self) -> None:
        if self._refresh_worker and self._refresh_worker.isRunning():
            return
        if not check_mlflow_installation():
            self._status_lbl.setText(
                "MLflow not installed. Install with: pip install mlflow"
            )
            self._update_mlflow_badge()
            return

        self._refresh_btn.setEnabled(False)
        self._status_lbl.setText("Loading runs…")
        self._refresh_worker = _RefreshWorker(parent=self)
        self._refresh_worker.finished.connect(self._on_runs_loaded)
        self._refresh_worker.error.connect(self._on_refresh_error)
        self._refresh_worker.start()

    @pyqtSlot(list)
    def _on_runs_loaded(self, runs: list) -> None:
        self._refresh_btn.setEnabled(True)
        self._populate_table(runs)
        self._update_mlflow_badge()

    @pyqtSlot(str)
    def _on_refresh_error(self, message: str) -> None:
        self._refresh_btn.setEnabled(True)
        self._status_lbl.setText(f"Error loading runs: {message}")

    @pyqtSlot()
    def _on_selection_changed(self) -> None:
        run_ids = self._selected_run_ids()

        if len(run_ids) == 0:
            self._detail_panel.clear()
            self._comparison_panel.clear()
            self._comparison_panel.hide()
            self._detail_panel.show()
            return

        if len(run_ids) == 1:
            run = self._run_by_id(run_ids[0])
            if run is not None:
                self._detail_panel.load_run(run)
            self._comparison_panel.hide()
            self._detail_panel.show()
            return

        # Multi-select: show comparison panel
        self._detail_panel.hide()
        self._comparison_panel.show()
        result = compare_runs(run_ids)
        self._comparison_panel.load_comparison(result)

    @pyqtSlot()
    def _launch_mlflow_ui(self) -> None:
        if not check_mlflow_installation():
            QMessageBox.warning(
                self,
                "MLflow Not Installed",
                "MLflow is not installed.\n\nInstall with:\n    pip install mlflow",
            )
            return
        # Terminate any existing UI process before spawning a new one
        if self._mlflow_proc and self._mlflow_proc.poll() is None:
            QMessageBox.information(
                self,
                "MLflow UI Already Running",
                "The MLflow UI is already running.\nOpen http://localhost:5000 in your browser.",
            )
            return
        self._mlflow_proc = launch_mlflow_ui(port=5000)
        if self._mlflow_proc is None:
            QMessageBox.critical(
                self,
                "Launch Failed",
                "Failed to launch the MLflow UI.\n"
                "Ensure mlflow is installed and the virtualenv is active:\n"
                "    pip install mlflow",
            )

    @pyqtSlot(str)
    def _on_deploy(self, run_id: str) -> None:
        run = self._run_by_id(run_id)
        run_label = (run.run_name if run else None) or run_id
        QMessageBox.information(
            self,
            "Deploy Model",
            f"Deployment requested for run:\n{run_label}\n\n"
            f"Run ID: {run_id}\n\n"
            "(Deployment pipeline integration is configured separately.)",
        )

    # ── public API ────────────────────────────────────────────────────────

    @pyqtSlot(str)
    def on_session_changed(self, session_id: str) -> None:
        """No-op.  This widget is session-agnostic."""
        pass

    def refresh(self) -> None:
        """Public refresh trigger (callable from main window toolbar etc.)."""
        self._refresh()
