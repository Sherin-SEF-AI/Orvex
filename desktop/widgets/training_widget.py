"""
desktop/widgets/training_widget.py — YOLOv8 model training widget.

Configures and launches fine-tuning, streams per-epoch metrics to
live charts, and handles cancellation + model export.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.models import TrainingConfig
from desktop.theme import (
    BG, PANEL, ACCENT, HI, TEXT, MUTED, SUCCESS, WARNING, BORDER, CARD,
    apply_plot_theme,
)
from desktop.workers import TrainingWorker

# Alias for local references
HIGHLIGHT = HI


class TrainingWidget(QWidget):
    """UI for YOLOv8 training with live metric charts."""

    def __init__(self, session_manager, parent=None) -> None:
        super().__init__(parent)
        self._sm = session_manager
        self._worker: TrainingWorker | None = None
        self._train_result = None
        self._epoch_data: dict[str, list] = {
            "epoch": [], "box_loss": [], "cls_loss": [], "dfl_loss": [],
            "map50": [], "map50_95": [], "precision": [], "recall": [],
        }
        self._build_ui()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ── Left config ────────────────────────────────────────────────
        cfg_w = QWidget()
        cfg_w.setMinimumWidth(300)
        cfg_w.setMaximumWidth(420)
        cfg_layout = QVBoxLayout(cfg_w)
        cfg_layout.setContentsMargins(8, 8, 8, 8)

        data_group = QGroupBox("Dataset")
        dg = QVBoxLayout(data_group)
        dataset_row = QHBoxLayout()
        self._dataset_edit = QLabel("No dataset selected.")
        self._dataset_edit.setWordWrap(True)
        dataset_row.addWidget(self._dataset_edit, stretch=1)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_dataset)
        dataset_row.addWidget(browse_btn)
        dg.addLayout(dataset_row)
        cfg_layout.addWidget(data_group)

        model_group = QGroupBox("Model")
        mg = QVBoxLayout(model_group)
        mg.addWidget(QLabel("Base model:"))
        self._model_combo = QComboBox()
        self._model_combo.addItems(["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"])
        mg.addWidget(self._model_combo)
        mg.addWidget(QLabel("Pretrained weights:"))
        self._weights_combo = QComboBox()
        self._weights_combo.addItems(["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
        self._weights_combo.setEditable(True)
        mg.addWidget(self._weights_combo)
        cfg_layout.addWidget(model_group)

        hp_group = QGroupBox("Hyperparameters")
        hg = QVBoxLayout(hp_group)
        for label, attr, range_vals, is_float in [
            ("Epochs:",       "_epochs_spin",   (1,   2000, 100),  False),
            ("Batch size:",   "_batch_spin",    (1,   128,  16),   False),
            ("Image size:",   "_imgsz_combo",   None,              None),
            ("Learning rate:","_lr_spin",       (1e-5, 0.1, 0.01), True),
        ]:
            hg.addWidget(QLabel(label))
            if label == "Image size:":
                self._imgsz_combo = QComboBox()
                self._imgsz_combo.addItems(["416", "512", "640", "1024"])
                self._imgsz_combo.setCurrentText("640")
                hg.addWidget(self._imgsz_combo)
            elif is_float:
                lo, hi, val = range_vals
                spin = QDoubleSpinBox()
                spin.setRange(lo, hi)
                spin.setValue(val)
                spin.setDecimals(5)
                spin.setSingleStep(0.001)
                setattr(self, attr, spin)
                hg.addWidget(spin)
            else:
                lo, hi, val = range_vals
                spin = QSpinBox()
                spin.setRange(lo, hi)
                spin.setValue(val)
                setattr(self, attr, spin)
                hg.addWidget(spin)
        hg.addWidget(QLabel("Device:"))
        self._device_combo = QComboBox()
        self._device_combo.addItems(["auto", "cpu", "cuda:0"])
        hg.addWidget(self._device_combo)
        cfg_layout.addWidget(hp_group)

        proj_group = QGroupBox("Project")
        pg = QVBoxLayout(proj_group)
        pg.addWidget(QLabel("Project name:"))
        self._proj_spin = QLabel()
        from PyQt6.QtWidgets import QLineEdit
        self._proj_edit = QLineEdit("rover_detection")
        self._run_edit = QLineEdit("run1")
        pg.addWidget(self._proj_edit)
        pg.addWidget(QLabel("Run name:"))
        pg.addWidget(self._run_edit)
        cfg_layout.addWidget(proj_group)

        btn_row = QHBoxLayout()
        self._start_btn = QPushButton("Start Training")
        self._start_btn.setObjectName("DangerBtn")
        self._start_btn.clicked.connect(self._start)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel)
        btn_row.addWidget(self._start_btn)
        btn_row.addWidget(self._cancel_btn)
        cfg_layout.addLayout(btn_row)

        cfg_layout.addStretch()
        splitter.addWidget(cfg_w)

        # ── Right panel ────────────────────────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 8, 8, 8)

        self._epoch_label = QLabel("Epoch: — / —")
        rl.addWidget(self._epoch_label)

        # Live charts
        charts_tabs = QTabWidget()
        self._loss_chart   = _MetricPlot("Losses",       ["box_loss", "cls_loss", "dfl_loss"])
        self._map_chart    = _MetricPlot("mAP",          ["map50", "map50_95"])
        self._prec_chart   = _MetricPlot("Precision/Recall", ["precision", "recall"])
        charts_tabs.addTab(self._loss_chart,  "Losses")
        charts_tabs.addTab(self._map_chart,   "mAP")
        charts_tabs.addTab(self._prec_chart,  "Precision/Recall")
        rl.addWidget(charts_tabs, stretch=2)

        # Results panel
        results_group = QGroupBox("Results")
        rg = QHBoxLayout(results_group)
        self._best_label = QLabel("Best mAP50: —")
        rg.addWidget(self._best_label)

        self._export_combo = QComboBox()
        self._export_combo.addItems(["onnx", "torchscript", "tflite"])
        rg.addWidget(QLabel("Export as:"))
        rg.addWidget(self._export_combo)
        self._export_btn = QPushButton("Export Model")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export)
        rg.addWidget(self._export_btn)

        self._inference_btn = QPushButton("Use for inference →")
        self._inference_btn.setEnabled(False)
        self._inference_btn.setToolTip("Switch to Auto-Label tab with this model")
        rg.addWidget(self._inference_btn)
        rl.addWidget(results_group)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 700])

        self._dataset_dir: str = ""

    @pyqtSlot(str)
    def on_session_changed(self, session_id: str) -> None:
        # Auto-set dataset dir from session augmentation output
        if session_id:
            candidate = self._sm.session_folder(session_id) / "augmented_dataset"
            if candidate.exists():
                self._dataset_dir = str(candidate)
                self._dataset_edit.setText(str(candidate))

    def _browse_dataset(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select YOLO dataset directory")
        if path:
            self._dataset_dir = path
            self._dataset_edit.setText(path)

    def _start(self) -> None:
        if not self._dataset_dir:
            QMessageBox.warning(self, "No Dataset", "Select a YOLO dataset directory first.")
            return
        if self._worker and self._worker.isRunning():
            return
        try:
            from core.trainer import prepare_training_config
            config = prepare_training_config(
                dataset_dir=self._dataset_dir,
                model_variant=self._model_combo.currentText(),
                pretrained_weights=self._weights_combo.currentText(),
                epochs=self._epochs_spin.value(),
                batch_size=self._batch_spin.value(),
                image_size=int(self._imgsz_combo.currentText()),
                learning_rate=self._lr_spin.value(),
                device=self._device_combo.currentText(),
                project_name=self._proj_edit.text() or "rover_detection",
                run_name=self._run_edit.text() or "run1",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Config Error", str(exc))
            return

        # Reset chart data
        for key in self._epoch_data:
            self._epoch_data[key].clear()

        self._worker = TrainingWorker(config=config)
        self._worker.status.connect(lambda m: None)  # logged elsewhere
        self._worker.epoch_metric.connect(self._on_epoch)
        self._worker.result.connect(self._on_result)
        self._worker.error.connect(lambda e: QMessageBox.critical(self, "Training Error", e))
        self._start_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._worker.start()

    def _cancel(self) -> None:
        if self._worker:
            self._worker.cancel()
        self._cancel_btn.setEnabled(False)
        self._start_btn.setEnabled(True)

    @pyqtSlot(object)
    def _on_epoch(self, em: Any) -> None:
        self._epoch_label.setText(
            f"Epoch: {em.epoch} / {self._worker._config.epochs if self._worker else '?'}  "
            f"| box={em.box_loss:.4f} cls={em.cls_loss:.4f} dfl={em.dfl_loss:.4f} "
            f"| mAP50={em.map50:.4f}"
        )
        for key in self._epoch_data:
            val = getattr(em, key, None)
            if val is not None:
                self._epoch_data[key].append(val)
        self._epoch_data["epoch"].append(em.epoch)
        self._loss_chart.update(self._epoch_data)
        self._map_chart.update(self._epoch_data)
        self._prec_chart.update(self._epoch_data)

    @pyqtSlot(object)
    def _on_result(self, result: Any) -> None:
        self._train_result = result
        self._start_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._export_btn.setEnabled(True)
        self._inference_btn.setEnabled(True)
        self._best_label.setText(
            f"Best mAP50: {result.final_map50:.4f} @ epoch {result.best_epoch}  |  "
            f"mAP50-95: {result.final_map50_95:.4f}  |  "
            f"Time: {result.training_time_minutes:.1f} min"
        )

    def _export(self) -> None:
        if not self._train_result:
            return
        fmt = self._export_combo.currentText()
        try:
            from core.trainer import export_model
            path = export_model(self._train_result.best_weights_path, fmt)
            QMessageBox.information(self, "Export Done", f"Model exported to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))


class _MetricPlot(QWidget):
    """Live metric chart using PyQtGraph."""

    def __init__(self, title: str, metrics: list[str], parent=None) -> None:
        super().__init__(parent)
        self._metrics = metrics
        layout = QVBoxLayout(self)
        self._placeholder = QLabel(f"{title} — training not started.")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._placeholder)
        self._plot = None
        self._curves: dict = {}
        self._colors = [
            (233, 69, 96), (79, 195, 247), (102, 187, 106),
            (255, 167, 38), (171, 71, 188), (239, 83, 80),
        ]

    def update(self, data: dict) -> None:
        try:
            import pyqtgraph as pg
            import numpy as np

            if self._plot is None:
                self._plot = pg.PlotWidget()
                self._plot.addLegend()
                self._plot.setLabel("bottom", "Epoch")
                apply_plot_theme(self._plot)
                layout = self.layout()
                layout.removeWidget(self._placeholder)
                self._placeholder.hide()
                layout.addWidget(self._plot)

            epochs = data.get("epoch", [])
            for i, metric in enumerate(self._metrics):
                vals = data.get(metric, [])
                if len(vals) != len(epochs):
                    continue
                color = self._colors[i % len(self._colors)]
                if metric not in self._curves:
                    pen = pg.mkPen(color=color, width=2)
                    curve = self._plot.plot([], [], pen=pen, name=metric)
                    self._curves[metric] = curve
                self._curves[metric].setData(epochs, vals)
        except Exception:
            self._placeholder.setText(
                f"pyqtgraph required for live charts.\n"
                + "\n".join(
                    f"{m}: {data.get(m, ['-'])[-1] if data.get(m) else '—'}"
                    for m in self._metrics
                )
            )
            self._placeholder.show()
