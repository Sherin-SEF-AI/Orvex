"""
desktop/widgets/edge_export_widget.py — Edge deployment export widget.

Provides a guided 4-step pipeline for exporting YOLOv8 models to ONNX and
TensorRT, benchmarking latency, and packaging for Jetson deployment.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from desktop.theme import ACCENT, BORDER, HI, MUTED, PANEL, SUCCESS, TEXT, WARNING
from core.edge_exporter import (
    benchmark_model,
    check_export_dependencies,
    export_to_onnx,
    export_to_tensorrt,
    package_jetson_deployment,
)
from core.models import BenchmarkResult, ONNXExportResult, TRTExportResult


# ─────────────────────────────────────────────────────────────────────────────
# Worker threads
# ─────────────────────────────────────────────────────────────────────────────

class _ONNXWorker(QThread):
    """Background thread for ONNX export."""

    status = pyqtSignal(str)
    result = pyqtSignal(object)   # ONNXExportResult
    error = pyqtSignal(str)

    def __init__(
        self,
        weights_path: str,
        output_path: str,
        image_size: int,
        batch_size: int,
        simplify: bool,
        opset_version: int,
        dynamic_axes: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._weights_path = weights_path
        self._output_path = output_path
        self._image_size = image_size
        self._batch_size = batch_size
        self._simplify = simplify
        self._opset_version = opset_version
        self._dynamic_axes = dynamic_axes

    def run(self) -> None:
        try:
            self.status.emit("Starting ONNX export…")
            result = export_to_onnx(
                weights_path=self._weights_path,
                output_path=self._output_path,
                image_size=self._image_size,
                batch_size=self._batch_size,
                simplify=self._simplify,
                opset_version=self._opset_version,
                dynamic_axes=self._dynamic_axes,
            )
            self.status.emit(
                f"ONNX export complete: {result.model_size_mb:.1f} MB, "
                f"latency {result.test_latency_ms:.1f} ms"
            )
            self.result.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class _TRTWorker(QThread):
    """Background thread for TensorRT conversion."""

    status = pyqtSignal(str)
    result = pyqtSignal(object)   # TRTExportResult
    error = pyqtSignal(str)

    def __init__(
        self,
        onnx_path: str,
        output_path: str,
        precision: str,
        workspace_gb: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._onnx_path = onnx_path
        self._output_path = output_path
        self._precision = precision
        self._workspace_gb = workspace_gb

    def run(self) -> None:
        try:
            self.status.emit(
                f"Starting TensorRT build ({self._precision.upper()})… "
                "This may take 20–40 minutes."
            )
            result = export_to_tensorrt(
                onnx_path=self._onnx_path,
                output_path=self._output_path,
                precision=self._precision,
                workspace_gb=self._workspace_gb,
            )
            self.status.emit(
                f"TRT build complete: {result.engine_size_mb:.1f} MB, "
                f"{result.build_time_minutes:.1f} min"
            )
            self.result.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class _BenchmarkWorker(QThread):
    """Background thread for model benchmarking."""

    status = pyqtSignal(str)
    result = pyqtSignal(object)   # BenchmarkResult
    error = pyqtSignal(str)

    def __init__(
        self,
        model_path: str,
        model_format: str,
        image_size: int,
        n_iterations: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._model_path = model_path
        self._model_format = model_format
        self._image_size = image_size
        self._n_iterations = n_iterations

    def run(self) -> None:
        try:
            self.status.emit(
                f"Benchmarking {self._model_format.upper()} model "
                f"({self._n_iterations} iterations)…"
            )
            result = benchmark_model(
                model_path=self._model_path,
                model_format=self._model_format,
                image_size=self._image_size,
                n_iterations=self._n_iterations,
            )
            self.status.emit(
                f"Benchmark done — mean {result.mean_latency_ms:.1f} ms, "
                f"{result.throughput_fps:.1f} FPS"
            )
            self.result.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class _PackageWorker(QThread):
    """Background thread for Jetson deployment packaging."""

    status = pyqtSignal(str)
    result = pyqtSignal(str)     # path to .tar.gz
    error = pyqtSignal(str)

    def __init__(
        self,
        weights_path: str,
        onnx_path: str,
        trt_path: str | None,
        class_names: list[str],
        output_dir: str,
        target_device: str,
        conf_threshold: float,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._weights_path = weights_path
        self._onnx_path = onnx_path
        self._trt_path = trt_path
        self._class_names = class_names
        self._output_dir = output_dir
        self._target_device = target_device
        self._conf_threshold = conf_threshold

    def run(self) -> None:
        try:
            self.status.emit("Building deployment package…")
            archive_path = package_jetson_deployment(
                weights_path=self._weights_path,
                onnx_path=self._onnx_path,
                trt_path=self._trt_path,
                class_names=self._class_names,
                output_dir=self._output_dir,
                target_device=self._target_device,
                conf_threshold=self._conf_threshold,
            )
            self.status.emit(f"Package ready: {archive_path}")
            self.result.emit(archive_path)
        except Exception as exc:
            self.error.emit(str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Step indicator bar
# ─────────────────────────────────────────────────────────────────────────────

class _StepIndicator(QWidget):
    """Horizontal row of step labels that highlights the active step."""

    _STEP_LABELS = [
        "1  ONNX Export",
        "2  TensorRT",
        "3  Benchmark",
        "4  Package",
    ]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._labels: list[QLabel] = []
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 4, 0, 4)
        row.setSpacing(0)
        for i, text in enumerate(self._STEP_LABELS):
            lbl = QLabel(text)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setMinimumHeight(28)
            row.addWidget(lbl, stretch=1)
            if i < len(self._STEP_LABELS) - 1:
                sep = QLabel(" › ")
                sep.setAlignment(Qt.AlignmentFlag.AlignCenter)
                sep.setStyleSheet(f"color: {MUTED}; font-size: 14px; background: transparent;")
                row.addWidget(sep)
            self._labels.append(lbl)
        self.set_active(0)

    def set_active(self, index: int) -> None:
        for i, lbl in enumerate(self._labels):
            if i == index:
                lbl.setStyleSheet(
                    f"color: {TEXT}; font-weight: bold; font-size: 12px; "
                    f"background: {ACCENT}; border-radius: 4px; padding: 2px 8px;"
                )
            elif i < index:
                lbl.setStyleSheet(
                    f"color: {SUCCESS}; font-size: 11px; "
                    f"background: transparent; padding: 2px 8px;"
                )
            else:
                lbl.setStyleSheet(
                    f"color: {MUTED}; font-size: 11px; "
                    f"background: transparent; padding: 2px 8px;"
                )


# ─────────────────────────────────────────────────────────────────────────────
# Individual step widgets
# ─────────────────────────────────────────────────────────────────────────────

class _NavRow(QWidget):
    """Back / Next navigation row used at the bottom of each step."""

    back_clicked = pyqtSignal()
    next_clicked = pyqtSignal()

    def __init__(self, show_back: bool = True, next_label: str = "Next Step →", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 8, 0, 0)
        if show_back:
            self._back = QPushButton("← Back")
            self._back.clicked.connect(self.back_clicked)
            row.addWidget(self._back)
        row.addStretch()
        self._next = QPushButton(next_label)
        self._next.setObjectName("PrimaryBtn")
        self._next.clicked.connect(self.next_clicked)
        row.addWidget(self._next)

    def set_next_enabled(self, enabled: bool) -> None:
        self._next.setEnabled(enabled)

    def set_next_label(self, label: str) -> None:
        self._next.setText(label)


class _Step1Widget(QWidget):
    """Step 1 — ONNX Export."""

    export_requested = pyqtSignal()
    next_clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        grp = QGroupBox("ONNX Export Settings")
        gl = QVBoxLayout(grp)

        # Image size
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Image size:"))
        self.img_size_combo = QComboBox()
        self.img_size_combo.addItems(["416", "512", "640", "1280"])
        self.img_size_combo.setCurrentText("640")
        row1.addWidget(self.img_size_combo)
        row1.addStretch()
        gl.addLayout(row1)

        # Batch size
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Batch size:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 8)
        self.batch_spin.setValue(1)
        row2.addWidget(self.batch_spin)
        row2.addStretch()
        gl.addLayout(row2)

        # Options
        opts = QHBoxLayout()
        self.simplify_cb = QCheckBox("Simplify (onnx-simplifier)")
        self.simplify_cb.setChecked(True)
        opts.addWidget(self.simplify_cb)
        opts.addStretch()
        gl.addLayout(opts)

        # Opset version
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Opset version:"))
        self.opset_spin = QSpinBox()
        self.opset_spin.setRange(11, 17)
        self.opset_spin.setValue(17)
        row3.addWidget(self.opset_spin)
        row3.addStretch()
        gl.addLayout(row3)

        # Export button
        btn_row = QHBoxLayout()
        self.export_btn = QPushButton("Export ONNX")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_requested)
        btn_row.addWidget(self.export_btn)
        btn_row.addStretch()
        gl.addLayout(btn_row)

        # Result label
        self.result_lbl = QLabel("No export yet.")
        self.result_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        gl.addWidget(self.result_lbl)

        layout.addWidget(grp)
        layout.addStretch()

        nav = _NavRow(show_back=False, next_label="Next Step →")
        nav.next_clicked.connect(self.next_clicked)
        nav.set_next_enabled(False)
        self._nav = nav
        layout.addWidget(nav)

    def set_model_selected(self, ok: bool) -> None:
        self.export_btn.setEnabled(ok)

    def set_result(self, result: ONNXExportResult) -> None:
        v = "✓" if result.verification_passed else "!"
        self.result_lbl.setText(
            f"{v}  {result.model_size_mb:.1f} MB | "
            f"latency {result.test_latency_ms:.1f} ms | "
            f"opset {result.onnx_opset} | "
            f"simplified={result.simplified}"
        )
        color = SUCCESS if result.verification_passed else WARNING
        self.result_lbl.setStyleSheet(f"color: {color}; font-size: 11px;")
        self._nav.set_next_enabled(True)

    def set_export_running(self, running: bool) -> None:
        self.export_btn.setEnabled(not running)
        self.export_btn.setText("Exporting…" if running else "Export ONNX")

    @property
    def image_size(self) -> int:
        return int(self.img_size_combo.currentText())

    @property
    def batch_size(self) -> int:
        return self.batch_spin.value()

    @property
    def simplify(self) -> bool:
        return self.simplify_cb.isChecked()

    @property
    def opset_version(self) -> int:
        return self.opset_spin.value()


class _Step2Widget(QWidget):
    """Step 2 — TensorRT Conversion."""

    convert_requested = pyqtSignal()
    back_clicked = pyqtSignal()
    next_clicked = pyqtSignal()

    def __init__(self, deps: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._trtexec_available = deps.get("trtexec", False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        grp = QGroupBox("TensorRT Conversion Settings")
        gl = QVBoxLayout(grp)

        # trtexec availability notice
        if not self._trtexec_available:
            warn_lbl = QLabel(
                "⚠  trtexec not found on PATH.\n"
                "On Jetson: sudo apt install tensorrt\n"
                "On desktop: install TensorRT from developer.nvidia.com/tensorrt\n"
                "then ensure trtexec is on your PATH."
            )
            warn_lbl.setStyleSheet(
                f"color: {WARNING}; font-size: 11px; "
                f"background: #2a1a00; border: 1px solid {WARNING}; "
                f"border-radius: 4px; padding: 6px;"
            )
            warn_lbl.setWordWrap(True)
            gl.addWidget(warn_lbl)

        # Precision
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Precision:"))
        self.precision_combo = QComboBox()
        self.precision_combo.addItems(["FP32", "FP16", "INT8"])
        self.precision_combo.setCurrentText("FP16")
        row1.addWidget(self.precision_combo)
        row1.addStretch()
        gl.addLayout(row1)

        # Workspace
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Workspace (GB):"))
        self.workspace_spin = QSpinBox()
        self.workspace_spin.setRange(1, 16)
        self.workspace_spin.setValue(4)
        row2.addWidget(self.workspace_spin)
        row2.addStretch()
        gl.addLayout(row2)

        # Convert button
        btn_row = QHBoxLayout()
        self.convert_btn = QPushButton("Convert to TRT")
        self.convert_btn.setEnabled(self._trtexec_available)
        self.convert_btn.clicked.connect(self.convert_requested)
        btn_row.addWidget(self.convert_btn)
        btn_row.addStretch()
        gl.addLayout(btn_row)

        # Result label
        self.result_lbl = QLabel(
            "Conversion not run."
            if self._trtexec_available
            else "TRT unavailable — you may skip to Step 3 or Step 4."
        )
        self.result_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        gl.addWidget(self.result_lbl)

        layout.addWidget(grp)
        layout.addStretch()

        nav = _NavRow(show_back=True, next_label="Next Step →")
        nav.back_clicked.connect(self.back_clicked)
        nav.next_clicked.connect(self.next_clicked)
        # Allow skipping TRT — user can proceed even without engine
        nav.set_next_enabled(True)
        self._nav = nav
        layout.addWidget(nav)

    def set_result(self, result: TRTExportResult) -> None:
        self.result_lbl.setText(
            f"✓  {result.engine_size_mb:.1f} MB | "
            f"{result.precision.upper()} | "
            f"build {result.build_time_minutes:.1f} min"
        )
        self.result_lbl.setStyleSheet(f"color: {SUCCESS}; font-size: 11px;")

    def set_convert_running(self, running: bool) -> None:
        self.convert_btn.setEnabled(not running and self._trtexec_available)
        self.convert_btn.setText("Converting…" if running else "Convert to TRT")

    @property
    def precision(self) -> str:
        return self.precision_combo.currentText().lower()

    @property
    def workspace_gb(self) -> int:
        return self.workspace_spin.value()


class _Step3Widget(QWidget):
    """Step 3 — Benchmark."""

    benchmark_requested = pyqtSignal()
    back_clicked = pyqtSignal()
    next_clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        grp = QGroupBox("Benchmark Settings")
        gl = QVBoxLayout(grp)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["PyTorch", "ONNX"])
        self.format_combo.setCurrentText("ONNX")
        row1.addWidget(self.format_combo)
        row1.addStretch()
        gl.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Iterations:"))
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(50, 500)
        self.iter_spin.setValue(200)
        self.iter_spin.setSingleStep(50)
        row2.addWidget(self.iter_spin)
        row2.addStretch()
        gl.addLayout(row2)

        btn_row = QHBoxLayout()
        self.bench_btn = QPushButton("Run Benchmark")
        self.bench_btn.setEnabled(False)
        self.bench_btn.clicked.connect(self.benchmark_requested)
        btn_row.addWidget(self.bench_btn)
        btn_row.addStretch()
        gl.addLayout(btn_row)

        # Results table
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Format", "Mean ms", "P95 ms", "FPS", "Memory MB"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(100)
        self.table.setMaximumHeight(180)
        gl.addWidget(self.table)

        layout.addWidget(grp)
        layout.addStretch()

        nav = _NavRow(show_back=True, next_label="Next Step →")
        nav.back_clicked.connect(self.back_clicked)
        nav.next_clicked.connect(self.next_clicked)
        nav.set_next_enabled(True)
        self._nav = nav
        layout.addWidget(nav)

    def set_model_ready(self, ok: bool) -> None:
        self.bench_btn.setEnabled(ok)

    def set_benchmark_running(self, running: bool) -> None:
        self.bench_btn.setEnabled(not running)
        self.bench_btn.setText("Running…" if running else "Run Benchmark")

    def add_result(self, result: BenchmarkResult) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        for col, val in enumerate([
            result.format.upper(),
            f"{result.mean_latency_ms:.2f}",
            f"{result.p95_latency_ms:.2f}",
            f"{result.throughput_fps:.1f}",
            f"{result.memory_mb:.1f}",
        ]):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, col, item)
        self.table.resizeColumnsToContents()

    @property
    def model_format(self) -> str:
        return self.format_combo.currentText().lower()

    @property
    def n_iterations(self) -> int:
        return self.iter_spin.value()


class _Step4Widget(QWidget):
    """Step 4 — Package for Jetson."""

    package_requested = pyqtSignal()
    back_clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        grp = QGroupBox("Jetson Deployment Package")
        gl = QVBoxLayout(grp)

        # Target device
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Target device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems([
            "Jetson Orin", "Jetson Xavier", "Jetson Nano", "Generic CPU"
        ])
        row1.addWidget(self.device_combo)
        row1.addStretch()
        gl.addLayout(row1)

        # Confidence threshold
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Confidence threshold:"))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 0.99)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)
        self.conf_spin.setDecimals(2)
        row2.addWidget(self.conf_spin)
        row2.addStretch()
        gl.addLayout(row2)

        # Output directory
        row3 = QHBoxLayout()
        self.out_dir_btn = QPushButton("Browse…")
        self.out_dir_btn.clicked.connect(self._browse_output)
        row3.addWidget(self.out_dir_btn)
        self.out_dir_lbl = QLabel("No output directory selected.")
        self.out_dir_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        row3.addWidget(self.out_dir_lbl, stretch=1)
        gl.addLayout(row3)

        # Build button
        self.build_btn = QPushButton("Build Package")
        self.build_btn.setObjectName("PrimaryBtn")
        self.build_btn.setEnabled(False)
        self.build_btn.clicked.connect(self.package_requested)
        gl.addWidget(self.build_btn)

        # Result text (file tree)
        self.result_edit = QTextEdit()
        self.result_edit.setReadOnly(True)
        self.result_edit.setPlaceholderText("Package contents will appear here after build.")
        self.result_edit.setMinimumHeight(120)
        gl.addWidget(self.result_edit)

        layout.addWidget(grp)
        layout.addStretch()

        nav = _NavRow(show_back=True, next_label="Finish")
        nav.back_clicked.connect(self.back_clicked)
        nav.set_next_enabled(False)
        nav._next.setEnabled(False)
        nav._next.setVisible(False)
        self._nav = nav
        layout.addWidget(nav)

        self._output_dir: str = ""

    def _browse_output(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Select output directory for deployment package"
        )
        if d:
            self._output_dir = d
            self.out_dir_lbl.setText(d)
            self._refresh_build_btn()

    def _refresh_build_btn(self) -> None:
        self.build_btn.setEnabled(bool(self._output_dir))

    def set_build_running(self, running: bool) -> None:
        self.build_btn.setEnabled(not running and bool(self._output_dir))
        self.build_btn.setText("Building…" if running else "Build Package")

    def set_result(self, archive_path: str) -> None:
        pkg_dir = archive_path.replace(".tar.gz", "")
        lines = [
            f"Archive: {archive_path}",
            "",
            f"{Path(pkg_dir).name}/",
            "  models/",
            "    model.pt",
            "    model.onnx",
            "    model.engine  (if TRT was built)",
            "  config/",
            "    classes.txt",
            "    deploy_config.json",
            "  inference/",
            "    infer_onnx.py",
            "    requirements.txt",
            "  README.md",
        ]
        self.result_edit.setPlainText("\n".join(lines))

    @property
    def target_device(self) -> str:
        return self.device_combo.currentText()

    @property
    def conf_threshold(self) -> float:
        return self.conf_spin.value()

    @property
    def output_dir(self) -> str:
        return self._output_dir


# ─────────────────────────────────────────────────────────────────────────────
# Main widget
# ─────────────────────────────────────────────────────────────────────────────

class EdgeExportWidget(QWidget):
    """Guided 4-step edge deployment export widget.

    Lets the user:
      1. Export a YOLOv8 .pt model to ONNX.
      2. Optionally convert the ONNX to a TensorRT engine.
      3. Benchmark latency/throughput.
      4. Package everything for deployment on Jetson hardware.

    Usage:
        widget = EdgeExportWidget(session_manager)
        # session_id changes are a no-op; model path is set manually.
    """

    def __init__(self, session_manager: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sm = session_manager
        self._session_id: str = ""
        self._weights_path: str = ""
        self._onnx_path: str = ""
        self._trt_path: str | None = None
        self._class_names: list[str] = []
        self._current_step: int = 0

        # Workers
        self._onnx_worker: _ONNXWorker | None = None
        self._trt_worker: _TRTWorker | None = None
        self._bench_worker: _BenchmarkWorker | None = None
        self._pkg_worker: _PackageWorker | None = None

        # Check dependencies (used to configure step 2)
        self._deps = check_export_dependencies()

        self._build_ui()
        self._log_deps()

    # ------------------------------------------------------------------
    # Dependency logging
    # ------------------------------------------------------------------

    def _log_deps(self) -> None:
        d = self._deps
        lines = [
            "Edge export dependencies:",
            f"  onnx:          {'OK' if d['onnx'] else 'MISSING (pip install onnx)'}",
            f"  onnxruntime:   {'OK' if d['onnxruntime'] else 'MISSING (pip install onnxruntime)'}",
            f"  onnxsim:       {'OK' if d['onnxsim'] else 'MISSING (pip install onnxsim)'}",
            f"  ultralytics:   {'OK' if d['ultralytics'] else 'MISSING (pip install ultralytics)'}",
            f"  torch:         {d['torch_version'] or 'NOT FOUND (pip install torch)'}",
            f"  CUDA:          {'available' if d['cuda_available'] else 'not available'}",
            f"  trtexec:       {'found on PATH' if d['trtexec'] else 'NOT FOUND — TRT disabled'}",
            f"  tensorrt pkg:  {d.get('tensorrt_version') or ('not installed' if not d['tensorrt'] else 'installed')}",
        ]
        for line in lines:
            self._append_log(line)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Model selector ───────────────────────────────────────────
        model_grp = QGroupBox("Model")
        mg = QHBoxLayout(model_grp)
        self._browse_btn = QPushButton("Browse .pt…")
        self._browse_btn.clicked.connect(self._browse_weights)
        mg.addWidget(self._browse_btn)
        self._model_path_edit = QLineEdit()
        self._model_path_edit.setPlaceholderText("No model selected")
        self._model_path_edit.setReadOnly(True)
        mg.addWidget(self._model_path_edit, stretch=1)
        self._model_info_lbl = QLabel("No model loaded.")
        self._model_info_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        mg.addWidget(self._model_info_lbl)
        root.addWidget(model_grp)

        # ── Step indicator ───────────────────────────────────────────
        self._step_indicator = _StepIndicator()
        root.addWidget(self._step_indicator)

        # ── Export pipeline (stacked steps) ─────────────────────────
        pipeline_grp = QGroupBox("Export Pipeline")
        pl = QVBoxLayout(pipeline_grp)

        self._stack = QStackedWidget()
        self._step1 = _Step1Widget()
        self._step2 = _Step2Widget(deps=self._deps)
        self._step3 = _Step3Widget()
        self._step4 = _Step4Widget()
        self._stack.addWidget(self._step1)
        self._stack.addWidget(self._step2)
        self._stack.addWidget(self._step3)
        self._stack.addWidget(self._step4)
        pl.addWidget(self._stack)
        root.addWidget(pipeline_grp, stretch=1)

        # ── Log ──────────────────────────────────────────────────────
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(100)
        self._log.setPlaceholderText("Log output…")
        root.addWidget(self._log)

        # ── Wire navigation signals ──────────────────────────────────
        self._step1.export_requested.connect(self._run_onnx_export)
        self._step1.next_clicked.connect(lambda: self._goto_step(1))

        self._step2.convert_requested.connect(self._run_trt_convert)
        self._step2.back_clicked.connect(lambda: self._goto_step(0))
        self._step2.next_clicked.connect(lambda: self._goto_step(2))

        self._step3.benchmark_requested.connect(self._run_benchmark)
        self._step3.back_clicked.connect(lambda: self._goto_step(1))
        self._step3.next_clicked.connect(lambda: self._goto_step(3))

        self._step4.package_requested.connect(self._run_package)
        self._step4.back_clicked.connect(lambda: self._goto_step(2))

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _goto_step(self, index: int) -> None:
        self._current_step = index
        self._stack.setCurrentIndex(index)
        self._step_indicator.set_active(index)

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------

    def _browse_weights(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLOv8 weights file",
            "",
            "PyTorch weights (*.pt);;All files (*)",
        )
        if not path:
            return
        self._weights_path = path
        self._model_path_edit.setText(path)
        self._model_info_lbl.setText(
            f"{Path(path).name}  —  {Path(path).stat().st_size / (1024 * 1024):.1f} MB"
        )
        self._model_info_lbl.setStyleSheet(f"color: {SUCCESS}; font-size: 11px;")
        self._step1.set_model_selected(True)
        self._step3.set_model_ready(True)
        self._append_log(f"Model selected: {path}")

        # Try to read class names from a companion data.yaml
        self._class_names = self._try_load_class_names(path)

    def _try_load_class_names(self, weights_path: str) -> list[str]:
        """Attempt to read class names from data.yaml next to the weights."""
        candidates = [
            Path(weights_path).parent / "data.yaml",
            Path(weights_path).parent.parent / "data.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                try:
                    import yaml  # type: ignore[import]
                    with open(candidate, "r") as f:
                        data = yaml.safe_load(f)
                    names = data.get("names", [])
                    if isinstance(names, dict):
                        names = [names[k] for k in sorted(names.keys())]
                    if names:
                        self._append_log(
                            f"Loaded {len(names)} class names from {candidate}"
                        )
                        return list(names)
                except Exception:
                    pass
        self._append_log(
            "No data.yaml found next to weights — class names will be auto-numbered."
        )
        return []

    # ------------------------------------------------------------------
    # Step 1: ONNX export
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _run_onnx_export(self) -> None:
        if not self._weights_path:
            QMessageBox.warning(self, "No model", "Select a .pt file first.")
            return
        if self._onnx_worker and self._onnx_worker.isRunning():
            return

        out_path = str(
            Path(self._weights_path).parent
            / (Path(self._weights_path).stem + "_exported.onnx")
        )
        self._onnx_path = out_path
        self._step1.set_export_running(True)

        self._onnx_worker = _ONNXWorker(
            weights_path=self._weights_path,
            output_path=out_path,
            image_size=self._step1.image_size,
            batch_size=self._step1.batch_size,
            simplify=self._step1.simplify,
            opset_version=self._step1.opset_version,
            dynamic_axes=False,
            parent=self,
        )
        self._onnx_worker.status.connect(self._on_status)
        self._onnx_worker.result.connect(self._on_onnx_result)
        self._onnx_worker.error.connect(self._on_error)
        self._onnx_worker.start()

    @pyqtSlot(object)
    def _on_onnx_result(self, result: ONNXExportResult) -> None:
        self._onnx_path = result.output_path
        self._step1.set_export_running(False)
        self._step1.set_result(result)
        self._append_log(
            f"ONNX export done: {result.output_path} "
            f"({result.model_size_mb:.1f} MB, "
            f"verified={result.verification_passed})"
        )

    # ------------------------------------------------------------------
    # Step 2: TRT conversion
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _run_trt_convert(self) -> None:
        if not self._onnx_path or not Path(self._onnx_path).exists():
            QMessageBox.warning(
                self,
                "No ONNX model",
                "Complete Step 1 (ONNX export) before converting to TensorRT.",
            )
            return
        if self._trt_worker and self._trt_worker.isRunning():
            return

        engine_path = str(Path(self._onnx_path).with_suffix(".engine"))
        self._step2.set_convert_running(True)

        self._trt_worker = _TRTWorker(
            onnx_path=self._onnx_path,
            output_path=engine_path,
            precision=self._step2.precision,
            workspace_gb=self._step2.workspace_gb,
            parent=self,
        )
        self._trt_worker.status.connect(self._on_status)
        self._trt_worker.result.connect(self._on_trt_result)
        self._trt_worker.error.connect(self._on_error)
        self._trt_worker.start()

    @pyqtSlot(object)
    def _on_trt_result(self, result: TRTExportResult) -> None:
        self._trt_path = result.output_path
        self._step2.set_convert_running(False)
        self._step2.set_result(result)
        self._append_log(
            f"TensorRT engine: {result.output_path} "
            f"({result.engine_size_mb:.1f} MB, {result.precision.upper()})"
        )

    # ------------------------------------------------------------------
    # Step 3: Benchmark
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _run_benchmark(self) -> None:
        fmt = self._step3.model_format
        if fmt == "onnx":
            if not self._onnx_path or not Path(self._onnx_path).exists():
                QMessageBox.warning(
                    self, "No ONNX model",
                    "Export the model to ONNX (Step 1) before benchmarking."
                )
                return
            model_path = self._onnx_path
        else:
            if not self._weights_path or not Path(self._weights_path).exists():
                QMessageBox.warning(self, "No model", "Select a .pt file first.")
                return
            model_path = self._weights_path

        if self._bench_worker and self._bench_worker.isRunning():
            return

        # Infer image size from step 1 setting
        image_size = self._step1.image_size

        self._step3.set_benchmark_running(True)
        self._bench_worker = _BenchmarkWorker(
            model_path=model_path,
            model_format=fmt,
            image_size=image_size,
            n_iterations=self._step3.n_iterations,
            parent=self,
        )
        self._bench_worker.status.connect(self._on_status)
        self._bench_worker.result.connect(self._on_bench_result)
        self._bench_worker.error.connect(self._on_error)
        self._bench_worker.start()

    @pyqtSlot(object)
    def _on_bench_result(self, result: BenchmarkResult) -> None:
        self._step3.set_benchmark_running(False)
        self._step3.add_result(result)
        self._append_log(
            f"Benchmark [{result.format.upper()}] "
            f"mean={result.mean_latency_ms:.1f}ms "
            f"p95={result.p95_latency_ms:.1f}ms "
            f"fps={result.throughput_fps:.1f}"
        )

    # ------------------------------------------------------------------
    # Step 4: Package
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _run_package(self) -> None:
        if not self._weights_path or not Path(self._weights_path).exists():
            QMessageBox.warning(self, "No model", "Select a .pt file first.")
            return
        if not self._onnx_path or not Path(self._onnx_path).exists():
            QMessageBox.warning(
                self, "No ONNX model",
                "Complete Step 1 (ONNX export) before packaging."
            )
            return
        if not self._step4.output_dir:
            QMessageBox.warning(
                self, "No output directory",
                "Select an output directory before building the package."
            )
            return
        if self._pkg_worker and self._pkg_worker.isRunning():
            return

        # Derive package subdirectory name from model stem
        model_stem = Path(self._weights_path).stem
        pkg_dir = str(Path(self._step4.output_dir) / f"{model_stem}_deploy")

        # Use numbered class names if none were loaded
        effective_classes = (
            self._class_names
            if self._class_names
            else [f"class_{i}" for i in range(80)]
        )

        self._step4.set_build_running(True)
        self._pkg_worker = _PackageWorker(
            weights_path=self._weights_path,
            onnx_path=self._onnx_path,
            trt_path=self._trt_path,
            class_names=effective_classes,
            output_dir=pkg_dir,
            target_device=self._step4.target_device,
            conf_threshold=self._step4.conf_threshold,
            parent=self,
        )
        self._pkg_worker.status.connect(self._on_status)
        self._pkg_worker.result.connect(self._on_pkg_result)
        self._pkg_worker.error.connect(self._on_error)
        self._pkg_worker.start()

    @pyqtSlot(str)
    def _on_pkg_result(self, archive_path: str) -> None:
        self._step4.set_build_running(False)
        self._step4.set_result(archive_path)
        self._append_log(f"Deployment package: {archive_path}")

    # ------------------------------------------------------------------
    # Shared signal handlers
    # ------------------------------------------------------------------

    @pyqtSlot(str)
    def _on_status(self, msg: str) -> None:
        self._append_log(msg)

    @pyqtSlot(str)
    def _on_error(self, msg: str) -> None:
        # Reset running state for whichever step was active
        self._step1.set_export_running(False)
        self._step2.set_convert_running(False)
        self._step3.set_benchmark_running(False)
        self._step4.set_build_running(False)
        self._append_log(f"ERROR: {msg}")
        QMessageBox.critical(self, "Edge Export Error", msg)

    def _append_log(self, msg: str) -> None:
        self._log.append(msg)
        # Keep scroll at bottom
        self._log.verticalScrollBar().setValue(
            self._log.verticalScrollBar().maximum()
        )

    # ------------------------------------------------------------------
    # Session integration (no-op — model path is set manually)
    # ------------------------------------------------------------------

    @pyqtSlot(str)
    def on_session_changed(self, session_id: str) -> None:
        """No-op: model path for edge export is set manually via Browse."""
        self._session_id = session_id
