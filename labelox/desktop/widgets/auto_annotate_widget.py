"""
labelox/desktop/widgets/auto_annotate_widget.py — Auto-annotation dialog (YOLOv8).
"""
from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from labelox.desktop.theme import MUTED


class _AutoAnnotateWorker(QThread):
    """Worker thread for running YOLOv8 auto-annotation."""

    progress = pyqtSignal(int, int)    # (current, total)
    status = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)     # result dict
    error = pyqtSignal(str)

    def __init__(self, project_id: str, model_path: str, conf: float, parent=None):
        super().__init__(parent)
        self._project_id = project_id
        self._model_path = model_path
        self._conf = conf

    def run(self) -> None:
        try:
            from labelox.core.auto_annotator import run_auto_annotate_project
            result = run_auto_annotate_project(
                self._project_id,
                model_path=self._model_path,
                conf_threshold=self._conf,
                progress_callback=lambda cur, tot: self.progress.emit(cur, tot),
                status_callback=lambda msg: self.status.emit(msg),
            )
            self.finished_ok.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class AutoAnnotateDialog(QDialog):
    """Modal dialog to run YOLOv8 auto-annotation on a project."""

    def __init__(self, project_id: str, parent=None) -> None:
        super().__init__(parent)
        self._project_id = project_id
        self._worker: _AutoAnnotateWorker | None = None
        self.setWindowTitle("Auto-Annotate (YOLOv8)")
        self.setMinimumWidth(450)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()

        self._model_combo = QComboBox()
        self._model_combo.addItems([
            "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
        ])
        form.addRow("Model:", self._model_combo)

        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.05, 0.95)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setValue(0.25)
        form.addRow("Confidence:", self._conf_spin)

        layout.addLayout(form)

        # Log
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(150)
        self._log.setStyleSheet(f"color:{MUTED}; font-size:10px; font-family:monospace;")
        layout.addWidget(self._log)

        # Progress
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet(f"color:{MUTED};")
        layout.addWidget(self._status_label)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("Close")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        self._run_btn = QPushButton("Run")
        self._run_btn.setObjectName("PrimaryBtn")
        self._run_btn.clicked.connect(self._start)
        btn_row.addWidget(self._run_btn)
        layout.addLayout(btn_row)

    @pyqtSlot()
    def _start(self) -> None:
        self._run_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)
        self._log.clear()
        self._status_label.setText("Loading model...")

        self._worker = _AutoAnnotateWorker(
            self._project_id,
            self._model_combo.currentText(),
            self._conf_spin.value(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.status.connect(self._on_status)
        self._worker.finished_ok.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    @pyqtSlot(int, int)
    def _on_progress(self, current: int, total: int) -> None:
        self._progress.setRange(0, total)
        self._progress.setValue(current)

    @pyqtSlot(str)
    def _on_status(self, msg: str) -> None:
        self._status_label.setText(msg)
        self._log.append(msg)

    @pyqtSlot(dict)
    def _on_done(self, result: dict) -> None:
        total = result.get("total_detections", 0)
        processed = result.get("processed", 0)
        self._progress.setRange(0, 1)
        self._progress.setValue(1)
        self._status_label.setText(f"Done — {total} detections across {processed} images")
        self._run_btn.setEnabled(True)
        QMessageBox.information(self, "Complete", f"{total} detections across {processed} images.")

    @pyqtSlot(str)
    def _on_error(self, msg: str) -> None:
        self._progress.setRange(0, 1)
        self._status_label.setText(f"Error: {msg}")
        self._run_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", msg)
