"""
desktop/widgets/inference_widget.py — Model registry + live inference UI.

Left panel  : model registry table + register/set-active/delete controls
              + single-image inference controls (path, conf, IoU)
Right panel : image preview with bbox overlay + results table + status bar
"""
from __future__ import annotations

import os
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.inference_server import (
    REGISTRY_FILE,
    delete_model,
    get_active_model,
    load_registry,
    register_model,
    set_active_model,
)
from core.models import InferenceRequest
from core.session_manager import SessionManager
from desktop.theme import BG, PANEL, ACCENT, HI, TEXT, MUTED, SUCCESS, WARNING, BORDER, CARD
from desktop.workers import InferenceWorker

# Semantic aliases used in this file
GREEN  = SUCCESS
ORANGE = WARNING

BBOX_COLORS = [
    QColor(229, 57, 96),   # HI red
    QColor(76, 175, 80),   # green
    QColor(33, 150, 243),  # blue
    QColor(255, 152, 0),   # orange
    QColor(156, 39, 176),  # purple
]


class InferenceWidget(QWidget):
    """Model registry browser + single/session inference UI."""

    def __init__(self, sm: SessionManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sm = sm
        self._session_id: str | None = None
        self._worker: InferenceWorker | None = None
        self._current_image_path: str | None = None
        self._last_detections: list = []

        self._build_ui()
        self._refresh_registry()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal, self)

        # ── Left panel ────────────────────────────────────────────────
        left = QWidget()
        left.setMinimumWidth(300)
        left.setMaximumWidth(420)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(6, 6, 6, 6)
        lv.setSpacing(8)

        # Registry group
        reg_grp = QGroupBox("Model Registry")
        rg = QVBoxLayout(reg_grp)

        self._reg_table = QTableWidget(0, 5)
        self._reg_table.setHorizontalHeaderLabels(["Name", "Variant", "mAP50", "Active", "ID"])
        self._reg_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._reg_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._reg_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._reg_table.setColumnWidth(1, 70)
        self._reg_table.setColumnWidth(2, 55)
        self._reg_table.setColumnWidth(3, 50)
        self._reg_table.setColumnHidden(4, True)
        self._reg_table.setMinimumHeight(180)
        rg.addWidget(self._reg_table)

        btn_row = QHBoxLayout()
        self._btn_register = QPushButton("Register…")
        self._btn_activate  = QPushButton("Set Active")
        self._btn_del_model = QPushButton("Delete")
        self._btn_del_model.setObjectName("DangerBtn")
        for b in (self._btn_register, self._btn_activate, self._btn_del_model):
            btn_row.addWidget(b)
        rg.addLayout(btn_row)
        lv.addWidget(reg_grp)

        # Run inference group
        infer_grp = QGroupBox("Run Inference")
        ig = QVBoxLayout(infer_grp)

        path_row = QHBoxLayout()
        self._img_path_edit = QLineEdit()
        self._img_path_edit.setPlaceholderText("Image path…")
        self._btn_browse = QPushButton("Browse…")
        path_row.addWidget(self._img_path_edit)
        path_row.addWidget(self._btn_browse)
        ig.addLayout(path_row)

        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel("Conf:"))
        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.01, 1.0)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setValue(0.25)
        self._conf_spin.setFixedWidth(65)
        conf_row.addWidget(self._conf_spin)
        conf_row.addSpacing(10)
        conf_row.addWidget(QLabel("IoU:"))
        self._iou_spin = QDoubleSpinBox()
        self._iou_spin.setRange(0.01, 1.0)
        self._iou_spin.setSingleStep(0.05)
        self._iou_spin.setValue(0.45)
        self._iou_spin.setFixedWidth(65)
        conf_row.addWidget(self._iou_spin)
        conf_row.addStretch()
        ig.addLayout(conf_row)

        btn_row2 = QHBoxLayout()
        self._btn_run_single  = QPushButton("Run Single")
        self._btn_run_session = QPushButton("Run on Session")
        btn_row2.addWidget(self._btn_run_single)
        btn_row2.addWidget(self._btn_run_session)
        ig.addLayout(btn_row2)

        lv.addWidget(infer_grp)
        lv.addStretch()

        # ── Right panel ───────────────────────────────────────────────
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(6, 6, 6, 6)
        rv.setSpacing(6)

        self._preview_label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setObjectName("preview")
        self._preview_label.setMinimumSize(480, 360)
        self._preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._preview_label.setText("No image loaded")
        rv.addWidget(self._preview_label)

        self._results_table = QTableWidget(0, 4)
        self._results_table.setHorizontalHeaderLabels(["Class", "Confidence", "x1,y1", "x2,y2"])
        self._results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._results_table.setMinimumHeight(100)
        self._results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        rv.addWidget(self._results_table)

        self._status_label = QLabel("Ready")
        rv.addWidget(self._status_label)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(splitter)

        # ── Connections ───────────────────────────────────────────────
        self._btn_register.clicked.connect(self._on_register)
        self._btn_activate.clicked.connect(self._on_set_active)
        self._btn_del_model.clicked.connect(self._on_delete_model)
        self._btn_browse.clicked.connect(self._on_browse)
        self._btn_run_single.clicked.connect(self._on_run_single)
        self._btn_run_session.clicked.connect(self._on_run_session)

    # ------------------------------------------------------------------
    # Public API (called by AutoLabel widget "Use for inference →")
    # ------------------------------------------------------------------

    def get_active_model_path(self) -> str | None:
        """Return the weights path of the currently active model, or None."""
        try:
            m = get_active_model(REGISTRY_FILE)
            return m.weights_path
        except Exception:
            return None

    @pyqtSlot(str)
    def on_session_changed(self, session_id: str) -> None:
        self._session_id = session_id
        self._btn_run_session.setEnabled(bool(session_id))

    # ------------------------------------------------------------------
    # Registry helpers
    # ------------------------------------------------------------------

    def _refresh_registry(self) -> None:
        models = load_registry(REGISTRY_FILE)
        self._reg_table.setRowCount(0)
        for m in models:
            row = self._reg_table.rowCount()
            self._reg_table.insertRow(row)
            map50 = m.metrics.get("map50", 0.0)
            active_txt = "✓" if m.is_active else ""
            for col, val in enumerate([m.name, m.model_variant, f"{map50:.3f}", active_txt, m.model_id]):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if m.is_active:
                    item.setForeground(QColor(GREEN))
                self._reg_table.setItem(row, col, item)

    def _selected_model_id(self) -> str | None:
        rows = self._reg_table.selectedItems()
        if not rows:
            return None
        row = self._reg_table.currentRow()
        id_item = self._reg_table.item(row, 4)
        return id_item.text() if id_item else None

    # ------------------------------------------------------------------
    # Slots — registry
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_register(self) -> None:
        weights_path, _ = QFileDialog.getOpenFileName(
            self, "Select weights file", str(Path.home()),
            "Weights (*.pt *.pth);;All Files (*)"
        )
        if not weights_path:
            return
        name, ok = _simple_input(self, "Model name", "Enter a short name for this model:")
        if not ok or not name.strip():
            return
        variant, ok = _simple_input(self, "Model variant", "Enter model variant (e.g. yolov8n):", "yolov8n")
        if not ok:
            return
        try:
            register_model(
                weights_path=weights_path,
                name=name.strip(),
                model_variant=variant.strip() or "yolov8n",
                registry_path=REGISTRY_FILE,
            )
            self._refresh_registry()
            self._status_label.setText(f"Registered: {name.strip()}")
        except Exception as exc:
            QMessageBox.critical(self, "Register failed", str(exc))

    @pyqtSlot()
    def _on_set_active(self) -> None:
        model_id = self._selected_model_id()
        if not model_id:
            QMessageBox.warning(self, "No selection", "Select a model in the table first.")
            return
        try:
            set_active_model(model_id, REGISTRY_FILE)
            self._refresh_registry()
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

    @pyqtSlot()
    def _on_delete_model(self) -> None:
        model_id = self._selected_model_id()
        if not model_id:
            QMessageBox.warning(self, "No selection", "Select a model first.")
            return
        ans = QMessageBox.question(
            self, "Delete model",
            "Remove this model from the registry?\n(Weights file is NOT deleted.)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return
        try:
            delete_model(model_id, REGISTRY_FILE)
            self._refresh_registry()
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

    # ------------------------------------------------------------------
    # Slots — inference
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select image", str(Path.home()),
            "Images (*.jpg *.jpeg *.png *.bmp);;All Files (*)"
        )
        if path:
            self._img_path_edit.setText(path)

    @pyqtSlot()
    def _on_run_single(self) -> None:
        img_path = self._img_path_edit.text().strip()
        if not img_path:
            QMessageBox.warning(self, "No image", "Enter or browse an image path.")
            return
        if not Path(img_path).exists():
            QMessageBox.warning(self, "File not found", f"Cannot read: {img_path}")
            return
        self._run_inference_worker([InferenceRequest(
            image_path=img_path,
            conf_threshold=self._conf_spin.value(),
            iou_threshold=self._iou_spin.value(),
        )])
        self._current_image_path = img_path

    @pyqtSlot()
    def _on_run_session(self) -> None:
        if not self._session_id:
            return
        frames_dir = self._sm.session_folder(self._session_id) / "frames"
        if not frames_dir.exists():
            QMessageBox.warning(
                self, "No frames",
                "No extracted frames found for this session.\n"
                "Run frame extraction first."
            )
            return
        jpg_paths = sorted(frames_dir.rglob("*.jpg"))[:50]  # cap at 50 for UI demo
        if not jpg_paths:
            QMessageBox.warning(self, "No frames", "No JPEG frames found.")
            return
        requests = [
            InferenceRequest(
                image_path=str(p),
                conf_threshold=self._conf_spin.value(),
                iou_threshold=self._iou_spin.value(),
            )
            for p in jpg_paths
        ]
        self._run_inference_worker(requests)
        self._current_image_path = str(jpg_paths[0])

    def _run_inference_worker(self, requests: list[InferenceRequest]) -> None:
        if self._worker and self._worker.isRunning():
            return
        self._btn_run_single.setEnabled(False)
        self._btn_run_session.setEnabled(False)
        self._status_label.setText("Running inference…")

        self._worker = InferenceWorker(requests, str(REGISTRY_FILE))
        self._worker.status.connect(self._status_label.setText)
        self._worker.result.connect(self._on_inference_done)
        self._worker.error.connect(self._on_inference_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    @pyqtSlot(object)
    def _on_inference_done(self, results: object) -> None:
        if not isinstance(results, list) or not results:
            return
        first = results[0]
        self._last_detections = first.detections
        elapsed = first.inference_time_ms
        self._status_label.setText(
            f"Done — {len(first.detections)} detection(s) | "
            f"{elapsed:.1f} ms | model: {first.model_variant}"
        )
        self._populate_results_table(first.detections)
        if self._current_image_path:
            self._draw_preview(self._current_image_path, first.detections)

    @pyqtSlot(str)
    def _on_inference_error(self, msg: str) -> None:
        self._status_label.setText(f"Error: {msg}")
        QMessageBox.critical(self, "Inference error", msg)

    @pyqtSlot()
    def _on_worker_finished(self) -> None:
        self._btn_run_single.setEnabled(True)
        self._btn_run_session.setEnabled(bool(self._session_id))

    # ------------------------------------------------------------------
    # Preview rendering
    # ------------------------------------------------------------------

    def _draw_preview(self, image_path: str, detections: list) -> None:
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self._preview_label.setText("Could not load image")
            return

        # Scale to fit label while keeping aspect ratio
        label_size = self._preview_label.size()
        scaled = pixmap.scaled(
            label_size, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        w, h = scaled.width(), scaled.height()
        orig_w, orig_h = pixmap.width(), pixmap.height()
        x_scale = w / orig_w
        y_scale = h / orig_h

        painter = QPainter(scaled)
        font = QFont("monospace", 9)
        painter.setFont(font)

        for i, det in enumerate(detections):
            color = BBOX_COLORS[i % len(BBOX_COLORS)]
            pen = QPen(color, 2)
            painter.setPen(pen)
            x1, y1, x2, y2 = det.bbox_xyxy
            rx1 = int(x1 * x_scale)
            ry1 = int(y1 * y_scale)
            rx2 = int(x2 * x_scale)
            ry2 = int(y2 * y_scale)
            painter.drawRect(rx1, ry1, rx2 - rx1, ry2 - ry1)
            label = f"{det.class_name} {det.confidence:.2f}"
            painter.fillRect(rx1, ry1 - 14, len(label) * 7, 14, color)
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(rx1 + 2, ry1 - 2, label)

        painter.end()
        self._preview_label.setPixmap(scaled)

    def _populate_results_table(self, detections: list) -> None:
        self._results_table.setRowCount(0)
        for det in detections:
            row = self._results_table.rowCount()
            self._results_table.insertRow(row)
            x1, y1, x2, y2 = det.bbox_xyxy
            for col, val in enumerate([
                det.class_name,
                f"{det.confidence:.3f}",
                f"{x1:.0f},{y1:.0f}",
                f"{x2:.0f},{y2:.0f}",
            ]):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._results_table.setItem(row, col, item)


# ---------------------------------------------------------------------------
# Minimal inline input dialog (avoids QInputDialog dependency quirk)
# ---------------------------------------------------------------------------

def _simple_input(parent: QWidget, title: str, label: str, default: str = "") -> tuple[str, bool]:
    from PyQt6.QtWidgets import QDialog, QDialogButtonBox
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    lv = QVBoxLayout(dlg)
    lv.addWidget(QLabel(label))
    edit = QLineEdit(default)
    lv.addWidget(edit)
    btns = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
    )
    btns.accepted.connect(dlg.accept)
    btns.rejected.connect(dlg.reject)
    lv.addWidget(btns)
    if dlg.exec() == QDialog.DialogCode.Accepted:
        return edit.text(), True
    return "", False
