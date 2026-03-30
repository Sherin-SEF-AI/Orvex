"""
labelox/desktop/widgets/label_panel.py — Right panel: class list + annotation list.
"""
from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from labelox.core.models import Annotation
from labelox.desktop.theme import ANNOTATION_COLORS, BORDER, HI, MUTED, TEXT


class LabelPanelWidget(QWidget):
    """Class selector + annotation list for current image."""

    label_selected = pyqtSignal(str, str)      # (label_id, label_name)
    annotation_clicked = pyqtSignal(str)        # annotation_id

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._classes: list[dict] = []
        self._annotations: list[Annotation] = []
        self._selected_class_idx: int = -1
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # ── Classes ──────────────────────────────────────────────────────
        cls_group = QGroupBox("Labels")
        cls_layout = QVBoxLayout(cls_group)

        self._class_list = QListWidget()
        self._class_list.currentRowChanged.connect(self._on_class_row_changed)
        cls_layout.addWidget(self._class_list)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("+ Add Class")
        add_btn.setObjectName("PrimaryBtn")
        add_btn.clicked.connect(self._add_class)
        btn_row.addWidget(add_btn)
        cls_layout.addLayout(btn_row)

        layout.addWidget(cls_group)

        # ── Annotations on current image ─────────────────────────────────
        ann_group = QGroupBox("Annotations")
        ann_layout = QVBoxLayout(ann_group)

        self._ann_list = QListWidget()
        self._ann_list.currentRowChanged.connect(self._on_ann_row_changed)
        ann_layout.addWidget(self._ann_list)

        del_btn = QPushButton("Delete Selected")
        del_btn.setObjectName("DangerBtn")
        del_btn.clicked.connect(self._delete_selected)
        ann_layout.addWidget(del_btn)

        layout.addWidget(ann_group)

    # ─── Class Management ────────────────────────────────────────────────

    def set_classes(self, classes: list[dict]) -> None:
        self._classes = classes
        self._class_list.clear()
        for i, cls in enumerate(classes):
            color = cls.get("color", ANNOTATION_COLORS[i % len(ANNOTATION_COLORS)])
            hotkey = cls.get("hotkey", "")
            name = cls.get("name", "unnamed")

            item = QListWidgetItem()
            text = f"[{hotkey}] {name}" if hotkey else name
            item.setText(text)
            item.setForeground(QColor(color))
            item.setData(Qt.ItemDataRole.UserRole, cls.get("id", name))
            item.setData(Qt.ItemDataRole.UserRole + 1, name)
            self._class_list.addItem(item)

        if classes:
            self._class_list.setCurrentRow(0)

    def select_by_index(self, idx: int) -> None:
        if 0 <= idx < self._class_list.count():
            self._class_list.setCurrentRow(idx)

    def _on_class_row_changed(self, row: int) -> None:
        if row < 0:
            return
        self._selected_class_idx = row
        item = self._class_list.item(row)
        if item:
            label_id = str(item.data(Qt.ItemDataRole.UserRole))
            label_name = str(item.data(Qt.ItemDataRole.UserRole + 1))
            self.label_selected.emit(label_id, label_name)

    def _add_class(self) -> None:
        from PyQt6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Add Class", "Class name:")
        if ok and name.strip():
            idx = len(self._classes)
            color = ANNOTATION_COLORS[idx % len(ANNOTATION_COLORS)]
            new_cls = {"name": name.strip(), "color": color, "id": f"cls_{name.strip()}"}
            self._classes.append(new_cls)
            self.set_classes(self._classes)

    # ─── Annotation List ─────────────────────────────────────────────────

    def set_annotations(self, annotations: list[Annotation]) -> None:
        self._annotations = annotations
        self._ann_list.clear()
        for i, ann in enumerate(annotations):
            auto_badge = " 🤖" if ann.is_auto else " ✋"
            conf = f" ({ann.confidence:.2f})" if ann.confidence is not None else ""
            text = f"{ann.label_name} [{ann.annotation_type.value}]{conf}{auto_badge}"

            item = QListWidgetItem(text)
            color = ANNOTATION_COLORS[i % len(ANNOTATION_COLORS)]
            item.setForeground(QColor(color))
            item.setData(Qt.ItemDataRole.UserRole, ann.id)
            self._ann_list.addItem(item)

    def highlight_annotation(self, ann_id: str) -> None:
        for i in range(self._ann_list.count()):
            item = self._ann_list.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == ann_id:
                self._ann_list.setCurrentRow(i)
                return

    def _on_ann_row_changed(self, row: int) -> None:
        if row < 0:
            return
        item = self._ann_list.item(row)
        if item:
            ann_id = str(item.data(Qt.ItemDataRole.UserRole))
            self.annotation_clicked.emit(ann_id)

    def _delete_selected(self) -> None:
        row = self._ann_list.currentRow()
        if row >= 0 and row < len(self._annotations):
            self._annotations.pop(row)
            self.set_annotations(self._annotations)
