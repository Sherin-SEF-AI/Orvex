"""
labelox/desktop/widgets/properties_panel.py — Selected annotation properties editor.
"""
from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from labelox.core.models import Annotation, AnnotationType
from labelox.desktop.theme import BORDER, HI, MUTED, TEXT


class PropertiesPanelWidget(QWidget):
    """Displays and edits properties of the currently selected annotation."""

    annotation_modified = pyqtSignal(object)  # Annotation

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._annotation: Annotation | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        group = QGroupBox("Properties")
        form = QFormLayout(group)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._id_label = QLabel("—")
        self._id_label.setStyleSheet(f"color:{MUTED}; font-size:10px;")
        form.addRow("ID:", self._id_label)

        self._type_label = QLabel("—")
        form.addRow("Type:", self._type_label)

        self._class_label = QLabel("—")
        form.addRow("Class:", self._class_label)

        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.0, 1.0)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setDecimals(2)
        self._conf_spin.setEnabled(False)
        form.addRow("Confidence:", self._conf_spin)

        self._source_label = QLabel("—")
        form.addRow("Source:", self._source_label)

        self._track_label = QLabel("—")
        form.addRow("Track ID:", self._track_label)

        # Coords group
        self._coords_label = QLabel("—")
        self._coords_label.setWordWrap(True)
        self._coords_label.setStyleSheet(f"color:{TEXT}; font-size:10px; font-family:monospace;")
        form.addRow("Coords:", self._coords_label)

        # Comment
        self._comment_edit = QLineEdit()
        self._comment_edit.setPlaceholderText("Add comment...")
        self._comment_edit.setEnabled(False)
        self._comment_edit.editingFinished.connect(self._on_comment_changed)
        form.addRow("Comment:", self._comment_edit)

        layout.addWidget(group)

        # Placeholder for when nothing is selected
        self._empty_label = QLabel("Select an annotation to view properties")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet(f"color:{MUTED}; font-style:italic;")
        layout.addWidget(self._empty_label)

        layout.addStretch()

    def set_annotation(self, ann: Annotation) -> None:
        self._annotation = ann
        self._empty_label.hide()

        self._id_label.setText(ann.id[:12] + "…")
        self._id_label.setToolTip(ann.id)
        self._type_label.setText(ann.annotation_type.value)
        self._class_label.setText(ann.label_name)

        if ann.confidence is not None:
            self._conf_spin.setEnabled(True)
            self._conf_spin.setValue(ann.confidence)
        else:
            self._conf_spin.setEnabled(False)
            self._conf_spin.setValue(0.0)

        self._source_label.setText("Auto 🤖" if ann.is_auto else "Manual ✋")
        self._track_label.setText(str(ann.track_id) if ann.track_id is not None else "—")

        self._coords_label.setText(self._format_coords(ann))

        self._comment_edit.setEnabled(True)
        self._comment_edit.setText(ann.comment or "")

    def clear(self) -> None:
        self._annotation = None
        self._id_label.setText("—")
        self._type_label.setText("—")
        self._class_label.setText("—")
        self._conf_spin.setEnabled(False)
        self._conf_spin.setValue(0.0)
        self._source_label.setText("—")
        self._track_label.setText("—")
        self._coords_label.setText("—")
        self._comment_edit.setEnabled(False)
        self._comment_edit.setText("")
        self._empty_label.show()

    def _format_coords(self, ann: Annotation) -> str:
        if ann.annotation_type == AnnotationType.BBOX and ann.bbox:
            b = ann.bbox
            return f"x={b.x:.4f} y={b.y:.4f}\nw={b.width:.4f} h={b.height:.4f}"
        elif ann.annotation_type == AnnotationType.POLYGON and ann.polyline:
            n = len(ann.polyline.points)
            return f"{n} vertices"
        elif ann.annotation_type == AnnotationType.POLYLINE and ann.polyline:
            n = len(ann.polyline.points)
            return f"{n} points"
        elif ann.annotation_type == AnnotationType.MASK and ann.mask:
            return "RLE mask"
        elif ann.annotation_type == AnnotationType.KEYPOINT and ann.keypoints:
            n = len(ann.keypoints.points)
            return f"{n} keypoints"
        elif ann.annotation_type == AnnotationType.CUBOID_3D and ann.cuboid_3d:
            d = ann.cuboid_3d.depth_estimate_m
            return f"8 corners\ndepth={d:.2f}m" if d else "8 corners"
        elif ann.annotation_type == AnnotationType.CLASSIFICATION:
            return ann.classification or "—"
        return "—"

    def _on_comment_changed(self) -> None:
        if self._annotation:
            self._annotation.comment = self._comment_edit.text() or None
            self.annotation_modified.emit(self._annotation)
