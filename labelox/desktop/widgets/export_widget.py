"""
labelox/desktop/widgets/export_widget.py — Export dialog for YOLO/COCO/CVAT/VOC/CSV.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from labelox.core.models import ExportConfig, ExportFormat
from labelox.desktop.theme import MUTED


class ExportDialog(QDialog):
    """Modal dialog to configure and run a project export."""

    def __init__(self, project_id: str, parent=None) -> None:
        super().__init__(parent)
        self._project_id = project_id
        self.setWindowTitle("Export Annotations")
        self.setMinimumWidth(420)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()

        self._format_combo = QComboBox()
        for fmt in ExportFormat:
            self._format_combo.addItem(fmt.value.upper(), fmt.value)
        form.addRow("Format:", self._format_combo)

        # Output directory
        dir_row = QHBoxLayout()
        self._dir_edit = QLineEdit()
        self._dir_edit.setPlaceholderText("Select output directory...")
        dir_row.addWidget(self._dir_edit, 1)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_dir)
        dir_row.addWidget(browse_btn)
        form.addRow("Output:", dir_row)

        self._include_images = QCheckBox("Copy images to output")
        self._include_images.setChecked(True)
        form.addRow("", self._include_images)

        self._only_reviewed = QCheckBox("Only reviewed images")
        form.addRow("", self._only_reviewed)

        self._split_check = QCheckBox("Split train/val")
        self._split_check.setChecked(True)
        form.addRow("", self._split_check)

        self._ratio_spin = QDoubleSpinBox()
        self._ratio_spin.setRange(0.5, 0.95)
        self._ratio_spin.setSingleStep(0.05)
        self._ratio_spin.setValue(0.8)
        form.addRow("Train ratio:", self._ratio_spin)

        layout.addLayout(form)

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
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        self._export_btn = QPushButton("Export")
        self._export_btn.setObjectName("PrimaryBtn")
        self._export_btn.clicked.connect(self._run_export)
        btn_row.addWidget(self._export_btn)
        layout.addLayout(btn_row)

    def _browse_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Output Directory")
        if d:
            self._dir_edit.setText(d)

    @pyqtSlot()
    def _run_export(self) -> None:
        out_dir = self._dir_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "Missing", "Select an output directory.")
            return

        fmt = self._format_combo.currentData()
        config = ExportConfig(
            project_id=self._project_id,
            format=ExportFormat(fmt),
            output_dir=out_dir,
            include_images=self._include_images.isChecked(),
            only_reviewed=self._only_reviewed.isChecked(),
            split_train_val=self._split_check.isChecked(),
            train_ratio=self._ratio_spin.value(),
        )

        self._export_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)  # indeterminate
        self._status_label.setText("Exporting...")

        try:
            from labelox.core.exporter import export_project
            result = export_project(config)
            self._progress.setRange(0, 1)
            self._progress.setValue(1)
            self._status_label.setText(
                f"Done — {result.total_images} images, "
                f"{result.total_annotations} annotations, "
                f"{len(result.files_created)} files"
            )
            QMessageBox.information(
                self, "Export Complete",
                f"Exported {result.total_images} images with "
                f"{result.total_annotations} annotations to:\n{out_dir}",
            )
        except Exception as exc:
            self._progress.setRange(0, 1)
            self._status_label.setText(f"Error: {exc}")
            QMessageBox.critical(self, "Export Error", str(exc))
        finally:
            self._export_btn.setEnabled(True)
