"""
labelox/desktop/widgets/stats_widget.py — Project statistics dialog.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from labelox.desktop.theme import HI, MUTED, SUCCESS, TEXT, WARNING


class StatsDialog(QDialog):
    """Modal dialog showing project statistics."""

    def __init__(self, project_id: str, parent=None) -> None:
        super().__init__(parent)
        self._project_id = project_id
        self.setWindowTitle("Project Statistics")
        self.setMinimumSize(550, 500)
        self._build_ui()
        self._load_stats()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Summary
        self._summary_group = QGroupBox("Summary")
        self._summary_layout = QVBoxLayout(self._summary_group)
        self._summary_labels: list[QLabel] = []
        for _ in range(6):
            lbl = QLabel("—")
            lbl.setStyleSheet(f"color:{TEXT}; font-size:12px;")
            self._summary_layout.addWidget(lbl)
            self._summary_labels.append(lbl)
        layout.addWidget(self._summary_group)

        # Class balance
        class_group = QGroupBox("Class Distribution")
        class_layout = QVBoxLayout(class_group)
        self._class_table = QTableWidget()
        self._class_table.setColumnCount(3)
        self._class_table.setHorizontalHeaderLabels(["Class", "Count", "%"])
        self._class_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        class_layout.addWidget(self._class_table)
        layout.addWidget(class_group)

        # Close
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def _load_stats(self) -> None:
        try:
            from labelox.core.stats_engine import compute_project_stats, compute_class_balance
            stats = compute_project_stats(self._project_id)
            balance = compute_class_balance(self._project_id)
        except Exception:
            self._summary_labels[0].setText("Error loading stats")
            return

        labels = [
            f"Total images: {stats.get('total_images', 0)}",
            f"Annotated: {stats.get('annotated_images', 0)}",
            f"Reviewed: {stats.get('reviewed_images', 0)}",
            f"Total annotations: {stats.get('total_annotations', 0)}",
            f"Auto annotations: {stats.get('auto_annotations', 0)}",
            f"Avg annotations/image: {stats.get('avg_annotations_per_image', 0):.1f}",
        ]
        for i, text in enumerate(labels):
            if i < len(self._summary_labels):
                self._summary_labels[i].setText(text)

        # Class table
        self._class_table.setRowCount(len(balance))
        total_ann = sum(item.get("count", 0) for item in balance) or 1
        for i, item in enumerate(balance):
            self._class_table.setItem(i, 0, QTableWidgetItem(item.get("class_name", "?")))
            count = item.get("count", 0)
            self._class_table.setItem(i, 1, QTableWidgetItem(str(count)))
            pct = count / total_ann * 100
            self._class_table.setItem(i, 2, QTableWidgetItem(f"{pct:.1f}%"))
