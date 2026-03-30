"""
labelox/desktop/widgets/review_widget.py — Review queue dialog.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from labelox.core.models import ReviewDecision
from labelox.desktop.theme import HI, MUTED, SUCCESS, WARNING


class ReviewDialog(QDialog):
    """Modal review queue — approve / reject images."""

    def __init__(self, project_id: str, parent=None) -> None:
        super().__init__(parent)
        self._project_id = project_id
        self._queue: list = []
        self._current_idx: int = -1
        self.setWindowTitle("Review Queue")
        self.setMinimumSize(600, 450)
        self._build_ui()
        self._load_queue()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Queue table
        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(["Image", "Annotations", "Confidence", "Status", "Reviewer"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.currentRowChanged.connect(self._on_row_changed)
        layout.addWidget(self._table)

        # Info
        self._info_label = QLabel("Select an image to review")
        self._info_label.setStyleSheet(f"color:{MUTED};")
        layout.addWidget(self._info_label)

        # Comment
        self._comment_edit = QLineEdit()
        self._comment_edit.setPlaceholderText("Review comment (optional)...")
        layout.addWidget(self._comment_edit)

        # Action buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        reject_btn = QPushButton("Reject")
        reject_btn.setObjectName("DangerBtn")
        reject_btn.clicked.connect(lambda: self._submit_review(ReviewDecision.REJECTED))
        btn_row.addWidget(reject_btn)

        needs_work_btn = QPushButton("Needs Work")
        needs_work_btn.setStyleSheet(f"background:{WARNING}; color:#000; font-weight:bold; border-radius:4px; padding:6px 16px;")
        needs_work_btn.clicked.connect(lambda: self._submit_review(ReviewDecision.NEEDS_WORK))
        btn_row.addWidget(needs_work_btn)

        approve_btn = QPushButton("Approve")
        approve_btn.setStyleSheet(f"background:{SUCCESS}; color:#000; font-weight:bold; border-radius:4px; padding:6px 16px;")
        approve_btn.clicked.connect(lambda: self._submit_review(ReviewDecision.APPROVED))
        btn_row.addWidget(approve_btn)

        layout.addLayout(btn_row)

        # Close
        close_row = QHBoxLayout()
        close_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_row.addWidget(close_btn)
        layout.addLayout(close_row)

    def _load_queue(self) -> None:
        from labelox.core.review_engine import get_review_queue
        self._queue = get_review_queue(self._project_id)
        self._table.setRowCount(len(self._queue))
        for i, item in enumerate(self._queue):
            self._table.setItem(i, 0, QTableWidgetItem(item.get("file_name", "?")))
            self._table.setItem(i, 1, QTableWidgetItem(str(item.get("annotation_count", 0))))
            conf = item.get("avg_confidence")
            self._table.setItem(i, 2, QTableWidgetItem(f"{conf:.2f}" if conf else "—"))
            self._table.setItem(i, 3, QTableWidgetItem(item.get("status", "?")))
            self._table.setItem(i, 4, QTableWidgetItem(item.get("reviewed_by", "—")))
        self._info_label.setText(f"{len(self._queue)} images in review queue")

    @pyqtSlot(int)
    def _on_row_changed(self, row: int) -> None:
        self._current_idx = row
        if 0 <= row < len(self._queue):
            item = self._queue[row]
            self._info_label.setText(
                f"{item.get('file_name', '?')} — "
                f"{item.get('annotation_count', 0)} annotations"
            )

    def _submit_review(self, decision: ReviewDecision) -> None:
        if self._current_idx < 0 or self._current_idx >= len(self._queue):
            QMessageBox.warning(self, "No Selection", "Select an image first.")
            return
        item = self._queue[self._current_idx]
        image_id = item.get("image_id", "")
        comment = self._comment_edit.text().strip()

        try:
            from labelox.core.review_engine import submit_review
            submit_review(
                image_id=image_id,
                decision=decision,
                reviewer_id="desktop_user",
                comment=comment or None,
            )
            self._comment_edit.clear()
            self._load_queue()
            # Auto-advance
            if self._current_idx < len(self._queue):
                self._table.setCurrentCell(self._current_idx, 0)
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))
