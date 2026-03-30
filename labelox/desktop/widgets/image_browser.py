"""
labelox/desktop/widgets/image_browser.py — Left panel thumbnail grid browser.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSize, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from labelox.core.database import DBImage, get_images, get_session
from labelox.desktop.theme import BORDER, CARD, HI, MUTED, SUCCESS, WARNING


_STATUS_COLORS = {
    "unlabeled": "#555566",
    "in_progress": WARNING,
    "annotated": "#f5a623",
    "reviewed": SUCCESS,
    "rejected": HI,
    "skipped": MUTED,
}


class ImageBrowserWidget(QWidget):
    """Thumbnail grid with filtering and status badges."""

    image_selected = pyqtSignal(str, str)  # (image_id, image_path)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project_id: str = ""
        self._images: list[DBImage] = []
        self._current_index: int = -1
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Search
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search images...")
        self._search.textChanged.connect(self._filter)
        layout.addWidget(self._search)

        # Filter row
        filter_row = QHBoxLayout()
        self._status_filter = QComboBox()
        self._status_filter.addItems(["All", "Unlabeled", "Annotated", "Reviewed", "Rejected"])
        self._status_filter.currentIndexChanged.connect(self._filter)
        filter_row.addWidget(QLabel("Status:"))
        filter_row.addWidget(self._status_filter)
        layout.addLayout(filter_row)

        # Count label
        self._count_label = QLabel("0 images")
        self._count_label.setStyleSheet(f"color:{MUTED}; font-size:10px;")
        layout.addWidget(self._count_label)

        # Thumbnail list
        self._list = QListWidget()
        self._list.setViewMode(QListWidget.ViewMode.IconMode)
        self._list.setIconSize(QSize(80, 80))
        self._list.setGridSize(QSize(90, 100))
        self._list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._list.setWrapping(True)
        self._list.setSpacing(4)
        self._list.currentRowChanged.connect(self._on_row_changed)
        layout.addWidget(self._list)

    # ─── Data Loading ────────────────────────────────────────────────────

    def set_project(self, project_id: str) -> None:
        self._project_id = project_id
        self._load_images()

    def refresh(self) -> None:
        self._load_images()

    def _load_images(self) -> None:
        if not self._project_id:
            return
        db = get_session()
        try:
            self._images = get_images(self._project_id, limit=99999, db=db)
        finally:
            db.close()
        self._populate_list()

    def _populate_list(self) -> None:
        self._list.clear()
        search_text = self._search.text().lower()
        status_filter = self._status_filter.currentText().lower()

        filtered: list[DBImage] = []
        for img in self._images:
            if search_text and search_text not in img.file_name.lower():
                continue
            if status_filter != "all" and img.status != status_filter:
                continue
            filtered.append(img)

        for img in filtered:
            icon = self._get_thumbnail_icon(img)
            item = QListWidgetItem(icon, img.file_name[:12])
            item.setData(Qt.ItemDataRole.UserRole, img.id)
            item.setData(Qt.ItemDataRole.UserRole + 1, img.file_path)
            item.setToolTip(f"{img.file_name}\n{img.width}x{img.height}\nStatus: {img.status}")

            # Status color
            color = _STATUS_COLORS.get(img.status, MUTED)
            item.setForeground(QColor(color))

            self._list.addItem(item)

        self._count_label.setText(f"{self._list.count()} / {len(self._images)} images")

    def _get_thumbnail_icon(self, img: DBImage) -> QIcon:
        """Load thumbnail or generate a placeholder."""
        if img.thumbnail_path and Path(img.thumbnail_path).exists():
            pm = QPixmap(img.thumbnail_path)
            if not pm.isNull():
                return QIcon(pm.scaled(80, 80, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        # Try loading from source (scaled down)
        pm = QPixmap(img.file_path)
        if not pm.isNull():
            return QIcon(pm.scaled(80, 80, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        # Placeholder
        pm = QPixmap(80, 80)
        pm.fill(QColor(CARD))
        return QIcon(pm)

    # ─── Filtering ───────────────────────────────────────────────────────

    @pyqtSlot()
    def _filter(self) -> None:
        self._populate_list()

    # ─── Selection ───────────────────────────────────────────────────────

    @pyqtSlot(int)
    def _on_row_changed(self, row: int) -> None:
        if row < 0:
            return
        item = self._list.item(row)
        if item:
            image_id = item.data(Qt.ItemDataRole.UserRole)
            image_path = item.data(Qt.ItemDataRole.UserRole + 1)
            self._current_index = row
            self.image_selected.emit(str(image_id), str(image_path))

    def select_previous(self) -> None:
        if self._list.count() == 0:
            return
        new_row = max(0, self._current_index - 1)
        self._list.setCurrentRow(new_row)

    def select_next(self) -> None:
        if self._list.count() == 0:
            return
        new_row = min(self._list.count() - 1, self._current_index + 1)
        self._list.setCurrentRow(new_row)

    def select_by_index(self, index: int) -> None:
        if 0 <= index < self._list.count():
            self._list.setCurrentRow(index)
