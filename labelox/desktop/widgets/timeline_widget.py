"""
labelox/desktop/widgets/timeline_widget.py — Sequence filmstrip / frame scrubber.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSlider, QWidget

from labelox.desktop.theme import ACCENT, BG, BORDER, HI, MUTED, PANEL, TEXT


class TimelineWidget(QWidget):
    """Compact timeline strip showing frame position within a sequence."""

    frame_changed = pyqtSignal(int)  # frame index

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._total_frames: int = 0
        self._current_frame: int = 0
        self._keyframes: set[int] = set()
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        self._label = QLabel("Frame:")
        self._label.setStyleSheet(f"color:{MUTED}; font-size:10px;")
        layout.addWidget(self._label)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self._slider, 1)

        self._pos_label = QLabel("0 / 0")
        self._pos_label.setStyleSheet(f"color:{TEXT}; font-size:10px; min-width:60px;")
        self._pos_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self._pos_label)

    def set_sequence(self, total_frames: int, keyframes: set[int] | None = None) -> None:
        self._total_frames = total_frames
        self._keyframes = keyframes or set()
        self._slider.setMaximum(max(0, total_frames - 1))
        self._update_label()

    def set_frame(self, index: int) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(index)
        self._slider.blockSignals(False)
        self._current_frame = index
        self._update_label()

    def _on_slider_changed(self, value: int) -> None:
        self._current_frame = value
        self._update_label()
        self.frame_changed.emit(value)

    def _update_label(self) -> None:
        self._pos_label.setText(f"{self._current_frame + 1} / {self._total_frames}")

    def add_keyframe(self, index: int) -> None:
        self._keyframes.add(index)
        self.update()

    def remove_keyframe(self, index: int) -> None:
        self._keyframes.discard(index)
        self.update()
