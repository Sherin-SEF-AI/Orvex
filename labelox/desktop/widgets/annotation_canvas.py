"""
labelox/desktop/widgets/annotation_canvas.py — Main annotation surface.

QGraphicsView-based canvas supporting multiple annotation tools.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QPainter,
    QPen,
    QPixmap,
    QTransform,
    QUndoCommand,
    QUndoStack,
)
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QWidget,
)

from labelox.core.models import (
    Annotation,
    AnnotationType,
    BBoxAnnotation,
    Point,
    PolylineAnnotation,
)
from labelox.desktop.theme import ANNOTATION_COLORS, CANVAS_BG, HI, MUTED, TEXT


class AnnotationCanvasWidget(QGraphicsView):
    """Main annotation drawing surface."""

    annotation_created = pyqtSignal(object)   # Annotation
    annotation_selected = pyqtSignal(object)  # Annotation or None
    annotation_modified = pyqtSignal(object)  # Annotation
    annotation_deleted = pyqtSignal(str)      # annotation id
    zoom_changed = pyqtSignal(float)          # zoom factor

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # Settings
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setBackgroundBrush(QBrush(QColor(CANVAS_BG)))

        # State
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._image_path: str = ""
        self._image_id: str = ""
        self._img_w: int = 0
        self._img_h: int = 0
        self._zoom_factor: float = 1.0

        # Tool state
        self._current_tool: str = "b"  # bbox
        self._current_label_id: str = ""
        self._current_label_name: str = ""
        self._drawing: bool = False
        self._draw_start: QPointF | None = None
        self._draw_rect_item: QGraphicsRectItem | None = None
        self._polygon_points: list[QPointF] = []
        self._polygon_items: list = []

        # Annotations
        self._annotations: list[Annotation] = []
        self._ann_items: dict[str, Any] = {}  # ann_id -> QGraphicsItem
        self._selected_ann_id: str = ""

        # Undo
        self._undo_stack = QUndoStack(self)

        # Panning
        self._panning: bool = False
        self._pan_start: QPointF | None = None

    # ─── Image Loading ───────────────────────────────────────────────────

    def load_image(self, path: str, image_id: str = "") -> None:
        """Load and display an image."""
        self._clear_drawing_state()
        self._image_path = path
        self._image_id = image_id

        pixmap = QPixmap(path)
        if pixmap.isNull():
            return

        self._img_w = pixmap.width()
        self._img_h = pixmap.height()

        self._scene.clear()
        self._ann_items.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._pixmap_item.setZValue(0)
        self._scene.setSceneRect(QRectF(0, 0, self._img_w, self._img_h))
        self.zoom_to_fit()

    # ─── Annotations ─────────────────────────────────────────────────────

    def set_annotations(self, annotations: list[Annotation]) -> None:
        """Set and render annotations on canvas."""
        self._annotations = list(annotations)
        self._selected_ann_id = ""
        self._render_all_annotations()

    def get_annotations(self) -> list[Annotation]:
        return list(self._annotations)

    def _render_all_annotations(self) -> None:
        # Remove old items
        for item in self._ann_items.values():
            self._scene.removeItem(item)
        self._ann_items.clear()

        for i, ann in enumerate(self._annotations):
            color = QColor(self._get_ann_color(ann, i))
            item = self._create_ann_item(ann, color)
            if item:
                item.setZValue(10 + i)
                self._scene.addItem(item)
                self._ann_items[ann.id] = item

    def _create_ann_item(self, ann: Annotation, color: QColor) -> Any:
        pen = QPen(color, 2)
        fill = QColor(color)
        fill.setAlpha(40)

        if ann.bbox:
            x = ann.bbox.x * self._img_w
            y = ann.bbox.y * self._img_h
            w = ann.bbox.width * self._img_w
            h = ann.bbox.height * self._img_h
            rect = QGraphicsRectItem(x, y, w, h)
            rect.setPen(pen)
            rect.setBrush(QBrush(fill))
            rect.setData(0, ann.id)
            return rect

        if ann.polyline and ann.polyline.points:
            from PyQt6.QtGui import QPolygonF
            pts = QPolygonF([
                QPointF(p.x * self._img_w, p.y * self._img_h)
                for p in ann.polyline.points
            ])
            poly = QGraphicsPolygonItem(pts)
            poly.setPen(pen)
            if ann.polyline.is_closed:
                poly.setBrush(QBrush(fill))
            poly.setData(0, ann.id)
            return poly

        return None

    def _get_ann_color(self, ann: Annotation, idx: int) -> str:
        # Try to match class color from label panel
        return ANNOTATION_COLORS[idx % len(ANNOTATION_COLORS)]

    # ─── Tool / Label Selection ──────────────────────────────────────────

    def set_tool(self, key: str) -> None:
        self._clear_drawing_state()
        self._current_tool = key
        if key == "b":
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif key in ("p", "l", "k"):
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif key == "m":
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.setCursor(Qt.CursorShape.CrossCursor)

    def set_current_label(self, label_id: str, label_name: str) -> None:
        self._current_label_id = label_id
        self._current_label_name = label_name

    # ─── Mouse Events ────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        # Middle button = pan
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        scene_pos = self.mapToScene(event.position().toPoint())
        if not self._is_in_image(scene_pos):
            # Check if clicked on an annotation
            item = self._scene.itemAt(scene_pos, QTransform())
            if item and item != self._pixmap_item:
                ann_id = item.data(0)
                if ann_id:
                    self._select_annotation_by_id(str(ann_id))
            return

        if self._current_tool == "b":
            self._start_bbox(scene_pos)
        elif self._current_tool in ("p", "l"):
            self._add_polygon_point(scene_pos)
        elif self._current_tool == "m":
            self._sam_click(scene_pos, positive=True)
        elif self._current_tool == "i":
            self._classification_click()
        else:
            # Select existing annotation
            item = self._scene.itemAt(scene_pos, QTransform())
            if item and item != self._pixmap_item:
                ann_id = item.data(0)
                if ann_id:
                    self._select_annotation_by_id(str(ann_id))

    def mouseMoveEvent(self, event) -> None:
        if self._panning and self._pan_start is not None:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
            return

        if self._drawing and self._current_tool == "b" and self._draw_start:
            scene_pos = self.mapToScene(event.position().toPoint())
            self._update_bbox_preview(scene_pos)
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self._pan_start = None
            self.set_tool(self._current_tool)
            return

        if self._drawing and self._current_tool == "b":
            scene_pos = self.mapToScene(event.position().toPoint())
            self._finish_bbox(scene_pos)
            return

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if self._current_tool in ("p", "l") and self._polygon_points:
            self._finish_polygon()
            return
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.zoom_by(factor)

    def contextMenuEvent(self, event) -> None:
        if self._current_tool == "m":
            scene_pos = self.mapToScene(event.pos())
            if self._is_in_image(scene_pos):
                self._sam_click(scene_pos, positive=False)
                return
        super().contextMenuEvent(event)

    # ─── BBox Tool ───────────────────────────────────────────────────────

    def _start_bbox(self, pos: QPointF) -> None:
        self._drawing = True
        self._draw_start = pos
        self._draw_rect_item = QGraphicsRectItem(pos.x(), pos.y(), 0, 0)
        self._draw_rect_item.setPen(QPen(QColor(HI), 2, Qt.PenStyle.DashLine))
        self._draw_rect_item.setZValue(100)
        self._scene.addItem(self._draw_rect_item)

    def _update_bbox_preview(self, pos: QPointF) -> None:
        if self._draw_rect_item and self._draw_start:
            x = min(self._draw_start.x(), pos.x())
            y = min(self._draw_start.y(), pos.y())
            w = abs(pos.x() - self._draw_start.x())
            h = abs(pos.y() - self._draw_start.y())
            self._draw_rect_item.setRect(x, y, w, h)

    def _finish_bbox(self, pos: QPointF) -> None:
        self._drawing = False
        if self._draw_rect_item:
            self._scene.removeItem(self._draw_rect_item)

        if self._draw_start is None or self._img_w == 0:
            return

        x = min(self._draw_start.x(), pos.x())
        y = min(self._draw_start.y(), pos.y())
        w = abs(pos.x() - self._draw_start.x())
        h = abs(pos.y() - self._draw_start.y())

        # Min size check
        if w < 3 or h < 3:
            return

        bbox = BBoxAnnotation(
            x=x / self._img_w,
            y=y / self._img_h,
            width=w / self._img_w,
            height=h / self._img_h,
        )
        ann = Annotation(
            image_id=self._image_id,
            label_id=self._current_label_id,
            label_name=self._current_label_name or "unlabeled",
            annotation_type=AnnotationType.BBOX,
            bbox=bbox,
            created_by="user",
        )
        self._annotations.append(ann)
        self._render_all_annotations()
        self.annotation_created.emit(ann)
        self._draw_start = None
        self._draw_rect_item = None

    # ─── Polygon / Polyline Tool ─────────────────────────────────────────

    def _add_polygon_point(self, pos: QPointF) -> None:
        self._polygon_points.append(pos)
        # Draw point marker
        dot = QGraphicsEllipseItem(pos.x() - 3, pos.y() - 3, 6, 6)
        dot.setPen(QPen(QColor(HI), 1))
        dot.setBrush(QBrush(QColor(HI)))
        dot.setZValue(100)
        self._scene.addItem(dot)
        self._polygon_items.append(dot)

        # Draw line to previous
        if len(self._polygon_points) > 1:
            prev = self._polygon_points[-2]
            line = QGraphicsLineItem(prev.x(), prev.y(), pos.x(), pos.y())
            line.setPen(QPen(QColor(HI), 2, Qt.PenStyle.DashLine))
            line.setZValue(100)
            self._scene.addItem(line)
            self._polygon_items.append(line)

    def _finish_polygon(self) -> None:
        if len(self._polygon_points) < 3:
            self._clear_drawing_state()
            return

        # Clear preview items
        for item in self._polygon_items:
            self._scene.removeItem(item)
        self._polygon_items.clear()

        is_closed = self._current_tool == "p"
        points = [
            Point(x=p.x() / self._img_w, y=p.y() / self._img_h)
            for p in self._polygon_points
        ]

        ann_type = AnnotationType.POLYGON if is_closed else AnnotationType.POLYLINE
        ann = Annotation(
            image_id=self._image_id,
            label_id=self._current_label_id,
            label_name=self._current_label_name or "unlabeled",
            annotation_type=ann_type,
            polyline=PolylineAnnotation(points=points, is_closed=is_closed),
            created_by="user",
        )
        self._annotations.append(ann)
        self._polygon_points.clear()
        self._render_all_annotations()
        self.annotation_created.emit(ann)

    # ─── SAM Tool ────────────────────────────────────────────────────────

    def _sam_click(self, pos: QPointF, positive: bool) -> None:
        """Handle SAM click — dispatch to worker for mask prediction."""
        if not self._image_path or self._img_w == 0:
            return

        nx = pos.x() / self._img_w
        ny = pos.y() / self._img_h

        # Draw click indicator
        color = QColor("#4ecca3") if positive else QColor("#e94560")
        dot = QGraphicsEllipseItem(pos.x() - 5, pos.y() - 5, 10, 10)
        dot.setPen(QPen(color, 2))
        dot.setBrush(QBrush(color))
        dot.setZValue(100)
        self._scene.addItem(dot)

        # Dispatch to SAM worker for async mask prediction
        from labelox.desktop.workers import SAMWorker
        label = 1 if positive else 0
        self._sam_worker = SAMWorker(
            image_path=self._image_path,
            points=[[nx, ny]],
            labels=[label],
        )
        self._sam_worker.result.connect(self._on_sam_result)
        self._sam_worker.error.connect(lambda msg: print(f"SAM error: {msg}"))
        self._sam_worker.start()

    def _on_sam_result(self, mask_ann) -> None:
        """Handle SAM worker result — add the predicted mask annotation."""
        if mask_ann is None:
            return
        ann = Annotation(
            image_id=self._image_id,
            label_id=self._current_label_id,
            label_name=self._current_label_name or "unlabeled",
            annotation_type=AnnotationType.MASK,
            mask=mask_ann if hasattr(mask_ann, "rle") else None,
            is_auto=True,
            created_by="sam",
        )
        self._annotations.append(ann)
        self._render_all_annotations()
        self.annotation_created.emit(ann)

    # ─── Classification Tool ─────────────────────────────────────────────

    def _classification_click(self) -> None:
        ann = Annotation(
            image_id=self._image_id,
            label_id=self._current_label_id,
            label_name=self._current_label_name or "unlabeled",
            annotation_type=AnnotationType.CLASSIFICATION,
            classification=self._current_label_name,
            created_by="user",
        )
        self._annotations.append(ann)
        self.annotation_created.emit(ann)

    # ─── Selection ───────────────────────────────────────────────────────

    def select_annotation(self, ann_id: str) -> None:
        self._select_annotation_by_id(ann_id)

    def _select_annotation_by_id(self, ann_id: str) -> None:
        self._selected_ann_id = ann_id
        # Highlight
        for aid, item in self._ann_items.items():
            if aid == ann_id:
                if hasattr(item, 'setPen'):
                    item.setPen(QPen(QColor(HI), 3))
            else:
                if hasattr(item, 'setPen'):
                    idx = next((i for i, a in enumerate(self._annotations) if a.id == aid), 0)
                    item.setPen(QPen(QColor(self._get_ann_color(None, idx)), 2))

        ann = next((a for a in self._annotations if a.id == ann_id), None)
        self.annotation_selected.emit(ann)

    def cycle_selection(self) -> None:
        if not self._annotations:
            return
        if not self._selected_ann_id:
            self._select_annotation_by_id(self._annotations[0].id)
        else:
            idx = next((i for i, a in enumerate(self._annotations) if a.id == self._selected_ann_id), -1)
            next_idx = (idx + 1) % len(self._annotations)
            self._select_annotation_by_id(self._annotations[next_idx].id)

    # ─── Delete ──────────────────────────────────────────────────────────

    def delete_selected(self) -> None:
        if not self._selected_ann_id:
            return
        self._annotations = [a for a in self._annotations if a.id != self._selected_ann_id]
        self._selected_ann_id = ""
        self._render_all_annotations()
        self.annotation_created.emit(None)  # trigger save

    def cancel_current(self) -> None:
        self._clear_drawing_state()

    # ─── Undo / Redo ─────────────────────────────────────────────────────

    def undo(self) -> None:
        self._undo_stack.undo()

    def redo(self) -> None:
        self._undo_stack.redo()

    # ─── Zoom ────────────────────────────────────────────────────────────

    def zoom_by(self, factor: float) -> None:
        new_zoom = self._zoom_factor * factor
        if 0.05 < new_zoom < 20.0:
            self.scale(factor, factor)
            self._zoom_factor = new_zoom
            self.zoom_changed.emit(self._zoom_factor)

    def zoom_to_fit(self) -> None:
        if self._pixmap_item:
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom_factor = self.transform().m11()
            self.zoom_changed.emit(self._zoom_factor)

    # ─── Helpers ─────────────────────────────────────────────────────────

    def _is_in_image(self, pos: QPointF) -> bool:
        return 0 <= pos.x() <= self._img_w and 0 <= pos.y() <= self._img_h

    def _clear_drawing_state(self) -> None:
        self._drawing = False
        self._draw_start = None
        if self._draw_rect_item:
            self._scene.removeItem(self._draw_rect_item)
            self._draw_rect_item = None
        for item in self._polygon_items:
            self._scene.removeItem(item)
        self._polygon_items.clear()
        self._polygon_points.clear()
