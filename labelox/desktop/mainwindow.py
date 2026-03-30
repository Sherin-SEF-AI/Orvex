"""
labelox/desktop/mainwindow.py — Main window with 3-panel annotation layout.

Layout:
┌────────────────────────────────────────────────────────────────┐
│  LABELOX  [Project ▼]    Toolbar (tools, zoom, nav)           │
├────────────┬──────────────────────────────┬────────────────────┤
│ IMAGE      │  ANNOTATION CANVAS           │ LABEL PANEL        │
│ BROWSER    │  (QGraphicsView)             │ + PROPERTIES       │
│            ├──────────────────────────────┤                    │
│            │  TIMELINE (sequences)        │                    │
├────────────┴──────────────────────────────┴────────────────────┤
│  Status bar                                                    │
└────────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, QPointF, QRectF, QSize, pyqtSlot
from PyQt6.QtGui import (
    QAction,
    QColor,
    QIcon,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from labelox.core.database import (
    create_project,
    delete_project,
    get_session,
    list_projects,
)
from labelox.desktop.theme import (
    ACCENT,
    BG,
    BORDER,
    HI,
    MUTED,
    PANEL,
    TEXT,
    section_header_style,
)


def _make_labelox_logo(size: int = 64) -> QIcon:
    """Draw a unique LABELOX logo: tag + pen nib."""
    pm = QPixmap(size, size)
    pm.fill(QColor(0, 0, 0, 0))
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    s = float(size)
    m = s * 0.1

    # Tag shape
    pen = QPen(QColor(HI), s * 0.06)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)

    # Rounded rectangle (tag body)
    body = QRectF(m + 2, m + 6, s - 2 * m - 4, s - 2 * m - 8)
    p.drawRoundedRect(body, s * 0.1, s * 0.1)

    # Tag hole (small circle top-left)
    p.setBrush(QColor(HI))
    p.drawEllipse(QPointF(m + 10, m + 14), s * 0.06, s * 0.06)
    p.setBrush(Qt.BrushStyle.NoBrush)

    # Pen nib (diagonal line from bottom-right)
    pen2 = QPen(QColor("#4ecca3"), s * 0.05)
    p.setPen(pen2)
    p.drawLine(
        QPointF(s * 0.6, s * 0.85),
        QPointF(s * 0.85, s * 0.55),
    )
    # Nib tip
    p.setBrush(QColor("#4ecca3"))
    p.drawEllipse(QPointF(s * 0.88, s * 0.52), s * 0.04, s * 0.04)

    p.end()
    return QIcon(pm)


class LabeloxMainWindow(QMainWindow):
    """Main LABELOX annotation window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LABELOX")
        self.setWindowIcon(_make_labelox_logo(64))

        self._current_project_id: str = ""
        self._current_image_id: str = ""

        self._build_menus()
        self._build_ui()
        self._build_status_bar()
        self._load_projects()

    # ─── Menu Bar ────────────────────────────────────────────────────────

    def _build_menus(self) -> None:
        mb = self.menuBar()

        # File menu
        file_menu = mb.addMenu("File")

        new_proj = QAction("New Project...", self)
        new_proj.setShortcut(QKeySequence("Ctrl+N"))
        new_proj.triggered.connect(self._new_project)
        file_menu.addAction(new_proj)

        import_imgs = QAction("Import Images...", self)
        import_imgs.setShortcut(QKeySequence("Ctrl+I"))
        import_imgs.triggered.connect(self._import_images)
        file_menu.addAction(import_imgs)

        import_folder = QAction("Import Folder...", self)
        import_folder.triggered.connect(self._import_folder)
        file_menu.addAction(import_folder)

        file_menu.addSeparator()

        import_ann = QAction("Import Annotations...", self)
        import_ann.triggered.connect(self._import_annotations)
        file_menu.addAction(import_ann)

        export_ann = QAction("Export Annotations...", self)
        export_ann.setShortcut(QKeySequence("Ctrl+E"))
        export_ann.triggered.connect(self._show_export)
        file_menu.addAction(export_ann)

        file_menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Edit menu
        edit_menu = mb.addMenu("Edit")

        undo = QAction("Undo", self)
        undo.setShortcut(QKeySequence("Ctrl+Z"))
        undo.triggered.connect(self._undo)
        edit_menu.addAction(undo)

        redo = QAction("Redo", self)
        redo.setShortcut(QKeySequence("Ctrl+Y"))
        redo.triggered.connect(self._redo)
        edit_menu.addAction(redo)

        edit_menu.addSeparator()

        prefs = QAction("Preferences...", self)
        prefs.triggered.connect(self._show_preferences)
        edit_menu.addAction(prefs)

        # View menu
        view_menu = mb.addMenu("View")

        zoom_in = QAction("Zoom In", self)
        zoom_in.setShortcut(QKeySequence("Ctrl+="))
        zoom_in.triggered.connect(lambda: self._zoom(1.25))
        view_menu.addAction(zoom_in)

        zoom_out = QAction("Zoom Out", self)
        zoom_out.setShortcut(QKeySequence("Ctrl+-"))
        zoom_out.triggered.connect(lambda: self._zoom(0.8))
        view_menu.addAction(zoom_out)

        zoom_fit = QAction("Fit to Window", self)
        zoom_fit.setShortcut(QKeySequence("Ctrl+0"))
        zoom_fit.triggered.connect(self._zoom_fit)
        view_menu.addAction(zoom_fit)

        # Tools menu
        tools_menu = mb.addMenu("Tools")

        auto_ann = QAction("Auto-Annotate...", self)
        auto_ann.setShortcut(QKeySequence("Ctrl+Shift+A"))
        auto_ann.triggered.connect(self._show_auto_annotate)
        tools_menu.addAction(auto_ann)

        classify = QAction("Classify Scenes...", self)
        classify.triggered.connect(self._classify_scenes)
        tools_menu.addAction(classify)

        similarity = QAction("Find Similar Images...", self)
        similarity.triggered.connect(self._find_similar)
        tools_menu.addAction(similarity)

        # Review menu
        review_menu = mb.addMenu("Review")

        review_queue = QAction("Review Queue", self)
        review_queue.setShortcut(QKeySequence("Ctrl+R"))
        review_queue.triggered.connect(self._show_review)
        review_menu.addAction(review_queue)

        stats = QAction("Statistics", self)
        stats.triggered.connect(self._show_stats)
        review_menu.addAction(stats)

    # ─── Main UI ─────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header bar ───────────────────────────────────────────────────
        header = QWidget()
        header.setFixedHeight(42)
        header.setStyleSheet(f"background:{PANEL}; border-bottom:1px solid {BORDER};")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(12, 0, 12, 0)

        # Logo + brand
        brand = QLabel("LABELOX")
        brand.setStyleSheet(f"color:{HI}; font-size:15px; font-weight:800; letter-spacing:2px;")
        hl.addWidget(brand)

        hl.addSpacing(20)

        # Project selector
        hl.addWidget(QLabel("Project:"))
        self._project_combo = QComboBox()
        self._project_combo.setMinimumWidth(200)
        self._project_combo.currentIndexChanged.connect(self._on_project_changed)
        hl.addWidget(self._project_combo)

        hl.addStretch()

        # Tool buttons in header
        tool_names = [
            ("B", "BBox", "b"),
            ("M", "Mask/SAM", "m"),
            ("P", "Polygon", "p"),
            ("L", "Polyline", "l"),
            ("K", "Keypoint", "k"),
            ("C", "Cuboid 3D", "c"),
            ("I", "Classification", "i"),
        ]
        self._tool_buttons: dict[str, QPushButton] = {}
        for short, tooltip, key in tool_names:
            btn = QPushButton(short)
            btn.setObjectName("ToolBtn")
            btn.setCheckable(True)
            btn.setToolTip(f"{tooltip} ({key.upper()})")
            btn.setFixedSize(32, 28)
            btn.clicked.connect(lambda checked, k=key: self._set_tool(k))
            hl.addWidget(btn)
            self._tool_buttons[key] = btn

        hl.addSpacing(12)

        # Zoom controls
        hl.addWidget(QLabel("Zoom:"))
        self._zoom_label = QLabel("100%")
        self._zoom_label.setMinimumWidth(40)
        hl.addWidget(self._zoom_label)

        # Nav buttons
        prev_btn = QPushButton("◀")
        prev_btn.setObjectName("ToolBtn")
        prev_btn.setFixedSize(28, 28)
        prev_btn.setToolTip("Previous image (A)")
        prev_btn.clicked.connect(self._prev_image)
        hl.addWidget(prev_btn)

        next_btn = QPushButton("▶")
        next_btn.setObjectName("ToolBtn")
        next_btn.setFixedSize(28, 28)
        next_btn.setToolTip("Next image (D)")
        next_btn.clicked.connect(self._next_image)
        hl.addWidget(next_btn)

        root.addWidget(header)

        # ── Main splitter (3 panels) ─────────────────────────────────────
        self._main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Image browser
        from labelox.desktop.widgets.image_browser import ImageBrowserWidget
        self._image_browser = ImageBrowserWidget()
        self._image_browser.image_selected.connect(self._on_image_selected)
        self._main_splitter.addWidget(self._image_browser)

        # Center: Canvas + Timeline (vertical splitter)
        center_splitter = QSplitter(Qt.Orientation.Vertical)

        from labelox.desktop.widgets.annotation_canvas import AnnotationCanvasWidget
        self._canvas = AnnotationCanvasWidget()
        self._canvas.annotation_created.connect(self._on_annotation_created)
        self._canvas.annotation_selected.connect(self._on_annotation_selected)
        self._canvas.zoom_changed.connect(self._on_zoom_changed)
        center_splitter.addWidget(self._canvas)

        from labelox.desktop.widgets.timeline_widget import TimelineWidget
        self._timeline = TimelineWidget()
        self._timeline.setFixedHeight(50)
        self._timeline.frame_changed.connect(self._on_frame_changed)
        center_splitter.addWidget(self._timeline)

        center_splitter.setStretchFactor(0, 1)
        center_splitter.setStretchFactor(1, 0)
        self._main_splitter.addWidget(center_splitter)

        # Right: Label panel + Properties (vertical splitter)
        right_splitter = QSplitter(Qt.Orientation.Vertical)

        from labelox.desktop.widgets.label_panel import LabelPanelWidget
        self._label_panel = LabelPanelWidget()
        self._label_panel.label_selected.connect(self._on_label_selected)
        self._label_panel.annotation_clicked.connect(self._on_annotation_list_clicked)
        right_splitter.addWidget(self._label_panel)

        from labelox.desktop.widgets.properties_panel import PropertiesPanelWidget
        self._properties_panel = PropertiesPanelWidget()
        right_splitter.addWidget(self._properties_panel)

        right_splitter.setSizes([400, 200])
        self._main_splitter.addWidget(right_splitter)

        # Set splitter sizes
        self._main_splitter.setSizes([220, 800, 260])
        self._main_splitter.setStretchFactor(0, 0)
        self._main_splitter.setStretchFactor(1, 1)
        self._main_splitter.setStretchFactor(2, 0)

        root.addWidget(self._main_splitter)

    # ─── Status Bar ──────────────────────────────────────────────────────

    def _build_status_bar(self) -> None:
        sb = QStatusBar()
        self.setStatusBar(sb)
        self._status_image = QLabel("No image loaded")
        self._status_tool = QLabel("Tool: BBox")
        self._status_count = QLabel("0 annotations")
        sb.addWidget(self._status_image, 1)
        sb.addWidget(self._status_tool)
        sb.addWidget(self._status_count)

    # ─── Project Management ──────────────────────────────────────────────

    def _load_projects(self) -> None:
        self._project_combo.blockSignals(True)
        self._project_combo.clear()
        projects = list_projects()
        for proj in projects:
            self._project_combo.addItem(proj.name, proj.id)
        self._project_combo.blockSignals(False)

        if projects:
            self._project_combo.setCurrentIndex(0)
            self._on_project_changed(0)

    def _new_project(self) -> None:
        from PyQt6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(self, "New Project", "Project name:")
        if ok and name.strip():
            default_classes = [
                {"name": "car", "color": "#e94560", "hotkey": "1", "id": "cls_car"},
                {"name": "person", "color": "#4ecca3", "hotkey": "2", "id": "cls_person"},
                {"name": "motorcycle", "color": "#4a9eff", "hotkey": "3", "id": "cls_moto"},
                {"name": "truck", "color": "#f5a623", "hotkey": "4", "id": "cls_truck"},
                {"name": "bus", "color": "#9b59b6", "hotkey": "5", "id": "cls_bus"},
            ]
            proj = create_project(name.strip(), "", default_classes)
            self._load_projects()
            # Select new project
            idx = self._project_combo.findData(proj.id)
            if idx >= 0:
                self._project_combo.setCurrentIndex(idx)

    @pyqtSlot(int)
    def _on_project_changed(self, index: int) -> None:
        pid = self._project_combo.itemData(index)
        if not pid:
            return
        self._current_project_id = pid

        from labelox.core.database import get_project
        proj = get_project(pid)
        if proj:
            self._label_panel.set_classes(proj.label_classes)
            self._image_browser.set_project(pid)
            self._status_image.setText(f"Project: {proj.name}")

    # ─── Image Navigation ────────────────────────────────────────────────

    @pyqtSlot(str, str)
    def _on_image_selected(self, image_id: str, image_path: str) -> None:
        self._current_image_id = image_id
        self._canvas.load_image(image_path, image_id)

        # Load annotations
        from labelox.core.annotation_engine import get_annotations
        anns = get_annotations(image_id)
        self._canvas.set_annotations(anns)
        self._label_panel.set_annotations(anns)
        self._properties_panel.clear()
        self._status_image.setText(Path(image_path).name)
        self._status_count.setText(f"{len(anns)} annotations")

    def _prev_image(self) -> None:
        self._image_browser.select_previous()

    def _next_image(self) -> None:
        self._image_browser.select_next()

    @pyqtSlot(int)
    def _on_frame_changed(self, frame_index: int) -> None:
        self._image_browser.select_by_index(frame_index)

    # ─── Tool Selection ──────────────────────────────────────────────────

    def _set_tool(self, key: str) -> None:
        for k, btn in self._tool_buttons.items():
            btn.setChecked(k == key)
        tool_names = {
            "b": "BBox", "m": "Mask/SAM", "p": "Polygon",
            "l": "Polyline", "k": "Keypoint", "c": "Cuboid 3D", "i": "Classification",
        }
        self._canvas.set_tool(key)
        self._status_tool.setText(f"Tool: {tool_names.get(key, key)}")

    # ─── Canvas Signals ──────────────────────────────────────────────────

    @pyqtSlot(object)
    def _on_annotation_created(self, ann: Any) -> None:
        # Save to DB
        if self._current_image_id:
            all_anns = self._canvas.get_annotations()
            from labelox.core.annotation_engine import save_annotations
            save_annotations(self._current_image_id, all_anns, "user")
            self._label_panel.set_annotations(all_anns)
            self._status_count.setText(f"{len(all_anns)} annotations")

    @pyqtSlot(object)
    def _on_annotation_selected(self, ann: Any) -> None:
        if ann:
            self._properties_panel.set_annotation(ann)
            self._label_panel.highlight_annotation(ann.id if hasattr(ann, 'id') else "")

    @pyqtSlot(float)
    def _on_zoom_changed(self, zoom: float) -> None:
        self._zoom_label.setText(f"{int(zoom * 100)}%")

    def _zoom(self, factor: float) -> None:
        self._canvas.zoom_by(factor)

    def _zoom_fit(self) -> None:
        self._canvas.zoom_to_fit()

    # ─── Label Panel Signals ─────────────────────────────────────────────

    @pyqtSlot(str, str)
    def _on_label_selected(self, label_id: str, label_name: str) -> None:
        self._canvas.set_current_label(label_id, label_name)

    @pyqtSlot(str)
    def _on_annotation_list_clicked(self, ann_id: str) -> None:
        self._canvas.select_annotation(ann_id)

    # ─── Import / Export ─────────────────────────────────────────────────

    def _import_images(self) -> None:
        if not self._current_project_id:
            QMessageBox.warning(self, "No Project", "Create or select a project first.")
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import Images", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tiff *.webp);;All Files (*)",
        )
        if paths:
            self._do_import(paths)

    def _import_folder(self) -> None:
        if not self._current_project_id:
            QMessageBox.warning(self, "No Project", "Create or select a project first.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Import Folder")
        if folder:
            self._do_import([folder])

    def _do_import(self, paths: list[str]) -> None:
        from labelox.core.image_manager import import_images
        db = get_session()
        try:
            thumb_dir = Path.home() / ".labelox" / "projects" / self._current_project_id / "thumbs"
            created = import_images(
                [Path(p) for p in paths],
                self._current_project_id,
                thumb_dir=thumb_dir,
                db=db,
            )
            self._image_browser.set_project(self._current_project_id)
            QMessageBox.information(
                self, "Import Complete",
                f"Imported {len(created)} images.",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Import Error", str(exc))
        finally:
            db.close()

    def _import_annotations(self) -> None:
        QMessageBox.information(self, "Import", "Use File > Import to load YOLO/COCO/CVAT/VOC annotations.")

    def _show_export(self) -> None:
        if not self._current_project_id:
            return
        from labelox.desktop.widgets.export_widget import ExportDialog
        dlg = ExportDialog(self._current_project_id, self)
        dlg.exec()

    # ─── Tools ───────────────────────────────────────────────────────────

    def _show_auto_annotate(self) -> None:
        if not self._current_project_id:
            return
        from labelox.desktop.widgets.auto_annotate_widget import AutoAnnotateDialog
        dlg = AutoAnnotateDialog(self._current_project_id, self)
        dlg.exec()
        self._image_browser.refresh()

    def _classify_scenes(self) -> None:
        QMessageBox.information(self, "Classify", "Scene classification runs in background.")

    def _find_similar(self) -> None:
        QMessageBox.information(self, "Similar", "Similarity search coming soon.")

    # ─── Review / Stats ──────────────────────────────────────────────────

    def _show_review(self) -> None:
        if not self._current_project_id:
            return
        from labelox.desktop.widgets.review_widget import ReviewDialog
        dlg = ReviewDialog(self._current_project_id, self)
        dlg.exec()
        self._image_browser.refresh()

    def _show_stats(self) -> None:
        if not self._current_project_id:
            return
        from labelox.desktop.widgets.stats_widget import StatsDialog
        dlg = StatsDialog(self._current_project_id, self)
        dlg.exec()

    def _show_preferences(self) -> None:
        QMessageBox.information(self, "Preferences", "Settings panel coming soon.")

    # ─── Undo / Redo ─────────────────────────────────────────────────────

    def _undo(self) -> None:
        self._canvas.undo()

    def _redo(self) -> None:
        self._canvas.redo()

    # ─── Keyboard Shortcuts ──────────────────────────────────────────────

    def keyPressEvent(self, event) -> None:
        key = event.key()
        mod = event.modifiers()

        if mod == Qt.KeyboardModifier.NoModifier:
            key_map = {
                Qt.Key.Key_B: "b",
                Qt.Key.Key_M: "m",
                Qt.Key.Key_P: "p",
                Qt.Key.Key_L: "l",
                Qt.Key.Key_K: "k",
                Qt.Key.Key_C: "c",
                Qt.Key.Key_I: "i",
            }
            if key in key_map:
                self._set_tool(key_map[key])
                return

            if key == Qt.Key.Key_A:
                self._prev_image()
                return
            if key == Qt.Key.Key_D:
                self._next_image()
                return
            if key == Qt.Key.Key_S:
                self._on_annotation_created(None)
                return
            if key == Qt.Key.Key_Space:
                # Mark complete + next
                self._next_image()
                return
            if key == Qt.Key.Key_Delete or key == Qt.Key.Key_Backspace:
                self._canvas.delete_selected()
                return
            if key == Qt.Key.Key_Escape:
                self._canvas.cancel_current()
                return
            if key == Qt.Key.Key_Tab:
                self._canvas.cycle_selection()
                return

            # Number keys 1-9 for class selection
            if Qt.Key.Key_1 <= key <= Qt.Key.Key_9:
                idx = key - Qt.Key.Key_1
                self._label_panel.select_by_index(idx)
                return

        super().keyPressEvent(event)
