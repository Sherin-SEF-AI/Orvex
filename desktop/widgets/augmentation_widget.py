"""
desktop/widgets/augmentation_widget.py — Data augmentation widget.

Configures and runs albumentations pipeline on extracted + annotated frames.
Shows before/after thumbnails. Outputs YOLO-format dataset.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.models import AugmentationConfig
from desktop.theme import BG, PANEL, ACCENT, HI, TEXT, MUTED, SUCCESS, WARNING, BORDER, CARD
from desktop.workers import AugmentationWorker

# Alias
HIGHLIGHT = HI


class AugmentationWidget(QWidget):
    """UI for configuring and running data augmentation."""

    def __init__(self, session_manager, parent=None) -> None:
        super().__init__(parent)
        self._sm = session_manager
        self._session_id: str = ""
        self._worker: AugmentationWorker | None = None
        self._annotations: list = []
        self._result = None
        self._build_ui()

    def set_annotations(self, annotations: list) -> None:
        self._annotations = annotations
        has_ann = bool(annotations)
        self._run_btn.setEnabled(has_ann and bool(self._session_id))
        if not has_ann:
            self._prereq_label.setText("Run Auto-Label first to get annotated frames.")
            self._prereq_label.setStyleSheet(f"color:{WARNING};")
        else:
            self._prereq_label.setText(f"✓ {len(annotations)} annotated frames ready.")
            self._prereq_label.setStyleSheet("color:#27ae60;")

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ── Left config ────────────────────────────────────────────────
        cfg_w = QWidget()
        cfg_w.setMinimumWidth(280)
        cfg_w.setMaximumWidth(400)
        cfg_layout = QVBoxLayout(cfg_w)
        cfg_layout.setContentsMargins(8, 8, 8, 8)

        self._prereq_label = QLabel("Run Auto-Label first to get annotated frames.")
        self._prereq_label.setStyleSheet(f"color:{WARNING}; font-size:11px;")
        self._prereq_label.setWordWrap(True)
        cfg_layout.addWidget(self._prereq_label)

        # Geometric group
        geo_group = QGroupBox("Geometric")
        gg = QVBoxLayout(geo_group)
        self._hflip_cb   = _cb("Horizontal flip",   True,  gg)
        self._vflip_cb   = _cb("Vertical flip",     False, gg)
        self._rot90_cb   = _cb("Random rotate 90°", True,  gg)
        cfg_layout.addWidget(geo_group)

        # Photometric group
        photo_group = QGroupBox("Photometric")
        pg = QVBoxLayout(photo_group)
        self._bc_cb      = _cb("Brightness/Contrast", True,  pg)
        self._hue_cb     = _cb("Hue/Saturation",      True,  pg)
        self._noise_cb   = _cb("Gaussian noise",       True,  pg)
        self._blur_cb    = _cb("Motion blur",          True,  pg)
        self._jpeg_cb    = _cb("JPEG compression",     True,  pg)
        cfg_layout.addWidget(photo_group)

        # Advanced group
        adv_group = QGroupBox("Advanced")
        ag = QVBoxLayout(adv_group)
        self._mosaic_cb  = _cb("Mosaic (YOLOv8 style)", True,  ag)
        self._rain_cb    = _cb("Rain simulation",        False, ag)
        self._fog_cb     = _cb("Fog simulation",         False, ag)
        cfg_layout.addWidget(adv_group)

        # Multiplier
        mult_group = QGroupBox("Dataset multiplier")
        mg = QVBoxLayout(mult_group)
        self._mult_spin = QSpinBox()
        self._mult_spin.setRange(2, 20)
        self._mult_spin.setValue(3)
        self._mult_spin.setToolTip("Output = N_original × multiplier images")
        mg.addWidget(self._mult_spin)
        cfg_layout.addWidget(mult_group)

        self._run_btn = QPushButton("Augment Dataset")
        self._run_btn.setEnabled(False)
        self._run_btn.setObjectName("DangerBtn")
        self._run_btn.clicked.connect(self._run)
        cfg_layout.addWidget(self._run_btn)

        self._preview_btn = QPushButton("Preview augmentations")
        self._preview_btn.setEnabled(False)
        self._preview_btn.clicked.connect(self._preview)
        cfg_layout.addWidget(self._preview_btn)

        cfg_layout.addStretch()
        splitter.addWidget(cfg_w)

        # ── Right panel ────────────────────────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 8, 8, 8)

        # Preview thumbnails
        preview_group = QGroupBox("Augmentation preview (6 variants)")
        pg_layout = QHBoxLayout(preview_group)
        self._preview_labels: list[QLabel] = []
        for _ in range(6):
            lbl = QLabel()
            lbl.setFixedSize(160, 120)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pg_layout.addWidget(lbl)
            self._preview_labels.append(lbl)
        rl.addWidget(preview_group)

        # Progress log
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        rl.addWidget(self._log, stretch=1)

        # Stats
        self._stats_label = QLabel("No augmentation run yet.")
        self._stats_label.setStyleSheet(
            f"background:{PANEL}; color:{TEXT}; padding:8px; border-radius:4px;"
        )
        rl.addWidget(self._stats_label)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([280, 720])

    @pyqtSlot(str)
    def on_session_changed(self, session_id: str) -> None:
        self._session_id = session_id
        has_session = bool(session_id)
        has_ann = bool(self._annotations)
        self._run_btn.setEnabled(has_session and has_ann)
        self._preview_btn.setEnabled(has_session and has_ann)

    def _build_config(self) -> AugmentationConfig:
        return AugmentationConfig(
            horizontal_flip=self._hflip_cb.isChecked(),
            vertical_flip=self._vflip_cb.isChecked(),
            random_rotate_90=self._rot90_cb.isChecked(),
            brightness_contrast=self._bc_cb.isChecked(),
            hue_saturation=self._hue_cb.isChecked(),
            gaussian_noise=self._noise_cb.isChecked(),
            motion_blur=self._blur_cb.isChecked(),
            jpeg_compression=self._jpeg_cb.isChecked(),
            mosaic=self._mosaic_cb.isChecked(),
            rain_simulation=self._rain_cb.isChecked(),
            fog_simulation=self._fog_cb.isChecked(),
            multiplier=self._mult_spin.value(),
        )

    def _run(self) -> None:
        if not self._annotations or (self._worker and self._worker.isRunning()):
            return
        config = self._build_config()
        out_dir = str(
            self._sm.session_folder(self._session_id) / "augmented_dataset"
        )
        self._worker = AugmentationWorker(
            session_id=self._session_id,
            sm=self._sm,
            config=config,
            annotations=self._annotations,
            output_dir=out_dir,
        )
        self._worker.status.connect(lambda m: self._log.append(m))
        self._worker.result.connect(self._on_result)
        self._worker.error.connect(lambda e: QMessageBox.critical(self, "Augmentation Error", e))
        self._run_btn.setEnabled(False)
        self._worker.start()

    @pyqtSlot(object)
    def _on_result(self, result: Any) -> None:
        self._result = result
        self._run_btn.setEnabled(True)
        self._stats_label.setText(
            f"✓ Original: {result.original_count}  →  "
            f"Augmented total: {result.augmented_count}  "
            f"({result.per_transform_counts.get('augmented', 0)} aug + "
            f"{result.per_transform_counts.get('mosaic', 0)} mosaic)  |  "
            f"Output: {result.output_dir}"
        )

    def _preview(self) -> None:
        if not self._annotations:
            return
        import random, cv2, numpy as np
        config = self._build_config()
        try:
            from core.augmentor import apply_augmentation, build_augmentation_pipeline
            pipeline = build_augmentation_pipeline(config)
            # Pick a random annotated frame
            ann = random.choice(self._annotations)
            previews = apply_augmentation(ann.frame_path, ann.detections, pipeline, 6)
            for i, (img, dets) in enumerate(previews[:6]):
                # Draw bboxes
                h, w = img.shape[:2]
                canvas = img.copy()
                for det in dets:
                    x1, y1, x2, y2 = (int(v) for v in det.bbox_xyxy)
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (233, 69, 96), 2)
                canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
                # Convert to QPixmap
                _, buf = cv2.imencode(".jpg", canvas_rgb, [cv2.IMWRITE_JPEG_QUALITY, 80])
                pix = QPixmap()
                pix.loadFromData(bytes(buf), "JPEG")
                self._preview_labels[i].setPixmap(
                    pix.scaled(160, 120,
                                Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation)
                )
        except Exception as exc:
            QMessageBox.warning(self, "Preview failed", str(exc))


def _cb(label: str, checked: bool, layout: QVBoxLayout) -> QCheckBox:
    cb = QCheckBox(label)
    cb.setChecked(checked)
    layout.addWidget(cb)
    return cb
