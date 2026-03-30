"""
desktop/widgets/annotation_review_widget.py — Human-in-the-loop annotation review.

Left  : session selector + stats + export/trigger controls
Center: frame image with bbox overlay + accept/reject/correct buttons
Right : frame list (color-coded by status) + filter + navigation
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.annotation_review import (
    export_corrected_dataset,
    get_review_stats,
    load_reviews,
    save_review,
)
from core.models import AnnotationReview, Detection, ReviewStatus
from core.session_manager import SessionManager
from desktop.theme import BG, PANEL, ACCENT, HI, TEXT, MUTED, SUCCESS, WARNING, BORDER, CARD
from desktop.workers import AnnotationReviewWorker

# Semantic aliases
GREEN  = SUCCESS
ORANGE = WARNING
RED    = HI

_STATUS_COLOR = {
    ReviewStatus.pending:   QColor(150, 150, 150),
    ReviewStatus.accepted:  QColor(76, 175, 80),
    ReviewStatus.corrected: QColor(255, 152, 0),
    ReviewStatus.rejected:  QColor(244, 67, 54),
}


class AnnotationReviewWidget(QWidget):
    """Human-in-the-loop annotation review UI."""

    def __init__(self, sm: SessionManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sm = sm
        self._session_id: str | None = None
        self._frame_paths: list[str] = []
        self._reviews: dict[str, AnnotationReview] = {}   # frame_path → review
        self._current_idx: int = -1
        self._worker = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal, self)

        # ── Left: stats + actions ─────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(220)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(6, 6, 6, 6)
        lv.setSpacing(8)

        stats_grp = QGroupBox("Review Stats")
        sg = QVBoxLayout(stats_grp)
        self._lbl_total    = QLabel("Total frames: —")
        self._lbl_pending  = QLabel("Pending: —")
        self._lbl_accepted = QLabel("Accepted: —")
        self._lbl_corrected= QLabel("Corrected: —")
        self._lbl_rejected = QLabel("Rejected: —")
        self._lbl_coverage = QLabel("Coverage: —%")
        for lbl in (self._lbl_total, self._lbl_pending, self._lbl_accepted,
                    self._lbl_corrected, self._lbl_rejected, self._lbl_coverage):
            sg.addWidget(lbl)
        lv.addWidget(stats_grp)

        act_grp = QGroupBox("Actions")
        ag = QVBoxLayout(act_grp)
        self._btn_export  = QPushButton("Export corrected dataset")
        self._btn_trigger = QPushButton("Trigger retraining")
        self._btn_trigger.setObjectName("DangerBtn")
        ag.addWidget(self._btn_export)
        ag.addWidget(self._btn_trigger)
        lv.addWidget(act_grp)

        lv.addStretch()

        # ── Center: image + controls ───────────────────────────────────
        center = QWidget()
        cv = QVBoxLayout(center)
        cv.setContentsMargins(6, 6, 6, 6)
        cv.setSpacing(6)

        self._preview = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self._preview.setObjectName("preview")
        self._preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._preview.setText("Select a frame →")
        cv.addWidget(self._preview)

        self._frame_info = QLabel("No frame loaded")
        cv.addWidget(self._frame_info)

        btn_row = QHBoxLayout()
        self._btn_accept  = QPushButton("✓ Accept")
        self._btn_accept.setObjectName("PrimaryBtn")
        self._btn_correct = QPushButton("✏ Mark corrected")
        self._btn_correct.setObjectName("WarningBtn")
        self._btn_reject  = QPushButton("✗ Reject")
        self._btn_reject.setObjectName("DangerBtn")
        for b in (self._btn_accept, self._btn_correct, self._btn_reject):
            b.setEnabled(False)
            btn_row.addWidget(b)
        cv.addLayout(btn_row)

        nav_row = QHBoxLayout()
        self._btn_prev = QPushButton("◀ Prev")
        self._btn_next = QPushButton("Next ▶")
        self._btn_prev.setEnabled(False)
        self._btn_next.setEnabled(False)
        nav_row.addWidget(self._btn_prev)
        nav_row.addStretch()
        nav_row.addWidget(self._btn_next)
        cv.addLayout(nav_row)

        # ── Right: frame list + filter ─────────────────────────────────
        right = QWidget()
        right.setFixedWidth(240)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(6, 6, 6, 6)
        rv.setSpacing(6)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter:"))
        self._filter_combo = QComboBox()
        self._filter_combo.addItems(["All", "Pending", "Accepted", "Corrected", "Rejected"])
        filter_row.addWidget(self._filter_combo)
        rv.addLayout(filter_row)

        self._frame_list = QListWidget()
        rv.addWidget(self._frame_list)

        splitter.addWidget(left)
        splitter.addWidget(center)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(splitter)

        # ── Connections ───────────────────────────────────────────────
        self._btn_export.clicked.connect(self._on_export)
        self._btn_trigger.clicked.connect(self._on_trigger_retrain)
        self._btn_accept.clicked.connect(lambda: self._on_review(ReviewStatus.accepted))
        self._btn_correct.clicked.connect(lambda: self._on_review(ReviewStatus.corrected))
        self._btn_reject.clicked.connect(lambda: self._on_review(ReviewStatus.rejected))
        self._btn_prev.clicked.connect(self._on_prev)
        self._btn_next.clicked.connect(self._on_next)
        self._frame_list.currentRowChanged.connect(self._on_list_selection)
        self._filter_combo.currentTextChanged.connect(self._on_filter_changed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @pyqtSlot(str)
    def on_session_changed(self, session_id: str) -> None:
        self._session_id = session_id
        self._load_session()

    # ------------------------------------------------------------------
    # Load / refresh
    # ------------------------------------------------------------------

    def _load_session(self) -> None:
        if not self._session_id:
            return
        # Collect frame paths from autolabel output
        session_dir = self._sm.session_folder(self._session_id)
        yolo_dir = session_dir / "autolabel" / "yolo"
        self._frame_paths = []
        if yolo_dir.exists():
            for txt in sorted(yolo_dir.glob("*.txt")):
                # Look for matching image in frames/
                for ext in ("jpg", "jpeg", "png"):
                    candidates = list((session_dir / "frames").rglob(f"{txt.stem}.{ext}"))
                    if candidates:
                        self._frame_paths.append(str(candidates[0]))
                        break

        # Load persisted reviews
        reviews_list = load_reviews(self._session_id, self._sm)
        self._reviews = {r.frame_path: r for r in reviews_list}

        # Ensure every frame has a pending review placeholder
        for fp in self._frame_paths:
            if fp not in self._reviews:
                dets = self._load_autolabel_dets(fp)
                self._reviews[fp] = AnnotationReview(
                    frame_path=fp,
                    original_detections=dets,
                    corrected_detections=dets,
                    status=ReviewStatus.pending,
                )

        self._refresh_stats()
        self._populate_frame_list()
        self._current_idx = 0 if self._frame_paths else -1
        if self._current_idx >= 0:
            self._show_frame(self._current_idx)

    def _load_autolabel_dets(self, frame_path: str) -> list[Detection]:
        """Read YOLO label file for a frame and convert to Detection list."""
        session_dir = self._sm.session_folder(self._session_id)
        stem = Path(frame_path).stem
        lbl = session_dir / "autolabel" / "yolo" / f"{stem}.txt"
        dets = []
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:])
                    dets.append(Detection(
                        class_id=cls_id,
                        class_name=str(cls_id),
                        confidence=1.0,
                        bbox_xyxy=[0.0, 0.0, 0.0, 0.0],
                        bbox_xywhn=[cx, cy, w, h],
                    ))
        return dets

    def _refresh_stats(self) -> None:
        if not self._session_id:
            return
        stats = get_review_stats(self._session_id, self._sm)
        self._lbl_total.setText(f"Total frames: {stats['total_frames']}")
        self._lbl_pending.setText(f"Pending: {stats['pending']}")
        self._lbl_accepted.setText(f"Accepted: {stats['accepted']}")
        self._lbl_corrected.setText(f"Corrected: {stats['corrected']}")
        self._lbl_rejected.setText(f"Rejected: {stats['rejected']}")
        self._lbl_coverage.setText(f"Coverage: {stats['coverage_percent']}%")

    def _populate_frame_list(self) -> None:
        filter_txt = self._filter_combo.currentText().lower()
        self._frame_list.blockSignals(True)
        self._frame_list.clear()
        for fp in self._frame_paths:
            review = self._reviews.get(fp)
            status = review.status if review else ReviewStatus.pending
            if filter_txt != "all" and status.value != filter_txt:
                continue
            item = QListWidgetItem(Path(fp).name)
            item.setForeground(_STATUS_COLOR[status])
            item.setData(Qt.ItemDataRole.UserRole, fp)
            self._frame_list.addItem(item)
        self._frame_list.blockSignals(False)

    # ------------------------------------------------------------------
    # Frame display
    # ------------------------------------------------------------------

    def _show_frame(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._frame_paths):
            return
        self._current_idx = idx
        fp = self._frame_paths[idx]
        review = self._reviews.get(fp)

        pixmap = QPixmap(fp)
        if pixmap.isNull():
            self._preview.setText(f"Cannot load: {Path(fp).name}")
        else:
            dets = review.corrected_detections if review else []
            self._preview.setPixmap(self._render_bboxes(pixmap, dets, review))

        status = review.status if review else ReviewStatus.pending
        self._frame_info.setText(
            f"{idx+1}/{len(self._frame_paths)}  {Path(fp).name}  [{status.value}]"
        )
        for b in (self._btn_accept, self._btn_correct, self._btn_reject):
            b.setEnabled(True)
        self._btn_prev.setEnabled(idx > 0)
        self._btn_next.setEnabled(idx < len(self._frame_paths) - 1)

    def _render_bboxes(
        self,
        pixmap: QPixmap,
        detections: list[Detection],
        review: AnnotationReview | None,
    ) -> QPixmap:
        label_size = self._preview.size()
        scaled = pixmap.scaled(
            label_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        if not detections:
            return scaled

        w, h = scaled.width(), scaled.height()
        orig_w, orig_h = pixmap.width(), pixmap.height()

        status = review.status if review else ReviewStatus.pending
        color = _STATUS_COLOR[status]

        painter = QPainter(scaled)
        painter.setPen(QPen(color, 2))
        font = QFont("monospace", 8)
        painter.setFont(font)

        for det in detections:
            cx, cy, bw, bh = det.bbox_xywhn
            x1 = int((cx - bw / 2) * orig_w * w / orig_w)
            y1 = int((cy - bh / 2) * orig_h * h / orig_h)
            bw_px = int(bw * w)
            bh_px = int(bh * h)
            painter.drawRect(x1, y1, bw_px, bh_px)
            painter.drawText(x1 + 2, y1 - 2, f"cls{det.class_id}")
        painter.end()
        return scaled

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    @pyqtSlot(ReviewStatus)
    def _on_review(self, status: ReviewStatus) -> None:
        if self._current_idx < 0 or not self._session_id:
            return
        fp = self._frame_paths[self._current_idx]
        review = self._reviews.get(fp)
        if review is None:
            return
        review.status = status
        review.reviewed_at = datetime.now(timezone.utc)
        self._reviews[fp] = review
        save_review(review, self._session_id, self._sm)
        self._refresh_stats()
        self._populate_frame_list()
        self._show_frame(self._current_idx)
        # Auto-advance to next pending
        self._advance_to_pending()

    def _advance_to_pending(self) -> None:
        start = self._current_idx + 1
        for i in range(start, len(self._frame_paths)):
            fp = self._frame_paths[i]
            if self._reviews.get(fp, None) is None or \
               self._reviews[fp].status == ReviewStatus.pending:
                self._show_frame(i)
                return

    @pyqtSlot()
    def _on_prev(self) -> None:
        self._show_frame(self._current_idx - 1)

    @pyqtSlot()
    def _on_next(self) -> None:
        self._show_frame(self._current_idx + 1)

    @pyqtSlot(int)
    def _on_list_selection(self, row: int) -> None:
        item = self._frame_list.item(row)
        if item is None:
            return
        fp = item.data(Qt.ItemDataRole.UserRole)
        try:
            idx = self._frame_paths.index(fp)
            self._show_frame(idx)
        except ValueError:
            pass

    @pyqtSlot(str)
    def _on_filter_changed(self, _: str) -> None:
        self._populate_frame_list()

    @pyqtSlot()
    def _on_export(self) -> None:
        if not self._session_id:
            QMessageBox.warning(self, "No session", "Select a session first.")
            return
        out_dir = QFileDialog.getExistingDirectory(
            self, "Select output directory", str(Path.home())
        )
        if not out_dir:
            return
        if self._worker and self._worker.isRunning():
            return
        self._btn_export.setEnabled(False)
        self._worker = AnnotationReviewWorker(self._session_id, self._sm, out_dir)
        self._worker.status.connect(lambda m: self._frame_info.setText(m))
        self._worker.result.connect(self._on_export_done)
        self._worker.error.connect(lambda e: QMessageBox.critical(self, "Export error", e))
        self._worker.finished.connect(lambda: self._btn_export.setEnabled(True))
        self._worker.start()

    @pyqtSlot(object)
    def _on_export_done(self, result: object) -> None:
        QMessageBox.information(
            self, "Export complete",
            f"Exported {result.augmented_count} frames to:\n{result.output_dir}"
        )

    @pyqtSlot()
    def _on_trigger_retrain(self) -> None:
        if not self._session_id:
            QMessageBox.warning(self, "No session", "Select a session first.")
            return
        from core.annotation_review import check_learning_trigger
        ready = check_learning_trigger(self._session_id, self._sm, threshold=1)
        if not ready:
            QMessageBox.information(
                self, "Not ready",
                "Not enough accepted/corrected frames yet.\n"
                "Accept or correct at least 1 frame first."
            )
            return
        # Switch mainwindow to ContinuousLearning tab
        mw = self.window()
        if hasattr(mw, "show_continuous_learning"):
            mw.show_continuous_learning(self._session_id)
        else:
            QMessageBox.information(
                self, "Ready to retrain",
                "Threshold met. Use the Auto-Retrain tab to start the cycle."
            )
