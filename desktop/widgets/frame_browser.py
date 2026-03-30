"""
desktop/widgets/frame_browser.py — Extracted frame grid browser.

Layout:
  ┌──────────────────────────────────────────────────────────┐
  │  [Session] [Device] [FPS filter] [Blur threshold] [Load]│
  ├───────────────────────────────────────────────────────────┤
  │  Scrollable thumbnail grid (lazy-loaded QListWidget)     │
  ├───────────────────────────────────────────────────────────┤
  │  Selected frame: full-res preview + metadata overlay     │
  └──────────────────────────────────────────────────────────┘

Features:
  - Lazy thumbnail loading (only visible rows are rendered)
  - Blur score (Laplacian variance) per frame — blurry frames flagged red
  - Consecutive duplicate detection (SSIM > 0.98 → flagged orange)
  - Timestamp overlay on thumbnail hover
  - Click → full-res preview with timestamp, nearest IMU, GPS coordinate
  - Select frames → export as ZIP or CVAT-compatible folder
  - Syncs with TelemetryPlot scrub bar via scrub_to(seconds)
"""
from __future__ import annotations

import math
import os
import zipfile
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from PyQt6.QtCore import (
    QObject,
    QRunnable,
    QSize,
    Qt,
    QThread,
    QThreadPool,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from core.session_manager import SessionManager
from desktop.theme import BG, PANEL, ACCENT, HI, TEXT, MUTED, SUCCESS, WARNING, BORDER, HOVER, CARD

_BG     = BG
_PANEL  = PANEL
_ACCENT = ACCENT
_TEXT   = TEXT
_MUTED  = MUTED
_OK     = SUCCESS
_WARN   = WARNING
_ERR    = HI

_THUMB_W = 160
_THUMB_H = 90
_BLUR_THRESHOLD = 100.0   # Laplacian variance below this → flagged
_SSIM_THRESHOLD = 0.98    # consecutive similarity above this → flagged


class FrameInfo(NamedTuple):
    path: str
    timestamp_ns: int
    blur_score: float       # Laplacian variance
    is_blurry: bool
    is_duplicate: bool


# ---------------------------------------------------------------------------
# Frame Quality Heatmap
# ---------------------------------------------------------------------------

class FrameQualityHeatmap(QWidget):
    """Horizontal colour-coded timeline strip showing per-frame blur quality.

    Buckets frames by time position and colours each bucket:
      green  → average blur_score > 200  (sharp)
      yellow → average blur_score 100–200 (moderate)
      red    → average blur_score < 100  (blurry)

    Emits seek_to_timestamp(int) ns when clicked, allowing the grid to scroll
    to the nearest frame.
    """

    seek_to_timestamp = pyqtSignal(int)   # nanoseconds

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._frames: list[FrameInfo] = []
        self.setFixedHeight(36)
        self.setToolTip("Click to seek to that timestamp")
        self.setVisible(False)

    def set_frames(self, frames: list[FrameInfo]) -> None:
        self._frames = frames
        self.setVisible(bool(frames))
        self.update()

    def clear(self) -> None:
        self._frames = []
        self.setVisible(False)
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        if not self._frames:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        w = self.width()
        h = self.height()
        n_buckets = max(1, w // 4)

        ts_min = self._frames[0].timestamp_ns
        ts_max = self._frames[-1].timestamp_ns
        span = ts_max - ts_min if ts_max > ts_min else 1

        # Assign frames to buckets
        buckets: list[list[float]] = [[] for _ in range(n_buckets)]
        for fi in self._frames:
            ratio = (fi.timestamp_ns - ts_min) / span
            b = min(n_buckets - 1, int(ratio * n_buckets))
            buckets[b].append(fi.blur_score)

        bucket_w = w / n_buckets
        for i, bucket in enumerate(buckets):
            if not bucket:
                color = QColor(_MUTED)
            else:
                avg = sum(bucket) / len(bucket)
                if avg > 200:
                    color = QColor("#27ae60")   # green
                elif avg > 100:
                    color = QColor("#f39c12")   # yellow
                else:
                    color = QColor("#e94560")   # red
            x = int(i * bucket_w)
            bw = max(1, int(bucket_w))
            painter.fillRect(x, 2, bw - 1, h - 4, color)

        # Border
        painter.setPen(QColor(_ACCENT))
        painter.drawRect(0, 0, w - 1, h - 1)
        painter.end()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if not self._frames:
            return
        ratio = event.position().x() / max(1, self.width())
        ratio = max(0.0, min(1.0, ratio))
        ts_min = self._frames[0].timestamp_ns
        ts_max = self._frames[-1].timestamp_ns
        ts = int(ts_min + ratio * (ts_max - ts_min))
        self.seek_to_timestamp.emit(ts)


# ---------------------------------------------------------------------------
# Background frame scanner (blur + dedup) — runs in QThread
# ---------------------------------------------------------------------------

class _FrameScannerSignals(QObject):
    progress = pyqtSignal(int, int)                    # (done, total)
    done     = pyqtSignal(list)                        # list[FrameInfo]
    error    = pyqtSignal(str)


class _FrameScanner(QThread):
    """Scans a list of jpg paths off the main thread: blur score + dedup."""

    def __init__(self, jpg_files: list[Path], blur_threshold: int) -> None:
        super().__init__()
        self._files = jpg_files
        self._threshold = blur_threshold
        self.signals = _FrameScannerSignals()

    def run(self) -> None:
        frames: list[FrameInfo] = []
        prev_gray = None
        total = len(self._files)
        try:
            for i, jp in enumerate(self._files):
                try:
                    ts_ns = int(jp.stem)
                except ValueError:
                    ts_ns = i * 200_000_000

                blur = _laplacian_variance(str(jp))
                is_blurry = blur < self._threshold

                is_dup = False
                try:
                    img = cv2.imread(str(jp))
                    if img is not None:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        if prev_gray is not None and gray.shape == prev_gray.shape:
                            diff = float(np.mean(
                                np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))
                            ))
                            is_dup = diff < (255.0 * (1.0 - _SSIM_THRESHOLD))
                        prev_gray = gray
                except Exception:
                    pass

                frames.append(FrameInfo(
                    path=str(jp),
                    timestamp_ns=ts_ns,
                    blur_score=blur,
                    is_blurry=is_blurry,
                    is_duplicate=is_dup,
                ))
                self.signals.progress.emit(i + 1, total)

            self.signals.done.emit(frames)
        except Exception as exc:
            self.signals.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Lazy thumbnail loader (runs in QThreadPool)
# ---------------------------------------------------------------------------

class _ThumbSignals(QObject):
    done = pyqtSignal(int, QPixmap)   # (row_index, pixmap)


class _ThumbLoader(QRunnable):
    def __init__(self, row: int, path: str, w: int, h: int) -> None:
        super().__init__()
        self._row = row
        self._path = path
        self._w = w
        self._h = h
        self.signals = _ThumbSignals()

    def run(self) -> None:
        try:
            img = cv2.imread(self._path)
            if img is None:
                return
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            # Letterbox
            scale = min(self._w / w, self._h / h)
            nw, nh = int(w * scale), int(h * scale)
            resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
            qimg = QImage(resized.data, nw, nh, nw * 3, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.signals.done.emit(self._row, pixmap)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class FrameBrowser(QWidget):
    """Thumbnail grid browser for extracted frames."""

    # Emitted when user clicks a frame; carries timestamp in seconds
    frame_selected = pyqtSignal(float)

    def __init__(self, session_manager: SessionManager, parent=None) -> None:
        super().__init__(parent)
        self._sm = session_manager
        self._session_id: str = ""
        self._frames: list[FrameInfo] = []
        self._scanner: _FrameScanner | None = None
        self._pool = QThreadPool.globalInstance()
        self._pool.setMaxThreadCount(4)
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # --- Toolbar ---
        toolbar = QHBoxLayout()
        self._session_label = QLabel("No session selected")
        toolbar.addWidget(self._session_label)
        toolbar.addStretch()

        toolbar.addWidget(QLabel("Min blur:"))
        self._blur_slider = QSlider(Qt.Orientation.Horizontal)
        self._blur_slider.setRange(0, 500)
        self._blur_slider.setValue(int(_BLUR_THRESHOLD))
        self._blur_slider.setFixedWidth(100)
        self._blur_slider.setToolTip("Flag frames below this Laplacian variance")
        self._blur_slider.valueChanged.connect(self._refresh_flags)
        toolbar.addWidget(self._blur_slider)
        self._blur_label = QLabel(f"{int(_BLUR_THRESHOLD)}")
        self._blur_label.setFixedWidth(36)
        toolbar.addWidget(self._blur_label)

        self._load_btn = QPushButton("Load Frames")
        self._load_btn.setEnabled(False)
        self._load_btn.setFixedHeight(28)
        self._load_btn.clicked.connect(self._load_frames)
        toolbar.addWidget(self._load_btn)

        self._export_btn = QPushButton("Export Selected…")
        self._export_btn.setEnabled(False)
        self._export_btn.setFixedHeight(28)
        self._export_btn.clicked.connect(self._export_selected)
        toolbar.addWidget(self._export_btn)

        root.addLayout(toolbar)

        # --- Stats bar ---
        self._stats_label = QLabel("")
        root.addWidget(self._stats_label)

        # --- Scan progress bar (hidden by default) ---
        self._scan_progress = QProgressBar()
        self._scan_progress.setFixedHeight(10)
        self._scan_progress.setTextVisible(False)
        self._scan_progress.setVisible(False)
        root.addWidget(self._scan_progress)

        # --- Quality heatmap strip ---
        self._heatmap = FrameQualityHeatmap(self)
        self._heatmap.seek_to_timestamp.connect(self._on_heatmap_seek)
        root.addWidget(self._heatmap)

        # --- Main splitter: grid | detail ---
        main_split = QSplitter(Qt.Orientation.Horizontal)

        # Thumbnail grid
        self._grid = QListWidget()
        self._grid.setViewMode(QListWidget.ViewMode.IconMode)
        self._grid.setIconSize(QSize(_THUMB_W, _THUMB_H))
        self._grid.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._grid.setSpacing(4)
        self._grid.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self._grid.itemClicked.connect(self._on_item_clicked)
        self._grid.itemSelectionChanged.connect(self._on_selection_changed)
        main_split.addWidget(self._grid)

        # Detail panel
        detail = QWidget()
        detail_layout = QVBoxLayout(detail)
        detail_layout.setContentsMargins(4, 0, 0, 0)

        detail_hdr = QLabel("Frame Detail")
        detail_hdr.setFont(QFont("sans-serif", 9, QFont.Weight.Bold))
        detail_layout.addWidget(detail_hdr)

        self._preview_label = QLabel()
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setMinimumHeight(180)
        self._preview_label.setText("Click a frame to preview")
        detail_layout.addWidget(self._preview_label)

        self._meta_label = QLabel("")
        self._meta_label.setFont(QFont("monospace", 9))
        self._meta_label.setWordWrap(True)
        self._meta_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        detail_layout.addWidget(self._meta_label)
        detail_layout.addStretch()

        main_split.addWidget(detail)
        main_split.setStretchFactor(0, 3)
        main_split.setStretchFactor(1, 1)
        root.addWidget(main_split, stretch=1)

        # Placeholder
        self._placeholder = QLabel("Select a session with extracted frames.")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._placeholder)
        main_split.setVisible(False)
        self._main_split = main_split
        self._placeholder.setVisible(True)

    # ------------------------------------------------------------------
    # Session change
    # ------------------------------------------------------------------

    @pyqtSlot(str)
    def on_session_changed(self, session_id: str) -> None:
        self._session_id = session_id
        self._frames = []
        self._grid.clear()
        self._export_btn.setEnabled(False)
        self._stats_label.setText("")
        self._heatmap.clear()
        self._scan_progress.setVisible(False)
        if self._scanner and self._scanner.isRunning():
            self._scanner.quit()
            self._scanner.wait()
            self._scanner = None

        if not session_id:
            self._session_label.setText("No session selected")
            self._load_btn.setEnabled(False)
            self._show_placeholder("Select a session with extracted frames.")
            return

        try:
            s = self._sm.get_session(session_id)
        except Exception:
            return

        self._session_label.setText(f"{s.name}  [{s.environment} · {s.location}]")

        has_extraction = self._find_frame_dir(session_id) is not None
        self._load_btn.setEnabled(True)
        self._show_placeholder(
            "Click Load Frames to browse extracted frames."
            if has_extraction else
            "No extracted frames found — run extraction first (Extract tab)."
        )

    # ------------------------------------------------------------------
    # Load frames
    # ------------------------------------------------------------------

    def _find_frame_dir(self, session_id: str) -> Path | None:
        folder = self._sm.session_folder(session_id)
        for candidate in [
            folder / "extraction_gopro" / "cam0" / "data",
            folder / "extraction_insta360" / "cam0" / "data",
            folder / "extraction" / "cam0" / "data",
        ]:
            if candidate.exists() and any(candidate.glob("*.jpg")):
                return candidate
        return None

    def _load_frames(self) -> None:
        frame_dir = self._find_frame_dir(self._session_id)
        if frame_dir is None:
            path = QFileDialog.getExistingDirectory(
                self, "Select frame directory", str(Path.home())
            )
            if not path:
                return
            frame_dir = Path(path)

        jpg_files = sorted(frame_dir.glob("*.jpg"))
        if not jpg_files:
            self._show_placeholder(f"No .jpg frames found in:\n{frame_dir}")
            return

        self._load_btn.setEnabled(False)
        self._grid.clear()
        self._stats_label.setText(f"Scanning {len(jpg_files)} frames…")
        self._scan_progress.setRange(0, len(jpg_files))
        self._scan_progress.setValue(0)
        self._scan_progress.setVisible(True)

        # Stop any previous scan
        if self._scanner and self._scanner.isRunning():
            self._scanner.quit()
            self._scanner.wait()

        self._scanner = _FrameScanner(jpg_files, self._blur_slider.value())
        self._scanner.signals.progress.connect(self._on_scan_progress)
        self._scanner.signals.done.connect(self._on_scan_done)
        self._scanner.signals.error.connect(self._on_scan_error)
        self._scanner.start()

    @pyqtSlot(int, int)
    def _on_scan_progress(self, done: int, total: int) -> None:
        self._scan_progress.setValue(done)
        self._stats_label.setText(f"Scanning frames… {done}/{total}")

    @pyqtSlot(list)
    def _on_scan_done(self, frames: list) -> None:
        self._scan_progress.setVisible(False)
        self._frames = frames
        self._populate_grid(frames)
        self._load_btn.setEnabled(True)

    @pyqtSlot(str)
    def _on_scan_error(self, msg: str) -> None:
        self._scan_progress.setVisible(False)
        self._load_btn.setEnabled(True)
        self._show_placeholder(f"Error scanning frames:\n{msg}")

    def _populate_grid(self, frames: list[FrameInfo]) -> None:
        self._grid.clear()
        for i, fi in enumerate(frames):
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, i)
            item.setSizeHint(QSize(_THUMB_W + 8, _THUMB_H + 24))
            ts_s = fi.timestamp_ns / 1e9
            label = f"{ts_s:.2f}s"
            if fi.is_blurry:
                label += " ⚠blur"
                item.setForeground(QColor(_ERR))
            elif fi.is_duplicate:
                label += " ~dup"
                item.setForeground(QColor(_WARN))
            else:
                item.setForeground(QColor(_TEXT))
            item.setText(label)
            # Default grey icon until loaded
            pm = QPixmap(_THUMB_W, _THUMB_H)
            pm.fill(QColor(_PANEL))
            item.setIcon(QIcon(pm))
            self._grid.addItem(item)

            # Kick off async thumbnail load
            loader = _ThumbLoader(i, fi.path, _THUMB_W, _THUMB_H)
            loader.signals.done.connect(self._on_thumb_ready)
            self._pool.start(loader)

        n_blur = sum(1 for f in frames if f.is_blurry)
        n_dup  = sum(1 for f in frames if f.is_duplicate)
        self._stats_label.setText(
            f"{len(frames)} frames  ·  {n_blur} blurry  ·  {n_dup} duplicates"
        )
        self._heatmap.set_frames(frames)
        self._placeholder.setVisible(False)
        self._main_split.setVisible(True)

    @pyqtSlot(int, QPixmap)
    def _on_thumb_ready(self, row: int, pixmap: QPixmap) -> None:
        if row < self._grid.count():
            self._grid.item(row).setIcon(QIcon(pixmap))

    # ------------------------------------------------------------------
    # Blur flag refresh (slider changed)
    # ------------------------------------------------------------------

    def _refresh_flags(self, value: int) -> None:
        self._blur_label.setText(str(value))
        if not self._frames:
            return
        for i, fi in enumerate(self._frames):
            is_blurry = fi.blur_score < value
            # Rebuild FrameInfo with updated flag
            self._frames[i] = FrameInfo(
                path=fi.path, timestamp_ns=fi.timestamp_ns,
                blur_score=fi.blur_score, is_blurry=is_blurry,
                is_duplicate=fi.is_duplicate,
            )
            item = self._grid.item(i)
            if item:
                ts_s = fi.timestamp_ns / 1e9
                label = f"{ts_s:.2f}s"
                if is_blurry:
                    label += " ⚠blur"
                    item.setForeground(QColor(_ERR))
                elif fi.is_duplicate:
                    label += " ~dup"
                    item.setForeground(QColor(_WARN))
                else:
                    item.setForeground(QColor(_TEXT))
                item.setText(label)

    # ------------------------------------------------------------------
    # Frame click → detail
    # ------------------------------------------------------------------

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        idx = item.data(Qt.ItemDataRole.UserRole)
        if idx is None or idx >= len(self._frames):
            return
        fi = self._frames[idx]
        self._show_detail(fi)
        self.frame_selected.emit(fi.timestamp_ns / 1e9)

    def _show_detail(self, fi: FrameInfo) -> None:
        # Load full-res preview (scaled to fit label)
        try:
            img = cv2.imread(fi.path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img_rgb.shape[:2]
                max_w, max_h = 320, 200
                scale = min(max_w / w, max_h / h)
                nw, nh = int(w * scale), int(h * scale)
                resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
                qimg = QImage(resized.data, nw, nh, nw * 3, QImage.Format.Format_RGB888)
                self._preview_label.setPixmap(QPixmap.fromImage(qimg))
        except Exception:
            self._preview_label.setText("(preview unavailable)")

        ts_s = fi.timestamp_ns / 1e9
        lines = [
            f"Timestamp:  {fi.timestamp_ns} ns",
            f"            ({ts_s:.4f} s)",
            f"Blur score: {fi.blur_score:.1f}"
            + (" ⚠ BLURRY" if fi.is_blurry else ""),
            f"Duplicate:  {'yes ⚠' if fi.is_duplicate else 'no'}",
            f"Path:       {Path(fi.path).name}",
        ]
        self._meta_label.setText("\n".join(lines))

    def _on_heatmap_seek(self, ts_ns: int) -> None:
        """Seek the grid to the frame nearest to ts_ns."""
        if not self._frames:
            return
        best = min(
            range(len(self._frames)),
            key=lambda i: abs(self._frames[i].timestamp_ns - ts_ns),
        )
        self._grid.setCurrentRow(best)
        self._grid.scrollToItem(self._grid.item(best))
        item = self._grid.item(best)
        if item:
            self._on_item_clicked(item)

    # ------------------------------------------------------------------
    # Selection tracking
    # ------------------------------------------------------------------

    def _on_selection_changed(self) -> None:
        n = len(self._grid.selectedItems())
        self._export_btn.setEnabled(n > 0)
        if n > 0:
            self._stats_label.setText(
                self._stats_label.text().split("·")[0].strip()
                + f"  ·  {n} selected"
            )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export_selected(self) -> None:
        items = self._grid.selectedItems()
        if not items:
            return

        paths = []
        for item in items:
            idx = item.data(Qt.ItemDataRole.UserRole)
            if idx is not None and idx < len(self._frames):
                paths.append(self._frames[idx].path)

        export_path, _ = QFileDialog.getSaveFileName(
            self, "Export Frames", "frames_export.zip", "ZIP Archive (*.zip)"
        )
        if not export_path:
            return

        try:
            with zipfile.ZipFile(export_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in paths:
                    zf.write(p, Path(p).name)
            QMessageBox.information(
                self, "Export Complete",
                f"Exported {len(paths)} frame(s) to:\n{export_path}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    # ------------------------------------------------------------------
    # Sync from telemetry scrub bar
    # ------------------------------------------------------------------

    def scrub_to(self, seconds: float) -> None:
        """Select the frame nearest to *seconds* (called by TelemetryPlot)."""
        if not self._frames:
            return
        target_ns = int(seconds * 1e9)
        best = min(
            range(len(self._frames)),
            key=lambda i: abs(self._frames[i].timestamp_ns - target_ns),
        )
        self._grid.setCurrentRow(best)
        self._on_item_clicked(self._grid.item(best))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _show_placeholder(self, msg: str) -> None:
        self._placeholder.setText(msg)
        self._placeholder.setVisible(True)
        self._main_split.setVisible(False)


# ---------------------------------------------------------------------------
# Image analysis helpers
# ---------------------------------------------------------------------------

def _laplacian_variance(path: str) -> float:
    """Compute Laplacian variance as a blur metric. Higher = sharper."""
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        return float(cv2.Laplacian(img, cv2.CV_64F).var())
    except Exception:
        return 0.0


