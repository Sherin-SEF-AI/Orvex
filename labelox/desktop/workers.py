"""
labelox/desktop/workers.py — QThread workers for background tasks.

Pattern: BaseWorker with progress/status/result/error signals + EMA-smoothed ETA.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from PyQt6.QtCore import QThread, pyqtSignal


class BaseWorker(QThread):
    """Base worker with standard signals and ETA tracking."""

    progress = pyqtSignal(int, int)      # (current, total)
    status = pyqtSignal(str)
    result = pyqtSignal(object)
    error = pyqtSignal(str)
    eta = pyqtSignal(float)              # seconds remaining

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._start_time: float = 0.0
        self._ema_rate: float = 0.0      # items/sec EMA
        self._alpha: float = 0.3         # EMA smoothing

    def _tick(self, current: int, total: int) -> None:
        """Call after each item to emit progress + ETA."""
        self.progress.emit(current, total)
        elapsed = time.time() - self._start_time
        if current > 0 and elapsed > 0:
            rate = current / elapsed
            self._ema_rate = self._alpha * rate + (1 - self._alpha) * self._ema_rate if self._ema_rate else rate
            remaining = (total - current) / self._ema_rate if self._ema_rate > 0 else 0
            self.eta.emit(remaining)

    def run(self) -> None:
        self._start_time = time.time()
        try:
            result = self._execute()
            self.result.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))

    def _execute(self) -> Any:
        raise NotImplementedError


class ImportWorker(BaseWorker):
    """Import images into a project."""

    def __init__(self, paths: list[Path], project_id: str, thumb_dir: Path, parent=None):
        super().__init__(parent)
        self._paths = paths
        self._project_id = project_id
        self._thumb_dir = thumb_dir

    def _execute(self) -> Any:
        from labelox.core.image_manager import import_images
        from labelox.core.database import get_session
        db = get_session()
        try:
            created = import_images(
                self._paths,
                self._project_id,
                thumb_dir=self._thumb_dir,
                db=db,
            )
            return {"imported": len(created)}
        finally:
            db.close()


class ExportWorker(BaseWorker):
    """Export annotations in a given format."""

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self._config = config

    def _execute(self) -> Any:
        from labelox.core.exporter import export_project
        return export_project(self._config)


class AutoAnnotateWorker(BaseWorker):
    """Run YOLOv8 auto-annotation on a project."""

    def __init__(self, project_id: str, model_path: str = "yolov8n.pt",
                 conf: float = 0.25, parent=None):
        super().__init__(parent)
        self._project_id = project_id
        self._model_path = model_path
        self._conf = conf

    def _execute(self) -> Any:
        from labelox.core.auto_annotator import run_auto_annotate_project
        return run_auto_annotate_project(
            self._project_id,
            model_path=self._model_path,
            conf_threshold=self._conf,
            progress_callback=lambda cur, tot: self._tick(cur, tot),
            status_callback=lambda msg: self.status.emit(msg),
        )


class SAMWorker(BaseWorker):
    """Run SAM prediction for a single image click/box."""

    def __init__(self, image_path: str, points: list | None = None,
                 box: list | None = None, labels: list | None = None,
                 parent=None):
        super().__init__(parent)
        self._image_path = image_path
        self._points = points
        self._box = box
        self._labels = labels

    def _execute(self) -> Any:
        from labelox.core.sam_engine import load_sam, set_image_for_sam
        load_sam()
        set_image_for_sam(self._image_path)
        if self._box:
            from labelox.core.sam_engine import predict_mask_from_box
            return predict_mask_from_box(self._image_path, self._box)
        elif self._points and self._labels:
            from labelox.core.sam_engine import predict_mask_from_points
            return predict_mask_from_points(
                self._image_path, self._points, self._labels,
            )
        elif self._points:
            from labelox.core.sam_engine import predict_mask_from_point
            return predict_mask_from_point(
                self._image_path, self._points[0][0], self._points[0][1],
            )
        return None


class ClassifyWorker(BaseWorker):
    """Run CLIP scene classification on project images."""

    def __init__(self, project_id: str, parent=None):
        super().__init__(parent)
        self._project_id = project_id

    def _execute(self) -> Any:
        from labelox.core.classifier import classify_batch, load_clip
        from labelox.core.database import get_images, get_session
        load_clip()
        db = get_session()
        try:
            images = get_images(self._project_id, limit=99999, db=db)
            paths = [img.file_path for img in images]
            results = classify_batch(paths, progress_callback=lambda c, t: self._tick(c, t))
            return {"classified": len(results), "results": results}
        finally:
            db.close()


class SimilarityWorker(BaseWorker):
    """Build similarity index for a project."""

    def __init__(self, project_id: str, parent=None):
        super().__init__(parent)
        self._project_id = project_id

    def _execute(self) -> Any:
        from labelox.core.similarity_engine import build_embedding_index
        from labelox.core.database import get_images, get_session
        db = get_session()
        try:
            images = get_images(self._project_id, limit=99999, db=db)
            paths = [img.file_path for img in images]
            ids = [img.id for img in images]
            index = build_embedding_index(paths, ids)
            return {"indexed": len(paths), "index": index}
        finally:
            db.close()


class TrackWorker(BaseWorker):
    """Run cross-frame tracking on a sequence."""

    def __init__(self, project_id: str, sequence_id: str, parent=None):
        super().__init__(parent)
        self._project_id = project_id
        self._sequence_id = sequence_id

    def _execute(self) -> Any:
        from labelox.core.tracker import track_sequence
        return track_sequence(self._project_id, self._sequence_id)


class ThumbnailWorker(BaseWorker):
    """Generate thumbnails for all images in a project."""

    def __init__(self, project_id: str, thumb_dir: Path, parent=None):
        super().__init__(parent)
        self._project_id = project_id
        self._thumb_dir = thumb_dir

    def _execute(self) -> Any:
        from labelox.core.database import get_images, get_session
        from labelox.core.utils import generate_thumbnail
        db = get_session()
        try:
            images = get_images(self._project_id, limit=99999, db=db)
            count = 0
            for i, img in enumerate(images):
                self._tick(i + 1, len(images))
                if img.thumbnail_path and Path(img.thumbnail_path).exists():
                    continue
                thumb = generate_thumbnail(img.file_path, self._thumb_dir)
                if thumb:
                    img.thumbnail_path = str(thumb)
                    count += 1
            db.commit()
            return {"generated": count}
        finally:
            db.close()
