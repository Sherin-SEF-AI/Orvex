"""
desktop/workers.py — QThread workers for all background operations.

Every long-running operation (audit, extract, calibrate, dataset build)
runs in a Worker subclass.  Workers emit:
  - progress(int)       — 0-100 percent
  - status(str)         — human-readable status line
  - result(object)      — final result payload (type varies per worker)
  - error(str)          — actionable error message on failure

The UI never blocks: all heavy lifting happens in run().
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QThread, pyqtSignal

from core.models import (
    AugmentationConfig, ExtractionConfig, TrainingConfig,
    InferenceRequest,
)
from core.session_manager import SessionManager
from core.utils import setup_logging


# ---------------------------------------------------------------------------
# Base worker
# ---------------------------------------------------------------------------

class BaseWorker(QThread):
    """Abstract base for all background workers.

    Subclasses must implement run().  They should call _emit_progress(),
    _emit_status(), and either _emit_result() or _emit_error() before
    returning from run().

    Timing is tracked automatically: every _emit_progress() call computes
    elapsed and ETA seconds and emits timing(elapsed_s, eta_s).
    """

    progress = pyqtSignal(int)            # 0-100
    status   = pyqtSignal(str)            # status line text
    result   = pyqtSignal(object)         # final result
    error    = pyqtSignal(str)            # actionable error message
    timing   = pyqtSignal(float, float)   # (elapsed_seconds, eta_seconds)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._start_time: float | None = None
        self._smoothed_eta: float = 0.0

    def _emit_progress(self, pct: int) -> None:
        pct = max(0, min(100, pct))
        now = time.monotonic()
        if self._start_time is None:
            self._start_time = now
        elapsed = now - self._start_time
        if pct > 0:
            raw_eta = elapsed / pct * (100 - pct)
        else:
            raw_eta = 0.0
        # EMA smoothing to prevent wild jumps at stage transitions
        if self._smoothed_eta <= 0 or pct <= 1:
            self._smoothed_eta = raw_eta
        else:
            self._smoothed_eta = 0.3 * raw_eta + 0.7 * self._smoothed_eta
        self.progress.emit(pct)
        self.timing.emit(elapsed, self._smoothed_eta)

    def _emit_status(self, msg: str) -> None:
        self.status.emit(msg)

    def _emit_result(self, payload: Any) -> None:
        self.result.emit(payload)

    def _emit_error(self, msg: str) -> None:
        self.error.emit(msg)


# ---------------------------------------------------------------------------
# Audit worker
# ---------------------------------------------------------------------------

class AuditWorker(BaseWorker):
    """Run file audits for all files in a session.

    Emits result(list[AuditResult]) on success.
    Updates the session's audit_results via SessionManager.
    """

    def __init__(
        self,
        session_id: str,
        session_manager: SessionManager,
        parent=None,
    ) -> None:
        super().__init__(parent)  # calls BaseWorker.__init__
        self._session_id = session_id
        self._sm = session_manager

    def run(self) -> None:
        from core.audit import audit_file
        from core.models import AuditResult

        try:
            session = self._sm.get_session(self._session_id)
            files = session.files
            if not files:
                self._emit_error(
                    "No files in session — add files before running audit."
                )
                return

            results: list[AuditResult] = []
            n = len(files)
            for i, fp in enumerate(files):
                self._emit_status(f"Auditing {Path(fp).name} ({i+1}/{n})…")
                # emit progress at start of file (gives ETA after first file)
                self._emit_progress(int(i / n * 90))
                audit_result = audit_file(fp)
                results.append(audit_result)
                # emit after file so elapsed captures real per-file cost
                self._emit_progress(int((i + 1) / n * 90))

            self._sm.set_audit_results(self._session_id, results)
            self._emit_progress(100)
            self._emit_status(f"Audit complete — {len(results)} file(s) processed.")
            self._emit_result(results)

        except Exception as exc:
            self._emit_error(str(exc))


# ---------------------------------------------------------------------------
# Extraction worker
# ---------------------------------------------------------------------------

class ExtractionWorker(BaseWorker):
    """Run GoPro telemetry + frame extraction for a session.

    Emits result(ExtractedSession) on success.
    """

    def __init__(
        self,
        session_id: str,
        config: ExtractionConfig,
        session_manager: SessionManager,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._session_id = session_id
        self._config = config
        self._sm = session_manager

    def run(self) -> None:
        from core.extractor_gopro import extract_gopro
        from core.models import DeviceType

        try:
            session = self._sm.get_session(self._session_id)
            gopro_files = [
                f for f in session.files
                if f.lower().endswith(".mp4")
            ]
            if not gopro_files:
                self._emit_error(
                    "No MP4 files found in session for GoPro extraction."
                )
                return

            self._sm.set_extraction_status(self._session_id, "running")
            self._emit_status("Starting GoPro extraction…")
            self._emit_progress(5)

            output_dir = self._sm.extraction_output_dir(self._session_id, "gopro")

            # Use the first file (chapter detection handles the rest)
            mp4_path = gopro_files[0]
            self._emit_status(f"Extracting {Path(mp4_path).name}…")

            def _progress_cb(pct: int, msg: str) -> None:
                self._emit_progress(pct)
                self._emit_status(msg)

            extracted = extract_gopro(mp4_path, self._config, output_dir,
                                      progress_callback=_progress_cb)

            self._sm.set_extraction_status(self._session_id, "done")
            self._emit_progress(100)
            self._emit_status(
                f"Extraction complete — {extracted.stats.get('frame_count', 0)} frames, "
                f"{extracted.stats.get('imu_count', 0)} IMU samples."
            )
            self._emit_result(extracted)

        except Exception as exc:
            self._sm.set_extraction_status(self._session_id, "failed")
            self._emit_error(str(exc))


# ---------------------------------------------------------------------------
# Calibration worker
# ---------------------------------------------------------------------------

class CalibrationWorker(BaseWorker):
    """Run a calibration step.

    step: "imu_static" | "camera_intrinsic" | "camera_imu_extrinsic"
    Emits result(dict) with step-specific output on success.
    """

    def __init__(
        self,
        calibration_session_id: str,
        step: str,
        file_path: str,
        extra: dict | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._cal_id = calibration_session_id
        self._step = step
        self._file_path = file_path
        self._extra = extra or {}

    def run(self) -> None:
        from core.calibration import run_calibration_step

        try:
            self._emit_status(f"Running calibration step: {self._step}…")
            self._emit_progress(5)
            result = run_calibration_step(
                self._step,
                self._file_path,
                self._extra,
                progress_callback=self._emit_progress,
                status_callback=self._emit_status,
            )
            self._emit_progress(100)
            self._emit_status(f"Calibration step '{self._step}' complete.")
            self._emit_result(result)
        except Exception as exc:
            self._emit_error(str(exc))


# ---------------------------------------------------------------------------
# Dataset build worker
# ---------------------------------------------------------------------------

class DatasetBuildWorker(BaseWorker):
    """Assemble a final dataset from one or more sessions.

    Emits result(str) — the path to the exported dataset root.
    """

    def __init__(
        self,
        session_ids: list[str],
        session_manager: SessionManager,
        output_format: str,
        output_dir: str | Path,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._session_ids = session_ids
        self._sm = session_manager
        self._output_format = output_format
        self._output_dir = Path(output_dir)

    def run(self) -> None:
        from core.dataset_builder import build_dataset

        try:
            self._emit_status("Building dataset…")
            self._emit_progress(5)
            output_path = build_dataset(
                self._session_ids,
                self._sm,
                self._output_format,
                self._output_dir,
                progress_callback=self._emit_progress,
                status_callback=self._emit_status,
            )
            self._emit_progress(100)
            self._emit_status(f"Dataset exported to: {output_path}")
            self._emit_result(str(output_path))
        except Exception as exc:
            self._emit_error(str(exc))


# ---------------------------------------------------------------------------
# Phase 2 workers
# ---------------------------------------------------------------------------

class AutoLabelWorker(BaseWorker):
    """Run YOLOv8 inference on extracted frames for a session.

    Emits:
        preview(QImage)  — annotated frame preview with drawn bounding boxes
        result(dict)     — {"annotations": [...], "stats": {...}, "preview_paths": [...]}
    """

    preview = pyqtSignal(object)  # QImage with drawn boxes

    def __init__(
        self,
        session_id: str,
        sm: SessionManager,
        model_path: str = "yolov8n.pt",
        conf: float = 0.25,
        iou: float = 0.45,
        batch_size: int = 16,
        export_format: str = "both",  # "cvat", "yolo", or "both"
        output_dir: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._session_id = session_id
        self._sm = sm
        self._model_path = model_path
        self._conf = conf
        self._iou = iou
        self._batch_size = batch_size
        self._export_format = export_format
        self._output_dir = output_dir

    def _draw_boxes_on_frame(self, frame_path: str, detections) -> "QImage | None":
        """Draw bounding boxes on a frame and return as QImage."""
        import cv2
        import numpy as np
        img = cv2.imread(frame_path)
        if img is None:
            return None

        # Color palette for classes
        colors = [
            (233, 69, 96), (78, 204, 163), (74, 158, 255), (245, 166, 35),
            (155, 89, 182), (26, 188, 156), (230, 126, 34), (52, 152, 219),
            (231, 76, 60), (46, 204, 113), (241, 196, 15), (142, 68, 173),
        ]

        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls_idx = det.class_id % len(colors)
            color = colors[cls_idx]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Convert BGR to RGB for QImage
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        from PyQt6.QtGui import QImage
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        return qimg

    def _save_annotated_frame(self, frame_path: str, detections, out_dir: str) -> str:
        """Save frame with drawn boxes to output directory."""
        import cv2
        img = cv2.imread(frame_path)
        if img is None:
            return ""

        colors = [
            (233, 69, 96), (78, 204, 163), (74, 158, 255), (245, 166, 35),
            (155, 89, 182), (26, 188, 156), (230, 126, 34), (52, 152, 219),
            (231, 76, 60), (46, 204, 113), (241, 196, 15), (142, 68, 173),
        ]

        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls_idx = det.class_id % len(colors)
            color = colors[cls_idx]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        preview_dir = Path(out_dir) / "preview"
        preview_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(preview_dir / Path(frame_path).name)
        cv2.imwrite(out_path, img)
        return out_path

    def run(self) -> None:
        from core.autolabel import (
            compute_annotation_stats,
            export_cvat_xml,
            export_yolo_format,
            load_model,
            run_inference_batch,
        )
        try:
            session = self._sm.get_session(self._session_id)
            ext_dir = self._sm.session_folder(self._session_id)
            frame_paths = sorted(str(p) for p in ext_dir.rglob("*.jpg"))
            if not frame_paths:
                self._emit_error(
                    f"No extracted frames found for session {session.name}. "
                    "Run extraction first to generate frames."
                )
                return

            self._emit_status(f"Loading model {self._model_path}…")
            model = load_model(self._model_path)
            self._emit_status(f"Running inference on {len(frame_paths)} frames…")

            annotations = run_inference_batch(
                frame_paths, model,
                conf_threshold=self._conf,
                iou_threshold=self._iou,
                batch_size=self._batch_size,
                progress_callback=self._emit_progress,
                status_callback=self._emit_status,
            )

            out_dir = self._output_dir or str(ext_dir / "autolabel")

            # Generate preview images with drawn bounding boxes
            self._emit_status("Generating annotated previews…")
            preview_paths = []
            for idx, ann in enumerate(annotations):
                # Emit live preview every 5th frame (or every frame if < 50 total)
                if idx % max(1, len(annotations) // 20) == 0 or len(annotations) < 50:
                    qimg = self._draw_boxes_on_frame(ann.frame_path, ann.detections)
                    if qimg is not None:
                        self.preview.emit(qimg)

                # Save all annotated frames to disk
                saved = self._save_annotated_frame(ann.frame_path, ann.detections, out_dir)
                if saved:
                    preview_paths.append(saved)

                pct = int((idx + 1) / len(annotations) * 100)
                self._emit_progress(pct)

            if self._export_format in ("cvat", "both"):
                export_cvat_xml(annotations, f"{out_dir}/annotations.xml",
                                task_name=session.name)
            if self._export_format in ("yolo", "both"):
                export_yolo_format(annotations, out_dir)

            stats = compute_annotation_stats(annotations)
            self._emit_result({
                "annotations": annotations,
                "stats": stats,
                "preview_paths": preview_paths,
            })
        except Exception as exc:
            self._emit_error(str(exc))


class DepthWorker(BaseWorker):
    """Estimate depth for all extracted frames in a session.

    Emits result(list[DepthResult]) on success.
    """

    def __init__(
        self,
        session_id: str,
        sm: SessionManager,
        model_name: str = "small",
        batch_size: int = 8,
        colorize: bool = True,
        metric_scale: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._session_id = session_id
        self._sm = sm
        self._model_name = model_name
        self._batch_size = batch_size
        self._colorize = colorize
        self._metric_scale = metric_scale

    def run(self) -> None:
        from core.depth_estimator import estimate_depth_batch, load_depth_model
        try:
            ext_dir = self._sm.session_folder(self._session_id)
            frame_paths = sorted(str(p) for p in ext_dir.rglob("*.jpg"))
            if not frame_paths:
                self._emit_error(
                    "No extracted frames found. Run extraction first."
                )
                return

            self._emit_status("Loading depth model…")
            model, processor = load_depth_model(self._model_name)
            self._emit_status(f"Estimating depth for {len(frame_paths)} frames…")

            output_dir = str(ext_dir / "depth")
            results = estimate_depth_batch(
                frame_paths, model, processor,
                batch_size=self._batch_size,
                output_dir=output_dir,
                colorize=self._colorize,
                progress_callback=self._emit_progress,
                status_callback=self._emit_status,
            )
            self._emit_result(results)
        except Exception as exc:
            self._emit_error(str(exc))


class SLAMWorker(BaseWorker):
    """Run ORBSLAM3 on a EuRoC-extracted session.

    Emits result(SLAMResult) on success.
    """

    def __init__(
        self,
        session_id: str,
        sm: SessionManager,
        vocabulary_path: str,
        config_yaml: str,
        mode: str = "mono_inertial",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._session_id = session_id
        self._sm = sm
        self._vocabulary_path = vocabulary_path
        self._config_yaml = config_yaml
        self._mode = mode

    def run(self) -> None:
        from core.slam_validator import run_orbslam3
        try:
            euroc_dir = str(self._sm.session_folder(self._session_id))
            result = run_orbslam3(
                euroc_dir,
                self._vocabulary_path,
                self._config_yaml,
                self._mode,
                progress_callback=self._emit_progress,
                status_callback=self._emit_status,
            )
            self._emit_result(result)
        except Exception as exc:
            self._emit_error(str(exc))


class ReconstructWorker(BaseWorker):
    """Run COLMAP SfM on extracted frames.

    Emits result(ColmapResult) on success.
    """

    def __init__(
        self,
        session_id: str,
        sm: SessionManager,
        every_nth: int = 6,
        use_gpu: bool = True,
        camera_model: str = "OPENCV_FISHEYE",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._session_id = session_id
        self._sm = sm
        self._every_nth = every_nth
        self._use_gpu = use_gpu
        self._camera_model = camera_model

    def run(self) -> None:
        from core.reconstructor import run_colmap_sfm
        import shutil, tempfile
        try:
            ext_dir = self._sm.session_folder(self._session_id)
            all_frames = sorted(ext_dir.rglob("cam0/data/*.jpg"))
            if not all_frames:
                all_frames = sorted(ext_dir.rglob("*.jpg"))
            sampled = all_frames[:: self._every_nth]
            if not sampled:
                self._emit_error("No frames found for reconstruction. Run extraction first.")
                return

            # Copy sampled frames to a temp image dir
            tmp_img = Path(tempfile.mkdtemp()) / "colmap_frames"
            tmp_img.mkdir()
            for fp in sampled:
                shutil.copy2(str(fp), str(tmp_img / fp.name))

            out_dir = str(ext_dir / "colmap")
            result = run_colmap_sfm(
                str(tmp_img), out_dir,
                camera_model=self._camera_model,
                use_gpu=self._use_gpu,
                progress_callback=self._emit_progress,
                status_callback=self._emit_status,
            )
            self._emit_result(result)
        except Exception as exc:
            self._emit_error(str(exc))


class ActiveLearningWorker(BaseWorker):
    """Score frames by uncertainty + diversity to select labeling candidates.

    Emits result(dict) with selected_frames, uncertainty_scores, diversity_scores.
    """

    def __init__(
        self,
        session_id: str,
        sm: SessionManager,
        annotations: list,
        method: str = "entropy",
        n_frames: int = 100,
        uncertainty_weight: float = 0.6,
        diversity_weight: float = 0.4,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._session_id = session_id
        self._sm = sm
        self._annotations = annotations
        self._method = method
        self._n_frames = n_frames
        self._unc_w = uncertainty_weight
        self._div_w = diversity_weight

    def run(self) -> None:
        from core.active_learning import (
            compute_diversity_scores,
            compute_label_budget_estimate,
            compute_uncertainty_scores,
            select_frames_for_labeling,
        )
        try:
            self._emit_status("Computing uncertainty scores…")
            unc_scores = compute_uncertainty_scores(self._annotations, self._method)
            self._emit_progress(40)

            self._emit_status("Computing diversity scores…")
            frame_paths = [a.frame_path for a in self._annotations]
            div_scores = compute_diversity_scores(
                frame_paths,
                progress_callback=self._emit_progress,
            )
            self._emit_progress(80)

            selected = select_frames_for_labeling(
                unc_scores, div_scores,
                self._n_frames,
                self._unc_w, self._div_w,
            )
            budget = compute_label_budget_estimate(
                len(frame_paths), len(selected)
            )
            self._emit_progress(100)
            self._emit_result({
                "selected_frames": selected,
                "uncertainty_scores": unc_scores,
                "diversity_scores": div_scores,
                "budget": budget,
            })
        except Exception as exc:
            self._emit_error(str(exc))


class AnalyticsWorker(BaseWorker):
    """Compute scene diversity and geographic coverage analytics.

    Emits result(dict) with diversity, coverage, class_distribution keys.
    """

    def __init__(
        self,
        session_id: str,
        sm: SessionManager,
        annotations: list | None = None,
        gps_samples: list | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._session_id = session_id
        self._sm = sm
        self._annotations = annotations or []
        self._gps_samples = gps_samples or []

    def run(self) -> None:
        from core.road_analytics import (
            compute_class_distribution,
            compute_geographic_coverage,
            compute_scene_diversity,
        )
        try:
            ext_dir = self._sm.session_folder(self._session_id)
            frame_paths = sorted(str(p) for p in ext_dir.rglob("*.jpg"))

            self._emit_status("Computing scene diversity…")
            diversity = compute_scene_diversity(
                frame_paths,
                progress_callback=self._emit_progress,
                status_callback=self._emit_status,
            )
            self._emit_progress(50)

            geo = None
            if self._gps_samples:
                self._emit_status("Computing geographic coverage…")
                geo = compute_geographic_coverage(
                    self._gps_samples, str(ext_dir / "analytics")
                )

            cls_dist = {}
            if self._annotations:
                self._emit_status("Computing class distribution…")
                cls_dist = compute_class_distribution(self._annotations)

            self._emit_progress(100)
            self._emit_result({
                "diversity": diversity,
                "coverage": geo,
                "class_distribution": cls_dist,
            })
        except Exception as exc:
            self._emit_error(str(exc))


class AugmentationWorker(BaseWorker):
    """Apply data augmentation pipeline to extracted frames.

    Emits result(AugmentationResult) on success.
    """

    def __init__(
        self,
        session_id: str,
        sm: SessionManager,
        config: AugmentationConfig,
        annotations: list,
        output_dir: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._session_id = session_id
        self._sm = sm
        self._config = config
        self._annotations = annotations
        self._output_dir = output_dir

    def run(self) -> None:
        from core.augmentor import augment_dataset
        try:
            ext_dir = self._sm.session_folder(self._session_id)
            frame_paths = [a.frame_path for a in self._annotations]
            if not frame_paths:
                self._emit_error(
                    "No annotated frames found. Run Auto-Label first."
                )
                return

            out_dir = self._output_dir or str(ext_dir / "augmented_dataset")
            result = augment_dataset(
                frame_paths, self._annotations,
                self._config, out_dir,
                progress_callback=self._emit_progress,
                status_callback=self._emit_status,
            )
            self._emit_result(result)
        except Exception as exc:
            self._emit_error(str(exc))


class TrainingWorker(BaseWorker):
    """Launch YOLOv8 training subprocess and stream per-epoch metrics.

    Emits result(TrainingRun) on completion.
    Extra signal: epoch_metric(object) — emits EpochMetrics after each epoch.
    Can be cancelled via cancel().
    """

    epoch_metric = pyqtSignal(object)  # EpochMetrics

    def __init__(self, config: TrainingConfig, parent=None) -> None:
        super().__init__(parent)
        self._config = config

    def run(self) -> None:
        from core.trainer import run_training
        try:
            def on_epoch(em: Any) -> None:
                self.epoch_metric.emit(em)
                # Approximate progress from epoch number
                total = self._config.epochs
                self._emit_progress(int(em.epoch / total * 100))
                self._emit_status(
                    f"Epoch {em.epoch}/{total} — "
                    f"box={em.box_loss:.4f} cls={em.cls_loss:.4f} "
                    f"mAP50={em.map50:.4f}"
                )

            result = run_training(
                self._config,
                status_callback=self._emit_status,
                epoch_callback=on_epoch,
            )
            self._emit_progress(100)
            self._emit_result(result)
        except Exception as exc:
            self._emit_error(str(exc))

    def cancel(self) -> None:
        """Send SIGTERM to the active training subprocess."""
        from core.trainer import cancel_training
        cancel_training()


# ---------------------------------------------------------------------------
# Phase 3 workers
# ---------------------------------------------------------------------------

class InferenceWorker(BaseWorker):
    """Run single-image or batch inference using the active (or specified) model."""

    def __init__(
        self,
        requests: list[InferenceRequest],
        registry_path: str | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._requests = requests
        self._registry_path = registry_path

    def run(self) -> None:
        from pathlib import Path
        from core.inference_server import run_inference_batch, REGISTRY_FILE
        reg = Path(self._registry_path) if self._registry_path else REGISTRY_FILE
        try:
            self._emit_status(f"Running inference on {len(self._requests)} image(s)…")
            results = run_inference_batch(
                self._requests,
                registry_path=reg,
                progress_callback=self._emit_progress,
            )
            self._emit_progress(100)
            self._emit_result(results)
        except Exception as exc:
            self._emit_error(str(exc))


class AnnotationReviewWorker(BaseWorker):
    """Export corrected dataset after a review session completes."""

    def __init__(
        self,
        session_id: str,
        sm,
        output_dir: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._session_id = session_id
        self._sm = sm
        self._output_dir = output_dir

    def run(self) -> None:
        from core.annotation_review import export_corrected_dataset
        try:
            self._emit_status("Exporting corrected dataset…")
            result = export_corrected_dataset(
                session_id=self._session_id,
                sm=self._sm,
                output_dir=self._output_dir,
                status_callback=self._emit_status,
            )
            self._emit_progress(100)
            self._emit_result(result)
        except Exception as exc:
            self._emit_error(str(exc))


class ContinuousLearningWorker(BaseWorker):
    """Full retrain-compare-promote cycle triggered by accumulated corrections."""

    epoch_metric = pyqtSignal(object)

    def __init__(
        self,
        session_id: str,
        sm,
        aug_config: AugmentationConfig,
        train_config: TrainingConfig,
        auto_promote: bool = True,
        trigger_type: str = "manual",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._session_id = session_id
        self._sm = sm
        self._aug_config = aug_config
        self._train_config = train_config
        self._auto_promote = auto_promote
        self._trigger_type = trigger_type

    def run(self) -> None:
        from core.continuous_learning import run_learning_cycle
        try:
            train_result, comparison = run_learning_cycle(
                session_id=self._session_id,
                sm=self._sm,
                trigger_type=self._trigger_type,
                augmentation_config=self._aug_config,
                training_config=self._train_config,
                auto_promote=self._auto_promote,
                progress_callback=self._emit_progress,
                status_callback=self._emit_status,
            )
            self._emit_progress(100)
            self._emit_result({"training": train_result, "comparison": comparison})
        except Exception as exc:
            self._emit_error(str(exc))


class Insta360Worker(BaseWorker):
    """QThread worker for the Insta360 X4 dual-fisheye → 4-perspective pipeline."""

    def __init__(self, insv_pair, config, output_dir: str, session_id: str, parent=None) -> None:
        super().__init__(parent)
        self._insv_pair = insv_pair
        self._config = config
        self._output_dir = output_dir
        self._session_id = session_id

    def run(self) -> None:
        from core.insta360_processor import run_full_insta360_pipeline

        def progress_cb(stage: str, pct: int) -> None:
            self._emit_status(stage)
            self._emit_progress(pct)

        try:
            result = run_full_insta360_pipeline(
                insv_pair=self._insv_pair,
                config=self._config,
                output_dir=self._output_dir,
                session_id=self._session_id,
                progress_callback=progress_cb,
            )
            self._emit_progress(100)
            self._emit_result(result)
        except Exception as exc:
            self._emit_error(str(exc))


class SensorLoggerWorker(BaseWorker):
    """Process a batch of Sensor Logger CSV files (audit + extraction)."""

    def __init__(self, csv_paths: list[str], output_dir: str, parent=None) -> None:
        super().__init__(parent)
        self._csv_paths = csv_paths
        self._output_dir = output_dir

    def run(self) -> None:
        from core.audit import audit_sensor_logger
        from core.extractor_sensorlogger import extract_sensor_logger
        from core.models import ExtractionConfig
        import uuid

        n = len(self._csv_paths)
        if not n:
            self._emit_error("No CSV files to process.")
            return

        audit_results, extracted, errors = [], [], []
        try:
            for i, csv_path in enumerate(self._csv_paths):
                name = Path(csv_path).name
                # Audit phase (0-50%)
                self._emit_status(f"Auditing {name} ({i+1}/{n})")
                self._emit_progress(int(i / n * 50))
                try:
                    ar = audit_sensor_logger(csv_path)
                    audit_results.append(ar)
                except Exception as exc:
                    errors.append({"file": csv_path, "stage": "audit", "error": str(exc)})
                    continue

                # Extract phase (50-95%)
                self._emit_status(f"Extracting {name} ({i+1}/{n})")
                self._emit_progress(int(50 + i / n * 45))
                try:
                    out = Path(self._output_dir) / Path(csv_path).stem
                    config = ExtractionConfig(session_id=str(uuid.uuid4()))
                    session = extract_sensor_logger(csv_path, config, str(out))
                    extracted.append(session)
                except Exception as exc:
                    errors.append({"file": csv_path, "stage": "extract", "error": str(exc)})

            self._emit_progress(100)
            self._emit_status(f"Done: {len(extracted)}/{n} files extracted")
            self._emit_result({
                "audit_results": audit_results,
                "extracted_sessions": extracted,
                "errors": errors,
            })
        except Exception as exc:
            self._emit_error(str(exc))
