"""
web/backend/tasks.py — Celery task definitions for long-running operations.

All tasks accept a task_id and push progress updates via Redis pub/sub
so the WebSocket handler can stream them to the client.
"""
from __future__ import annotations

import json
from pathlib import Path

import redis
from celery import Celery

from core.audit import audit_file
from core.calibration import run_calibration_step
from core.dataset_builder import build_dataset
from core.extractor_gopro import extract_gopro
from core.models import ExtractionConfig
from core.session_manager import SessionManager

REDIS_URL = "redis://localhost:6379/0"

celery_app = Celery("roverdatakit", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)

_redis_client: redis.Redis | None = None


def _redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(REDIS_URL)
    return _redis_client


def _push(task_id: str, status: str, progress: int, message: str) -> None:
    """Publish a progress event to the task's Redis channel."""
    payload = json.dumps({
        "task_id": task_id,
        "status": status,
        "progress": progress,
        "message": message,
    })
    try:
        _redis().publish(f"task:{task_id}", payload)
    except Exception:
        pass


def _sessions_root() -> Path:
    """Default sessions root — same as desktop default."""
    return Path.home() / ".roverdatakit" / "data" / "sessions"


# ---------------------------------------------------------------------------
# Audit task
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="tasks.run_audit")
def run_audit(self, session_id: str, task_id: str) -> dict:
    """Audit all files in a session and persist results."""
    sm = SessionManager(_sessions_root())
    session = sm.get_session(session_id)
    results = []
    total = len(session.files)

    _push(task_id, "running", 0, f"Auditing {total} file(s)…")

    for i, fp in enumerate(session.files):
        _push(task_id, "running", int(i / total * 90), f"Auditing {Path(fp).name}…")
        try:
            result = audit_file(fp)
            results.append(result)
        except Exception as exc:
            _push(task_id, "running", int(i / total * 90),
                  f"ERROR auditing {Path(fp).name}: {exc}")

    sm.set_audit_results(session_id, results)
    _push(task_id, "done", 100, f"Audit complete: {len(results)} file(s) processed.")
    return {"session_id": session_id, "file_count": len(results)}


# ---------------------------------------------------------------------------
# Extraction task
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="tasks.run_extraction")
def run_extraction(self, session_id: str, config_dict: dict, task_id: str) -> dict:
    """Extract telemetry and frames for all .mp4 files in a session."""
    sm = SessionManager(_sessions_root())
    session = sm.get_session(session_id)
    config = ExtractionConfig(**config_dict)

    mp4_files = [f for f in session.files if f.lower().endswith(".mp4")]
    if not mp4_files:
        _push(task_id, "failed", 0, "No .mp4 files found in session.")
        return {"error": "no mp4 files"}

    sm.set_extraction_status(session_id, "running")
    _push(task_id, "running", 0, f"Starting extraction of {len(mp4_files)} file(s)…")

    output_dir = sm.extraction_output_dir(session_id)

    try:
        total = len(mp4_files)
        for i, fp in enumerate(mp4_files):
            base_pct = int(i / total * 90)
            end_pct = int((i + 1) / total * 90)

            def progress_cb(pct: int, _b=base_pct, _e=end_pct) -> None:
                mapped = _b + int(pct / 100 * (_e - _b))
                _push(task_id, "running", mapped, f"Extracting {Path(fp).name}…")

            extract_gopro(fp, config, output_dir, progress_callback=progress_cb)

        sm.set_extraction_status(session_id, "done")
        _push(task_id, "done", 100, "Extraction complete.")
        return {"session_id": session_id, "output_dir": str(output_dir)}

    except Exception as exc:
        sm.set_extraction_status(session_id, "failed")
        _push(task_id, "failed", 0, f"Extraction failed: {exc}")
        raise


# ---------------------------------------------------------------------------
# Dataset build task
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="tasks.run_dataset_build")
def run_dataset_build(
    self,
    session_ids: list[str],
    output_format: str,
    output_dir: str,
    task_id: str,
) -> dict:
    """Build a dataset from extracted sessions."""
    sm = SessionManager(_sessions_root())

    def progress_cb(pct: int) -> None:
        _push(task_id, "running", pct, "")

    def status_cb(msg: str) -> None:
        _push(task_id, "running", -1, msg)

    _push(task_id, "running", 0, f"Building {output_format.upper()} dataset…")

    try:
        result = build_dataset(
            session_ids=session_ids,
            session_manager=sm,
            output_format=output_format,
            output_dir=Path(output_dir),
            progress_callback=progress_cb,
            status_callback=status_cb,
        )
        _push(task_id, "done", 100,
              f"Done: {result.total_frames} frames, {result.total_imu_samples} IMU samples.")
        return {
            "total_frames": result.total_frames,
            "total_imu_samples": result.total_imu_samples,
            "manifest_path": str(result.manifest_path),
            "warnings": result.warnings,
        }
    except Exception as exc:
        _push(task_id, "failed", 0, f"Build failed: {exc}")
        raise


# ---------------------------------------------------------------------------
# Calibration step task
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="tasks.run_calibration_step")
def run_calibration_step_task(
    self,
    cal_session_id: str,
    step: str,
    file_path: str,
    extra: dict,
    task_id: str,
) -> dict:
    """Run a single calibration step and stream stdout to Redis."""

    def progress_cb(pct: int) -> None:
        _push(task_id, "running", pct, "")

    def status_cb(msg: str) -> None:
        _push(task_id, "running", -1, msg)

    _push(task_id, "running", 0, f"Starting calibration step: {step}…")

    try:
        result = run_calibration_step(
            step=step,
            file_path=file_path,
            extra=extra,
            progress_callback=progress_cb,
            status_callback=status_cb,
        )
        _push(task_id, "done", 100, f"Step {step} complete.")
        return {"step": step, "result": result}
    except Exception as exc:
        _push(task_id, "failed", 0, f"Step {step} failed: {exc}")
        raise


# ---------------------------------------------------------------------------
# Auto-label task
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="tasks.run_autolabel")
def run_autolabel(self, session_id: str, config_dict: dict, task_id: str) -> dict:
    """Run YOLOv8 inference on all extracted frames in a session."""
    sm = SessionManager(_sessions_root())

    def progress_cb(pct: int) -> None:
        _push(task_id, "running", pct, "")

    def status_cb(msg: str) -> None:
        _push(task_id, "running", -1, msg)

    _push(task_id, "running", 0, "Starting auto-labeling…")
    try:
        from core.autolabel import load_model, run_inference_batch, export_cvat_xml, export_yolo_format, compute_annotation_stats, ROVER_CLASSES

        frames_dir = sm.session_folder(session_id) / "cam0" / "data"
        if not frames_dir.exists():
            frames_dir = sm.session_folder(session_id) / "frames"
        if not frames_dir.exists():
            _push(task_id, "failed", 0, "No extracted frames found. Run extraction first.")
            return {"error": "no frames"}

        frame_paths = sorted(str(p) for p in frames_dir.glob("*.jpg"))
        frame_paths += sorted(str(p) for p in frames_dir.glob("*.png"))
        if not frame_paths:
            _push(task_id, "failed", 0, "No frame files found in session.")
            return {"error": "no frames"}

        _push(task_id, "running", 5, f"Found {len(frame_paths)} frames. Loading model…")
        model = load_model(config_dict["model_path"], config_dict["device"])
        _push(task_id, "running", 10, "Model loaded. Running inference…")

        annotations = run_inference_batch(
            frame_paths=frame_paths,
            model=model,
            conf_threshold=config_dict["conf_threshold"],
            iou_threshold=config_dict["iou_threshold"],
            batch_size=config_dict["batch_size"],
            progress_callback=progress_cb,
            status_callback=status_cb,
        )

        output_dir = sm.session_folder(session_id) / "autolabel"
        output_dir.mkdir(parents=True, exist_ok=True)

        fmt = config_dict.get("export_format", "both")
        if fmt in ("cvat", "both"):
            export_cvat_xml(annotations, str(output_dir / "annotations.xml"),
                            task_name=session_id, labels=ROVER_CLASSES)
        if fmt in ("yolo", "both"):
            export_yolo_format(annotations, str(output_dir / "yolo"), ROVER_CLASSES)

        stats = compute_annotation_stats(annotations)
        (output_dir / "annotation_stats.json").write_text(json.dumps(stats, indent=2))

        _push(task_id, "done", 100,
              f"Auto-label done: {stats['total_frames']} frames, "
              f"{stats['total_detections']} detections.")
        return {"session_id": session_id, "total_frames": stats["total_frames"]}
    except Exception as exc:
        _push(task_id, "failed", 0, f"Auto-label failed: {exc}")
        raise


# ---------------------------------------------------------------------------
# Depth estimation task
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="tasks.run_depth_estimation")
def run_depth_estimation(self, session_id: str, config_dict: dict, task_id: str) -> dict:
    """Run Depth-Anything-v2 on all session frames."""
    sm = SessionManager(_sessions_root())

    def progress_cb(pct: int) -> None:
        _push(task_id, "running", pct, "")

    def status_cb(msg: str) -> None:
        _push(task_id, "running", -1, msg)

    _push(task_id, "running", 0, "Starting depth estimation…")
    try:
        from core.depth_estimator import load_depth_model, estimate_depth_batch, MODEL_VARIANTS

        frames_dir = sm.session_folder(session_id) / "cam0" / "data"
        if not frames_dir.exists():
            frames_dir = sm.session_folder(session_id) / "frames"
        if not frames_dir.exists():
            _push(task_id, "failed", 0, "No extracted frames found.")
            return {"error": "no frames"}

        frame_paths = sorted(str(p) for p in frames_dir.glob("*.jpg"))
        frame_paths += sorted(str(p) for p in frames_dir.glob("*.png"))

        variant = config_dict.get("model_variant", "small")
        model_name = MODEL_VARIANTS.get(variant, MODEL_VARIANTS["small"])
        _push(task_id, "running", 5, f"Loading depth model ({variant})…")
        model, processor = load_depth_model(model_name, config_dict["device"])
        _push(task_id, "running", 10, f"Model loaded. Processing {len(frame_paths)} frames…")

        output_dir = sm.session_folder(session_id) / "depth"
        output_dir.mkdir(parents=True, exist_ok=True)

        results = estimate_depth_batch(
            image_paths=frame_paths,
            model=model,
            processor=processor,
            batch_size=config_dict["batch_size"],
            output_dir=str(output_dir),
            colorize=config_dict["colorize"],
            progress_callback=progress_cb,
            status_callback=status_cb,
        )

        summary = {
            "frame_count": len(results),
            "mean_depth_avg": sum(r.mean_depth for r in results) / max(len(results), 1),
            "is_metric": False,
        }
        (output_dir / "depth_summary.json").write_text(json.dumps(summary, indent=2))

        _push(task_id, "done", 100, f"Depth estimation complete: {len(results)} frames.")
        return {"session_id": session_id, "frame_count": len(results)}
    except Exception as exc:
        _push(task_id, "failed", 0, f"Depth estimation failed: {exc}")
        raise


# ---------------------------------------------------------------------------
# SLAM validation task
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="tasks.run_slam_validation")
def run_slam_validation(self, session_id: str, config_dict: dict, task_id: str) -> dict:
    """Run ORBSLAM3 on an EuRoC-format session directory."""
    sm = SessionManager(_sessions_root())

    def progress_cb(pct: int) -> None:
        _push(task_id, "running", pct, "")

    def status_cb(msg: str) -> None:
        _push(task_id, "running", -1, msg)

    _push(task_id, "running", 0, "Starting SLAM validation…")
    try:
        from core.slam_validator import run_orbslam3

        euroc_dir = str(sm.session_folder(session_id))
        slam_dir = sm.session_folder(session_id) / "slam"
        slam_dir.mkdir(parents=True, exist_ok=True)

        result = run_orbslam3(
            euroc_session_dir=euroc_dir,
            vocabulary_path=config_dict["vocabulary_path"],
            config_yaml_path=config_dict["config_yaml_path"],
            mode=config_dict["mode"],
            progress_callback=progress_cb,
            status_callback=status_cb,
        )

        result_dict = result.model_dump()
        (slam_dir / "slam_result.json").write_text(json.dumps(result_dict, indent=2))

        msg = (f"SLAM done: {result.metrics.total_distance_m:.1f}m, "
               f"{result.metrics.keyframe_count} keyframes, "
               f"{'success' if result.success else 'FAILED'}")
        _push(task_id, "done" if result.success else "failed", 100, msg)
        return {"session_id": session_id, "success": result.success}
    except Exception as exc:
        _push(task_id, "failed", 0, f"SLAM failed: {exc}")
        raise


# ---------------------------------------------------------------------------
# 3D Reconstruction task
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="tasks.run_reconstruction")
def run_reconstruction(self, session_id: str, config_dict: dict, task_id: str) -> dict:
    """Run COLMAP SfM on sampled session frames."""
    sm = SessionManager(_sessions_root())

    def progress_cb(pct: int) -> None:
        _push(task_id, "running", pct, "")

    def status_cb(msg: str) -> None:
        _push(task_id, "running", -1, msg)

    _push(task_id, "running", 0, "Starting 3D reconstruction…")
    try:
        import shutil, tempfile
        from core.reconstructor import run_colmap_sfm, export_point_cloud_ply

        frames_dir = sm.session_folder(session_id) / "cam0" / "data"
        if not frames_dir.exists():
            frames_dir = sm.session_folder(session_id) / "frames"
        if not frames_dir.exists():
            _push(task_id, "failed", 0, "No extracted frames found.")
            return {"error": "no frames"}

        every_nth = config_dict.get("every_nth", 6)
        all_frames = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
        sampled = all_frames[::every_nth]
        _push(task_id, "running", 5,
              f"Sampled {len(sampled)} / {len(all_frames)} frames (every {every_nth}th).")

        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = Path(tmpdir) / "images"
            img_dir.mkdir()
            for f in sampled:
                shutil.copy(f, img_dir / f.name)

            out_dir = sm.session_folder(session_id) / "reconstruction"
            out_dir.mkdir(parents=True, exist_ok=True)

            result = run_colmap_sfm(
                image_dir=str(img_dir),
                output_dir=str(out_dir),
                camera_model=config_dict.get("camera_model", "OPENCV_FISHEYE"),
                use_gpu=config_dict.get("use_gpu", True),
                max_image_size=config_dict.get("max_image_size", 1600),
                progress_callback=progress_cb,
                status_callback=status_cb,
            )

        ply_path = export_point_cloud_ply(result, str(out_dir / "point_cloud.ply"))
        result_dict = result.model_dump()
        result_dict["ply_path"] = ply_path
        (out_dir / "colmap_result.json").write_text(json.dumps(result_dict, indent=2))

        coverage = result.num_images_registered / max(result.num_images_total, 1) * 100
        _push(task_id, "done", 100,
              f"Reconstruction done: {result.num_points3d} points, "
              f"{coverage:.1f}% coverage, "
              f"reprojection error {result.mean_reprojection_error:.3f}px")
        return {"session_id": session_id, "num_points3d": result.num_points3d}
    except Exception as exc:
        _push(task_id, "failed", 0, f"Reconstruction failed: {exc}")
        raise


# ---------------------------------------------------------------------------
# Active learning task
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="tasks.run_active_learning")
def run_active_learning(self, session_id: str, config_dict: dict, task_id: str) -> dict:
    """Score frames by uncertainty + diversity for labeling prioritization."""
    sm = SessionManager(_sessions_root())

    _push(task_id, "running", 0, "Loading annotations…")
    try:
        from core.active_learning import (
            compute_uncertainty_scores, compute_diversity_scores,
            select_frames_for_labeling, compute_label_budget_estimate,
        )
        from core.models import FrameAnnotation

        autolabel_dir = sm.session_folder(session_id) / "autolabel"
        ann_files = list((autolabel_dir / "yolo").glob("*.json")) if (autolabel_dir / "yolo").exists() else []

        annotations: list[FrameAnnotation] = []
        for f in ann_files:
            try:
                annotations.append(FrameAnnotation.model_validate_json(f.read_text()))
            except Exception:
                pass

        if not annotations:
            _push(task_id, "failed", 0, "No annotations found. Run auto-label first.")
            return {"error": "no annotations"}

        _push(task_id, "running", 20, f"Scoring {len(annotations)} frames…")
        unc_scores = compute_uncertainty_scores(annotations, method=config_dict["method"])
        _push(task_id, "running", 50, "Computing diversity scores…")
        frame_paths = [a.frame_path for a in annotations]
        div_scores = compute_diversity_scores(frame_paths)
        _push(task_id, "running", 75, "Selecting frames…")
        selected = select_frames_for_labeling(
            uncertainty_scores=unc_scores,
            diversity_scores=div_scores,
            n_frames=config_dict["n_frames"],
            uncertainty_weight=config_dict["uncertainty_weight"],
            diversity_weight=config_dict["diversity_weight"],
        )
        budget = compute_label_budget_estimate(len(annotations), selected)

        out_dir = sm.session_folder(session_id) / "active_learning"
        out_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "selected_frames": selected,
            "budget": budget,
            "total_frames": len(annotations),
            "selected_count": len(selected),
        }
        (out_dir / "selection_result.json").write_text(json.dumps(result, indent=2))

        _push(task_id, "done", 100,
              f"Selected {len(selected)} / {len(annotations)} frames for labeling.")
        return {"session_id": session_id, "selected_count": len(selected)}
    except Exception as exc:
        _push(task_id, "failed", 0, f"Active learning failed: {exc}")
        raise


# ---------------------------------------------------------------------------
# Analytics task
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="tasks.run_analytics")
def run_analytics(self, session_id: str, config_dict: dict, task_id: str) -> dict:
    """Compute scene diversity and geographic coverage analytics."""
    sm = SessionManager(_sessions_root())

    def status_cb(msg: str) -> None:
        _push(task_id, "running", -1, msg)

    _push(task_id, "running", 0, "Starting analytics…")
    try:
        from core.road_analytics import (
            compute_scene_diversity, compute_geographic_coverage, compute_class_distribution
        )

        frames_dir = sm.session_folder(session_id) / "cam0" / "data"
        if not frames_dir.exists():
            frames_dir = sm.session_folder(session_id) / "frames"

        frame_paths = []
        if frames_dir.exists():
            frame_paths = [str(p) for p in sorted(frames_dir.glob("*.jpg"))]
            frame_paths += [str(p) for p in sorted(frames_dir.glob("*.png"))]

        out_dir = sm.session_folder(session_id) / "analytics"
        out_dir.mkdir(parents=True, exist_ok=True)

        result: dict = {}

        if frame_paths:
            _push(task_id, "running", 20, f"Computing scene diversity ({len(frame_paths)} frames)…")
            diversity = compute_scene_diversity(
                frame_paths, sample_n=config_dict.get("sample_n", 500)
            )
            result["diversity"] = diversity.model_dump()

        # GPS coverage
        gps_path = sm.session_folder(session_id) / "gps.csv"
        if gps_path.exists():
            _push(task_id, "running", 60, "Computing geographic coverage…")
            import csv
            from core.models import GPSSample
            gps_samples = []
            with open(gps_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        gps_samples.append(GPSSample(
                            timestamp_ns=int(row["timestamp_ns"]),
                            latitude=float(row["latitude"]),
                            longitude=float(row["longitude"]),
                            altitude_m=float(row.get("altitude_m", 0)),
                            speed_mps=float(row.get("speed_mps", 0)),
                            fix_type=int(row.get("fix_type", 3)),
                        ))
                    except Exception:
                        pass
            if gps_samples:
                coverage = compute_geographic_coverage(gps_samples, str(out_dir))
                result["coverage"] = coverage.model_dump()

        (out_dir / "analytics_result.json").write_text(json.dumps(result, indent=2, default=str))
        _push(task_id, "done", 100, "Analytics complete.")
        return {"session_id": session_id}
    except Exception as exc:
        _push(task_id, "failed", 0, f"Analytics failed: {exc}")
        raise


# ---------------------------------------------------------------------------
# Augmentation task
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="tasks.run_augmentation")
def run_augmentation(self, session_id: str, config_dict: dict, task_id: str) -> dict:
    """Augment annotated frames using albumentations pipeline."""
    sm = SessionManager(_sessions_root())

    def progress_cb(pct: int) -> None:
        _push(task_id, "running", pct, "")

    def status_cb(msg: str) -> None:
        _push(task_id, "running", -1, msg)

    _push(task_id, "running", 0, "Loading annotations for augmentation…")
    try:
        from core.augmentor import augment_dataset
        from core.models import AugmentationConfig, FrameAnnotation

        autolabel_dir = sm.session_folder(session_id) / "autolabel"
        yolo_dir = autolabel_dir / "yolo"

        annotations: list[FrameAnnotation] = []
        if yolo_dir.exists():
            for f in yolo_dir.glob("*.json"):
                try:
                    annotations.append(FrameAnnotation.model_validate_json(f.read_text()))
                except Exception:
                    pass

        if not annotations:
            _push(task_id, "failed", 0, "No annotations found. Run auto-label first.")
            return {"error": "no annotations"}

        config = AugmentationConfig(**{
            k: v for k, v in config_dict.items() if k != "multiplier"
        }, multiplier=config_dict.get("multiplier", 3))

        frame_paths = [a.frame_path for a in annotations]
        out_dir = str(sm.session_folder(session_id) / "augmented_dataset")

        _push(task_id, "running", 10,
              f"Augmenting {len(frame_paths)} frames × {config.multiplier}…")
        result = augment_dataset(
            frame_paths=frame_paths,
            annotations=annotations,
            config=config,
            output_dir=out_dir,
            multiplier=config.multiplier,
            progress_callback=progress_cb,
            status_callback=status_cb,
        )

        result_dict = result.model_dump()
        (Path(out_dir) / "augmentation_result.json").write_text(
            json.dumps(result_dict, indent=2)
        )
        _push(task_id, "done", 100,
              f"Augmentation done: {result.original_count} → {result.augmented_count} images.")
        return {"session_id": session_id, "augmented_count": result.augmented_count}
    except Exception as exc:
        _push(task_id, "failed", 0, f"Augmentation failed: {exc}")
        raise


# ---------------------------------------------------------------------------
# Training task
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="tasks.run_training")
def run_training(self, config_dict: dict, task_id: str) -> dict:
    """Run YOLOv8 fine-tuning with per-epoch metric streaming."""

    def progress_cb(pct: int) -> None:
        _push(task_id, "running", pct, "")

    def status_cb(msg: str) -> None:
        _push(task_id, "running", -1, msg)

    def epoch_cb(em) -> None:
        _push(task_id, "running", -1, json.dumps({
            "type": "epoch",
            "epoch": em.epoch,
            "box_loss": em.box_loss,
            "cls_loss": em.cls_loss,
            "dfl_loss": em.dfl_loss,
            "map50": em.map50,
            "map50_95": em.map50_95,
        }))

    _push(task_id, "running", 0, "Preparing training configuration…")
    try:
        from core.trainer import prepare_training_config, run_training as core_train

        config = prepare_training_config(**config_dict)
        _push(task_id, "running", 5, f"Training {config.model_variant} for {config.epochs} epochs…")

        result = core_train(
            config=config,
            progress_callback=progress_cb,
            status_callback=status_cb,
            epoch_callback=epoch_cb,
        )

        _push(task_id, "done", 100,
              f"Training done: best mAP50={result.final_map50:.4f} "
              f"@ epoch {result.best_epoch}, "
              f"weights: {result.best_weights_path}")
        return {
            "best_weights_path": result.best_weights_path,
            "final_map50": result.final_map50,
            "best_epoch": result.best_epoch,
        }
    except Exception as exc:
        _push(task_id, "failed", 0, f"Training failed: {exc}")
        raise


# ---------------------------------------------------------------------------
# Phase 3: Inference batch task
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="tasks.run_inference_batch")
def run_inference_batch_task(self, requests_dicts: list[dict], task_id: str) -> dict:
    """Run batch inference using the active model."""
    from core.inference_server import REGISTRY_FILE, run_inference_batch
    from core.models import InferenceRequest

    def progress_cb(pct: int) -> None:
        _push(task_id, "running", pct, "")

    _push(task_id, "running", 0, f"Running inference on {len(requests_dicts)} image(s)…")
    try:
        requests = [InferenceRequest(**r) for r in requests_dicts]
        results = run_inference_batch(requests, REGISTRY_FILE, progress_callback=progress_cb)
        _push(task_id, "done", 100, f"Inference complete: {len(results)} result(s).")
        return {"count": len(results), "results": [r.model_dump() for r in results]}
    except Exception as exc:
        _push(task_id, "failed", 0, f"Inference failed: {exc}")
        raise


# ---------------------------------------------------------------------------
# Phase 3: Export corrected dataset task
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="tasks.run_export_corrected")
def run_export_corrected(self, session_id: str, output_dir: str, task_id: str) -> dict:
    """Export accepted + corrected frames to YOLO dataset format."""
    import os
    from pathlib import Path
    _sessions_root = Path(os.environ.get("ROVERDATAKIT_DATA", Path.home() / ".roverdatakit" / "data")) / "sessions"
    _sessions_root.mkdir(parents=True, exist_ok=True)
    sm = SessionManager(_sessions_root)

    def status_cb(msg: str) -> None:
        _push(task_id, "running", -1, msg)

    _push(task_id, "running", 0, "Exporting corrected dataset…")
    try:
        from core.annotation_review import export_corrected_dataset
        result = export_corrected_dataset(session_id, sm, output_dir, status_callback=status_cb)
        _push(task_id, "done", 100,
              f"Export complete: {result.augmented_count} frames → {result.output_dir}")
        return {"output_dir": result.output_dir, "frame_count": result.augmented_count}
    except Exception as exc:
        _push(task_id, "failed", 0, f"Export failed: {exc}")
        raise


# ---------------------------------------------------------------------------
# Phase 3: Full continuous learning cycle task
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="tasks.run_learning_cycle")
def run_learning_cycle_task(
    self,
    task_id: str,
    session_id: str,
    trigger_type: str,
    aug_config: dict,
    train_config: dict,
    auto_promote: bool = True,
) -> dict:
    """Full retrain-compare-promote cycle."""
    import os
    from pathlib import Path
    _sessions_root = Path(os.environ.get("ROVERDATAKIT_DATA", Path.home() / ".roverdatakit" / "data")) / "sessions"
    _sessions_root.mkdir(parents=True, exist_ok=True)
    sm = SessionManager(_sessions_root)

    def progress_cb(pct: int) -> None:
        _push(task_id, "running", pct, "")

    def status_cb(msg: str) -> None:
        _push(task_id, "running", -1, msg)

    _push(task_id, "running", 0, "Starting continuous learning cycle…")
    try:
        from core.continuous_learning import run_learning_cycle
        from core.models import AugmentationConfig, TrainingConfig

        aug = AugmentationConfig(**aug_config)
        train = TrainingConfig(**train_config)

        train_result, comparison = run_learning_cycle(
            session_id=session_id,
            sm=sm,
            trigger_type=trigger_type,
            augmentation_config=aug,
            training_config=train,
            auto_promote=auto_promote,
            progress_callback=progress_cb,
            status_callback=status_cb,
        )

        _push(task_id, "done", 100,
              f"Learning cycle complete. mAP50: "
              f"{comparison.baseline_map50:.4f} → {comparison.candidate_map50:.4f} "
              f"(Δ{comparison.delta_map50:+.4f})")
        return {
            "run_id": train_result.run_id,
            "final_map50": train_result.final_map50,
            "delta_map50": comparison.delta_map50,
            "promoted": comparison.improved and auto_promote,
        }
    except Exception as exc:
        _push(task_id, "failed", 0, f"Learning cycle failed: {exc}")
        raise


# ---------------------------------------------------------------------------
# Phase 3 perception: Segmentation task
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="tasks.run_segmentation")
def run_segmentation(self, session_id: str, config_dict: dict, task_id: str) -> dict:
    """Run SegFormer semantic segmentation on all session frames."""
    sm = SessionManager(_sessions_root())

    def progress_cb(pct: int) -> None:
        _push(task_id, "running", pct, "")

    _push(task_id, "running", 0, "Starting semantic segmentation…")
    try:
        from core.segmentation import load_segmentation_model, run_segmentation_batch

        frames_dir = sm.session_folder(session_id) / "cam0" / "data"
        if not frames_dir.exists():
            frames_dir = sm.session_folder(session_id) / "frames"
        if not frames_dir.exists():
            _push(task_id, "failed", 0, "No extracted frames found. Run extraction first.")
            return {"error": "no frames"}

        frame_paths = sorted(str(p) for p in frames_dir.glob("*.jpg"))
        frame_paths += sorted(str(p) for p in frames_dir.glob("*.png"))
        if not frame_paths:
            _push(task_id, "failed", 0, "No frame files found in session.")
            return {"error": "no frames"}

        _push(task_id, "running", 5, f"Loading segmentation model: {config_dict['model_name']}…")
        model, processor, id2label = load_segmentation_model(
            config_dict["model_name"], config_dict.get("device", "cpu")
        )
        _push(task_id, "running", 10, f"Model loaded. Processing {len(frame_paths)} frames…")

        output_dir = sm.session_folder(session_id) / "segmentation"
        output_dir.mkdir(parents=True, exist_ok=True)

        results = run_segmentation_batch(
            image_paths=frame_paths,
            model=model,
            processor=processor,
            id2label=id2label,
            batch_size=config_dict.get("batch_size", 4),
            output_dir=str(output_dir),
            overlay_alpha=config_dict.get("overlay_alpha", 0.6),
            progress_callback=progress_cb,
        )

        summary = {
            "frame_count": len(results),
            "mean_road_coverage": sum(r.road_area_percent for r in results) / max(len(results), 1),
        }
        (output_dir / "segmentation_summary.json").write_text(json.dumps(summary, indent=2))

        _push(task_id, "done", 100, f"Segmentation complete: {len(results)} frames.")
        return {"session_id": session_id, "frame_count": len(results)}
    except Exception as exc:
        _push(task_id, "failed", 0, f"Segmentation failed: {exc}")
        raise


# ---------------------------------------------------------------------------
# Phase 3 perception: Occupancy grid task
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="tasks.run_occupancy")
def run_occupancy(self, session_id: str, config_dict: dict, task_id: str) -> dict:
    """Build BEV occupancy grids from depth maps and detections."""
    sm = SessionManager(_sessions_root())

    def progress_cb(pct: int) -> None:
        _push(task_id, "running", pct, "")

    _push(task_id, "running", 0, "Starting occupancy grid generation…")
    try:
        from core.occupancy import run_occupancy_pipeline, OccupancyConfig
        from core.models import OccupancyConfig as OccModel

        depth_dir = sm.session_folder(session_id) / "depth"
        if not depth_dir.exists():
            _push(task_id, "failed", 0, "No depth maps found. Run depth estimation first.")
            return {"error": "no depth maps"}

        depth_paths = sorted(str(p) for p in depth_dir.glob("*.npy"))
        if not depth_paths:
            _push(task_id, "failed", 0, "No .npy depth files found in depth/ directory.")
            return {"error": "no depth npy files"}

        autolabel_dir = sm.session_folder(session_id) / "autolabel" / "yolo"
        annotations = []
        if autolabel_dir.exists():
            from core.models import FrameAnnotation
            for f in autolabel_dir.glob("*.json"):
                try:
                    annotations.append(FrameAnnotation.model_validate_json(f.read_text()))
                except Exception:
                    pass

        occ_config = OccModel(**{k: v for k, v in config_dict.items()
                                 if k in OccModel.model_fields})

        output_dir = sm.session_folder(session_id) / "occupancy"
        output_dir.mkdir(parents=True, exist_ok=True)

        results = run_occupancy_pipeline(
            depth_results=depth_paths,
            annotations=annotations,
            fx=config_dict.get("fx", 500.0),
            fy=config_dict.get("fy", 500.0),
            cx=config_dict.get("cx", 960.0),
            cy=config_dict.get("cy", 540.0),
            config=occ_config,
            output_dir=str(output_dir),
            progress_callback=progress_cb,
        )

        summary = {"frame_count": len(results)}
        (output_dir / "occupancy_summary.json").write_text(json.dumps(summary, indent=2))

        _push(task_id, "done", 100, f"Occupancy grids complete: {len(results)} frames.")
        return {"session_id": session_id, "frame_count": len(results)}
    except Exception as exc:
        _push(task_id, "failed", 0, f"Occupancy failed: {exc}")
        raise


# ---------------------------------------------------------------------------
# Phase 3 perception: Lane detection task
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="tasks.run_lane_detection")
def run_lane_detection(self, session_id: str, config_dict: dict, task_id: str) -> dict:
    """Detect lanes in all session frames (UFLD model or classical CV fallback)."""
    sm = SessionManager(_sessions_root())

    def progress_cb(pct: int) -> None:
        _push(task_id, "running", pct, "")

    _push(task_id, "running", 0, "Starting lane detection…")
    try:
        from core.lane_detector import load_lane_model, run_lane_pipeline, generate_lane_departure_report
        from core.models import LaneConfig

        frames_dir = sm.session_folder(session_id) / "cam0" / "data"
        if not frames_dir.exists():
            frames_dir = sm.session_folder(session_id) / "frames"
        if not frames_dir.exists():
            _push(task_id, "failed", 0, "No extracted frames found. Run extraction first.")
            return {"error": "no frames"}

        frame_paths = sorted(str(p) for p in frames_dir.glob("*.jpg"))
        frame_paths += sorted(str(p) for p in frames_dir.glob("*.png"))
        if not frame_paths:
            _push(task_id, "failed", 0, "No frame files found in session.")
            return {"error": "no frames"}

        model_path = config_dict.get("model_path")
        device = config_dict.get("device", "cpu")
        _push(task_id, "running", 5, "Loading lane model…")
        model = load_lane_model(model_path, device)
        method = "classical" if model is None else "ufld"
        _push(task_id, "running", 10,
              f"Using {method} detection on {len(frame_paths)} frames…")

        lane_config = LaneConfig(**{k: v for k, v in config_dict.items()
                                    if k in LaneConfig.model_fields})

        output_dir = sm.session_folder(session_id) / "lanes"
        output_dir.mkdir(parents=True, exist_ok=True)

        results = run_lane_pipeline(
            frame_paths=frame_paths,
            model=model,
            config=lane_config,
            output_dir=str(output_dir),
            progress_callback=progress_cb,
        )

        report = generate_lane_departure_report(results)
        (output_dir / "lane_report.json").write_text(json.dumps(report, indent=2))

        _push(task_id, "done", 100,
              f"Lane detection complete: {len(results)} frames, "
              f"departure rate {report.get('departure_rate_percent', 0):.1f}%.")
        return {"session_id": session_id, "frame_count": len(results), "report": report}
    except Exception as exc:
        _push(task_id, "failed", 0, f"Lane detection failed: {exc}")
        raise


# ---------------------------------------------------------------------------
# Phase 3 perception: Multi-object tracking task
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="tasks.run_tracking")
def run_tracking(self, session_id: str, config_dict: dict, task_id: str) -> dict:
    """Run ByteTrack multi-object tracking on annotated session frames."""
    sm = SessionManager(_sessions_root())

    def progress_cb(pct: int) -> None:
        _push(task_id, "running", pct, "")

    _push(task_id, "running", 0, "Starting multi-object tracking…")
    try:
        from core.tracker import build_tracker, run_tracking_session, compute_tracking_statistics

        autolabel_dir = sm.session_folder(session_id) / "autolabel" / "yolo"
        if not autolabel_dir.exists():
            _push(task_id, "failed", 0, "No annotations found. Run auto-label first.")
            return {"error": "no annotations"}

        from core.models import FrameAnnotation
        annotations = []
        for f in sorted(autolabel_dir.glob("*.json")):
            try:
                annotations.append(FrameAnnotation.model_validate_json(f.read_text()))
            except Exception:
                pass

        if not annotations:
            _push(task_id, "failed", 0, "No valid annotation files found.")
            return {"error": "no annotations"}

        frame_paths = [a.frame_path for a in annotations]
        _push(task_id, "running", 5,
              f"Tracking across {len(frame_paths)} frames…")

        tracker = build_tracker(
            track_thresh=config_dict.get("track_thresh", 0.5),
            track_buffer=config_dict.get("track_buffer", 30),
            match_thresh=config_dict.get("match_thresh", 0.8),
            frame_rate=config_dict.get("frame_rate", 10),
        )

        output_dir = sm.session_folder(session_id) / "tracking"
        output_dir.mkdir(parents=True, exist_ok=True)

        result = run_tracking_session(
            frame_paths=frame_paths,
            annotations=annotations,
            tracker=tracker,
            output_dir=str(output_dir),
            progress_callback=progress_cb,
        )

        stats = compute_tracking_statistics(result, frame_rate=config_dict.get("frame_rate", 10))
        (output_dir / "tracking_stats.json").write_text(
            json.dumps(stats.model_dump(), indent=2)
        )

        _push(task_id, "done", 100,
              f"Tracking complete: {stats.total_unique_tracks} unique tracks, "
              f"{stats.mean_track_length:.1f} mean length.")
        return {
            "session_id": session_id,
            "total_unique_tracks": stats.total_unique_tracks,
            "mean_track_length": stats.mean_track_length,
        }
    except Exception as exc:
        _push(task_id, "failed", 0, f"Tracking failed: {exc}")
        raise


# ---------------------------------------------------------------------------
# Insta360 X4 full 360° pipeline task
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="tasks.run_insta360_pipeline")
def run_insta360_pipeline(
    self,
    pair_dict: dict,
    config_dict: dict,
    output_dir: str,
    session_id: str,
    task_id: str,
) -> dict:
    """Run the full Insta360 X4 360° processing pipeline (stitch → perspective → frames → dataset)."""
    from core.insta360_processor import run_full_insta360_pipeline
    from core.models import INSVPair, Insta360ProcessingConfig

    pair = INSVPair(**pair_dict)
    config = Insta360ProcessingConfig(**config_dict)

    def progress_cb(stage: str, pct: int) -> None:
        _push(task_id, "running", pct, stage)

    _push(task_id, "running", 0, f"Starting Insta360 pipeline for {pair.base_name}…")
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        result = run_full_insta360_pipeline(
            insv_pair=pair,
            config=config,
            output_dir=output_dir,
            session_id=session_id,
            progress_callback=progress_cb,
        )
        _push(
            task_id, "done", 100,
            f"Pipeline complete: {result.total_frames_per_view} frames/view, "
            f"{result.gps_samples} GPS, {result.imu_samples} IMU samples, "
            f"{result.disk_usage_gb:.2f} GB"
        )
        return result.model_dump()
    except Exception as exc:
        _push(task_id, "failed", 0, f"Insta360 pipeline failed: {exc}")
        raise
