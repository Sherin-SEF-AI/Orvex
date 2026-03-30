"""
core/trainer.py — YOLOv8 model fine-tuning via subprocess.

Launches training as a subprocess (not via the ultralytics Python API directly)
so it can be cancelled via SIGTERM without killing the whole application.
Streams per-epoch metrics by parsing stdout and results.csv.

No UI imports — pure Python business logic.

Dependencies:
    pip install ultralytics
"""
from __future__ import annotations

import csv
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Callable

import yaml
from loguru import logger

from core.models import EpochMetrics, TrainingConfig, TrainingRun
from core import experiment_tracker as _mlflow

ProgressCB = Callable[[int], None]
StatusCB = Callable[[str], None]
MetricCB = Callable[[EpochMetrics], None]

# Regex to parse YOLO training stdout lines:
# e.g.  "      1/100      3.2G      1.234      2.345      0.678        12      640"
_EPOCH_RE = re.compile(
    r"^\s*(\d+)/(\d+)\s+[\d.]+G\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+\s+\d+"
)


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------

def prepare_training_config(
    dataset_dir: str,
    model_variant: str = "yolov8n",
    pretrained_weights: str = "yolov8n.pt",
    epochs: int = 100,
    batch_size: int = 16,
    image_size: int = 640,
    learning_rate: float = 0.01,
    device: str = "auto",
    project_name: str = "rover_detection",
    run_name: str = "run1",
) -> TrainingConfig:
    """Validate dataset structure and return a TrainingConfig.

    Args:
        dataset_dir: Root directory containing dataset.yaml, images/, labels/.
        ...other args map directly to TrainingConfig fields.

    Returns:
        Validated TrainingConfig.

    Raises:
        FileNotFoundError: if dataset_dir or expected subdirs are missing.
        ValueError: if dataset.yaml is malformed (missing nc or names).
    """
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}. "
            "Run augmentation or dataset builder first to create a YOLO dataset."
        )

    yaml_path = root / "dataset.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"dataset.yaml not found in {dataset_dir}. "
            "The directory must contain a valid YOLO dataset.yaml file."
        )

    with yaml_path.open() as f:
        ds_cfg = yaml.safe_load(f)

    if "nc" not in ds_cfg:
        raise ValueError(
            f"dataset.yaml is missing 'nc' (number of classes) field: {yaml_path}. "
            "Add 'nc: <number>' to the file."
        )
    if "names" not in ds_cfg:
        raise ValueError(
            f"dataset.yaml is missing 'names' field: {yaml_path}. "
            "Add 'names: [class1, class2, ...]' to the file."
        )

    train_img_dir = root / "images" / "train"
    if not train_img_dir.exists():
        raise FileNotFoundError(
            f"Training images directory not found: {train_img_dir}. "
            "Expected structure: {dataset_dir}/images/train/. "
            "Run augmentation to create this structure."
        )

    train_lbl_dir = root / "labels" / "train"
    if not train_lbl_dir.exists():
        raise FileNotFoundError(
            f"Training labels directory not found: {train_lbl_dir}. "
            "Expected structure: {dataset_dir}/labels/train/."
        )

    return TrainingConfig(
        dataset_dir=str(root.resolve()),
        model_variant=model_variant,
        pretrained_weights=pretrained_weights,
        epochs=epochs,
        batch_size=batch_size,
        image_size=image_size,
        learning_rate=learning_rate,
        device=device,
        project_name=project_name,
        run_name=run_name,
    )


# ---------------------------------------------------------------------------
# Training runner
# ---------------------------------------------------------------------------

def run_training(
    config: TrainingConfig,
    progress_callback: ProgressCB | None = None,
    status_callback: StatusCB | None = None,
    epoch_callback: MetricCB | None = None,
) -> TrainingRun:
    """Launch YOLOv8 training as a subprocess and stream metrics.

    The subprocess is stored in module state so cancel() can send SIGTERM.
    Per-epoch metrics are parsed from stdout in real time and from
    results.csv after training completes.

    Args:
        config:            Validated TrainingConfig from prepare_training_config().
        progress_callback: 0-100 based on epoch progress.
        status_callback:   Human-readable status strings.
        epoch_callback:    Called after each epoch with EpochMetrics.

    Returns:
        TrainingRun with full metrics and best weights path.

    Raises:
        RuntimeError: if training subprocess exits with non-zero code.
        FileNotFoundError: if ultralytics Python is not available.
    """
    _check_ultralytics()

    run_id = str(uuid.uuid4())[:8]
    dataset_yaml = str(Path(config.dataset_dir) / "dataset.yaml")
    device_arg = _resolve_device(config.device)

    # Auto-log to MLflow (no-op if mlflow not installed)
    _mlflow_run_id: str | None = None
    try:
        _mlflow_run_id = _mlflow.start_training_run(
            experiment_name=config.project_name,
            run_name=config.run_name,
            config={
                "model_variant": config.model_variant,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "image_size": config.image_size,
                "learning_rate": config.learning_rate,
                "device": device_arg,
            },
        )
    except Exception as _exc:
        logger.debug("trainer: MLflow start skipped: {}", _exc)

    cmd = [
        sys.executable, "-m", "ultralytics",
        "yolo", "detect", "train",
        f"data={dataset_yaml}",
        f"model={config.pretrained_weights}",
        f"epochs={config.epochs}",
        f"batch={config.batch_size}",
        f"imgsz={config.image_size}",
        f"lr0={config.learning_rate}",
        f"device={device_arg}",
        f"project={config.project_name}",
        f"name={config.run_name}",
        "exist_ok=True",
        "verbose=True",
    ]

    logger.info("trainer: starting training: {}", " ".join(cmd))
    t_start = time.monotonic()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    _ACTIVE_PROCESS[0] = proc

    live_metrics: list[EpochMetrics] = []
    epoch_times: list[float] = []
    last_epoch_start = time.monotonic()

    try:
        for line in proc.stdout:  # type: ignore[union-attr]
            line = line.rstrip()
            if status_callback:
                status_callback(line)

            m = _EPOCH_RE.match(line)
            if m:
                epoch = int(m.group(1))
                total_epochs = int(m.group(2))
                box_loss = float(m.group(3))
                cls_loss = float(m.group(4))
                dfl_loss = float(m.group(5))

                now = time.monotonic()
                epoch_time = now - last_epoch_start
                last_epoch_start = now
                epoch_times.append(epoch_time)

                em = EpochMetrics(
                    epoch=epoch,
                    box_loss=box_loss,
                    cls_loss=cls_loss,
                    dfl_loss=dfl_loss,
                    precision=0.0,   # filled from results.csv after training
                    recall=0.0,
                    map50=0.0,
                    map50_95=0.0,
                    lr=config.learning_rate,
                    epoch_time_seconds=epoch_time,
                )
                live_metrics.append(em)

                if epoch_callback:
                    epoch_callback(em)
                if progress_callback:
                    progress_callback(int(epoch / total_epochs * 100))

                # Log epoch to MLflow
                if _mlflow_run_id:
                    try:
                        _mlflow.log_epoch_metrics(_mlflow_run_id, {
                            "train/box_loss": box_loss,
                            "train/cls_loss": cls_loss,
                            "train/dfl_loss": dfl_loss,
                            "lr": config.learning_rate,
                        })
                    except Exception:
                        pass

        proc.wait()
    except Exception as exc:
        proc.kill()
        raise RuntimeError(
            f"Training subprocess error: {exc}. "
            "Check that ultralytics is properly installed."
        ) from exc
    finally:
        _ACTIVE_PROCESS[0] = None

    training_time = (time.monotonic() - t_start) / 60.0

    if proc.returncode not in (0, -signal.SIGTERM):
        raise RuntimeError(
            f"Training failed with exit code {proc.returncode}. "
            "Check the log above for details. "
            "Common causes: GPU OOM (reduce batch_size), invalid dataset.yaml."
        )

    # Locate output directory
    run_dir = Path(config.project_name) / config.run_name
    if not run_dir.exists():
        # Try current working dir fallback
        run_dir = Path.cwd() / config.project_name / config.run_name

    best_weights = str(run_dir / "weights" / "best.pt")
    epoch_metrics = _load_results_csv(run_dir / "results.csv", live_metrics)

    # Best epoch by mAP50
    best_epoch = 1
    best_map50 = 0.0
    final_map50_95 = 0.0
    for em in epoch_metrics:
        if em.map50 > best_map50:
            best_map50 = em.map50
            best_epoch = em.epoch
            final_map50_95 = em.map50_95

    status = "cancelled" if proc.returncode == -signal.SIGTERM else "done"

    logger.info(
        "trainer: training {} — best mAP50={:.4f} at epoch {}",
        status, best_map50, best_epoch,
    )

    # Finish MLflow run with final metrics
    if _mlflow_run_id:
        try:
            _mlflow.finish_training_run(
                _mlflow_run_id,
                result={
                    "val/mAP50": best_map50,
                    "val/mAP50-95": final_map50_95,
                    "best_epoch": best_epoch,
                    "training_time_minutes": round(training_time, 2),
                    "status": status,
                },
                weights_path=best_weights,
            )
        except Exception as _exc:
            logger.debug("trainer: MLflow finish skipped: {}", _exc)

    return TrainingRun(
        run_id=run_id,
        config=config,
        best_weights_path=best_weights,
        best_epoch=best_epoch,
        final_map50=best_map50,
        final_map50_95=final_map50_95,
        epoch_metrics=epoch_metrics,
        training_time_minutes=round(training_time, 2),
        status=status,
    )


# Active process reference for cancellation
_ACTIVE_PROCESS: list[subprocess.Popen | None] = [None]


def cancel_training() -> None:
    """Send SIGTERM to the active training subprocess."""
    proc = _ACTIVE_PROCESS[0]
    if proc and proc.poll() is None:
        logger.info("trainer: cancelling training (SIGTERM)")
        proc.terminate()


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    weights_path: str,
    val_dir: str,
    conf_threshold: float = 0.25,
) -> dict:
    """Run YOLOv8 validation on a held-out set.

    Args:
        weights_path:   Path to .pt weights file.
        val_dir:        Directory with images/val/ and labels/val/ subdirs.
        conf_threshold: Confidence threshold for validation.

    Returns:
        Dict with keys: map50, map50_95, precision, recall, per_class_map.

    Raises:
        FileNotFoundError: if weights or val_dir do not exist.
    """
    _check_ultralytics()
    from ultralytics import YOLO

    if not Path(weights_path).exists():
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}. "
            "Train the model first or provide a valid .pt path."
        )

    model = YOLO(weights_path)
    val_yaml = str(Path(val_dir) / "dataset.yaml")
    if not Path(val_yaml).exists():
        # Try parent
        val_yaml = str(Path(val_dir).parent / "dataset.yaml")

    metrics = model.val(data=val_yaml, conf=conf_threshold, verbose=False)
    results = {
        "map50":     float(metrics.box.map50),
        "map50_95":  float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall":    float(metrics.box.mr),
        "per_class_map": {},
    }
    if hasattr(metrics.box, "ap_class_index") and metrics.box.ap_class_index is not None:
        for i, cls_idx in enumerate(metrics.box.ap_class_index):
            results["per_class_map"][int(cls_idx)] = float(metrics.box.ap[i])

    return results


# ---------------------------------------------------------------------------
# Model export
# ---------------------------------------------------------------------------

def export_model(
    weights_path: str,
    format: str = "onnx",
    image_size: int = 640,
) -> str:
    """Export a trained YOLOv8 model to a deployment format.

    Args:
        weights_path: Path to best.pt.
        format:       "onnx", "torchscript", or "tflite".
        image_size:   Input image size for export.

    Returns:
        Path to the exported model file.

    Raises:
        FileNotFoundError: if weights_path does not exist.
        ValueError: if format is not supported.
    """
    _check_ultralytics()
    from ultralytics import YOLO

    supported = ("onnx", "torchscript", "tflite")
    if format not in supported:
        raise ValueError(
            f"Unsupported export format '{format}'. "
            f"Choose one of: {supported}"
        )

    if not Path(weights_path).exists():
        raise FileNotFoundError(
            f"Weights file not found: {weights_path}. "
            "Complete training before exporting."
        )

    model = YOLO(weights_path)
    exported = model.export(format=format, imgsz=image_size)
    logger.info("trainer: model exported as {} to {}", format, exported)
    return str(exported)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_ultralytics() -> None:
    try:
        import ultralytics  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "ultralytics is not installed. "
            "Install with: pip install ultralytics"
        ) from exc


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _load_results_csv(
    csv_path: Path,
    live_metrics: list[EpochMetrics],
) -> list[EpochMetrics]:
    """Parse results.csv written by ultralytics and merge with live metrics."""
    if not csv_path.exists():
        logger.warning("trainer: results.csv not found at {}, using live metrics", csv_path)
        return live_metrics

    updated: list[EpochMetrics] = []
    try:
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                def _f(key: str) -> float:
                    for k in row:
                        if key in k.strip():
                            try:
                                return float(row[k])
                            except (ValueError, TypeError):
                                return 0.0
                    return 0.0

                epoch = i + 1
                existing = live_metrics[i] if i < len(live_metrics) else None
                updated.append(
                    EpochMetrics(
                        epoch=epoch,
                        box_loss=_f("box_loss"),
                        cls_loss=_f("cls_loss"),
                        dfl_loss=_f("dfl_loss"),
                        precision=_f("precision"),
                        recall=_f("recall"),
                        map50=_f("mAP50)"),
                        map50_95=_f("mAP50-95"),
                        lr=_f("lr/pg0") or (existing.lr if existing else 0.0),
                        epoch_time_seconds=existing.epoch_time_seconds if existing else 0.0,
                    )
                )
    except Exception as exc:
        logger.warning("trainer: failed to parse results.csv: {}", exc)
        return live_metrics

    return updated
