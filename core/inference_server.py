"""
core/inference_server.py — Model registry and live inference server.

Manages trained YOLOv8 weights, runs single/batch inference, and handles
model versioning (register → activate → predict → compare).

Registry persisted at ~/.roverdatakit/models/registry.toml
No UI imports — pure Python business logic.
"""
from __future__ import annotations

import base64
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import toml
from loguru import logger

from core.models import (
    Detection, InferenceRequest, InferenceResult,
    ModelComparison, ModelRegistry,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REGISTRY_ROOT = Path.home() / ".roverdatakit" / "models"
REGISTRY_FILE = REGISTRY_ROOT / "registry.toml"

# Module-level model cache — keyed by model_id
_LOADED_MODELS: dict[str, object] = {}   # model_id → YOLO instance


# ---------------------------------------------------------------------------
# Registry persistence
# ---------------------------------------------------------------------------

def load_registry(registry_path: Path = REGISTRY_FILE) -> list[ModelRegistry]:
    """Load model registry from TOML. Returns empty list if file absent."""
    if not registry_path.exists():
        return []
    data = toml.loads(registry_path.read_text())
    entries = data.get("models", [])
    result = []
    for e in entries:
        # toml drops empty dicts — restore metrics default if missing
        e.setdefault("metrics", {})
        try:
            result.append(ModelRegistry(**e))
        except Exception as exc:
            logger.warning(f"Skipping malformed registry entry: {exc}")
    return result


def save_registry(models: list[ModelRegistry], registry_path: Path = REGISTRY_FILE) -> None:
    """Persist model registry to TOML atomically."""
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    data = {"models": [m.model_dump() for m in models]}
    # Serialize datetimes to ISO strings for TOML
    for entry in data["models"]:
        entry["created_at"] = entry["created_at"].isoformat()
    tmp = registry_path.with_suffix(".tmp")
    tmp.write_text(toml.dumps(data))
    tmp.replace(registry_path)
    logger.debug(f"Saved {len(models)} model(s) to registry: {registry_path}")


# ---------------------------------------------------------------------------
# Registry CRUD
# ---------------------------------------------------------------------------

def register_model(
    weights_path: str,
    name: str,
    model_variant: str,
    training_run_id: str | None = None,
    metrics: dict | None = None,
    registry_path: Path = REGISTRY_FILE,
) -> ModelRegistry:
    """Register a trained weights file in the registry."""
    weights = Path(weights_path)
    if not weights.exists():
        raise FileNotFoundError(
            f"Weights file not found: {weights_path}\n"
            "Run a training job first to produce best.pt weights."
        )
    models = load_registry(registry_path)
    entry = ModelRegistry(
        model_id=str(uuid.uuid4()),
        name=name,
        weights_path=str(weights.resolve()),
        model_variant=model_variant,
        training_run_id=training_run_id,
        created_at=datetime.now(timezone.utc),
        metrics=metrics or {},
        is_active=len(models) == 0,   # first model auto-activated
    )
    models.append(entry)
    save_registry(models, registry_path)
    logger.info(f"Registered model '{name}' ({model_variant}) id={entry.model_id}")
    return entry


def set_active_model(model_id: str, registry_path: Path = REGISTRY_FILE) -> None:
    """Set a specific model as the active inference model."""
    models = load_registry(registry_path)
    found = False
    for m in models:
        m.is_active = (m.model_id == model_id)
        if m.model_id == model_id:
            found = True
    if not found:
        raise ValueError(f"Model {model_id} not found in registry.")
    save_registry(models, registry_path)
    logger.info(f"Active model set to {model_id}")


def get_active_model(registry_path: Path = REGISTRY_FILE) -> ModelRegistry:
    """Return the currently active model. Raises if none set."""
    models = load_registry(registry_path)
    for m in models:
        if m.is_active:
            return m
    raise RuntimeError(
        "No active model in registry.\n"
        "Register a trained weights file first: "
        "register_model(weights_path, name, model_variant)"
    )


def delete_model(model_id: str, registry_path: Path = REGISTRY_FILE) -> None:
    """Remove a model entry from the registry (does not delete weights file)."""
    models = load_registry(registry_path)
    before = len(models)
    models = [m for m in models if m.model_id != model_id]
    if len(models) == before:
        raise ValueError(f"Model {model_id} not found in registry.")
    # If deleted model was active, activate most recent remaining
    if models and not any(m.is_active for m in models):
        models[-1].is_active = True
    save_registry(models, registry_path)
    # Evict from cache
    _LOADED_MODELS.pop(model_id, None)
    logger.info(f"Deleted model {model_id} from registry")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_yolo(model_entry: ModelRegistry):
    """Load and cache a YOLO model for a registry entry."""
    if model_entry.model_id in _LOADED_MODELS:
        return _LOADED_MODELS[model_entry.model_id]
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics is required for inference.\n"
            "Install: pip install ultralytics>=8.2.0"
        )
    logger.info(f"Loading model {model_entry.model_id} from {model_entry.weights_path}")
    model = YOLO(model_entry.weights_path)
    _LOADED_MODELS[model_entry.model_id] = model
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _resolve_image(request: InferenceRequest) -> tuple[str, bool]:
    """
    Resolve image input to a file path.
    Returns (path, is_temp) — caller must delete temp file if is_temp=True.
    """
    if request.image_path:
        if not Path(request.image_path).exists():
            raise FileNotFoundError(f"Image not found: {request.image_path}")
        return request.image_path, False

    if request.image_base64:
        data = base64.b64decode(request.image_base64)
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.write(data)
        tmp.flush()
        tmp.close()
        return tmp.name, True

    raise ValueError("InferenceRequest must have either image_path or image_base64.")


def run_inference(
    request: InferenceRequest,
    registry_path: Path = REGISTRY_FILE,
) -> InferenceResult:
    """Run inference on a single image using the specified or active model."""
    if request.model_id:
        models = load_registry(registry_path)
        entry = next((m for m in models if m.model_id == request.model_id), None)
        if entry is None:
            raise ValueError(f"Model {request.model_id} not found in registry.")
    else:
        entry = get_active_model(registry_path)

    model = _load_yolo(entry)
    image_path, is_temp = _resolve_image(request)

    try:
        t0 = time.perf_counter()
        results = model(
            image_path,
            conf=request.conf_threshold,
            iou=request.iou_threshold,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        detections: list[Detection] = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            names = r.names
            for box in boxes:
                cls_id = int(box.cls[0])
                detections.append(Detection(
                    class_id=cls_id,
                    class_name=names.get(cls_id, str(cls_id)),
                    confidence=float(box.conf[0]),
                    bbox_xyxy=[float(v) for v in box.xyxy[0].tolist()],
                    bbox_xywhn=[float(v) for v in box.xywhn[0].tolist()],
                ))

        return InferenceResult(
            model_id=entry.model_id,
            image_path=request.image_path,
            detections=detections,
            inference_time_ms=elapsed_ms,
            model_variant=entry.model_variant,
        )
    finally:
        if is_temp:
            Path(image_path).unlink(missing_ok=True)


def run_inference_batch(
    requests: list[InferenceRequest],
    registry_path: Path = REGISTRY_FILE,
    progress_callback: Callable[[int], None] | None = None,
) -> list[InferenceResult]:
    """Run inference on a list of requests sequentially. Same model for all."""
    results = []
    total = len(requests)
    for i, req in enumerate(requests):
        results.append(run_inference(req, registry_path))
        if progress_callback:
            progress_callback(int((i + 1) / total * 100))
    return results


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def compare_models(
    baseline_id: str,
    candidate_id: str,
    val_dir: str,
    conf_threshold: float = 0.25,
    registry_path: Path = REGISTRY_FILE,
) -> ModelComparison:
    """
    Evaluate both models on val_dir and compare mAP50.
    val_dir must contain images/ and labels/ sub-directories (YOLO format).
    """
    from core.trainer import evaluate_model as _eval

    models = load_registry(registry_path)
    baseline = next((m for m in models if m.model_id == baseline_id), None)
    candidate = next((m for m in models if m.model_id == candidate_id), None)
    if baseline is None:
        raise ValueError(f"Baseline model {baseline_id} not found.")
    if candidate is None:
        raise ValueError(f"Candidate model {candidate_id} not found.")

    logger.info(f"Evaluating baseline {baseline_id}…")
    b_metrics = _eval(baseline.weights_path, val_dir, conf_threshold)
    b_map50 = b_metrics.get("map50", 0.0)

    logger.info(f"Evaluating candidate {candidate_id}…")
    c_metrics = _eval(candidate.weights_path, val_dir, conf_threshold)
    c_map50 = c_metrics.get("map50", 0.0)

    delta = c_map50 - b_map50
    return ModelComparison(
        baseline_model_id=baseline_id,
        candidate_model_id=candidate_id,
        val_dir=val_dir,
        baseline_map50=b_map50,
        candidate_map50=c_map50,
        improved=delta >= 0.01,
        delta_map50=delta,
    )
