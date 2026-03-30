"""
labelox/core/classifier.py — CLIP-based zero-shot scene classification.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
from loguru import logger
from PIL import Image

ProgressCB = Callable[[int, int], None]

# ─── Default Scene Classes ───────────────────────────────────────────────────

SCENE_CLASSES = [
    "urban_road",
    "highway",
    "residential_street",
    "rural_road",
    "unpaved_track",
    "intersection",
    "parking_area",
    "indoor",
    "construction_zone",
    "waterlogged_road",
    "speed_bump_zone",
]

# ─── Cache ───────────────────────────────────────────────────────────────────

_CLIP_CACHE: dict[str, tuple[Any, Any]] = {}


def _resolve_device(device: str) -> str:
    import torch
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def load_clip(
    model_name: str = "openai/clip-vit-base-patch32",
    device: str = "auto",
) -> tuple[Any, Any]:
    """Load CLIP model + processor. Cached globally."""
    resolved = _resolve_device(device)
    key = f"{model_name}:{resolved}"

    if key in _CLIP_CACHE:
        return _CLIP_CACHE[key]

    from transformers import CLIPModel, CLIPProcessor

    logger.info("Loading CLIP model {} on {}", model_name, resolved)
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.to(resolved)
    model.eval()

    _CLIP_CACHE[key] = (model, processor)
    return model, processor


# ─── Classification ──────────────────────────────────────────────────────────

def classify_scene(
    image_path: str,
    candidate_labels: list[str] | None = None,
    model: Any = None,
    processor: Any = None,
    model_name: str = "openai/clip-vit-base-patch32",
    device: str = "auto",
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """Zero-shot classify a single image.

    Returns top_k (label, confidence) pairs, sorted by confidence descending.
    """
    import torch

    if candidate_labels is None:
        candidate_labels = SCENE_CLASSES

    if model is None or processor is None:
        model, processor = load_clip(model_name, device)

    resolved = _resolve_device(device)

    image = Image.open(image_path).convert("RGB")
    text_prompts = [f"a photo of {label.replace('_', ' ')}" for label in candidate_labels]

    inputs = processor(
        text=text_prompts,
        images=image,
        return_tensors="pt",
        padding=True,
    ).to(resolved)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image[0]  # [num_labels]
        probs = logits.softmax(dim=0).cpu().numpy()

    scored = list(zip(candidate_labels, probs.tolist()))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def classify_batch(
    image_paths: list[str],
    candidate_labels: list[str] | None = None,
    model: Any = None,
    processor: Any = None,
    model_name: str = "openai/clip-vit-base-patch32",
    device: str = "auto",
    batch_size: int = 32,
    progress_callback: ProgressCB | None = None,
) -> dict[str, str]:
    """Batch classification. Returns {image_path: top_scene_class}."""
    if candidate_labels is None:
        candidate_labels = SCENE_CLASSES

    if model is None or processor is None:
        model, processor = load_clip(model_name, device)

    results: dict[str, str] = {}
    n = len(image_paths)

    for start in range(0, n, batch_size):
        batch_paths = image_paths[start: start + batch_size]

        for path in batch_paths:
            try:
                scored = classify_scene(
                    path, candidate_labels,
                    model=model, processor=processor,
                    model_name=model_name, device=device,
                    top_k=1,
                )
                results[path] = scored[0][0] if scored else "unknown"
            except Exception as exc:
                logger.warning("Classification failed for {}: {}", path, exc)
                results[path] = "unknown"

        if progress_callback:
            progress_callback(min(start + batch_size, n), n)

    return results


# ─── Embedding Extraction (for similarity engine) ───────────────────────────

def compute_image_embedding(
    image_path: str,
    model: Any = None,
    processor: Any = None,
    model_name: str = "openai/clip-vit-base-patch32",
    device: str = "auto",
) -> np.ndarray:
    """Extract CLIP feature vector for an image. Returns 512-d numpy array."""
    import torch

    if model is None or processor is None:
        model, processor = load_clip(model_name, device)

    resolved = _resolve_device(device)
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(resolved)

    with torch.no_grad():
        features = model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)

    return features[0].cpu().numpy()
