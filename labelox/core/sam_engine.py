"""
labelox/core/sam_engine.py — SAM (Segment Anything) integration via HuggingFace transformers.

Model loaded lazily and cached. Image embeddings cached per image path.
Target: <200ms per mask prediction after embedding is cached.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from PIL import Image

from labelox.core.models import MaskAnnotation

# ─── Cache ───────────────────────────────────────────────────────────────────

_SAM_CACHE: dict[str, tuple[Any, Any]] = {}  # model_name -> (model, processor)
_EMBEDDING_CACHE: dict[str, dict] = {}  # image_path -> {inputs, image_size}


def _resolve_device(device: str) -> str:
    import torch
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


# ─── Model Loading ───────────────────────────────────────────────────────────

def load_sam(
    model_name: str = "facebook/sam-vit-base",
    device: str = "auto",
) -> tuple[Any, Any]:
    """Load SAM model + processor. Cached globally.

    Returns (model, processor) tuple.
    """
    resolved = _resolve_device(device)
    key = f"{model_name}:{resolved}"

    if key in _SAM_CACHE:
        return _SAM_CACHE[key]

    from transformers import SamModel, SamProcessor

    logger.info("Loading SAM model {} on {}", model_name, resolved)
    processor = SamProcessor.from_pretrained(model_name)
    model = SamModel.from_pretrained(model_name)
    model.to(resolved)
    model.eval()

    _SAM_CACHE[key] = (model, processor)
    return model, processor


# ─── Image Embedding ─────────────────────────────────────────────────────────

def set_image_for_sam(
    image_path: str,
    model: Any = None,
    processor: Any = None,
    model_name: str = "facebook/sam-vit-base",
    device: str = "auto",
) -> dict:
    """Precompute image embedding (~1-2s). Cached per image path.

    Call before predict_mask_* functions.
    Returns dict with 'image_embeddings' and 'image_size'.
    """
    import torch

    if image_path in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[image_path]

    if model is None or processor is None:
        model, processor = load_sam(model_name, device)

    resolved = _resolve_device(device)
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    # Process image for embedding
    inputs = processor(image, return_tensors="pt").to(resolved)

    with torch.no_grad():
        image_embeddings = model.get_image_embeddings(inputs["pixel_values"])

    cache_entry = {
        "image_embeddings": image_embeddings,
        "original_sizes": inputs.get("original_sizes"),
        "reshaped_input_sizes": inputs.get("reshaped_input_sizes"),
        "image_size": (w, h),
        "device": resolved,
    }
    _EMBEDDING_CACHE[image_path] = cache_entry
    logger.debug("SAM embedding cached for {}", Path(image_path).name)
    return cache_entry


def clear_embedding_cache() -> None:
    """Free cached embeddings to release GPU memory."""
    _EMBEDDING_CACHE.clear()


# ─── Mask Prediction ─────────────────────────────────────────────────────────

def predict_mask_from_point(
    image_path: str,
    point_x: float,  # normalised 0-1
    point_y: float,  # normalised 0-1
    point_label: int = 1,  # 1=foreground, 0=background
    model: Any = None,
    processor: Any = None,
    model_name: str = "facebook/sam-vit-base",
    device: str = "auto",
) -> MaskAnnotation:
    """Single click → mask prediction."""
    return predict_mask_from_points(
        image_path,
        positive_points=[(point_x, point_y)] if point_label == 1 else [],
        negative_points=[(point_x, point_y)] if point_label == 0 else [],
        model=model,
        processor=processor,
        model_name=model_name,
        device=device,
    )


def predict_mask_from_box(
    image_path: str,
    x: float, y: float, width: float, height: float,  # normalised
    model: Any = None,
    processor: Any = None,
    model_name: str = "facebook/sam-vit-base",
    device: str = "auto",
) -> MaskAnnotation:
    """Bounding box prompt → precise mask."""
    import torch

    if model is None or processor is None:
        model, processor = load_sam(model_name, device)

    embedding = set_image_for_sam(image_path, model, processor, model_name, device)
    w, h = embedding["image_size"]
    resolved = embedding["device"]

    # Convert normalised box to pixel coords
    input_boxes = [[[x * w, y * h, (x + width) * w, (y + height) * h]]]

    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, input_boxes=input_boxes, return_tensors="pt").to(resolved)

    with torch.no_grad():
        inputs["image_embeddings"] = embedding["image_embeddings"]
        if embedding.get("original_sizes") is not None:
            inputs["original_sizes"] = embedding["original_sizes"]
        if embedding.get("reshaped_input_sizes") is not None:
            inputs["reshaped_input_sizes"] = embedding["reshaped_input_sizes"]
        outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )

    return _best_mask_to_annotation(masks, outputs.iou_scores, h, w)


def predict_mask_from_points(
    image_path: str,
    positive_points: list[tuple[float, float]],
    negative_points: list[tuple[float, float]] | None = None,
    model: Any = None,
    processor: Any = None,
    model_name: str = "facebook/sam-vit-base",
    device: str = "auto",
) -> MaskAnnotation:
    """Multi-point prompting. Positive = inside object, negative = outside."""
    import torch

    if model is None or processor is None:
        model, processor = load_sam(model_name, device)

    embedding = set_image_for_sam(image_path, model, processor, model_name, device)
    w, h = embedding["image_size"]
    resolved = embedding["device"]

    if negative_points is None:
        negative_points = []

    # Build point lists in pixel coords
    all_points = []
    all_labels = []
    for px, py in positive_points:
        all_points.append([px * w, py * h])
        all_labels.append(1)
    for px, py in negative_points:
        all_points.append([px * w, py * h])
        all_labels.append(0)

    if not all_points:
        raise ValueError("At least one point required for SAM prediction.")

    input_points = [[all_points]]
    input_labels = [[all_labels]]

    image = Image.open(image_path).convert("RGB")
    inputs = processor(
        image,
        input_points=input_points,
        input_labels=input_labels,
        return_tensors="pt",
    ).to(resolved)

    with torch.no_grad():
        inputs["image_embeddings"] = embedding["image_embeddings"]
        if embedding.get("original_sizes") is not None:
            inputs["original_sizes"] = embedding["original_sizes"]
        if embedding.get("reshaped_input_sizes") is not None:
            inputs["reshaped_input_sizes"] = embedding["reshaped_input_sizes"]
        outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )

    return _best_mask_to_annotation(masks, outputs.iou_scores, h, w)


def refine_mask_with_click(
    image_path: str,
    existing_positive: list[tuple[float, float]],
    existing_negative: list[tuple[float, float]],
    new_point: tuple[float, float],
    is_positive: bool,
    model: Any = None,
    processor: Any = None,
    model_name: str = "facebook/sam-vit-base",
    device: str = "auto",
) -> MaskAnnotation:
    """Add one more point to refine existing prediction."""
    if is_positive:
        pos = existing_positive + [new_point]
        neg = existing_negative
    else:
        pos = existing_positive
        neg = existing_negative + [new_point]

    return predict_mask_from_points(
        image_path, pos, neg,
        model=model, processor=processor,
        model_name=model_name, device=device,
    )


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _best_mask_to_annotation(masks, iou_scores, h: int, w: int) -> MaskAnnotation:
    """Select highest-IoU mask from SAM's 3 candidates, convert to RLE."""
    # masks shape: list of tensors [1, 3, H, W]
    mask_tensor = masks[0]  # first image in batch
    scores = iou_scores[0].cpu().numpy()  # [1, 3]

    # Pick best mask
    best_idx = int(scores[0].argmax())
    mask_np = mask_tensor[0, best_idx].numpy().astype(np.uint8)  # [H, W]

    rle = _mask_to_rle(mask_np)
    return MaskAnnotation(rle=rle)


def _mask_to_rle(mask: np.ndarray) -> dict:
    """Convert binary mask to COCO RLE format."""
    h, w = mask.shape
    flat = mask.flatten()
    changes = np.diff(flat)
    change_idx = np.where(changes != 0)[0] + 1
    runs = np.concatenate([[0], change_idx, [len(flat)]])
    lengths = np.diff(runs).tolist()

    if flat[0] == 1:
        lengths = [0] + lengths

    return {"counts": lengths, "size": [h, w]}
