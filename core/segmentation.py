"""
core/segmentation.py — Semantic segmentation using SegFormer.

Runs HuggingFace SegFormer models (cityscapes-finetuned) on extracted rover
frames.  Produces per-pixel class masks, colorized overlays, and per-frame
statistics compatible with the SegmentationResult / SegmentationStats models.

No UI imports.  All paths use pathlib.Path.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
from loguru import logger

from core.models import SegmentationResult, SegmentationStats

# ---------------------------------------------------------------------------
# Class definitions
# ---------------------------------------------------------------------------

SEGMENTATION_CLASSES: dict[int, str] = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic_light",
    7: "traffic_sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "motorcycle",
    17: "bicycle",
    18: "autorickshaw",
    19: "pothole_region",
    20: "unpaved_road",
}

CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "road":           (128,  64,  64),
    "sidewalk":       (160, 144,  96),
    "building":       ( 64,  64, 128),
    "sky":            ( 64, 128, 192),
    "vegetation":     ( 64,  96,  32),
    "person":         (224,  64,  64),
    "car":            (  0,  96, 192),
    "truck":          (  0,   0, 192),
    "bus":            (  0,  60, 200),
    "motorcycle":     (192, 144,  64),
    "traffic_light":  (  0, 192,   0),
    "autorickshaw":   (224, 128,   0),
    "pothole_region": (255,   0,   0),
}
_DEFAULT_COLOR: tuple[int, int, int] = (100, 100, 100)

# Mapping from Cityscapes label names to our internal class names.
# Keys are exact strings that HuggingFace SegFormer id2label may return.
_CITYSCAPES_TO_OURS: dict[str, str] = {
    "road":          "road",
    "sidewalk":      "sidewalk",
    "building":      "building",
    "wall":          "wall",
    "fence":         "fence",
    "pole":          "pole",
    "traffic light": "traffic_light",
    "traffic sign":  "traffic_sign",
    "vegetation":    "vegetation",
    "terrain":       "unpaved_road",   # terrain → unpaved_road
    "sky":           "sky",
    "person":        "person",
    "rider":         "rider",
    "car":           "car",
    "truck":         "truck",
    "bus":           "bus",
    "train":         "building",       # map train → building (not in our set)
    "motorcycle":    "motorcycle",
    "bicycle":       "bicycle",
}


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

def check_segmentation_dependencies() -> dict[str, bool]:
    """Return availability of each required package.

    Returns:
        dict with keys 'transformers', 'torch', 'cv2', 'numpy'.
    """
    result: dict[str, bool] = {}

    try:
        import transformers  # noqa: F401
        result["transformers"] = True
    except ImportError:
        result["transformers"] = False

    try:
        import torch  # noqa: F401
        result["torch"] = True
    except ImportError:
        result["torch"] = False

    try:
        import cv2  # noqa: F401
        result["cv2"] = True
    except ImportError:
        result["cv2"] = False

    try:
        import numpy  # noqa: F401
        result["numpy"] = True
    except ImportError:
        result["numpy"] = False

    return result


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_segmentation_model(
    model_name: str = "nvidia/segformer-b2-finetuned-cityscapes-512-1024",
    device: str = "auto",
) -> tuple:
    """Load a SegFormer model and feature extractor from HuggingFace.

    Args:
        model_name: HuggingFace model identifier.
        device:     'auto' selects CUDA if available, otherwise CPU.
                    'cpu', 'cuda:0', etc. are passed through directly.

    Returns:
        (model, processor, id2label) where id2label maps int → class name string.

    Raises:
        ImportError: if transformers or torch are not installed.
        RuntimeError: if the model download or load fails.
    """
    try:
        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    except ImportError as exc:
        raise ImportError(
            "transformers and torch are required for segmentation. "
            "Install with: pip install transformers torch"
        ) from exc

    if device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = device

    logger.info("Loading segmentation model '{}' on device '{}'", model_name, resolved_device)

    try:
        processor = SegformerImageProcessor.from_pretrained(model_name)
        model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        model = model.to(resolved_device)
        model.eval()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model '{model_name}'. "
            f"Check your internet connection and HuggingFace token if the model is gated. "
            f"Underlying error: {exc}"
        ) from exc

    id2label: dict[int, str] = model.config.id2label
    logger.info(
        "Segmentation model loaded — {} classes, device={}",
        len(id2label),
        resolved_device,
    )
    return model, processor, id2label


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _map_label(cityscapes_name: str) -> str:
    """Normalise a Cityscapes label name to our internal class name."""
    name = cityscapes_name.lower().strip()
    return _CITYSCAPES_TO_OURS.get(name, name.replace(" ", "_"))


def _build_color_map(id2label: dict[int, str]) -> np.ndarray:
    """Build an (N, 3) uint8 array mapping label id → RGB colour."""
    n = max(id2label.keys()) + 1
    color_map = np.full((n, 3), _DEFAULT_COLOR, dtype=np.uint8)
    for idx, cityscapes_name in id2label.items():
        our_name = _map_label(cityscapes_name)
        color = CLASS_COLORS.get(our_name, _DEFAULT_COLOR)
        color_map[idx] = color
    return color_map


def _compute_class_percent(
    mask: np.ndarray,
    id2label: dict[int, str],
) -> dict[str, float]:
    """Return per-class pixel percentage for a 2-D label mask."""
    total = mask.size
    if total == 0:
        return {}
    percents: dict[str, float] = {}
    for idx, cityscapes_name in id2label.items():
        our_name = _map_label(cityscapes_name)
        count = int(np.sum(mask == idx))
        pct = count / total * 100.0
        # Accumulate into our class name (multiple cityscapes ids might map to one)
        percents[our_name] = percents.get(our_name, 0.0) + pct
    return percents


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

def run_segmentation_batch(
    image_paths: list[str],
    model,
    processor,
    id2label: dict[int, str],
    batch_size: int = 4,
    output_dir: str | None = None,
    overlay_alpha: float = 0.5,
    progress_callback: Callable[[int], None] | None = None,
) -> list[SegmentationResult]:
    """Run SegFormer on a list of images and produce masks + overlays.

    For each image:
      1. Run SegFormer inference.
      2. Upsample logits → argmax → semantic mask (H×W uint8).
      3. Save raw mask PNG to output_dir/masks/{stem}_mask.png.
      4. Generate colorized overlay (alpha blend) saved to
         output_dir/overlays/{stem}_overlay.jpg.
      5. Compute per-class pixel percent.
      6. road_area_percent  = road + unpaved_road pixels %
      7. sky_area_percent   from sky class %
      8. is_valid_rover_frame: road_area_percent >= 10.0

    Args:
        image_paths:       Absolute paths to input images (must exist).
        model:             Loaded SegFormer model.
        processor:         Matching SegformerImageProcessor.
        id2label:          Model's int → class-name mapping.
        batch_size:        Number of images per inference batch.
        output_dir:        Root directory for masks/ and overlays/ subdirs.
                           Defaults to parent of first image / 'segmentation'.
        overlay_alpha:     Blend factor for colour overlay (0=original, 1=pure colour).
        progress_callback: Optional callable(int 0-100) for progress reporting.

    Returns:
        List of SegmentationResult — one per input image, in order.

    Raises:
        ImportError: if torch or cv2 are missing.
        FileNotFoundError: if any image_path does not exist.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required for segmentation inference. "
            "Install with: pip install torch"
        ) from exc

    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required for image I/O. "
            "Install with: pip install opencv-python"
        ) from exc

    if not image_paths:
        return []

    # Resolve output directories
    if output_dir is None:
        output_dir = str(Path(image_paths[0]).parent.parent / "segmentation")
    out_root = Path(output_dir)
    masks_dir = out_root / "masks"
    overlays_dir = out_root / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    color_map = _build_color_map(id2label)
    results: list[SegmentationResult] = []
    n_total = len(image_paths)
    device = next(model.parameters()).device

    for batch_start in range(0, n_total, batch_size):
        batch_paths = image_paths[batch_start : batch_start + batch_size]

        # Load images — raise immediately if a file is missing
        batch_images = []
        for p in batch_paths:
            if not Path(p).exists():
                raise FileNotFoundError(
                    f"Image not found: {p}. "
                    "Ensure extraction has completed before running segmentation."
                )
            img_bgr = cv2.imread(str(p))
            if img_bgr is None:
                raise RuntimeError(
                    f"cv2.imread returned None for '{p}'. "
                    "The file may be corrupt or in an unsupported format."
                )
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            batch_images.append(img_rgb)

        # Processor expects list of PIL or numpy images
        t_start = time.monotonic()
        inputs = processor(images=batch_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # outputs.logits: (B, num_classes, H/4, W/4)
        logits = outputs.logits  # (B, C, h, w)

        for i, img_path in enumerate(batch_paths):
            img_rgb = batch_images[i]
            h_orig, w_orig = img_rgb.shape[:2]
            stem = Path(img_path).stem

            # Upsample single image logits to original resolution
            single_logit = logits[i : i + 1]  # (1, C, h, w)
            upsampled = torch.nn.functional.interpolate(
                single_logit,
                size=(h_orig, w_orig),
                mode="bilinear",
                align_corners=False,
            )  # (1, C, H, W)
            mask_tensor = upsampled.argmax(dim=1).squeeze(0)  # (H, W)
            mask_np = mask_tensor.cpu().numpy().astype(np.uint8)  # (H, W) uint8

            inf_time_ms = (time.monotonic() - t_start) / len(batch_paths) * 1000.0

            # Save raw mask PNG
            mask_path = masks_dir / f"{stem}_mask.png"
            cv2.imwrite(str(mask_path), mask_np)

            # Build colorized overlay
            color_overlay = color_map[mask_np]  # (H, W, 3) RGB
            # Blend with original image
            img_float = img_rgb.astype(np.float32)
            ov_float = color_overlay.astype(np.float32)
            blended = (
                (1.0 - overlay_alpha) * img_float + overlay_alpha * ov_float
            ).clip(0, 255).astype(np.uint8)
            blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
            overlay_path = overlays_dir / f"{stem}_overlay.jpg"
            cv2.imwrite(
                str(overlay_path),
                blended_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )

            # Per-class percentages
            class_pct = _compute_class_percent(mask_np, id2label)

            road_area = class_pct.get("road", 0.0) + class_pct.get("unpaved_road", 0.0)
            sky_area = class_pct.get("sky", 0.0)
            is_valid = road_area >= 10.0

            results.append(
                SegmentationResult(
                    frame_path=str(img_path),
                    mask_path=str(mask_path),
                    overlay_path=str(overlay_path),
                    class_pixel_percent=class_pct,
                    road_area_percent=road_area,
                    sky_area_percent=sky_area,
                    inference_time_ms=inf_time_ms,
                    is_valid_rover_frame=is_valid,
                )
            )

        # Progress callback
        done = min(batch_start + batch_size, n_total)
        if progress_callback is not None:
            progress_callback(int(done / n_total * 100))

        logger.debug(
            "Segmentation batch {}/{}: {} frames processed",
            done,
            n_total,
            len(batch_paths),
        )

    return results


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_segmentation_statistics(
    results: list[SegmentationResult],
) -> SegmentationStats:
    """Aggregate per-frame SegmentationResults into dataset-level statistics.

    Args:
        results: List of SegmentationResult from run_segmentation_batch.

    Returns:
        SegmentationStats with mean/std per class, road coverage, and
        a list of invalid frame paths (is_valid_rover_frame == False).
    """
    if not results:
        return SegmentationStats(
            per_class_mean_percent={},
            per_class_std_percent={},
            frames_with_road=0,
            frames_without_road=0,
            mean_road_coverage=0.0,
            invalid_frames=[],
        )

    # Collect all class names seen across all frames
    all_classes: set[str] = set()
    for r in results:
        all_classes.update(r.class_pixel_percent.keys())

    per_class_values: dict[str, list[float]] = {cls: [] for cls in all_classes}
    road_coverages: list[float] = []
    invalid_frames: list[str] = []
    frames_with_road = 0
    frames_without_road = 0

    for r in results:
        for cls in all_classes:
            per_class_values[cls].append(r.class_pixel_percent.get(cls, 0.0))

        road_coverages.append(r.road_area_percent)
        if r.road_area_percent > 0.0:
            frames_with_road += 1
        else:
            frames_without_road += 1

        if not r.is_valid_rover_frame:
            invalid_frames.append(r.frame_path)

    per_class_mean: dict[str, float] = {}
    per_class_std: dict[str, float] = {}
    for cls, vals in per_class_values.items():
        arr = np.array(vals, dtype=np.float64)
        per_class_mean[cls] = float(arr.mean())
        per_class_std[cls] = float(arr.std())

    mean_road = float(np.mean(road_coverages)) if road_coverages else 0.0

    return SegmentationStats(
        per_class_mean_percent=per_class_mean,
        per_class_std_percent=per_class_std,
        frames_with_road=frames_with_road,
        frames_without_road=frames_without_road,
        mean_road_coverage=mean_road,
        invalid_frames=invalid_frames,
    )
