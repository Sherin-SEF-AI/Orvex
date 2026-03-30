"""
core/augmentor.py — Data augmentation pipeline for rover datasets.

Applies albumentations transforms to extracted frames + YOLO labels.
Mosaic augmentation (YOLOv8-style) is implemented from scratch.

No UI imports — pure Python business logic.

Dependencies:
    pip install albumentations Pillow
"""
from __future__ import annotations

import random
import shutil
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from loguru import logger

from core.models import AugmentationConfig, AugmentationResult, Detection, FrameAnnotation

ProgressCB = Callable[[int], None]
StatusCB = Callable[[str], None]


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_augmentation_pipeline(config: AugmentationConfig) -> object:
    """Build an albumentations Compose pipeline from AugmentationConfig.

    BBox format: pascal_voc [x_min, y_min, x_max, y_max] (absolute pixels).
    Albumentations handles coordinate transform for all geometric ops.

    Args:
        config: AugmentationConfig with boolean toggles + probabilities.

    Returns:
        albumentations.Compose instance.

    Raises:
        ImportError: if albumentations is not installed.
    """
    try:
        import albumentations as A
    except ImportError as exc:
        raise ImportError(
            "albumentations is not installed. "
            "Install with: pip install albumentations"
        ) from exc

    transforms: list = []

    # ── Geometric ──────────────────────────────────────────────────────────
    if config.horizontal_flip:
        transforms.append(A.HorizontalFlip(p=0.5))
    if config.vertical_flip:
        transforms.append(A.VerticalFlip(p=0.0))  # spec default
    if config.random_rotate_90:
        transforms.append(A.RandomRotate90(p=0.3))

    # ── Photometric ────────────────────────────────────────────────────────
    if config.brightness_contrast:
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=(-0.2, 0.2),
                contrast_limit=(-0.2, 0.2),
                p=0.5,
            )
        )
    if config.hue_saturation:
        transforms.append(A.HueSaturationValue(p=0.3))
    if config.gaussian_noise:
        transforms.append(A.GaussNoise(var_limit=(10.0, 50.0), p=0.2))
    if config.motion_blur:
        transforms.append(A.MotionBlur(blur_limit=(3, 7), p=0.2))
    if config.jpeg_compression:
        transforms.append(A.ImageCompression(quality_lower=60, quality_upper=95, p=0.3))

    # ── Weather simulation ─────────────────────────────────────────────────
    if config.rain_simulation:
        transforms.append(A.RandomRain(p=0.1))
    if config.fog_simulation:
        transforms.append(A.RandomFog(p=0.1))

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_visibility=0.1,   # drop boxes that become < 10% visible
        ),
    )


# ---------------------------------------------------------------------------
# Single-image augmentation
# ---------------------------------------------------------------------------

def apply_augmentation(
    image_path: str,
    labels: list[Detection],
    pipeline: object,
    n_augmentations: int = 3,
) -> list[tuple[np.ndarray, list[Detection]]]:
    """Apply augmentation pipeline N times to one image.

    Args:
        image_path:      Path to source JPEG/PNG.
        labels:          Detection list with bbox_xyxy (absolute pixels).
        pipeline:        albumentations Compose pipeline.
        n_augmentations: Number of augmented variants to produce.

    Returns:
        List of (augmented_image_numpy, augmented_detections) tuples.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(
            f"Failed to read image for augmentation: {image_path}. "
            "Check that the file exists and is a valid image."
        )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    bboxes = [[d.bbox_xyxy[0], d.bbox_xyxy[1], d.bbox_xyxy[2], d.bbox_xyxy[3]]
              for d in labels]
    class_labels = [d.class_name for d in labels]
    # Clip bboxes to image bounds
    bboxes = [[max(0, x1), max(0, y1), min(w, x2), min(h, y2)]
              for x1, y1, x2, y2 in bboxes]

    results: list[tuple[np.ndarray, list[Detection]]] = []

    for _ in range(n_augmentations):
        try:
            transformed = pipeline(
                image=img, bboxes=bboxes, class_labels=class_labels
            )
        except Exception as exc:
            logger.warning("augmentor: transform failed for {}: {}", image_path, exc)
            continue

        aug_img = transformed["image"]
        aug_bboxes = transformed["bboxes"]
        aug_labels = transformed["class_labels"]
        aug_h, aug_w = aug_img.shape[:2]

        aug_dets: list[Detection] = []
        for (x1, y1, x2, y2), cls_name in zip(aug_bboxes, aug_labels):
            cx = (x1 + x2) / 2 / aug_w
            cy = (y1 + y2) / 2 / aug_h
            bw = (x2 - x1) / aug_w
            bh = (y2 - y1) / aug_h
            # Find original class_id
            orig = next((d for d in labels if d.class_name == cls_name), None)
            aug_dets.append(
                Detection(
                    class_id=orig.class_id if orig else 0,
                    class_name=cls_name,
                    confidence=orig.confidence if orig else 1.0,
                    bbox_xyxy=[x1, y1, x2, y2],
                    bbox_xywhn=[cx, cy, bw, bh],
                )
            )
        results.append((aug_img, aug_dets))

    return results


# ---------------------------------------------------------------------------
# Mosaic augmentation
# ---------------------------------------------------------------------------

def run_mosaic_augmentation(
    image_paths: list[str],
    annotations: list[list[Detection]],
    output_size: tuple[int, int] = (640, 640),
) -> tuple[np.ndarray, list[Detection]]:
    """YOLOv8-style mosaic: combine 4 random images into one.

    Picks 4 images, resizes each to a quadrant of output_size, and merges
    them into a single image.  BBox coordinates are adjusted accordingly.

    Args:
        image_paths:  Pool of image paths to sample from.
        annotations:  Corresponding detection lists (parallel to image_paths).
        output_size:  (height, width) of the output mosaic.

    Returns:
        (mosaic_image, merged_detections) tuple.

    Raises:
        ValueError: if fewer than 4 images are available.
    """
    if len(image_paths) < 4:
        raise ValueError(
            f"Mosaic augmentation requires at least 4 images, got {len(image_paths)}. "
            "Add more frames or disable mosaic augmentation."
        )

    out_h, out_w = output_size
    mosaic = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    # Quadrant centers (cut point at mid)
    mid_x, mid_y = out_w // 2, out_h // 2
    quadrants = [
        (0,     0,     mid_x, mid_y),    # top-left
        (mid_x, 0,     out_w, mid_y),    # top-right
        (0,     mid_y, mid_x, out_h),    # bottom-left
        (mid_x, mid_y, out_w, out_h),    # bottom-right
    ]

    indices = random.sample(range(len(image_paths)), 4)
    all_dets: list[Detection] = []

    for idx, (x1, y1, x2, y2) in zip(indices, quadrants):
        fp = image_paths[idx]
        img = cv2.imread(fp)
        if img is None:
            raise RuntimeError(
                f"Mosaic: failed to read image {fp}. "
                "Verify the file exists and is readable."
            )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qw, qh = x2 - x1, y2 - y1
        img_r = cv2.resize(img, (qw, qh))
        mosaic[y1:y2, x1:x2] = img_r

        orig_h, orig_w = img.shape[:2]
        sx = qw / orig_w
        sy = qh / orig_h

        for det in annotations[idx]:
            ax1, ay1, ax2, ay2 = det.bbox_xyxy
            # Scale to quadrant, then offset to mosaic coords
            mx1 = ax1 * sx + x1
            my1 = ay1 * sy + y1
            mx2 = ax2 * sx + x1
            my2 = ay2 * sy + y1
            # Clip to quadrant bounds
            mx1, my1, mx2, my2 = (
                max(x1, mx1), max(y1, my1),
                min(x2, mx2), min(y2, my2),
            )
            if mx2 <= mx1 or my2 <= my1:
                continue
            cx = (mx1 + mx2) / 2 / out_w
            cy = (my1 + my2) / 2 / out_h
            bw = (mx2 - mx1) / out_w
            bh = (my2 - my1) / out_h
            all_dets.append(
                Detection(
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox_xyxy=[mx1, my1, mx2, my2],
                    bbox_xywhn=[cx, cy, bw, bh],
                )
            )

    return mosaic, all_dets


# ---------------------------------------------------------------------------
# Full dataset augmentation
# ---------------------------------------------------------------------------

def augment_dataset(
    frame_paths: list[str],
    annotations: list[FrameAnnotation],
    config: AugmentationConfig,
    output_dir: str,
    multiplier: int | None = None,
    progress_callback: ProgressCB | None = None,
    status_callback: StatusCB | None = None,
) -> AugmentationResult:
    """Augment a full dataset by the configured multiplier.

    Input:  N frames + annotations.
    Output: N * multiplier total images (originals + augmented).

    YOLO format: writes images to {output_dir}/images/train/
                 labels to  {output_dir}/labels/train/

    Also copies original frames and labels into the output structure
    so the output directory is a self-contained YOLO dataset.

    Args:
        frame_paths:       Source frame JPEG/PNG paths.
        annotations:       FrameAnnotation list (parallel to frame_paths).
        config:            AugmentationConfig.
        output_dir:        Root output directory.
        multiplier:        Override config.multiplier if provided.
        progress_callback: 0-100.
        status_callback:   Status strings.

    Returns:
        AugmentationResult with counts and output paths.
    """
    eff_multiplier = multiplier if multiplier is not None else config.multiplier
    if not frame_paths:
        raise RuntimeError(
            "No frame paths provided to augment_dataset. "
            "Run extraction first to generate frames."
        )
    if len(frame_paths) != len(annotations):
        raise ValueError(
            f"frame_paths ({len(frame_paths)}) and annotations ({len(annotations)}) "
            "must be the same length."
        )

    out = Path(output_dir)
    img_out = out / "images" / "train"
    lbl_out = out / "labels" / "train"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    pipeline = build_augmentation_pipeline(config)
    per_transform_counts: dict[str, int] = {"original": 0, "augmented": 0, "mosaic": 0}

    det_lists: list[list[Detection]] = [ann.detections for ann in annotations]
    n = len(frame_paths)
    total_ops = n  # one pass per original frame

    for i, (fp, ann) in enumerate(zip(frame_paths, annotations)):
        stem = Path(fp).stem

        # Copy original
        dst_img = img_out / f"{stem}.jpg"
        dst_lbl = lbl_out / f"{stem}.txt"
        shutil.copy2(fp, dst_img)
        _write_yolo_label(dst_lbl, ann.detections)
        per_transform_counts["original"] += 1

        # Apply standard augmentations
        n_aug = eff_multiplier - 1  # -1 because original counts as 1
        if n_aug > 0:
            aug_results = apply_augmentation(fp, ann.detections, pipeline, n_aug)
            for aug_idx, (aug_img, aug_dets) in enumerate(aug_results):
                aug_stem = f"{stem}_aug{aug_idx:02d}"
                aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    str(img_out / f"{aug_stem}.jpg"),
                    aug_img_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )
                _write_yolo_label(lbl_out / f"{aug_stem}.txt", aug_dets)
                per_transform_counts["augmented"] += 1

        # Mosaic augmentation (once per original if enabled)
        if config.mosaic and n >= 4:
            try:
                mosaic_img, mosaic_dets = run_mosaic_augmentation(
                    frame_paths, det_lists, output_size=(640, 640)
                )
                mosaic_stem = f"{stem}_mosaic"
                mosaic_bgr = cv2.cvtColor(mosaic_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    str(img_out / f"{mosaic_stem}.jpg"),
                    mosaic_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )
                _write_yolo_label(lbl_out / f"{mosaic_stem}.txt", mosaic_dets)
                per_transform_counts["mosaic"] += 1
            except Exception as exc:
                logger.warning("augmentor: mosaic failed for {}: {}", fp, exc)

        if progress_callback:
            progress_callback(int((i + 1) / n * 100))
        if status_callback and i % max(1, n // 10) == 0:
            status_callback(f"Augmenting frame {i + 1}/{n}")

    total_out = sum(per_transform_counts.values())
    logger.info(
        "augmentor: {} originals → {} total images ({}x) in {}",
        n, total_out, eff_multiplier, output_dir,
    )
    return AugmentationResult(
        original_count=n,
        augmented_count=total_out,
        output_dir=str(out.resolve()),
        per_transform_counts=per_transform_counts,
    )


def _write_yolo_label(path: Path, detections: list[Detection]) -> None:
    """Write YOLO .txt label file from Detection list."""
    lines: list[str] = []
    for det in detections:
        cx, cy, w, h = det.bbox_xywhn
        lines.append(f"{det.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    path.write_text("\n".join(lines))
