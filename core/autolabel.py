"""
core/autolabel.py — YOLOv8 inference on extracted frames.

Runs batch inference, exports annotations as CVAT XML 1.1 or YOLO .txt format.
No UI imports — pure Python business logic.

Dependencies:
    pip install ultralytics
"""
from __future__ import annotations

import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable

from loguru import logger

from core.models import Detection, FrameAnnotation

# ---------------------------------------------------------------------------
# Class constants
# ---------------------------------------------------------------------------

ROVER_CLASSES: list[str] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "autorickshaw",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
    "dog",
    "cow",
    "pothole_region",
    "speed_bump",
]

# COCO class IDs that map to ROVER_CLASSES when using a stock COCO model.
# autorickshaw, pothole_region, speed_bump have no COCO equivalent — they
# require a custom fine-tuned model. Stock inference will simply not produce
# those detections; the class list is kept for CVAT label scaffolding.
_COCO_TO_ROVER: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic light",
    11: "stop sign",
    16: "dog",
    19: "cow",
}

# Module-level model cache — keyed by "<model_path>:<device>"
_MODEL_CACHE: dict[str, object] = {}

ProgressCB = Callable[[int], None]
StatusCB = Callable[[str], None]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_path: str = "yolov8n.pt",
    device: str = "auto",
) -> object:
    """Load (or retrieve from cache) a YOLOv8 model.

    Args:
        model_path: Path to .pt weights file, or a shorthand like "yolov8n.pt".
                    Shorthand names are downloaded by ultralytics on first use.
        device:     "auto" → CUDA if available else CPU.
                    "cpu", "cuda:0", etc. are passed through directly.

    Returns:
        ultralytics.YOLO instance.

    Raises:
        ImportError: if ultralytics is not installed.
        FileNotFoundError: if a custom model_path does not exist on disk.
    """
    try:
        from ultralytics import YOLO  # deferred — optional dependency
    except ImportError as exc:
        raise ImportError(
            "ultralytics is not installed. "
            "Install it with: pip install ultralytics"
        ) from exc

    resolved_device = _resolve_device(device)
    cache_key = f"{model_path}:{resolved_device}"

    if cache_key in _MODEL_CACHE:
        logger.debug("autolabel: model cache hit for {}", cache_key)
        return _MODEL_CACHE[cache_key]

    # Check for custom path existence
    p = Path(model_path)
    if p.suffix == ".pt" and not p.name.startswith("yolov8") and not p.exists():
        raise FileNotFoundError(
            f"Custom model weights not found: {model_path}. "
            "Provide a valid .pt path or use a standard name like 'yolov8n.pt'."
        )

    logger.info("autolabel: loading model {} on {}", model_path, resolved_device)
    model = YOLO(model_path)
    model.to(resolved_device)
    _MODEL_CACHE[cache_key] = model
    return model


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

def run_inference_batch(
    frame_paths: list[str],
    model: object,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    batch_size: int = 16,
    progress_callback: ProgressCB | None = None,
    status_callback: StatusCB | None = None,
) -> list[FrameAnnotation]:
    """Run YOLOv8 inference on a list of frame image paths.

    Args:
        frame_paths:       Paths to JPEG/PNG frames to annotate.
        model:             Loaded YOLO model (from load_model()).
        conf_threshold:    Minimum confidence to keep a detection.
        iou_threshold:     NMS IoU threshold.
        batch_size:        Number of images per inference call.
        progress_callback: Called with 0-100 progress int.
        status_callback:   Called with human-readable status strings.

    Returns:
        List of FrameAnnotation, one per input frame.

    Raises:
        RuntimeError: if inference fails on a batch.
    """
    if not frame_paths:
        return []

    model_version = getattr(model, "ckpt_path", str(model)) or "yolov8"
    # Use just the file name for brevity
    model_version = Path(str(model_version)).name

    annotations: list[FrameAnnotation] = []
    n = len(frame_paths)

    for batch_start in range(0, n, batch_size):
        batch = frame_paths[batch_start : batch_start + batch_size]
        if status_callback:
            status_callback(
                f"Annotating frames {batch_start + 1}–{min(batch_start + batch_size, n)} / {n}"
            )

        t0 = time.perf_counter()
        try:
            results = model(
                batch,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Inference failed on batch starting at frame {batch_start}: {exc}"
            ) from exc
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        ms_per_frame = elapsed_ms / len(batch)

        for i, result in enumerate(results):
            fp = batch[i]
            detections: list[Detection] = []

            if result.boxes is not None:
                boxes_xyxy = result.boxes.xyxy.cpu().tolist()    # [[x1,y1,x2,y2], ...]
                boxes_xywhn = result.boxes.xywhn.cpu().tolist()  # normalized
                confs = result.boxes.conf.cpu().tolist()
                cls_ids = result.boxes.cls.cpu().tolist()

                img_h, img_w = result.orig_shape[:2]

                for j in range(len(boxes_xyxy)):
                    raw_cls_id = int(cls_ids[j])
                    # Map COCO id to rover class name if possible
                    class_name = _COCO_TO_ROVER.get(
                        raw_cls_id,
                        result.names.get(raw_cls_id, str(raw_cls_id)),
                    )
                    detections.append(
                        Detection(
                            class_id=raw_cls_id,
                            class_name=class_name,
                            confidence=float(confs[j]),
                            bbox_xyxy=[float(v) for v in boxes_xyxy[j]],
                            bbox_xywhn=[float(v) for v in boxes_xywhn[j]],
                        )
                    )

            annotations.append(
                FrameAnnotation(
                    frame_path=fp,
                    detections=detections,
                    inference_time_ms=ms_per_frame,
                    model_version=model_version,
                )
            )

        if progress_callback:
            pct = int((batch_start + len(batch)) / n * 100)
            progress_callback(pct)

    logger.info(
        "autolabel: annotated {} frames, {} total detections",
        n,
        sum(len(a.detections) for a in annotations),
    )
    return annotations


# ---------------------------------------------------------------------------
# CVAT XML 1.1 export
# ---------------------------------------------------------------------------

def export_cvat_xml(
    annotations: list[FrameAnnotation],
    output_path: str,
    task_name: str = "rover_annotations",
    labels: list[str] | None = None,
) -> str:
    """Export annotations as CVAT XML 1.1 for direct import into a CVAT task.

    Args:
        annotations:  Inference output from run_inference_batch().
        output_path:  Path to write the .xml file.
        task_name:    Task name embedded in the XML metadata.
        labels:       Class names to include in <labels> block.
                      Defaults to ROVER_CLASSES.

    Returns:
        Absolute path to the written XML file.
    """
    if labels is None:
        labels = ROVER_CLASSES

    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"

    # Meta
    meta = ET.SubElement(root, "meta")
    task_el = ET.SubElement(meta, "task")
    ET.SubElement(task_el, "name").text = task_name
    labels_el = ET.SubElement(task_el, "labels")
    for lname in labels:
        lbl = ET.SubElement(labels_el, "label")
        ET.SubElement(lbl, "name").text = lname
        ET.SubElement(lbl, "color").text = "#000000"
        ET.SubElement(lbl, "attributes")

    # Images + boxes
    for img_id, ann in enumerate(annotations):
        # Try to get image dimensions from the file
        w, h = _get_image_dims(ann.frame_path)
        img_el = ET.SubElement(
            root,
            "image",
            id=str(img_id),
            name=Path(ann.frame_path).name,
            width=str(w),
            height=str(h),
        )
        for det in ann.detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            box_el = ET.SubElement(
                img_el,
                "box",
                label=det.class_name,
                xtl=f"{x1:.2f}",
                ytl=f"{y1:.2f}",
                xbr=f"{x2:.2f}",
                ybr=f"{y2:.2f}",
                occluded="0",
                z_order="0",
            )
            attr = ET.SubElement(box_el, "attribute", name="confidence")
            attr.text = f"{det.confidence:.4f}"

    # Write
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(str(out), encoding="utf-8", xml_declaration=True)
    logger.info("autolabel: CVAT XML written to {}", out)
    return str(out.resolve())


def _get_image_dims(path: str) -> tuple[int, int]:
    """Return (width, height) of an image, fallback to (0, 0) on error."""
    try:
        from PIL import Image as PILImage
        with PILImage.open(path) as img:
            return img.size  # (width, height)
    except Exception:
        return 0, 0


# ---------------------------------------------------------------------------
# YOLO format export
# ---------------------------------------------------------------------------

def export_yolo_format(
    annotations: list[FrameAnnotation],
    output_dir: str,
    class_names: list[str] | None = None,
) -> None:
    """Export annotations as YOLO .txt label files.

    Writes:
      {output_dir}/labels/{frame_stem}.txt  — one line per detection:
          class_id cx cy w h  (normalized, space-separated)
      {output_dir}/classes.txt              — class names, one per line
      {output_dir}/dataset.yaml             — stub YOLO dataset config

    Args:
        annotations:  Inference output.
        output_dir:   Root directory for YOLO export.
        class_names:  Class list. Defaults to ROVER_CLASSES.
    """
    if class_names is None:
        class_names = ROVER_CLASSES

    out = Path(output_dir)
    labels_dir = out / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Build class_id → YOLO index mapping using class_names list
    name_to_idx = {name: i for i, name in enumerate(class_names)}

    for ann in annotations:
        stem = Path(ann.frame_path).stem
        txt_path = labels_dir / f"{stem}.txt"
        lines: list[str] = []
        for det in ann.detections:
            yolo_cls = name_to_idx.get(det.class_name, det.class_id)
            cx, cy, w, h = det.bbox_xywhn
            lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        txt_path.write_text("\n".join(lines))

    # classes.txt
    (out / "classes.txt").write_text("\n".join(class_names))

    # dataset.yaml (stub — user fills in train/val paths)
    yaml_lines = [
        f"nc: {len(class_names)}",
        "names:",
    ] + [f"  - {n}" for n in class_names] + [
        "",
        "# Fill in paths below:",
        "train: images/train",
        "val:   images/val",
    ]
    (out / "dataset.yaml").write_text("\n".join(yaml_lines))
    logger.info("autolabel: YOLO format exported to {}", out)


# ---------------------------------------------------------------------------
# Annotation statistics
# ---------------------------------------------------------------------------

def compute_annotation_stats(annotations: list[FrameAnnotation]) -> dict:
    """Compute aggregate statistics over a set of FrameAnnotation objects.

    Returns a dict with:
        total_detections           int
        detections_per_class       dict[str, int]
        avg_confidence_per_class   dict[str, float]
        frames_with_no_detection   int
        avg_detections_per_frame   float
        class_distribution_percent dict[str, float]
    """
    total = 0
    per_class_count: dict[str, int] = {}
    per_class_conf_sum: dict[str, float] = {}
    frames_empty = 0

    for ann in annotations:
        if not ann.detections:
            frames_empty += 1
        for det in ann.detections:
            total += 1
            per_class_count[det.class_name] = per_class_count.get(det.class_name, 0) + 1
            per_class_conf_sum[det.class_name] = (
                per_class_conf_sum.get(det.class_name, 0.0) + det.confidence
            )

    avg_conf: dict[str, float] = {}
    for cls, cnt in per_class_count.items():
        avg_conf[cls] = per_class_conf_sum[cls] / cnt if cnt > 0 else 0.0

    dist_pct: dict[str, float] = {}
    if total > 0:
        for cls, cnt in per_class_count.items():
            dist_pct[cls] = cnt / total * 100.0

    n_frames = len(annotations)
    return {
        "total_detections": total,
        "detections_per_class": per_class_count,
        "avg_confidence_per_class": avg_conf,
        "frames_with_no_detection": frames_empty,
        "avg_detections_per_frame": total / n_frames if n_frames > 0 else 0.0,
        "class_distribution_percent": dist_pct,
    }
