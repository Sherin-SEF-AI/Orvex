"""
labelox/core/tracker.py — Cross-frame tracking for annotation sequences.

Uses IoU-based matching + Kalman filter for bbox propagation.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Callable

import numpy as np
from loguru import logger

from labelox.core.models import Annotation, AnnotationType, BBoxAnnotation

ProgressCB = Callable[[int, int], None]


# ─── Kalman Box Tracker ─────────────────────────────────────────────────────

class KalmanBoxTracker:
    """Simple Kalman filter for bbox tracking (normalised coords)."""

    _count = 0

    def __init__(self, bbox: BBoxAnnotation, track_id: int | None = None) -> None:
        # State: [cx, cy, w, h, dcx, dcy, dw, dh]
        cx = bbox.x + bbox.width / 2
        cy = bbox.y + bbox.height / 2
        self.state = np.array([cx, cy, bbox.width, bbox.height, 0, 0, 0, 0], dtype=np.float64)
        self.age = 0
        self.hits = 1
        self.time_since_update = 0

        if track_id is not None:
            self.id = track_id
        else:
            KalmanBoxTracker._count += 1
            self.id = KalmanBoxTracker._count

    def predict(self) -> BBoxAnnotation:
        """Predict next position using constant velocity model."""
        self.state[:4] += self.state[4:]
        self.age += 1
        self.time_since_update += 1
        cx, cy, w, h = self.state[:4]
        w = max(0.01, w)
        h = max(0.01, h)
        return BBoxAnnotation(x=cx - w / 2, y=cy - h / 2, width=w, height=h)

    def update(self, bbox: BBoxAnnotation) -> None:
        """Update state with observed bbox."""
        cx = bbox.x + bbox.width / 2
        cy = bbox.y + bbox.height / 2
        new_state = np.array([cx, cy, bbox.width, bbox.height], dtype=np.float64)
        velocity = new_state - self.state[:4]
        # Exponential moving average for velocity
        self.state[4:] = 0.7 * self.state[4:] + 0.3 * velocity
        self.state[:4] = new_state
        self.hits += 1
        self.time_since_update = 0

    def get_bbox(self) -> BBoxAnnotation:
        cx, cy, w, h = self.state[:4]
        w = max(0.01, w)
        h = max(0.01, h)
        return BBoxAnnotation(x=cx - w / 2, y=cy - h / 2, width=w, height=h)


# ─── IoU Matching ────────────────────────────────────────────────────────────

def _iou(a: BBoxAnnotation, b: BBoxAnnotation) -> float:
    ax1, ay1 = a.x, a.y
    ax2, ay2 = a.x + a.width, a.y + a.height
    bx1, by1 = b.x, b.y
    bx2, by2 = b.x + b.width, b.y + b.height

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


def _match_detections(
    trackers: list[KalmanBoxTracker],
    detections: list[BBoxAnnotation],
    iou_threshold: float = 0.3,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian matching between trackers and detections.

    Returns (matches, unmatched_trackers, unmatched_detections).
    """
    if not trackers or not detections:
        return [], list(range(len(trackers))), list(range(len(detections)))

    n_t = len(trackers)
    n_d = len(detections)
    iou_matrix = np.zeros((n_t, n_d))

    for t in range(n_t):
        for d in range(n_d):
            iou_matrix[t, d] = _iou(trackers[t].get_bbox(), detections[d])

    # Greedy matching (good enough for annotation; avoids scipy dependency)
    matches = []
    matched_t = set()
    matched_d = set()

    while True:
        if not np.any(iou_matrix > iou_threshold):
            break
        t, d = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
        if iou_matrix[t, d] < iou_threshold:
            break
        matches.append((int(t), int(d)))
        matched_t.add(int(t))
        matched_d.add(int(d))
        iou_matrix[t, :] = 0
        iou_matrix[:, d] = 0

    unmatched_t = [i for i in range(n_t) if i not in matched_t]
    unmatched_d = [i for i in range(n_d) if i not in matched_d]
    return matches, unmatched_t, unmatched_d


# ─── Propagation ─────────────────────────────────────────────────────────────

def propagate_annotations_forward(
    current_annotations: list[Annotation],
    next_image_path: str,
    next_image_id: str,
    detector_model: Any | None = None,
    strategy: str = "interpolate",
) -> list[Annotation]:
    """Predict annotations for next frame given current frame annotations.

    strategy:
    - "interpolate": Kalman predict (fast, no re-detection)
    - "detect_and_match": Run YOLO + match to tracks
    - "copy": Direct copy (no motion model)
    """
    now = datetime.utcnow()

    if strategy == "copy":
        result = []
        for ann in current_annotations:
            new = ann.model_copy(deep=True)
            new.id = str(uuid.uuid4())
            new.image_id = next_image_id
            new.is_auto = True
            new.created_at = now
            new.updated_at = now
            result.append(new)
        return result

    # Build trackers from current annotations
    trackers: list[tuple[KalmanBoxTracker, Annotation]] = []
    for ann in current_annotations:
        if ann.bbox:
            tracker = KalmanBoxTracker(ann.bbox, track_id=ann.track_id)
            trackers.append((tracker, ann))

    if strategy == "interpolate":
        result = []
        for tracker, ann in trackers:
            pred_bbox = tracker.predict()
            new = ann.model_copy(deep=True)
            new.id = str(uuid.uuid4())
            new.image_id = next_image_id
            new.bbox = pred_bbox
            new.is_auto = True
            new.confidence = None
            new.created_at = now
            new.updated_at = now
            result.append(new)
        return result

    if strategy == "detect_and_match" and detector_model is not None:
        from labelox.core.auto_annotator import run_yolo_on_image

        # Run detection on next frame
        detections = run_yolo_on_image(
            next_image_path, detector_model,
            label_classes=[{"name": ann.label_name} for _, ann in trackers],
        )
        det_bboxes = [d.bbox for d in detections if d.bbox]

        # Predict tracker positions
        pred_bboxes = [t.predict() for t, _ in trackers]

        # Match
        tracker_objs = [t for t, _ in trackers]
        matches, unmatched_t, unmatched_d = _match_detections(
            tracker_objs, det_bboxes,
        )

        result = []
        for t_idx, d_idx in matches:
            tracker, ann = trackers[t_idx]
            tracker.update(det_bboxes[d_idx])
            new = ann.model_copy(deep=True)
            new.id = str(uuid.uuid4())
            new.image_id = next_image_id
            new.bbox = tracker.get_bbox()
            new.is_auto = True
            new.track_id = tracker.id
            new.confidence = detections[d_idx].confidence
            new.created_at = now
            new.updated_at = now
            result.append(new)

        # Unmatched trackers — use predicted position
        for t_idx in unmatched_t:
            tracker, ann = trackers[t_idx]
            if tracker.time_since_update <= 3:
                new = ann.model_copy(deep=True)
                new.id = str(uuid.uuid4())
                new.image_id = next_image_id
                new.bbox = tracker.get_bbox()
                new.is_auto = True
                new.track_id = tracker.id
                new.confidence = None
                new.created_at = now
                new.updated_at = now
                result.append(new)

        return result

    # Fallback: copy
    return propagate_annotations_forward(
        current_annotations, next_image_path, next_image_id, strategy="copy",
    )


# ─── Sequence Tracking ──────────────────────────────────────────────────────

def track_sequence(
    sequence_images: list[tuple[str, str]],  # [(image_id, image_path), ...]
    initial_annotations: list[Annotation],
    detector_model: Any | None = None,
    strategy: str = "detect_and_match",
    progress_callback: ProgressCB | None = None,
) -> dict[str, list[Annotation]]:
    """Track annotations through a sequence starting from first frame.

    Returns: {image_id: [annotations]}
    """
    if not sequence_images:
        return {}

    result: dict[str, list[Annotation]] = {}
    current_anns = initial_annotations

    for i, (img_id, img_path) in enumerate(sequence_images):
        if i == 0:
            result[img_id] = current_anns
        else:
            next_anns = propagate_annotations_forward(
                current_anns, img_path, img_id,
                detector_model=detector_model,
                strategy=strategy,
            )
            result[img_id] = next_anns
            current_anns = next_anns

        if progress_callback:
            progress_callback(i + 1, len(sequence_images))

    return result


# ─── Keyframe Interpolation ─────────────────────────────────────────────────

def interpolate_between_keyframes(
    keyframe_a: tuple[str, list[Annotation]],  # (image_id, annotations)
    keyframe_b: tuple[str, list[Annotation]],
    intermediate_image_ids: list[str],
) -> dict[str, list[Annotation]]:
    """Linear interpolation of bbox positions between two keyframes.

    Matches annotations by track_id between keyframes.
    """
    id_a, anns_a = keyframe_a
    id_b, anns_b = keyframe_b

    # Build track maps
    tracks_a = {a.track_id: a for a in anns_a if a.track_id is not None and a.bbox}
    tracks_b = {a.track_id: a for a in anns_b if a.track_id is not None and a.bbox}

    common_tracks = set(tracks_a.keys()) & set(tracks_b.keys())
    n = len(intermediate_image_ids) + 1  # number of intervals

    result: dict[str, list[Annotation]] = {}
    now = datetime.utcnow()

    for i, img_id in enumerate(intermediate_image_ids):
        t = (i + 1) / n  # 0 < t < 1
        frame_anns: list[Annotation] = []

        for tid in common_tracks:
            a_ann = tracks_a[tid]
            b_ann = tracks_b[tid]
            a_box = a_ann.bbox
            b_box = b_ann.bbox

            # Linear interpolation
            interp_box = BBoxAnnotation(
                x=a_box.x + t * (b_box.x - a_box.x),
                y=a_box.y + t * (b_box.y - a_box.y),
                width=a_box.width + t * (b_box.width - a_box.width),
                height=a_box.height + t * (b_box.height - a_box.height),
            )

            frame_anns.append(Annotation(
                image_id=img_id,
                label_id=a_ann.label_id,
                label_name=a_ann.label_name,
                annotation_type=AnnotationType.BBOX,
                bbox=interp_box,
                track_id=tid,
                is_auto=True,
                created_by="interpolation",
                created_at=now,
                updated_at=now,
            ))

        result[img_id] = frame_anns

    return result
