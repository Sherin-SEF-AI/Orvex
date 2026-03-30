"""
core/tracker.py — ByteTrack multi-object tracker (from-scratch implementation).

No external ByteTrack/filterpy dependency.  All Kalman filter math is done
with NumPy directly.  This module is pure Python — no UI imports.

Public surface:
  KalmanBoxTracker   — per-track state machine
  BYTETracker        — two-stage IoU matching
  build_tracker()    — factory
  run_tracking_session()
  compute_tracking_statistics()
  generate_heatmap_from_tracks()
  export_tracks_mot_format()
"""
from __future__ import annotations

import colorsys
import csv
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger
from scipy.optimize import linear_sum_assignment

from core.models import (
    Detection,
    FrameAnnotation,
    TrackHistory,
    TrackingResult,
    TrackingStats,
)

# ---------------------------------------------------------------------------
# Kalman filter for a single bounding box
# ---------------------------------------------------------------------------

class KalmanBoxTracker:
    """Constant-velocity Kalman filter for axis-aligned bounding boxes.

    State vector (8-dim):  [cx, cy, w, h, dcx, dcy, dw, dh]
    Observation (4-dim):   [cx, cy, w, h]

    All matrix math is vanilla NumPy — no filterpy or scipy.kalman.
    """

    count: int = 0  # class-level monotonic ID counter

    def __init__(
        self,
        bbox_xyxy: list[float],
        class_id: int,
        class_name: str,
        conf: float,
    ) -> None:
        KalmanBoxTracker.count += 1
        self.id: int = KalmanBoxTracker.count
        self.class_id: int = class_id
        self.class_name: str = class_name
        self.confidence: float = conf

        # bookkeeping
        self.hits: int = 1
        self.hit_streak: int = 1
        self.age: int = 0
        self.time_since_update: int = 0

        # dimension of state / observation
        dim_x = 8
        dim_z = 4

        # state: [cx, cy, w, h, dcx, dcy, dw, dh]
        self.x = np.zeros((dim_x, 1), dtype=np.float64)
        cx, cy, w, h = _xyxy_to_cxcywh(bbox_xyxy)
        self.x[:4, 0] = [cx, cy, w, h]

        # State transition matrix  F  (constant velocity)
        self.F = np.eye(dim_x, dtype=np.float64)
        for i in range(dim_z):
            self.F[i, i + dim_z] = 1.0

        # Observation matrix  H  (we observe cx, cy, w, h)
        self.H = np.zeros((dim_z, dim_x), dtype=np.float64)
        for i in range(dim_z):
            self.H[i, i] = 1.0

        # Process noise  Q
        self.Q = np.eye(dim_x, dtype=np.float64)
        self.Q[4:, 4:] *= 0.01        # velocity noise is small

        # Measurement noise  R
        self.R = np.eye(dim_z, dtype=np.float64)
        self.R[2:, 2:] *= 10.0        # w, h measurement is noisier

        # Posterior covariance  P  (initialise with high uncertainty on velocity)
        self.P = np.eye(dim_x, dtype=np.float64)
        self.P[4:, 4:] *= 1000.0
        self.P *= 10.0

    # ------------------------------------------------------------------

    def predict(self) -> np.ndarray:
        """Advance state by one timestep.  Returns predicted [x1, y1, x2, y2]."""
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.age += 1

        # prevent w/h going negative
        if self.x[2, 0] + self.x[6, 0] <= 0:
            self.x[6, 0] = 0.0
        if self.x[3, 0] + self.x[7, 0] <= 0:
            self.x[7, 0] = 0.0

        # x = F @ x
        self.x = self.F @ self.x
        # P = F @ P @ F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.get_state()

    def update(self, bbox_xyxy: list[float], conf: float) -> None:
        """Correct state with a new observation."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.confidence = conf

        z = np.array(_xyxy_to_cxcywh(bbox_xyxy), dtype=np.float64).reshape(4, 1)

        # Innovation:  y = z - H @ x
        y = z - self.H @ self.x

        # Innovation covariance:  S = H @ P @ H^T + R
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain:  K = P @ H^T @ S^{-1}
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Updated state:  x = x + K @ y
        self.x = self.x + K @ y

        # Updated covariance:  P = (I - K @ H) @ P
        I = np.eye(self.P.shape[0], dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P

    def get_state(self) -> np.ndarray:
        """Return current bounding box as [x1, y1, x2, y2]."""
        return _cxcywh_to_xyxy(self.x[:4, 0].tolist())


# ---------------------------------------------------------------------------
# IoU helpers
# ---------------------------------------------------------------------------

def iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute N x M IoU matrix.

    Args:
        boxes_a: (N, 4) array of [x1, y1, x2, y2]
        boxes_b: (M, 4) array of [x1, y1, x2, y2]

    Returns:
        (N, M) float64 IoU matrix.
    """
    n = len(boxes_a)
    m = len(boxes_b)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float64)

    # broadcast to (N, M)
    # expand dims: boxes_a → (N,1,4), boxes_b → (1,M,4)
    ba = boxes_a[:, np.newaxis, :]   # (N, 1, 4)
    bb = boxes_b[np.newaxis, :, :]   # (1, M, 4)

    inter_x1 = np.maximum(ba[..., 0], bb[..., 0])
    inter_y1 = np.maximum(ba[..., 1], bb[..., 1])
    inter_x2 = np.minimum(ba[..., 2], bb[..., 2])
    inter_y2 = np.minimum(ba[..., 3], bb[..., 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    union = area_a[:, np.newaxis] + area_b[np.newaxis, :] - intersection
    union = np.maximum(union, 1e-9)

    return intersection / union


# ---------------------------------------------------------------------------
# Hungarian-based matching
# ---------------------------------------------------------------------------

def _linear_assignment(cost_matrix: np.ndarray) -> list[tuple[int, int]]:
    """Wrap scipy linear_sum_assignment, returning list of (row, col) pairs."""
    if cost_matrix.size == 0:
        return []
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return list(zip(row_ind.tolist(), col_ind.tolist()))


def _match_detections_to_tracks(
    tracks: list[KalmanBoxTracker],
    detections: np.ndarray,   # (K, 5) — [x1,y1,x2,y2,conf]
    iou_threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Return (matches, unmatched_track_ids, unmatched_det_indices).

    matches: list of (track_index, det_index) pairs that exceed iou_threshold.
    """
    if not tracks or len(detections) == 0:
        return [], list(range(len(tracks))), list(range(len(detections)))

    track_boxes = np.array([t.get_state() for t in tracks], dtype=np.float64)
    det_boxes = detections[:, :4]

    iou_mat = iou_batch(track_boxes, det_boxes)       # (T, D)
    cost_mat = 1.0 - iou_mat

    assigned = _linear_assignment(cost_mat)

    matched: list[tuple[int, int]] = []
    unmatched_tracks = set(range(len(tracks)))
    unmatched_dets = set(range(len(detections)))

    for ti, di in assigned:
        if iou_mat[ti, di] >= iou_threshold:
            matched.append((ti, di))
            unmatched_tracks.discard(ti)
            unmatched_dets.discard(di)
        # else: below threshold — leave both unmatched

    return matched, list(unmatched_tracks), list(unmatched_dets)


# ---------------------------------------------------------------------------
# BYTETracker
# ---------------------------------------------------------------------------

class BYTETracker:
    """ByteTrack two-stage multi-object tracker.

    Stage 1 — match *active* tracks against *high-confidence* detections.
    Stage 2 — match *lost* tracks against *low-confidence* detections.

    New tracks are created for unmatched high-conf detections that survive
    ``min_hits`` frames.  Tracks are removed after ``track_buffer`` frames
    without a match.
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: float = 30.0,
        min_hits: int = 3,
    ) -> None:
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        self.min_hits = min_hits

        # max frames a track can be unmatched before removal
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)

        self._active_tracks: list[KalmanBoxTracker] = []
        self._lost_tracks: list[KalmanBoxTracker] = []
        self.frame_count: int = 0

        # reset class counter so IDs are session-scoped
        KalmanBoxTracker.count = 0

    # ------------------------------------------------------------------

    def update(
        self,
        detections: np.ndarray,       # (N, 5): [x1,y1,x2,y2,conf]
        class_ids: list[int],
        class_names: list[str],
        frame_size: tuple[int, int],  # (W, H) — for clipping; not strictly required
    ) -> list[dict[str, Any]]:
        """Run one tracking step.

        Returns:
            List of dicts, one per confirmed active track:
            {track_id, bbox_xyxy, class_id, class_name, confidence}
        """
        self.frame_count += 1

        if detections is None or len(detections) == 0:
            detections = np.empty((0, 5), dtype=np.float64)
            class_ids = []
            class_names = []

        # ── Predict all existing tracks ──────────────────────────────
        for t in self._active_tracks:
            t.predict()
        for t in self._lost_tracks:
            t.predict()

        # ── Split detections by confidence ───────────────────────────
        high_mask = detections[:, 4] >= self.track_thresh if len(detections) else np.array([], bool)
        low_mask = ~high_mask if len(detections) else np.array([], bool)

        high_dets = detections[high_mask]
        low_dets = detections[low_mask]
        high_cids = [class_ids[i] for i in np.where(high_mask)[0].tolist()]
        high_cnames = [class_names[i] for i in np.where(high_mask)[0].tolist()]
        low_cids = [class_ids[i] for i in np.where(low_mask)[0].tolist()]
        low_cnames = [class_names[i] for i in np.where(low_mask)[0].tolist()]

        # ── Stage 1: match active tracks ↔ high-conf detections ──────
        matched1, unmatched_tracks1, unmatched_high = _match_detections_to_tracks(
            self._active_tracks, high_dets, self.match_thresh
        )

        for ti, di in matched1:
            self._active_tracks[ti].update(high_dets[di, :4].tolist(), float(high_dets[di, 4]))

        # ── Stage 2: match remaining active + lost ↔ low-conf dets ──
        second_pool = (
            [self._active_tracks[ti] for ti in unmatched_tracks1]
            + self._lost_tracks
        )
        n_active_unmatched = len(unmatched_tracks1)

        matched2, unmatched_pool2, unmatched_low = _match_detections_to_tracks(
            second_pool, low_dets, self.match_thresh
        )

        for pi, di in matched2:
            second_pool[pi].update(low_dets[di, :4].tolist(), float(low_dets[di, 4]))

        matched_second_pool_indices = {pi for pi, _ in matched2}

        # ── Handle unmatched active tracks from stage 1 ──────────────
        # Active tracks unmatched after both stages → move to lost
        unmatched_active_after = [
            idx for idx, pool_idx in enumerate(unmatched_tracks1)
            if idx not in matched_second_pool_indices  # stage 2 index in second_pool
        ]
        # Rebuild: which original active indices are truly unmatched
        unmatched_active_orig = []
        for pool_idx, orig_ti in enumerate(unmatched_tracks1):
            # pool_idx maps to second_pool position
            if pool_idx not in matched_second_pool_indices:
                unmatched_active_orig.append(orig_ti)

        new_lost: list[KalmanBoxTracker] = [self._active_tracks[ti] for ti in unmatched_active_orig]
        # matched active tracks (stage 1 + stage 2)
        kept_active_indices = set(range(len(self._active_tracks))) - set(unmatched_active_orig)

        # Lost tracks matched in stage 2 → bring back to active
        n_lost = len(self._lost_tracks)
        recovered_from_lost: list[KalmanBoxTracker] = []
        for pool_idx, _ in matched2:
            if pool_idx >= n_active_unmatched:
                lost_idx = pool_idx - n_active_unmatched
                recovered_from_lost.append(self._lost_tracks[lost_idx])

        recovered_lost_indices = set()
        for pool_idx, _ in matched2:
            if pool_idx >= n_active_unmatched:
                recovered_lost_indices.add(pool_idx - n_active_unmatched)

        # ── Create new tracks from unmatched high-conf detections ─────
        new_tracks: list[KalmanBoxTracker] = []
        for di in unmatched_high:
            t = KalmanBoxTracker(
                bbox_xyxy=high_dets[di, :4].tolist(),
                class_id=high_cids[di],
                class_name=high_cnames[di],
                conf=float(high_dets[di, 4]),
            )
            new_tracks.append(t)

        # ── Rebuild active list ────────────────────────────────────────
        next_active: list[KalmanBoxTracker] = []
        for ti in sorted(kept_active_indices):
            next_active.append(self._active_tracks[ti])
        next_active.extend(recovered_from_lost)
        next_active.extend(new_tracks)

        # ── Rebuild lost list ─────────────────────────────────────────
        still_lost: list[KalmanBoxTracker] = []
        for i, t in enumerate(self._lost_tracks):
            if i not in recovered_lost_indices:
                still_lost.append(t)
        still_lost.extend(new_lost)

        # prune tracks that have been lost too long
        self._lost_tracks = [
            t for t in still_lost
            if t.time_since_update <= self.max_time_lost
        ]
        self._active_tracks = next_active

        # ── Return only confirmed tracks (hit_streak >= min_hits) ─────
        output: list[dict[str, Any]] = []
        for t in self._active_tracks:
            if t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                box = t.get_state().tolist()
                output.append({
                    "track_id": t.id,
                    "bbox_xyxy": box,
                    "class_id": t.class_id,
                    "class_name": t.class_name,
                    "confidence": t.confidence,
                })

        return output


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_tracker(
    track_thresh: float = 0.5,
    track_buffer: int = 30,
    match_thresh: float = 0.8,
    frame_rate: float = 30.0,
) -> BYTETracker:
    """Create and return a BYTETracker with the given parameters."""
    return BYTETracker(
        track_thresh=track_thresh,
        track_buffer=track_buffer,
        match_thresh=match_thresh,
        frame_rate=frame_rate,
    )


# ---------------------------------------------------------------------------
# Session runner
# ---------------------------------------------------------------------------

def _track_color_bgr(track_id: int) -> tuple[int, int, int]:
    """Return a visually distinct BGR color for a given track_id.

    Hue is spaced by 37 degrees so consecutive IDs are clearly different.
    """
    hue = (track_id * 37) % 360
    r, g, b = colorsys.hsv_to_rgb(hue / 360.0, 0.8, 0.9)
    return (int(b * 255), int(g * 255), int(r * 255))  # OpenCV is BGR


def run_tracking_session(
    frame_paths: list[str],
    annotations: list,          # list[FrameAnnotation]
    tracker: BYTETracker,
    output_dir: str,
    progress_callback=None,
) -> TrackingResult:
    """Run the tracker over a sequence of pre-annotated frames.

    For each frame:
      1. Look up the matching FrameAnnotation by filename stem.
      2. Convert detections to Nx5 array.
      3. Call tracker.update().
      4. Update TrackHistory records.
      5. Draw overlay (bbox + 15-frame trail) and save to
         <output_dir>/tracking/<stem>_track.jpg.

    Args:
        frame_paths:       Ordered list of absolute image paths.
        annotations:       FrameAnnotation objects (order doesn't have to match).
        tracker:           An initialised BYTETracker (reset externally if reusing).
        output_dir:        Root directory for the session; overlays go in
                           <output_dir>/tracking/.
        progress_callback: Optional callable(int, int) — (current, total).

    Returns:
        TrackingResult with all histories populated.
    """
    # Build annotation lookup by stem (filename without extension)
    ann_by_stem: dict[str, FrameAnnotation] = {}
    for ann in annotations:
        stem = Path(ann.frame_path).stem
        ann_by_stem[stem] = ann

    out_tracking = Path(output_dir) / "tracking"
    out_tracking.mkdir(parents=True, exist_ok=True)

    # --- per-track bookkeeping ---
    histories: dict[int, _TrackBuildState] = {}

    active_tracks_per_frame: list[int] = []
    total = len(frame_paths)

    for frame_idx, frame_path in enumerate(frame_paths):
        if progress_callback is not None:
            progress_callback(frame_idx, total)

        stem = Path(frame_path).stem
        ann: FrameAnnotation | None = ann_by_stem.get(stem)

        # Build detection array
        dets = np.empty((0, 5), dtype=np.float64)
        cids: list[int] = []
        cnames: list[str] = []

        if ann and ann.detections:
            rows = []
            for det in ann.detections:
                x1, y1, x2, y2 = det.bbox_xyxy
                rows.append([x1, y1, x2, y2, det.confidence])
                cids.append(det.class_id)
                cnames.append(det.class_name)
            dets = np.array(rows, dtype=np.float64)

        # Determine frame size from image
        img = cv2.imread(frame_path)
        if img is None:
            logger.warning("tracker: cannot read frame '{}'", frame_path)
            active_tracks_per_frame.append(0)
            continue
        H, W = img.shape[:2]

        active = tracker.update(dets, cids, cnames, (W, H))
        active_tracks_per_frame.append(len(active))

        # Update histories
        for trk in active:
            tid = trk["track_id"]
            box = trk["bbox_xyxy"]
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0

            if tid not in histories:
                histories[tid] = _TrackBuildState(
                    track_id=tid,
                    class_name=trk["class_name"],
                    first_frame=frame_idx,
                    confidence_sum=trk["confidence"],
                    hit_count=1,
                    trajectory=[(cx, cy)],
                    last_frame=frame_idx,
                )
            else:
                h = histories[tid]
                h.last_frame = frame_idx
                h.confidence_sum += trk["confidence"]
                h.hit_count += 1
                h.trajectory.append((cx, cy))

        # Draw overlay
        _draw_tracking_overlay(img, active, histories, out_tracking, stem)

    if progress_callback is not None:
        progress_callback(total, total)

    # Build final TrackingResult
    track_histories: dict[int, TrackHistory] = {}
    class_track_counts: dict[str, int] = {}

    for tid, state in histories.items():
        pts = state.trajectory
        if len(pts) >= 2:
            disp = ((pts[-1][0] - pts[0][0]) ** 2 + (pts[-1][1] - pts[0][1]) ** 2) ** 0.5
        else:
            disp = 0.0

        th = TrackHistory(
            track_id=tid,
            class_name=state.class_name,
            first_frame=state.first_frame,
            last_frame=state.last_frame,
            duration_frames=state.last_frame - state.first_frame + 1,
            center_trajectory=pts,
            mean_confidence=state.confidence_sum / max(state.hit_count, 1),
            is_static=disp < 20.0,
        )
        track_histories[tid] = th
        class_track_counts[state.class_name] = class_track_counts.get(state.class_name, 0) + 1

    # Determine session_id from output_dir (best effort)
    session_id = Path(output_dir).name

    return TrackingResult(
        session_id=session_id,
        total_tracks=len(track_histories),
        active_tracks_per_frame=active_tracks_per_frame,
        track_histories=track_histories,
        class_track_counts=class_track_counts,
    )


# ---------------------------------------------------------------------------
# Overlay drawing helper
# ---------------------------------------------------------------------------

def _draw_tracking_overlay(
    img: np.ndarray,
    active: list[dict[str, Any]],
    histories: dict[int, "_TrackBuildState"],
    out_dir: Path,
    stem: str,
) -> None:
    overlay = img.copy()

    for trk in active:
        tid = trk["track_id"]
        box = trk["bbox_xyxy"]
        color = _track_color_bgr(tid)

        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        label = f"#{tid} {trk['class_name']} {trk['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(overlay, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            overlay, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )

        # Trail — last 15 center points
        if tid in histories:
            trail = histories[tid].trajectory[-15:]
            for i in range(1, len(trail)):
                pt1 = (int(trail[i - 1][0]), int(trail[i - 1][1]))
                pt2 = (int(trail[i][0]), int(trail[i][1]))
                alpha = i / len(trail)
                c = tuple(int(ch * alpha) for ch in color)
                cv2.line(overlay, pt1, pt2, c, 2)

    out_path = str(out_dir / f"{stem}_track.jpg")
    cv2.imwrite(out_path, overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_tracking_statistics(
    result: TrackingResult,
    frame_rate: float = 30.0,
) -> TrackingStats:
    """Derive all TrackingStats fields from a TrackingResult.

    Args:
        result:     Completed TrackingResult.
        frame_rate: Video frame rate (used for duration conversion).

    Returns:
        Populated TrackingStats.
    """
    histories = result.track_histories

    if not histories:
        return TrackingStats(
            total_unique_objects=0,
            mean_track_duration_seconds=0.0,
            max_simultaneous_tracks=0,
            static_objects_percent=0.0,
            objects_per_class={},
            mean_objects_per_frame=0.0,
            track_fragmentation_rate=0.0,
            high_density_frames=[],
        )

    durations_frames = [th.duration_frames for th in histories.values()]
    durations_secs = [d / max(frame_rate, 1e-9) for d in durations_frames]

    static_count = sum(1 for th in histories.values() if th.is_static)
    static_pct = 100.0 * static_count / len(histories)

    objects_per_class: dict[str, int] = {}
    for th in histories.values():
        objects_per_class[th.class_name] = objects_per_class.get(th.class_name, 0) + 1

    atpf = result.active_tracks_per_frame
    max_simult = max(atpf) if atpf else 0
    mean_per_frame = float(np.mean(atpf)) if atpf else 0.0

    # High-density frames: frames where active tracks > mean + 1 std
    if len(atpf) > 1:
        arr = np.array(atpf, dtype=float)
        threshold = arr.mean() + arr.std()
        high_density = [i for i, v in enumerate(atpf) if v > threshold]
    else:
        high_density = []

    # Track fragmentation rate: ratio of tracks shorter than 5 frames
    short_tracks = sum(1 for d in durations_frames if d < 5)
    frag_rate = short_tracks / len(histories) if histories else 0.0

    return TrackingStats(
        total_unique_objects=len(histories),
        mean_track_duration_seconds=float(np.mean(durations_secs)),
        max_simultaneous_tracks=max_simult,
        static_objects_percent=static_pct,
        objects_per_class=objects_per_class,
        mean_objects_per_frame=mean_per_frame,
        track_fragmentation_rate=frag_rate,
        high_density_frames=high_density,
    )


# ---------------------------------------------------------------------------
# Heatmap generation
# ---------------------------------------------------------------------------

def generate_heatmap_from_tracks(
    track_histories: dict,          # dict[int, TrackHistory]
    image_size: tuple[int, int],    # (W, H)
    class_filter: list[str] | None = None,
) -> np.ndarray:
    """Accumulate trajectory points into a smoothed, colorized heatmap.

    Args:
        track_histories: Mapping track_id → TrackHistory.
        image_size:      (width, height) of the source video/image.
        class_filter:    If provided, only include tracks of these classes.

    Returns:
        (H, W, 3) uint8 RGB array with jet colormap applied.
    """
    W, H = image_size
    heatmap = np.zeros((H, W), dtype=np.float32)

    for th in track_histories.values():
        if class_filter is not None and th.class_name not in class_filter:
            continue
        for cx, cy in th.center_trajectory:
            xi, yi = int(round(cx)), int(round(cy))
            if 0 <= xi < W and 0 <= yi < H:
                heatmap[yi, xi] += 1.0

    # Gaussian smoothing
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=15, sigmaY=15)

    # Normalize to 0-255
    max_val = heatmap.max()
    if max_val > 0:
        heatmap = heatmap / max_val * 255.0
    heatmap_u8 = heatmap.astype(np.uint8)

    # Apply jet colormap (OpenCV COLORMAP_JET)
    colored = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    # OpenCV returns BGR; convert to RGB for Qt display
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return colored_rgb


# ---------------------------------------------------------------------------
# MOT export
# ---------------------------------------------------------------------------

def export_tracks_mot_format(result: TrackingResult, output_path: str) -> str:
    """Write tracking results in MOT Challenge CSV format.

    Format: frame_id, track_id, x, y, w, h, conf, -1, -1, -1

    Args:
        result:      TrackingResult to export.
        output_path: Destination .csv file path.

    Returns:
        Absolute path to the written file.
    """
    rows: list[list] = []

    for tid, th in result.track_histories.items():
        for fi, (cx, cy) in enumerate(th.center_trajectory):
            # We only have centers; approximate w/h as 0 since per-frame
            # bbox isn't stored in TrackHistory (only center trajectory).
            frame_id = th.first_frame + fi
            rows.append([frame_id, tid, cx, cy, 0, 0, th.mean_confidence, -1, -1, -1])

    rows.sort(key=lambda r: (r[0], r[1]))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "track_id", "x", "y", "w", "h", "conf",
                         "cx1", "cx2", "cx3"])
        writer.writerows(rows)

    logger.info("Exported MOT CSV to '{}'", out)
    return str(out.resolve())


# ---------------------------------------------------------------------------
# Internal bookkeeping dataclass (not exported)
# ---------------------------------------------------------------------------

class _TrackBuildState:
    """Mutable accumulator used during run_tracking_session."""

    __slots__ = (
        "track_id", "class_name", "first_frame", "last_frame",
        "confidence_sum", "hit_count", "trajectory",
    )

    def __init__(
        self,
        track_id: int,
        class_name: str,
        first_frame: int,
        confidence_sum: float,
        hit_count: int,
        trajectory: list[tuple[float, float]],
        last_frame: int,
    ) -> None:
        self.track_id = track_id
        self.class_name = class_name
        self.first_frame = first_frame
        self.last_frame = last_frame
        self.confidence_sum = confidence_sum
        self.hit_count = hit_count
        self.trajectory = trajectory


# ---------------------------------------------------------------------------
# Private geometry helpers
# ---------------------------------------------------------------------------

def _xyxy_to_cxcywh(bbox: list[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1


def _cxcywh_to_xyxy(cxcywh: list[float]) -> np.ndarray:
    cx, cy, w, h = cxcywh
    hw, hh = w / 2.0, h / 2.0
    return np.array([cx - hw, cy - hh, cx + hw, cy + hh], dtype=np.float64)
