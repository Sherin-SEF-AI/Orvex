"""
core/active_learning.py — Frame selection for efficient labeling.

Scores extracted frames by model uncertainty and visual diversity,
then selects the most informative subset to send for human annotation.

No UI imports — pure Python business logic.

Dependencies:
    pip install imagehash Pillow
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import numpy as np
from loguru import logger

from core.models import FrameAnnotation, UncertaintyScore

ProgressCB = Callable[[int], None]
StatusCB = Callable[[str], None]


# ---------------------------------------------------------------------------
# Uncertainty scoring
# ---------------------------------------------------------------------------

def compute_uncertainty_scores(
    annotations: list[FrameAnnotation],
    method: str = "entropy",
) -> list[UncertaintyScore]:
    """Score each frame by model uncertainty using annotation confidence values.

    Args:
        annotations: FrameAnnotation list from autolabel.run_inference_batch().
        method:      "entropy"          — Shannon entropy of confidence scores.
                     "margin"           — 1 - (top1_conf - top2_conf).
                     "least_confidence" — 1 - max_confidence.

    Returns:
        List of UncertaintyScore, parallel to input annotations.

    Raises:
        ValueError: if method is unknown.
    """
    valid_methods = ("entropy", "margin", "least_confidence")
    if method not in valid_methods:
        raise ValueError(
            f"Unknown uncertainty method '{method}'. "
            f"Choose one of: {valid_methods}"
        )

    scores: list[UncertaintyScore] = []
    for ann in annotations:
        dets = ann.detections
        n_dets = len(dets)
        high_unc = sum(1 for d in dets if d.confidence < 0.5)

        if n_dets == 0:
            # No detections: model is uncertain (could be empty or missed)
            score = 0.5
        elif method == "entropy":
            score = _entropy_score(dets)
        elif method == "margin":
            score = _margin_score(dets)
        else:  # least_confidence
            score = _least_confidence_score(dets)

        scores.append(
            UncertaintyScore(
                frame_path=ann.frame_path,
                score=float(score),
                method=method,
                num_detections=n_dets,
                high_uncertainty_detections=high_unc,
            )
        )

    logger.info(
        "active_learning: uncertainty scored {} frames via '{}'",
        len(scores),
        method,
    )
    return scores


def _entropy_score(detections: list) -> float:
    """Mean Shannon entropy of confidence scores across all detections."""
    entropies: list[float] = []
    for det in detections:
        p = det.confidence
        # Binary entropy: -p*log(p) - (1-p)*log(1-p)
        e = -(p * math.log(p + 1e-9) + (1 - p) * math.log(1 - p + 1e-9))
        entropies.append(e)
    return float(np.mean(entropies)) if entropies else 0.5


def _margin_score(detections: list) -> float:
    """Mean margin score: 1 - (top1_conf - top2_conf).
    For single-detection frames, margin = 1 - confidence."""
    margins: list[float] = []
    if len(detections) >= 2:
        sorted_confs = sorted([d.confidence for d in detections], reverse=True)
        margin = 1.0 - (sorted_confs[0] - sorted_confs[1])
        margins.append(margin)
    elif len(detections) == 1:
        margins.append(1.0 - detections[0].confidence)
    return float(np.mean(margins)) if margins else 0.5


def _least_confidence_score(detections: list) -> float:
    """1 - max confidence across all detections."""
    if not detections:
        return 0.5
    max_conf = max(d.confidence for d in detections)
    return 1.0 - max_conf


# ---------------------------------------------------------------------------
# Diversity scoring
# ---------------------------------------------------------------------------

def compute_diversity_scores(
    frame_paths: list[str],
    embeddings: np.ndarray | None = None,
    progress_callback: ProgressCB | None = None,
) -> list[float]:
    """Score frames by visual diversity to avoid redundant labeling.

    If embeddings are provided (Nx D float32 array), uses cosine distance
    from cluster centers.  Otherwise uses perceptual hashing (imagehash).

    Args:
        frame_paths:  Paths to frame images.
        embeddings:   Optional Nx D visual embeddings (e.g. from a CNN).
        progress_callback: 0-100 progress.

    Returns:
        Diversity score per frame (higher = more visually unique).
    """
    if not frame_paths:
        return []

    if embeddings is not None and len(embeddings) == len(frame_paths):
        return _diversity_from_embeddings(embeddings)

    return _diversity_from_phash(frame_paths, progress_callback)


def _diversity_from_phash(
    frame_paths: list[str],
    progress_callback: ProgressCB | None = None,
) -> list[float]:
    """Compute diversity via perceptual hash Hamming distances."""
    try:
        import imagehash
        from PIL import Image as PILImage
    except ImportError as exc:
        raise ImportError(
            "imagehash is not installed. "
            "Install with: pip install imagehash Pillow"
        ) from exc

    n = len(frame_paths)
    hashes: list[int] = []

    for i, fp in enumerate(frame_paths):
        try:
            with PILImage.open(fp) as img:
                h = imagehash.phash(img)
                hashes.append(int(h.hash.flatten().dot(
                    2 ** np.arange(h.hash.size, dtype=np.uint64)
                )))
        except Exception:
            hashes.append(0)

        if progress_callback and i % max(1, n // 20) == 0:
            progress_callback(int(i / n * 100))

    # Greedy farthest-point diversity: score = min Hamming distance to
    # all previously selected frames (higher = more diverse)
    hash_arr = np.array(hashes, dtype=np.uint64)
    diversity: list[float] = []

    # First frame: max diversity by convention
    selected_hashes: list[int] = [hashes[0]]
    diversity.append(1.0)

    for i in range(1, n):
        min_dist = min(
            _hamming_distance(hashes[i], sh) for sh in selected_hashes
        )
        # Normalize by hash length (64 bits)
        score = min_dist / 64.0
        diversity.append(float(score))
        selected_hashes.append(hashes[i])

    # Normalize to [0, 1]
    arr = np.array(diversity, dtype=float)
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr.tolist()


def _hamming_distance(a: int, b: int) -> int:
    """Count differing bits between two integers."""
    xor = a ^ b
    count = 0
    while xor:
        count += xor & 1
        xor >>= 1
    return count


def _diversity_from_embeddings(embeddings: np.ndarray) -> list[float]:
    """Cosine-distance-based diversity from pre-computed embeddings."""
    # Normalize rows
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = embeddings / norms

    # For each frame, score = mean cosine distance to all others
    # Use dot product: similarity = normed @ normed.T
    sim = normed @ normed.T  # N x N
    # Distance = 1 - similarity; mean across row excluding self
    n = len(embeddings)
    np.fill_diagonal(sim, 1.0)  # exclude self
    dist_sum = (1.0 - sim).sum(axis=1) - 0.0  # self already excluded
    scores = dist_sum / max(n - 1, 1)
    # Normalize to [0, 1]
    if scores.max() > 0:
        scores = scores / scores.max()
    return scores.tolist()


# ---------------------------------------------------------------------------
# Frame selection
# ---------------------------------------------------------------------------

def select_frames_for_labeling(
    uncertainty_scores: list[UncertaintyScore],
    diversity_scores: list[float],
    n_frames: int,
    uncertainty_weight: float = 0.6,
    diversity_weight: float = 0.4,
) -> list[str]:
    """Select the top N frames most worth labeling.

    Combined score: final = uncertainty_weight * unc + diversity_weight * div

    Args:
        uncertainty_scores: From compute_uncertainty_scores().
        diversity_scores:   From compute_diversity_scores(), parallel list.
        n_frames:           Number of frames to select.
        uncertainty_weight: Weight for uncertainty component.
        diversity_weight:   Weight for diversity component.

    Returns:
        List of frame paths, sorted best-first.

    Raises:
        ValueError: if score lists have different lengths.
    """
    if len(uncertainty_scores) != len(diversity_scores):
        raise ValueError(
            f"uncertainty_scores ({len(uncertainty_scores)}) and "
            f"diversity_scores ({len(diversity_scores)}) must have the same length."
        )

    if n_frames <= 0:
        return []

    combined: list[tuple[float, str]] = []
    for unc, div in zip(uncertainty_scores, diversity_scores):
        score = uncertainty_weight * unc.score + diversity_weight * div
        combined.append((score, unc.frame_path))

    combined.sort(key=lambda x: x[0], reverse=True)
    selected = [fp for _, fp in combined[:n_frames]]
    logger.info(
        "active_learning: selected {}/{} frames for labeling",
        len(selected),
        len(uncertainty_scores),
    )
    return selected


# ---------------------------------------------------------------------------
# Budget estimation
# ---------------------------------------------------------------------------

def compute_label_budget_estimate(
    total_frames: int,
    selected_frames: int,
    avg_annotation_time_minutes: float = 2.0,
) -> dict:
    """Estimate annotation cost and time for the selected frame set.

    Args:
        total_frames:                Total frames in the dataset.
        selected_frames:             Frames chosen for labeling.
        avg_annotation_time_minutes: Industry average per frame.

    Returns:
        dict with keys:
            selected_frames            int
            skipped_frames             int
            estimated_annotation_hours float
            estimated_cost_usd         float  ($0.05/annotation estimate)
            coverage_percent           float
    """
    hours = selected_frames * avg_annotation_time_minutes / 60.0
    cost_usd = selected_frames * 0.05  # $0.05 per annotation (industry estimate)
    coverage = (selected_frames / total_frames * 100.0) if total_frames > 0 else 0.0

    return {
        "selected_frames": selected_frames,
        "skipped_frames": max(0, total_frames - selected_frames),
        "estimated_annotation_hours": round(hours, 2),
        "estimated_cost_usd": round(cost_usd, 2),
        "coverage_percent": round(coverage, 1),
    }
