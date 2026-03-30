"""
core/lane_detector.py — Lane detection pipeline for rover datasets.

Supports two modes:
  1. Classical CV (always available): Canny + HoughLinesP + polyfit.
  2. UFLD (Ultra Fast Lane Detection): optional, requires torch + user-provided weights.

UFLD weights are NOT pip-installable; this module works fully without them.
The classical pipeline is the production default.

No UI imports.  All functions are pure Python / NumPy / OpenCV.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from loguru import logger

# ── Optional imports ─────────────────────────────────────────────────────────

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
    logger.warning(
        "OpenCV (cv2) is not installed. "
        "Install it with: pip install opencv-python>=4.10.0"
    )

try:
    import numpy as np  # type: ignore
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False
    logger.warning("NumPy is not installed.  Install it with: pip install numpy>=1.26.0")

try:
    from scipy.signal import savgol_filter as _savgol_filter  # type: ignore
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    import torch  # type: ignore
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from core.models import LaneCurve, LaneConfig, LaneDepartureStatus, LaneFrame


# ── Internal result container ─────────────────────────────────────────────────

@dataclass
class LaneDetectionResult:
    """Internal result object for a single-frame detection pass."""

    detected: bool
    lanes: list[LaneCurve] = field(default_factory=list)
    method: str = "none"          # "ufld" | "classical" | "none"
    inference_time_ms: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Dependency check
# ─────────────────────────────────────────────────────────────────────────────

def check_lane_dependencies() -> dict[str, bool]:
    """Return availability status for each runtime dependency.

    Returns:
        Dict with keys ``cv2``, ``numpy``, ``scipy``, ``torch``.
    """
    return {
        "cv2":   _HAS_CV2,
        "numpy": _HAS_NUMPY,
        "scipy": _HAS_SCIPY,
        "torch": _HAS_TORCH,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_lane_model(
    model_path: str | None = None,
    device: str = "auto",
) -> Any | None:
    """Attempt to load a UFLD torch model from *model_path*.

    Args:
        model_path: Path to a ``.pth`` or ``.pt`` weights file.  If ``None``
                    or the file does not exist, returns ``None`` so the caller
                    falls back to the classical pipeline.
        device:     ``"auto"``, ``"cpu"``, or ``"cuda"``.  ``"auto"`` picks
                    CUDA when available.

    Returns:
        A loaded torch model (``eval()`` mode) or ``None``.
    """
    if model_path is None:
        logger.info("load_lane_model: no model_path given — using classical fallback.")
        return None

    if not Path(model_path).exists():
        logger.warning(
            "load_lane_model: model file '{}' not found — using classical fallback. "
            "Provide a valid UFLD weights path to enable neural lane detection.",
            model_path,
        )
        return None

    if not _HAS_TORCH:
        logger.warning(
            "load_lane_model: PyTorch is not installed — using classical fallback. "
            "Install with: pip install torch>=2.0.0"
        )
        return None

    try:
        if device == "auto":
            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            _device = torch.device(device)

        model = torch.load(model_path, map_location=_device, weights_only=False)
        model.eval()
        logger.info("UFLD model loaded from '{}' on device '{}'.", model_path, _device)
        return model
    except Exception as exc:
        logger.warning(
            "load_lane_model: failed to load '{}' — {}. Using classical fallback.",
            model_path,
            exc,
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Classical CV pipeline
# ─────────────────────────────────────────────────────────────────────────────

def detect_lanes_classical(
    image: Any,           # np.ndarray
    roi_top_percent: float = 0.55,
    hough_threshold: int = 50,
) -> LaneDetectionResult:
    """Detect lane markings using a classical computer-vision pipeline.

    Steps:
    1. Crop the region-of-interest: the bottom ``(1 - roi_top_percent)`` of the
       image (typically the lower 45 % contains the road surface).
    2. Convert to grayscale.
    3. Gaussian blur 5×5 to suppress noise.
    4. Canny edge detection (low=50, high=150).
    5. Probabilistic Hough transform (HoughLinesP).
    6. Cluster line segments by angle:
       - angle < −10 ° → left candidate
       - angle >  +10 ° → right candidate
       (near-horizontal lines are road noise and are discarded.)
    7. Fit a 1st-degree polynomial (linear) to each cluster with ``np.polyfit``.
    8. Extrapolate sampled points from the top of the ROI to the image bottom.
    9. Package results as ``LaneCurve`` objects (marking_type="unknown").

    If OpenCV or NumPy is absent, returns a result with ``detected=False``.
    "No lines found" is a valid, non-error outcome.

    Args:
        image:             BGR image as a NumPy array (H×W×3).
        roi_top_percent:   Top boundary of the ROI expressed as a fraction of
                           image height (0.55 → road starts at 55 % from top).
        hough_threshold:   Minimum vote count for a Hough line.

    Returns:
        :class:`LaneDetectionResult`.
    """
    if not _HAS_CV2 or not _HAS_NUMPY:
        logger.error(
            "detect_lanes_classical: OpenCV and NumPy are required. "
            "Install with: pip install opencv-python numpy"
        )
        return LaneDetectionResult(detected=False, method="none")

    t0 = time.perf_counter()

    h, w = image.shape[:2]
    roi_top = int(h * roi_top_percent)

    # ── Step 1: crop ROI ──────────────────────────────────────────────────────
    roi = image[roi_top:h, 0:w]

    # ── Step 2–4: grayscale → blur → edges ────────────────────────────────────
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # ── Step 5: Hough lines ───────────────────────────────────────────────────
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=40,
        maxLineGap=20,
    )

    if lines is None or len(lines) == 0:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return LaneDetectionResult(
            detected=False,
            method="classical",
            inference_time_ms=elapsed_ms,
        )

    # ── Step 6: cluster by angle ──────────────────────────────────────────────
    left_pts: list[tuple[float, float]] = []
    right_pts: list[tuple[float, float]] = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue  # vertical — skip
        angle_deg = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle_deg < -10:
            left_pts.extend([(float(x1), float(y1)), (float(x2), float(y2))])
        elif angle_deg > 10:
            right_pts.extend([(float(x1), float(y1)), (float(x2), float(y2))])

    # ── Step 7–8: fit polynomial, generate lane curves ────────────────────────
    lanes: list[LaneCurve] = []

    def _build_curve(pts: list[tuple[float, float]], lane_type: str) -> LaneCurve | None:
        if len(pts) < 2:
            return None
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])  # y is within ROI coordinate space
        try:
            # Fit y = f(x) using a linear polynomial (degree 1)
            coeffs = np.polyfit(xs, ys, 1)
        except (np.linalg.LinAlgError, ValueError):
            return None

        # Pad polynomial to 4 coefficients (cubic format) [a, b, c, d]
        # For a linear fit f(x) = mx + b → [0, 0, m, b]
        poly4 = [0.0, 0.0, float(coeffs[0]), float(coeffs[1])]

        # Extrapolate: sample points from top of ROI to bottom of image.
        # Convert y back to full-image coordinates by adding roi_top.
        x_min = float(np.clip(np.min(xs), 0, w - 1))
        x_max = float(np.clip(np.max(xs), 0, w - 1))
        x_samples = np.linspace(x_min, x_max, 20)
        poly_fn = np.poly1d(coeffs)
        sampled_pts: list[tuple[float, float]] = [
            (float(x), float(poly_fn(x)) + roi_top)
            for x in x_samples
            if 0 <= float(poly_fn(x)) + roi_top <= h
        ]

        if not sampled_pts:
            return None

        confidence = min(1.0, len(pts) / (5.0 * 2))  # pts are pairs; 5 lines = full conf

        return LaneCurve(
            lane_type=lane_type,
            points=sampled_pts,
            polynomial=poly4,
            confidence=confidence,
            marking_type="unknown",
        )

    for _pts, _type in ((left_pts, "ego_left"), (right_pts, "ego_right")):
        curve = _build_curve(_pts, _type)
        if curve is not None:
            lanes.append(curve)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return LaneDetectionResult(
        detected=bool(lanes),
        lanes=lanes,
        method="classical",
        inference_time_ms=elapsed_ms,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Lane departure estimation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_lane_departure(
    lanes: list[LaneCurve],
    image_width: int,
    image_height: int,
) -> LaneDepartureStatus:
    """Compute lateral offset and departure status from detected lane curves.

    Finds the ``ego_left`` and ``ego_right`` curves and evaluates their
    x-positions at the bottom row of the image (highest y value in each
    ``points`` list).  The lane centre is the midpoint of those two x values.
    The lateral offset is normalised to [−1, +1] where 0 is perfectly centred.

    Status thresholds (signed offset):
    - ``|offset| < 0.15``                   → ``"centered"``
    - ``0.15 ≤ |offset| < 0.4``             → ``"drifting_left"`` / ``"drifting_right"``
    - ``|offset| ≥ 0.4``                    → ``"drifting_left"`` / ``"drifting_right"``

    If neither ego lane is found: ``status="no_lane"``, ``offset=0.0``,
    ``confidence=0.0``.

    Args:
        lanes:        List of :class:`LaneCurve` objects (may be empty).
        image_width:  Width of the source image in pixels.
        image_height: Height of the source image in pixels.

    Returns:
        :class:`LaneDepartureStatus`.
    """
    ego_left: LaneCurve | None = None
    ego_right: LaneCurve | None = None

    for lane in lanes:
        if lane.lane_type == "ego_left":
            ego_left = lane
        elif lane.lane_type == "ego_right":
            ego_right = lane

    if ego_left is None and ego_right is None:
        return LaneDepartureStatus(
            status="no_lane",
            lateral_offset_percent=0.0,
            confidence=0.0,
        )

    def _bottom_x(curve: LaneCurve) -> float:
        """Return the x of the point with the largest y (closest to bottom)."""
        return max(curve.points, key=lambda p: p[1])[0]

    img_centre_x = image_width / 2.0

    if ego_left is not None and ego_right is not None:
        lx = _bottom_x(ego_left)
        rx = _bottom_x(ego_right)
        lane_centre_x = (lx + rx) / 2.0
        lane_half_width = abs(rx - lx) / 2.0
        if lane_half_width < 1.0:
            lane_half_width = image_width / 4.0  # fallback: assume quarter-width
        lateral_offset = (lane_centre_x - img_centre_x) / lane_half_width
        confidence = (ego_left.confidence + ego_right.confidence) / 2.0
    elif ego_left is not None:
        # Only left lane — vehicle is probably to the right of centre.
        lx = _bottom_x(ego_left)
        lane_half_width = image_width / 4.0
        lateral_offset = (img_centre_x - lx - lane_half_width) / lane_half_width
        confidence = ego_left.confidence * 0.6  # single-lane estimate, lower confidence
    else:  # only right lane
        rx = _bottom_x(ego_right)  # type: ignore[union-attr]
        lane_half_width = image_width / 4.0
        lateral_offset = (rx - lane_half_width - img_centre_x) / lane_half_width
        confidence = ego_right.confidence * 0.6  # type: ignore[union-attr]

    abs_off = abs(lateral_offset)
    if abs_off < 0.15:
        status = "centered"
    elif lateral_offset < 0:
        status = "drifting_left"
    else:
        status = "drifting_right"

    return LaneDepartureStatus(
        status=status,
        lateral_offset_percent=float(lateral_offset),
        confidence=float(max(0.0, min(1.0, confidence))),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Overlay drawing
# ─────────────────────────────────────────────────────────────────────────────

def draw_lane_overlay(
    image: Any,           # np.ndarray
    lanes: list[LaneCurve],
    departure: LaneDepartureStatus,
) -> Any:                 # np.ndarray
    """Draw lane curves and a departure indicator onto a copy of *image*.

    Lane rendering:
    - ``ego_left`` / ``ego_right``: thick yellow polylines (thickness 3).
    - ``adjacent_left`` / ``adjacent_right``: thin white polylines (thickness 1).

    Departure indicator: a filled circle at the bottom-centre of the image.
    - Green  if ``centered``
    - Yellow if ``drifting_left`` or ``drifting_right``
    - Red    if ``no_lane``

    Args:
        image:     Source BGR image (not modified).
        lanes:     Lane curves to render.
        departure: Departure status for the indicator dot.

    Returns:
        A new BGR image with overlays drawn.
    """
    if not _HAS_CV2 or not _HAS_NUMPY:
        return image

    out = image.copy()
    h, w = out.shape[:2]

    _EGO_COLOR = (0, 220, 220)       # yellow in BGR
    _ADJ_COLOR = (220, 220, 220)     # white in BGR
    _EGO_THICKNESS = 3
    _ADJ_THICKNESS = 1

    for lane in lanes:
        if len(lane.points) < 2:
            continue
        pts = np.array([(int(p[0]), int(p[1])) for p in lane.points], dtype=np.int32)
        is_ego = lane.lane_type in ("ego_left", "ego_right")
        color = _EGO_COLOR if is_ego else _ADJ_COLOR
        thickness = _EGO_THICKNESS if is_ego else _ADJ_THICKNESS

        if is_ego:
            cv2.polylines(out, [pts], isClosed=False, color=color, thickness=thickness)
        else:
            # Draw adjacent lanes as dashed by skipping every other segment
            for i in range(0, len(pts) - 1, 2):
                cv2.line(out, tuple(pts[i]), tuple(pts[i + 1]), color, thickness)

    # Departure dot at bottom centre
    dot_x = w // 2
    dot_y = h - 16
    if departure.status == "centered":
        dot_color = (0, 200, 50)       # green
    elif departure.status in ("drifting_left", "drifting_right"):
        dot_color = (0, 180, 220)      # yellow
    else:
        dot_color = (0, 50, 220)       # red

    cv2.circle(out, (dot_x, dot_y), radius=10, color=dot_color, thickness=-1)
    cv2.circle(out, (dot_x, dot_y), radius=10, color=(255, 255, 255), thickness=1)

    # Offset needle: shift dot horizontally according to lateral_offset_percent
    offset_x = int(departure.lateral_offset_percent * (w / 4.0))
    needle_x = max(0, min(w - 1, dot_x + offset_x))
    cv2.line(out, (needle_x, dot_y - 14), (needle_x, dot_y + 14), (255, 255, 255), 2)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_lane_pipeline(
    frame_paths: list[str],
    model: Any | None,
    config: LaneConfig,
    output_dir: str,
    progress_callback: Callable[[int], None] | None = None,
) -> list[LaneFrame]:
    """Run the lane detection pipeline on a list of frame image paths.

    Per-frame processing:
    1. Load the image with ``cv2.imread``.
    2. If *model* is not ``None``, attempt UFLD inference.
    3. If no UFLD result (or low confidence), fall through to the classical
       pipeline when ``config.classical_fallback`` is ``True``.
    4. Compute :func:`estimate_lane_departure` from detected lanes.
    5. Draw the overlay with :func:`draw_lane_overlay`.
    6. Save the overlay to ``output_dir/overlays/{stem}_lanes.jpg``.
    7. Append a :class:`LaneFrame` to the output list.

    Args:
        frame_paths:       Ordered list of absolute frame image paths.
        model:             UFLD torch model or ``None`` for classical-only mode.
        config:            :class:`LaneConfig` controlling thresholds and ROI.
        output_dir:        Directory where overlay images will be saved.
        progress_callback: Optional callable receiving ``int`` in [0, 100].

    Returns:
        List of :class:`LaneFrame` objects, one per input frame.
    """
    if not _HAS_CV2 or not _HAS_NUMPY:
        raise RuntimeError(
            "run_lane_pipeline requires OpenCV and NumPy. "
            "Install with: pip install opencv-python>=4.10.0 numpy>=1.26.0"
        )

    overlay_dir = Path(output_dir) / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    results: list[LaneFrame] = []
    total = len(frame_paths)

    for idx, frame_path in enumerate(frame_paths):
        fp = Path(frame_path)
        if not fp.exists():
            logger.warning(
                "run_lane_pipeline: frame '{}' does not exist — skipping.", frame_path
            )
            continue

        image = cv2.imread(str(fp))
        if image is None:
            logger.warning(
                "run_lane_pipeline: cv2.imread returned None for '{}'. "
                "The file may be corrupt or in an unsupported format.",
                frame_path,
            )
            continue

        h, w = image.shape[:2]
        detection: LaneDetectionResult | None = None

        # ── UFLD inference ────────────────────────────────────────────────────
        if model is not None:
            detection = _run_ufld_inference(image, model, config)

        # ── Classical fallback ────────────────────────────────────────────────
        use_classical = (
            detection is None
            or not detection.detected
            or (detection.lanes and max(l.confidence for l in detection.lanes) < config.ufld_conf_threshold)
        )
        if use_classical and config.classical_fallback:
            detection = detect_lanes_classical(
                image,
                roi_top_percent=config.roi_top_percent,
            )

        if detection is None:
            detection = LaneDetectionResult(detected=False, method="none")

        # ── Departure estimation ──────────────────────────────────────────────
        departure = estimate_lane_departure(detection.lanes, w, h)

        # ── Overlay + save ────────────────────────────────────────────────────
        annotated = draw_lane_overlay(image, detection.lanes, departure)
        overlay_name = f"{fp.stem}_lanes.jpg"
        overlay_path = str(overlay_dir / overlay_name)
        cv2.imwrite(overlay_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])

        results.append(
            LaneFrame(
                frame_path=frame_path,
                overlay_path=overlay_path,
                lanes=detection.lanes,
                departure=departure,
                detection_method=detection.method,
                inference_time_ms=detection.inference_time_ms,
            )
        )

        if progress_callback is not None and total > 0:
            progress_callback(int((idx + 1) / total * 100))

    logger.info(
        "run_lane_pipeline: processed {} / {} frames, {} with detections.",
        len(results),
        total,
        sum(1 for r in results if r.lanes),
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_lane_departure_report(frames: list[LaneFrame]) -> dict[str, Any]:
    """Summarise lane detection results across a sequence of frames.

    Args:
        frames: List of :class:`LaneFrame` objects from :func:`run_lane_pipeline`.

    Returns:
        Dictionary with the following keys:

        - ``frames_with_lanes`` (int)
        - ``frames_without_lanes`` (int)
        - ``detection_method_distribution`` (dict[str, int])
        - ``departure_distribution`` (dict[str, int])
        - ``mean_lateral_offset`` (float)  — mean of ``lateral_offset_percent``
        - ``max_lateral_offset`` (float)   — maximum absolute offset
        - ``unmarked_road_percent`` (float) — % of frames with no lanes at all
    """
    if not frames:
        return {
            "frames_with_lanes": 0,
            "frames_without_lanes": 0,
            "detection_method_distribution": {},
            "departure_distribution": {},
            "mean_lateral_offset": 0.0,
            "max_lateral_offset": 0.0,
            "unmarked_road_percent": 0.0,
        }

    total = len(frames)
    frames_with = sum(1 for f in frames if f.lanes)
    frames_without = total - frames_with

    method_dist: dict[str, int] = {}
    depart_dist: dict[str, int] = {}
    offsets: list[float] = []

    for f in frames:
        method_dist[f.detection_method] = method_dist.get(f.detection_method, 0) + 1
        depart_dist[f.departure.status] = depart_dist.get(f.departure.status, 0) + 1
        offsets.append(f.departure.lateral_offset_percent)

    mean_offset = float(sum(offsets) / len(offsets)) if offsets else 0.0
    max_offset = float(max(abs(o) for o in offsets)) if offsets else 0.0
    unmarked_pct = (frames_without / total * 100.0) if total > 0 else 0.0

    return {
        "frames_with_lanes": frames_with,
        "frames_without_lanes": frames_without,
        "detection_method_distribution": method_dist,
        "departure_distribution": depart_dist,
        "mean_lateral_offset": mean_offset,
        "max_lateral_offset": max_offset,
        "unmarked_road_percent": round(unmarked_pct, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_ufld_inference(
    image: Any,  # np.ndarray
    model: Any,
    config: LaneConfig,
) -> LaneDetectionResult:
    """Run UFLD inference on a single image.

    This is a best-effort wrapper around a generic torch lane-detection model.
    Because UFLD model variants differ, we attempt a standard call pattern and
    handle failures gracefully so the caller can fall back to the classical
    pipeline.

    Args:
        image:  BGR NumPy image.
        model:  Loaded torch model (``eval()`` mode).
        config: :class:`LaneConfig` for threshold.

    Returns:
        :class:`LaneDetectionResult` — ``detected=False`` on any failure.
    """
    if not _HAS_TORCH or not _HAS_CV2 or not _HAS_NUMPY:
        return LaneDetectionResult(detected=False, method="ufld")

    t0 = time.perf_counter()
    try:
        # Resize to the canonical UFLD input size (800×288)
        input_h, input_w = 288, 800
        resized = cv2.resize(image, (input_w, input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = (
            torch.from_numpy(rgb.transpose(2, 0, 1))
            .float()
            .unsqueeze(0)
            .div(255.0)
        )

        with torch.no_grad():
            out = model(tensor)

        # UFLD output is typically a list/tensor of row anchors × lane probabilities.
        # Without the exact model arch we cannot decode it; return detected=False
        # so the caller falls back to classical.
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.debug(
            "_run_ufld_inference: inference succeeded but output decoding is "
            "model-specific (shape={}). Using classical fallback.",
            getattr(out, "shape", type(out)),
        )
        return LaneDetectionResult(
            detected=False,
            method="ufld",
            inference_time_ms=elapsed_ms,
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.warning(
            "_run_ufld_inference: inference failed — {}. "
            "Check that the model is a valid UFLD weights file.",
            exc,
        )
        return LaneDetectionResult(
            detected=False,
            method="ufld",
            inference_time_ms=elapsed_ms,
        )
