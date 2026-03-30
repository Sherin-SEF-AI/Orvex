"""
core/occupancy.py — Bird's-Eye-View occupancy grid pipeline.

Converts monocular depth maps (from Depth-Anything or similar) and
2D detections into a top-down occupancy grid suitable for rover navigation
and obstacle awareness.

Pipeline per frame:
  depth map (.npy or .png)
  → back-project to 3D point cloud (camera frame)
  → project to BEV occupancy grid
  → fuse detected bboxes as occupied regions
  → visualize as coloured RGB overhead map
  → temporal fusion over a sliding window

All outputs are written to disk.  No UI imports.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

import numpy as np
from loguru import logger

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    logger.warning("cv2 not available — PNG depth loading will fail gracefully.")

from core.models import (
    CalibrationResult,
    Detection,
    DepthResult,
    FrameAnnotation,
    OccupancyConfig,
    OccupancyFrame,
)

# ---------------------------------------------------------------------------
# Class-specific obstacle radii for bbox-to-grid projection
# ---------------------------------------------------------------------------
_CLASS_RADIUS_M: dict[str, float] = {
    "person":     0.5,
    "car":        2.0,
    "truck":      3.0,
    "bus":        3.0,
    "motorcycle": 1.0,
    "bicycle":    0.8,
}
_DEFAULT_RADIUS_M = 1.5

# ---------------------------------------------------------------------------
# Grid cell values
# ---------------------------------------------------------------------------
FREE     = 0.0
UNKNOWN  = 0.5
OCCUPIED = 1.0


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def depth_to_point_cloud(
    depth_map: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    max_depth_m: float = 30.0,
    depth_scale: float = 1.0,
) -> np.ndarray:
    """Back-project a depth map to a 3D point cloud in camera frame.

    Camera convention: Z=forward, X=right, Y=down.

    Args:
        depth_map:    H×W float array of depth values (before depth_scale).
        fx, fy:       Focal lengths in pixels.
        cx, cy:       Principal point in pixels.
        max_depth_m:  Points beyond this distance are discarded.
        depth_scale:  Multiply raw depth values by this to get metres.

    Returns:
        Nx3 float32 array of (X, Y, Z) points.  Empty array if no valid points.
    """
    if depth_map.ndim != 2:
        raise ValueError(
            f"depth_map must be 2-D (H×W), got shape {depth_map.shape}."
        )
    if fx == 0.0 or fy == 0.0:
        h, w = depth_map.shape
        fx = fy = max(h, w) * 0.8
        cx, cy = w / 2.0, h / 2.0
        logger.warning(
            "fx/fy are 0 — using default pinhole estimate fx=fy={:.1f}, "
            "cx={:.1f}, cy={:.1f}",
            fx, cx, cy,
        )

    depth = depth_map.astype(np.float32) * depth_scale

    h, w = depth.shape
    u_coords, v_coords = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )

    # Valid mask: positive and within max range
    valid = (depth > 0.0) & (depth <= max_depth_m)
    z = depth[valid]
    u = u_coords[valid]
    v = v_coords[valid]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack([x, y, z], axis=1).astype(np.float32)
    return points


def point_cloud_to_bev(
    points: np.ndarray,
    grid_resolution_m: float = 0.1,
    grid_width_m: float = 20.0,
    grid_height_m: float = 30.0,
    camera_height_m: float = 1.0,
) -> np.ndarray:
    """Project a 3D point cloud to a BEV occupancy grid.

    Grid layout:
      - Camera is placed at the bottom-centre of the grid.
      - X-axis (right/left) maps to grid columns.
      - Z-axis (forward) maps to grid rows (row 0 = farthest from camera).
      - Y-axis (down in camera frame) encodes height above ground.

    Cell values:
      0.0 (FREE)    — ground-plane point (Y ∈ [cam_height−0.3, cam_height+0.3])
      0.5 (UNKNOWN) — unvisited
      1.0 (OCCUPIED) — non-ground point above 0.1 m clearance

    Args:
        points:             Nx3 (X, Y, Z) float32 array in camera frame.
        grid_resolution_m:  Metres per cell.
        grid_width_m:       Total grid width in metres (centred on camera).
        grid_height_m:      Total grid depth in metres (forward from camera).
        camera_height_m:    Camera height above ground in metres.

    Returns:
        H×W float32 array with values in {0.0, 0.5, 1.0}.
        H = grid_height_m / grid_resolution_m
        W = grid_width_m  / grid_resolution_m
    """
    grid_w = int(round(grid_width_m  / grid_resolution_m))
    grid_h = int(round(grid_height_m / grid_resolution_m))
    grid = np.full((grid_h, grid_w), UNKNOWN, dtype=np.float32)

    if points.shape[0] == 0:
        return grid

    x_pts = points[:, 0]
    y_pts = points[:, 1]   # down in camera = toward ground
    z_pts = points[:, 2]   # forward

    # Camera origin is at the bottom-centre of the grid
    col_origin = grid_w // 2
    row_origin = grid_h      # rows grow from bottom; Z forward reduces row index

    # Convert world coords to grid indices
    col_idx = (x_pts / grid_resolution_m + col_origin).astype(np.int32)
    row_idx = (row_origin - z_pts / grid_resolution_m).astype(np.int32)

    # Keep only indices that fall within the grid
    in_bounds = (
        (col_idx >= 0) & (col_idx < grid_w) &
        (row_idx >= 0) & (row_idx < grid_h)
    )
    col_idx = col_idx[in_bounds]
    row_idx = row_idx[in_bounds]
    y_filtered = y_pts[in_bounds]

    # Ground plane tolerance: Y in camera frame ≈ camera_height_m when on floor
    ground_low  = camera_height_m - 0.3
    ground_high = camera_height_m + 0.3

    is_ground = (y_filtered >= ground_low) & (y_filtered <= ground_high)

    # Mark ground cells FREE first, then occupied cells on top
    # (so that obstacles override ground marks)
    ground_rows = row_idx[is_ground]
    ground_cols = col_idx[is_ground]
    grid[ground_rows, ground_cols] = FREE

    obs_rows = row_idx[~is_ground]
    obs_cols = col_idx[~is_ground]
    grid[obs_rows, obs_cols] = OCCUPIED

    return grid


def fuse_detections_into_grid(
    grid: np.ndarray,
    detections: list,  # list[Detection]
    depth_map: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    grid_resolution_m: float = 0.1,
    grid_width_m: float = 20.0,
    camera_height_m: float = 1.0,
    max_depth_m: float = 30.0,
) -> np.ndarray:
    """Mark detected object bounding boxes as OCCUPIED in the BEV grid.

    Samples the depth at the centre of each detection's bbox, estimates the
    3D position in camera frame, then marks a class-specific disc radius as
    OCCUPIED in the grid.

    Args:
        grid:               H×W float32 BEV grid (modified in place).
        detections:         list[Detection] from FrameAnnotation.
        depth_map:          H×W float depth array (same as used for BEV).
        fx, fy, cx, cy:     Camera intrinsics.
        grid_resolution_m:  Metres per grid cell.
        grid_width_m:       Total grid width in metres.
        camera_height_m:    Camera height above ground.
        max_depth_m:        Ignore detections whose depth sample exceeds this.

    Returns:
        Modified grid (same object, also returned for convenience).
    """
    if not detections or depth_map is None:
        return grid

    grid_h, grid_w = grid.shape
    col_origin = grid_w // 2
    row_origin = grid_h

    dh, dw = depth_map.shape

    for det in detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        px = int(round((x1 + x2) / 2.0))
        py = int(round((y1 + y2) / 2.0))

        # Clamp to depth map bounds
        px = max(0, min(dw - 1, px))
        py = max(0, min(dh - 1, py))

        depth_val = float(depth_map[py, px])
        if depth_val <= 0.0 or depth_val > max_depth_m:
            continue

        # 3D position in camera frame
        x3d = (px - cx) * depth_val / fx
        z3d = depth_val

        # BEV grid position
        col_c = int(round(x3d / grid_resolution_m + col_origin))
        row_c = int(round(row_origin - z3d / grid_resolution_m))

        if not (0 <= col_c < grid_w and 0 <= row_c < grid_h):
            continue

        radius_m = _CLASS_RADIUS_M.get(det.class_name.lower(), _DEFAULT_RADIUS_M)
        radius_cells = max(1, int(round(radius_m / grid_resolution_m)))

        # Draw filled circle of occupied cells
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                if dr * dr + dc * dc <= radius_cells * radius_cells:
                    rr = row_c + dr
                    cc = col_c + dc
                    if 0 <= rr < grid_h and 0 <= cc < grid_w:
                        grid[rr, cc] = OCCUPIED

    return grid


def temporal_grid_fusion(
    grids: list[np.ndarray],
    decay_factor: float = 0.95,
) -> np.ndarray:
    """Fuse a list of BEV grids with exponential temporal decay.

    The most recent grid (last in the list) has weight 1.0.  Each older
    grid's weight is multiplied by decay_factor per step back.

    Unknown cells (0.5) are excluded from the weighted sum so they do not
    pull occupied/free probabilities toward 0.5.

    Args:
        grids:        List of H×W float32 grids, oldest first.
        decay_factor: Weight multiplier per step back in time (0.5–1.0).

    Returns:
        Fused H×W float32 grid, clipped to [0, 1].
    """
    if not grids:
        raise ValueError("temporal_grid_fusion: grids list is empty.")

    if len(grids) == 1:
        return grids[0].copy()

    n = len(grids)
    # weights: most recent = 1.0, each older step * decay_factor
    weights = [decay_factor ** (n - 1 - i) for i in range(n)]

    h, w = grids[0].shape
    fused = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)

    for g, wt in zip(grids, weights):
        if g.shape != (h, w):
            logger.warning(
                "temporal_grid_fusion: grid shape mismatch {} vs {}; skipping.",
                g.shape, (h, w),
            )
            continue
        known_mask = (g != UNKNOWN)
        fused       += np.where(known_mask, g * wt, 0.0)
        weight_sum  += np.where(known_mask, wt,     0.0)

    # Where no frame ever visited, leave as UNKNOWN
    visited = weight_sum > 0.0
    result = np.full((h, w), UNKNOWN, dtype=np.float32)
    result[visited] = np.clip(fused[visited] / weight_sum[visited], 0.0, 1.0)
    return result


def generate_bev_visualization(
    grid: np.ndarray,
    grid_resolution_m: float = 0.1,
) -> np.ndarray:
    """Render a BEV occupancy grid as an RGB image.

    Colour scheme (from theme palette):
      FREE     (≤ 0.1):  #2d5a27 dark green
      UNKNOWN  (0.1–0.9):#1a1a2e dark background (matches app bg)
      OCCUPIED (≥ 0.9):  #e94560 red highlight

    A white filled triangle is drawn at the bottom-centre to represent the
    rover's current position and forward direction.

    Args:
        grid:               H×W float32 BEV grid.
        grid_resolution_m:  Metres per cell (used for scale annotations).

    Returns:
        H×W×3 uint8 RGB image.
    """
    h, w = grid.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # FREE
    free_mask = grid <= 0.1
    img[free_mask] = (45, 90, 39)      # #2d5a27 in RGB

    # UNKNOWN (already black — just set background)
    unk_mask = (grid > 0.1) & (grid < 0.9)
    img[unk_mask] = (26, 26, 46)       # #1a1a2e

    # OCCUPIED
    occ_mask = grid >= 0.9
    img[occ_mask] = (233, 69, 96)      # #e94560

    # Rover triangle — white, bottom-centre pointing upward (forward)
    col_c = w // 2
    row_c = h - 1
    size  = max(4, min(12, h // 20))
    tri_pts = np.array([
        [col_c,          row_c - size * 2],   # apex (forward)
        [col_c - size,   row_c],              # bottom-left
        [col_c + size,   row_c],              # bottom-right
    ], dtype=np.int32)

    if _CV2_AVAILABLE:
        cv2.fillPoly(img, [tri_pts], color=(255, 255, 255))
    else:
        # Fallback: draw a small cross without cv2
        for r in range(max(0, row_c - size * 2), min(h, row_c + 1)):
            for c in range(max(0, col_c - size), min(w, col_c + size + 1)):
                img[r, c] = (255, 255, 255)

    return img


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_occupancy_pipeline(
    depth_results: list,         # list[DepthResult]
    annotations: list,           # list[FrameAnnotation]
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    config: OccupancyConfig,
    output_dir: str,
    progress_callback: Callable[[int], None] | None = None,
) -> list:  # list[OccupancyFrame]
    """Run the full per-frame occupancy grid pipeline.

    Steps per frame:
      1. Load depth from DepthResult.depth_raw_path (.npy preferred, .png fallback)
      2. depth_to_point_cloud
      3. point_cloud_to_bev
      4. fuse_detections_into_grid (matching annotation by frame_path)
      5. generate_bev_visualization
      6. Save grid as .npy and visualization as .png
      7. After all frames: apply temporal_grid_fusion over sliding window

    Calibration fallback:
      If fx == 0 (calibration not available), depth_to_point_cloud will
      automatically estimate intrinsics from the depth map size.

    Args:
        depth_results:     list[DepthResult] with depth_raw_path set.
        annotations:       list[FrameAnnotation] (may be shorter / empty).
        fx, fy, cx, cy:    Camera intrinsics.  Pass 0.0 to use auto-estimate.
        config:            OccupancyConfig.
        output_dir:        Root directory for output.
                           Grids saved to <output_dir>/grids/
                           Visualizations to <output_dir>/viz/
        progress_callback: Optional callable(int 0-100).

    Returns:
        list[OccupancyFrame], one per input DepthResult.
    """
    if not depth_results:
        logger.warning("run_occupancy_pipeline called with empty depth_results.")
        return []

    out = Path(output_dir)
    grids_dir = out / "grids"
    viz_dir   = out / "viz"
    grids_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Build annotation lookup by frame_path stem for fast access
    ann_by_stem: dict[str, FrameAnnotation] = {}
    for ann in annotations:
        stem = Path(ann.frame_path).stem
        ann_by_stem[stem] = ann

    if fx == 0.0:
        logger.warning(
            "Calibration not available (fx=0). "
            "Will use default pinhole estimate from depth map size. "
            "Provide camera intrinsics for accurate results."
        )

    n = len(depth_results)
    occupancy_frames: list[OccupancyFrame] = []
    raw_grids: list[np.ndarray] = []   # for temporal fusion

    for i, dr in enumerate(depth_results):
        pct = int(i / n * 85)
        if progress_callback:
            progress_callback(pct)

        # ── 1. Load depth ──────────────────────────────────────────────────
        depth_map = _load_depth(dr.depth_raw_path, config.max_depth_m)
        if depth_map is None:
            logger.warning(
                "Frame {}/{}: could not load depth from '{}' — skipping.",
                i + 1, n, dr.depth_raw_path,
            )
            continue

        dh, dw = depth_map.shape

        # ── 2. Back-project to point cloud ─────────────────────────────────
        points = depth_to_point_cloud(
            depth_map, fx, fy, cx, cy,
            max_depth_m=config.max_depth_m,
        )

        # ── 3. BEV grid ────────────────────────────────────────────────────
        grid = point_cloud_to_bev(
            points,
            grid_resolution_m=config.grid_resolution_m,
            grid_width_m=config.grid_width_m,
            grid_height_m=config.grid_height_m,
            camera_height_m=config.camera_height_m,
        )

        # ── 4. Fuse detections ─────────────────────────────────────────────
        frame_stem = Path(dr.frame_path).stem
        ann = ann_by_stem.get(frame_stem)
        if ann and ann.detections:
            grid = fuse_detections_into_grid(
                grid,
                ann.detections,
                depth_map,
                fx, fy, cx, cy,
                grid_resolution_m=config.grid_resolution_m,
                grid_width_m=config.grid_width_m,
                camera_height_m=config.camera_height_m,
                max_depth_m=config.max_depth_m,
            )

        raw_grids.append(grid.copy())

        # ── 5. Temporal fusion on sliding window ───────────────────────────
        window_start = max(0, len(raw_grids) - config.temporal_fusion_window)
        window = raw_grids[window_start:]
        fused_grid = temporal_grid_fusion(window, config.decay_factor)

        # ── 6. Visualize ───────────────────────────────────────────────────
        viz_img = generate_bev_visualization(
            fused_grid, config.grid_resolution_m
        )

        # ── 7. Save ────────────────────────────────────────────────────────
        grid_stem = f"{i:06d}_{frame_stem}"
        grid_path = str(grids_dir / f"{grid_stem}.npy")
        viz_path  = str(viz_dir   / f"{grid_stem}.png")

        np.save(grid_path, fused_grid)

        if _CV2_AVAILABLE:
            # cv2 expects BGR
            bgr = viz_img[:, :, ::-1]
            cv2.imwrite(viz_path, bgr)
        else:
            try:
                from PIL import Image as _PILImage
                _PILImage.fromarray(viz_img, "RGB").save(viz_path)
            except ImportError:
                logger.error(
                    "Neither cv2 nor PIL available — cannot save visualization PNG."
                )
                viz_path = ""

        # ── 8. Compute stats ───────────────────────────────────────────────
        total_cells = fused_grid.size
        occ_cells   = int(np.sum(fused_grid >= 0.9))
        free_cells  = int(np.sum(fused_grid <= 0.1))
        unk_cells   = total_cells - occ_cells - free_cells
        occ_pct     = float(occ_cells) / total_cells * 100.0

        # Derive timestamp: prefer FrameAnnotation, else use index * 33ms
        ts_ns = _get_timestamp(dr, ann, i)

        of = OccupancyFrame(
            frame_path=dr.frame_path,
            grid_path=grid_path,
            visualization_path=viz_path,
            timestamp_ns=ts_ns,
            occupied_cells=occ_cells,
            free_cells=free_cells,
            unknown_cells=unk_cells,
            occupancy_percent=occ_pct,
        )
        occupancy_frames.append(of)
        logger.debug(
            "Frame {}/{}: occ={:.1f}% occ_cells={} free_cells={}",
            i + 1, n, occ_pct, occ_cells, free_cells,
        )

    if progress_callback:
        progress_callback(100)

    logger.info(
        "run_occupancy_pipeline complete — {} frames processed.",
        len(occupancy_frames),
    )
    return occupancy_frames


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_depth(path: str, max_depth_m: float) -> np.ndarray | None:
    """Load a depth map from a .npy or image (.png/.jpg) file.

    For .npy: loaded directly as float32.
    For image: loaded via cv2 as float, normalized to [0, max_depth_m].

    Returns None on failure (logs warning, does not raise).
    """
    p = Path(path)
    if not p.exists():
        logger.warning("Depth file not found: '{}'", path)
        return None

    if p.suffix.lower() == ".npy":
        try:
            arr = np.load(str(p)).astype(np.float32)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            return arr
        except Exception as exc:
            logger.warning("Failed to load .npy '{}': {}", path, exc)
            # Fall through and try cv2 as image

    # PNG / JPG or failed .npy
    if not _CV2_AVAILABLE:
        logger.error(
            "cv2 not available — cannot load depth image '{}'. "
            "Install opencv-python.",
            path,
        )
        return None

    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        logger.warning("cv2 could not read depth image: '{}'", path)
        return None

    arr = img.astype(np.float32)
    if arr.ndim == 3:
        arr = arr[:, :, 0]

    # Normalize to [0, max_depth_m]
    arr_max = arr.max()
    if arr_max > 0.0:
        arr = arr / arr_max * max_depth_m
    else:
        logger.warning("Depth image '{}' is all zeros — no valid depth.", path)
        return None

    return arr


def _get_timestamp(
    dr: DepthResult,
    ann: FrameAnnotation | None,
    frame_index: int,
) -> int:
    """Extract or estimate a frame timestamp in nanoseconds.

    Order of preference:
      1. Parse integer timestamp from frame filename stem.
      2. Fallback: frame_index * 33_333_333 ns (~30 fps).
    """
    stem = Path(dr.frame_path).stem
    # Frame files are often named by their timestamp in nanoseconds
    try:
        return int(stem)
    except ValueError:
        pass
    # Fallback: 30 fps estimate
    return frame_index * 33_333_333
