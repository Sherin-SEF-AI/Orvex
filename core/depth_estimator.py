"""
core/depth_estimator.py — Monocular depth estimation via Depth-Anything-v2.

Produces relative (affine-invariant) depth maps from RGB frames.
Output is NEVER metric depth unless GPS-based scale estimation is applied.

No UI imports — pure Python business logic.

Dependencies:
    pip install transformers torch torchvision Pillow
    (model weights downloaded from HuggingFace on first use)
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
from loguru import logger

from core.models import DepthResult
from core.utils import ffmpeg_run

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
MODEL_VARIANTS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base":  "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}

# Module-level model cache — keyed by model_name
_DEPTH_PIPELINE: dict[str, tuple] = {}   # model_name → (model, processor)

ProgressCB = Callable[[int], None]
StatusCB = Callable[[str], None]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_depth_model(
    model_name: str = DEFAULT_MODEL,
    device: str = "auto",
) -> tuple:
    """Load Depth-Anything-v2 from HuggingFace (or cache).

    Args:
        model_name: HuggingFace model ID or "small"/"base"/"large" shorthand.
        device:     "auto" → CUDA if available else CPU.

    Returns:
        (model, processor) tuple.

    Raises:
        ImportError: if transformers / torch are not installed.
    """
    # Resolve shorthand
    resolved_name = MODEL_VARIANTS.get(model_name, model_name)

    resolved_device = _resolve_device(device)
    cache_key = f"{resolved_name}:{resolved_device}"

    if cache_key in _DEPTH_PIPELINE:
        logger.debug("depth_estimator: cache hit for {}", cache_key)
        return _DEPTH_PIPELINE[cache_key]

    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        import torch
    except ImportError as exc:
        raise ImportError(
            "transformers and torch are required for depth estimation. "
            "Install with: pip install transformers torch torchvision"
        ) from exc

    logger.info(
        "depth_estimator: loading {} on {} (downloading if needed)…",
        resolved_name,
        resolved_device,
    )
    processor = AutoImageProcessor.from_pretrained(resolved_name)
    model = AutoModelForDepthEstimation.from_pretrained(resolved_name)
    model.to(resolved_device)
    model.eval()

    _DEPTH_PIPELINE[cache_key] = (model, processor)
    return model, processor


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ---------------------------------------------------------------------------
# Batch depth estimation
# ---------------------------------------------------------------------------

def estimate_depth_batch(
    image_paths: list[str],
    model: object,
    processor: object,
    batch_size: int = 8,
    output_dir: str | None = None,
    colorize: bool = True,
    progress_callback: ProgressCB | None = None,
    status_callback: StatusCB | None = None,
) -> list[DepthResult]:
    """Run depth estimation on a list of frame images.

    Depth maps are RELATIVE (affine-invariant). They are NOT metric depth.
    All DepthResult.is_metric will be False.

    Args:
        image_paths:       Paths to RGB frames.
        model:             Loaded depth model (from load_depth_model()).
        processor:         Corresponding HuggingFace processor.
        batch_size:        Images per inference batch.
        output_dir:        Where to save depth maps. If None, nothing is saved.
        colorize:          If True, save plasma-colorized JPEG alongside raw PNG.
        progress_callback: 0-100 int callback.
        status_callback:   Human-readable status strings.

    Returns:
        List of DepthResult, one per input image.
    """
    try:
        import torch
        from PIL import Image as PILImage
    except ImportError as exc:
        raise ImportError(
            "torch and Pillow are required. "
            "Install with: pip install torch Pillow"
        ) from exc

    if not image_paths:
        return []

    if output_dir:
        raw_dir = Path(output_dir) / "depth_raw"
        color_dir = Path(output_dir) / "depth_color"
        raw_dir.mkdir(parents=True, exist_ok=True)
        if colorize:
            color_dir.mkdir(parents=True, exist_ok=True)

    results: list[DepthResult] = []
    n = len(image_paths)

    for batch_start in range(0, n, batch_size):
        batch_paths = image_paths[batch_start : batch_start + batch_size]
        if status_callback:
            status_callback(
                f"Estimating depth {batch_start + 1}–{min(batch_start + batch_size, n)} / {n}"
            )

        images: list = []
        for fp in batch_paths:
            try:
                img = PILImage.open(fp).convert("RGB")
                images.append(img)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to open image for depth estimation: {fp} — {exc}"
                ) from exc

        t0 = time.perf_counter()
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Interpolate to original sizes
        predicted_depths = torch.nn.functional.interpolate(
            outputs.predicted_depth.unsqueeze(1),
            size=images[0].size[::-1],  # (H, W)
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)  # shape: (B, H, W)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        ms_per = elapsed_ms / len(batch_paths)

        for i, fp in enumerate(batch_paths):
            depth_np = predicted_depths[i].cpu().numpy()  # (H, W) float32

            min_d = float(depth_np.min())
            max_d = float(depth_np.max())
            mean_d = float(depth_np.mean())

            raw_path: str = ""
            color_path: str | None = None

            if output_dir:
                stem = Path(fp).stem
                # Save 16-bit raw depth PNG
                raw_file = raw_dir / f"{stem}_depth.png"
                _save_depth_raw(depth_np, str(raw_file))
                raw_path = str(raw_file)

                if colorize:
                    color_file = color_dir / f"{stem}_depth_color.jpg"
                    _save_depth_colorized(depth_np, str(color_file))
                    color_path = str(color_file)

            results.append(
                DepthResult(
                    frame_path=fp,
                    depth_raw_path=raw_path,
                    depth_color_path=color_path,
                    min_depth=min_d,
                    max_depth=max_d,
                    mean_depth=mean_d,
                    inference_time_ms=ms_per,
                    is_metric=False,  # always relative
                )
            )

        if progress_callback:
            pct = int((batch_start + len(batch_paths)) / n * 100)
            progress_callback(pct)

    logger.info("depth_estimator: processed {} frames", n)
    return results


def _save_depth_raw(depth_np: np.ndarray, path: str) -> None:
    """Save float32 depth map as 16-bit grayscale PNG."""
    from PIL import Image as PILImage

    # Normalize to 0-65535
    d_min, d_max = depth_np.min(), depth_np.max()
    if d_max > d_min:
        normalized = (depth_np - d_min) / (d_max - d_min) * 65535.0
    else:
        normalized = np.zeros_like(depth_np)
    depth_uint16 = normalized.astype(np.uint16)
    PILImage.fromarray(depth_uint16, mode="I;16").save(path)


def _save_depth_colorized(depth_np: np.ndarray, path: str) -> None:
    """Save depth map as plasma-colorized JPEG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm
    from PIL import Image as PILImage

    d_min, d_max = depth_np.min(), depth_np.max()
    if d_max > d_min:
        normalized = (depth_np - d_min) / (d_max - d_min)
    else:
        normalized = np.zeros_like(depth_np)

    colored = (cm.plasma(normalized)[:, :, :3] * 255).astype(np.uint8)
    PILImage.fromarray(colored).save(path, quality=90)


# ---------------------------------------------------------------------------
# Metric scale estimation
# ---------------------------------------------------------------------------

def estimate_metric_scale(
    depth_maps: list[np.ndarray],
    gps_speed_mps: list[float],
    camera_fps: float,
) -> float:
    """Estimate approximate metric scale factor from GPS speed and optical flow.

    Method:
        1. Compute optical flow magnitude between consecutive frame pairs.
        2. Correlate mean optical flow magnitude with GPS speed (m/s).
        3. Return scale_factor such that: metric_depth = relative_depth * scale_factor.

    This is a heuristic estimate, NOT ground truth.

    Args:
        depth_maps:    List of depth map numpy arrays (H x W float32).
        gps_speed_mps: GPS speed per frame (same length as depth_maps).
        camera_fps:    Frame rate (used to convert speed to per-frame motion).

    Returns:
        scale_factor (float). Multiply relative depth by this to approximate metric depth.

    Raises:
        ValueError: if fewer than 2 frames are provided.
    """
    import cv2

    if len(depth_maps) < 2:
        raise ValueError(
            "At least 2 depth maps are required to estimate metric scale. "
            "Provide more frames."
        )

    if len(depth_maps) != len(gps_speed_mps):
        raise ValueError(
            f"depth_maps ({len(depth_maps)}) and gps_speed_mps ({len(gps_speed_mps)}) "
            "must have the same length."
        )

    flow_magnitudes: list[float] = []
    speed_per_frame: list[float] = []

    for i in range(1, len(depth_maps)):
        # Normalize depth maps to 0-255 uint8 for optical flow
        d_prev = _normalize_u8(depth_maps[i - 1])
        d_curr = _normalize_u8(depth_maps[i])
        flow = cv2.calcOpticalFlowFarneback(
            d_prev, d_curr, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        mag = float(np.linalg.norm(flow, axis=2).mean())
        flow_magnitudes.append(mag)
        # Distance per frame = speed * (1/fps)
        speed_per_frame.append(gps_speed_mps[i] / camera_fps)

    flow_arr = np.array(flow_magnitudes)
    dist_arr = np.array(speed_per_frame)

    # Least-squares scale: dist = scale * flow_magnitude
    valid = (flow_arr > 0.01) & (dist_arr > 0.0)
    if not valid.any():
        logger.warning(
            "depth_estimator: no valid flow/GPS pairs for scale estimation, returning 1.0"
        )
        return 1.0

    scale = float(np.median(dist_arr[valid] / flow_arr[valid]))
    logger.info("depth_estimator: estimated metric scale factor = {:.4f}", scale)
    return scale


def _normalize_u8(arr: np.ndarray) -> np.ndarray:
    """Normalize float array to uint8 [0, 255]."""
    a_min, a_max = arr.min(), arr.max()
    if a_max > a_min:
        out = ((arr - a_min) / (a_max - a_min) * 255).astype(np.uint8)
    else:
        out = np.zeros_like(arr, dtype=np.uint8)
    return out


# ---------------------------------------------------------------------------
# Depth video generation
# ---------------------------------------------------------------------------

def generate_depth_video(
    depth_color_paths: list[str],
    output_path: str,
    fps: float = 10.0,
) -> str:
    """Compile colorized depth frames into an MP4 video.

    Args:
        depth_color_paths: Ordered list of JPEG paths.
        output_path:       Path for output MP4.
        fps:               Playback frame rate.

    Returns:
        Absolute path to the written MP4.

    Raises:
        RuntimeError: if ffmpeg fails or no frames provided.
    """
    if not depth_color_paths:
        raise RuntimeError(
            "No depth color frames provided to generate_depth_video. "
            "Run estimate_depth_batch with colorize=True first."
        )

    # Write a temp file list for ffmpeg concat demuxer
    import tempfile, os
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as tmp:
        for fp in depth_color_paths:
            tmp.write(f"file '{fp}'\n")
            tmp.write(f"duration {1.0 / fps:.6f}\n")
        list_path = tmp.name

    try:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_run([
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "23",
            str(out),
        ])
    finally:
        os.unlink(list_path)

    logger.info("depth_estimator: depth video written to {}", output_path)
    return str(Path(output_path).resolve())
