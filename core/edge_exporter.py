"""
core/edge_exporter.py — Edge deployment export pipeline for RoverDataKit.

Handles YOLOv8 → ONNX export, ONNX → TensorRT conversion, model benchmarking,
standalone inference script generation, and Jetson deployment packaging.

No UI imports. Pure Python / subprocess only.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from core.models import BenchmarkResult, ONNXExportResult, TRTExportResult


# ─────────────────────────────────────────────────────────────────────────────
# Dependency detection
# ─────────────────────────────────────────────────────────────────────────────

def check_export_dependencies() -> dict[str, Any]:
    """Check which export/inference dependencies are available.

    Returns a dict with boolean flags per dependency plus version strings
    where applicable.

    Keys:
        onnx (bool), onnxruntime (bool), onnxsim (bool),
        tensorrt (bool), tensorrt_version (str | None),
        cuda_available (bool), torch_version (str | None),
        ultralytics (bool), trtexec (bool)
    """
    result: dict[str, Any] = {
        "onnx": False,
        "onnxruntime": False,
        "onnxsim": False,
        "tensorrt": False,
        "tensorrt_version": None,
        "cuda_available": False,
        "torch_version": None,
        "ultralytics": False,
        "trtexec": False,
    }

    try:
        import onnx  # noqa: F401
        result["onnx"] = True
    except ImportError:
        pass

    try:
        import onnxruntime  # noqa: F401
        result["onnxruntime"] = True
    except ImportError:
        pass

    try:
        import onnxsim  # noqa: F401
        result["onnxsim"] = True
    except ImportError:
        pass

    try:
        import tensorrt as trt  # noqa: F401
        result["tensorrt"] = True
        result["tensorrt_version"] = getattr(trt, "__version__", "unknown")
    except ImportError:
        pass

    try:
        import torch
        result["torch_version"] = torch.__version__
        result["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        pass

    try:
        import ultralytics  # noqa: F401
        result["ultralytics"] = True
    except ImportError:
        pass

    result["trtexec"] = shutil.which("trtexec") is not None

    return result


# ─────────────────────────────────────────────────────────────────────────────
# ONNX export
# ─────────────────────────────────────────────────────────────────────────────

def export_to_onnx(
    weights_path: str,
    output_path: str,
    image_size: int = 640,
    batch_size: int = 1,
    simplify: bool = True,
    opset_version: int = 17,
    dynamic_axes: bool = False,
) -> ONNXExportResult:
    """Export a YOLOv8 .pt model to ONNX format.

    Uses ultralytics to drive the export, then verifies the model and runs
    a quick latency test with onnxruntime.

    Args:
        weights_path: Path to the YOLOv8 .pt file.
        output_path: Desired output path for the .onnx file.
        image_size: Input image size (square). Default 640.
        batch_size: Static batch size baked into the model. Default 1.
        simplify: Whether to run onnx-simplifier. Default True.
        opset_version: ONNX opset version. Default 17.
        dynamic_axes: Export with dynamic batch/spatial axes. Default False.

    Returns:
        ONNXExportResult with size, shapes, latency, and verification status.

    Raises:
        ImportError: If ultralytics is not installed.
        FileNotFoundError: If weights_path does not exist.
        RuntimeError: If the export or verification fails.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required for ONNX export but is not installed. "
            "Fix: pip install ultralytics"
        ) from exc

    weights_path = str(weights_path)
    output_path = str(output_path)

    if not Path(weights_path).exists():
        raise FileNotFoundError(
            f"Weights file not found: {weights_path}. "
            "Provide the path to a valid YOLOv8 .pt file."
        )

    logger.info(f"Loading YOLOv8 model: {weights_path}")
    model = YOLO(weights_path)

    logger.info(
        f"Exporting to ONNX — imgsz={image_size}, opset={opset_version}, "
        f"simplify={simplify}, dynamic={dynamic_axes}"
    )
    try:
        exported = model.export(
            format="onnx",
            imgsz=image_size,
            opset=opset_version,
            simplify=simplify,
            dynamic=dynamic_axes,
            batch=batch_size,
        )
    except Exception as exc:
        raise RuntimeError(
            f"ONNX export failed for {weights_path}: {exc}. "
            "Ensure the model file is a valid YOLOv8 checkpoint and ultralytics "
            "is up to date (pip install -U ultralytics)."
        ) from exc

    # ultralytics writes the .onnx next to the .pt; resolve the actual path
    if exported and Path(str(exported)).exists():
        generated_path = str(exported)
    else:
        # fallback: same name/location as .pt but .onnx extension
        generated_path = str(Path(weights_path).with_suffix(".onnx"))

    if not Path(generated_path).exists():
        raise RuntimeError(
            f"Export appeared to succeed but no .onnx file was found at "
            f"{generated_path}. Check ultralytics output for errors."
        )

    # Move to requested output path
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    if generated_path != output_path:
        shutil.copy2(generated_path, output_path)
        logger.info(f"Copied ONNX to {output_path}")
    else:
        logger.info(f"ONNX written to {output_path}")

    # --- Verify with onnx checker ---
    verification_passed = False
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        verification_passed = True
        logger.info("ONNX model verification passed.")
    except ImportError:
        logger.warning(
            "onnx package not installed — skipping model verification. "
            "Fix: pip install onnx"
        )
    except Exception as exc:
        logger.warning(f"ONNX model verification failed: {exc}")

    # --- Extract shapes from the model proto ---
    input_shape: list[int] = []
    output_shapes: list[list[int]] = []
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        graph = onnx_model.graph
        if graph.input:
            inp = graph.input[0]
            shape_proto = inp.type.tensor_type.shape
            input_shape = [
                d.dim_value if d.HasField("dim_value") else -1
                for d in shape_proto.dim
            ]
        for out in graph.output:
            shape_proto = out.type.tensor_type.shape
            out_shape = [
                d.dim_value if d.HasField("dim_value") else -1
                for d in shape_proto.dim
            ]
            output_shapes.append(out_shape)
    except Exception as exc:
        logger.warning(f"Could not read ONNX shapes: {exc}")
        # Construct expected shape from params
        input_shape = [batch_size, 3, image_size, image_size]

    # --- Latency test with onnxruntime ---
    test_latency_ms = 0.0
    try:
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            sess = ort.InferenceSession(output_path, providers=providers)
        except Exception:
            sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])

        in_name = sess.get_inputs()[0].name
        in_shape = sess.get_inputs()[0].shape
        # Replace dynamic dims with concrete values
        concrete_shape = []
        for dim in in_shape:
            if isinstance(dim, int) and dim > 0:
                concrete_shape.append(dim)
            elif isinstance(dim, str) or dim is None or dim <= 0:
                concrete_shape.append(batch_size if len(concrete_shape) == 0 else image_size)
            else:
                concrete_shape.append(dim)
        dummy = np.random.rand(*concrete_shape).astype(np.float32)

        # Warmup
        for _ in range(3):
            sess.run(None, {in_name: dummy})

        latencies = []
        for _ in range(10):
            t0 = time.perf_counter()
            sess.run(None, {in_name: dummy})
            latencies.append((time.perf_counter() - t0) * 1000.0)
        test_latency_ms = float(np.mean(latencies))
        logger.info(f"ONNX test latency: {test_latency_ms:.2f} ms (mean of 10 runs)")
    except ImportError:
        logger.warning(
            "onnxruntime not installed — skipping latency test. "
            "Fix: pip install onnxruntime  or  pip install onnxruntime-gpu"
        )
    except Exception as exc:
        logger.warning(f"Latency test failed: {exc}")

    model_size_mb = Path(output_path).stat().st_size / (1024 * 1024)

    return ONNXExportResult(
        output_path=output_path,
        model_size_mb=round(model_size_mb, 3),
        input_shape=input_shape,
        output_shapes=output_shapes,
        onnx_opset=opset_version,
        simplified=simplify,
        test_latency_ms=round(test_latency_ms, 3),
        verification_passed=verification_passed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TensorRT export
# ─────────────────────────────────────────────────────────────────────────────

def export_to_tensorrt(
    onnx_path: str,
    output_path: str,
    precision: str = "fp16",
    workspace_gb: int = 4,
    min_batch: int = 1,
    opt_batch: int = 1,
    max_batch: int = 4,
) -> TRTExportResult:
    """Convert an ONNX model to a TensorRT engine via trtexec.

    Streams trtexec stdout to the loguru logger in real time so the caller
    can see progress without a frozen UI.

    Args:
        onnx_path: Path to the source .onnx file.
        output_path: Desired output path for the .engine file.
        precision: "fp32", "fp16", or "int8". Default "fp16".
        workspace_gb: Max GPU workspace in GB. Default 4.
        min_batch: Minimum batch size for dynamic shape profile.
        opt_batch: Optimal batch size for dynamic shape profile.
        max_batch: Maximum batch size for dynamic shape profile.

    Returns:
        TRTExportResult with engine size and build time.

    Raises:
        RuntimeError: If trtexec is not found or the build fails.
        FileNotFoundError: If onnx_path does not exist.
    """
    trtexec = shutil.which("trtexec")
    if trtexec is None:
        raise RuntimeError(
            "trtexec not found on PATH. "
            "On Jetson: sudo apt install tensorrt. "
            "On desktop: install TensorRT from developer.nvidia.com/tensorrt "
            "and ensure trtexec is on your PATH."
        )

    onnx_path = str(onnx_path)
    output_path = str(output_path)

    if not Path(onnx_path).exists():
        raise FileNotFoundError(
            f"ONNX file not found: {onnx_path}. "
            "Run ONNX export first before converting to TensorRT."
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={output_path}",
        f"--workspace={workspace_gb * 1024}",
    ]

    precision_lower = precision.lower()
    if precision_lower == "fp16":
        cmd.append("--fp16")
    elif precision_lower == "int8":
        cmd.append("--int8")
    elif precision_lower == "fp32":
        pass  # default; no flag needed
    else:
        raise ValueError(
            f"Unknown precision '{precision}'. Must be one of: fp32, fp16, int8."
        )

    logger.info(f"Starting TensorRT build: {' '.join(cmd)}")
    build_start = time.perf_counter()

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            stripped = line.rstrip()
            if stripped:
                logger.info(f"[trtexec] {stripped}")
        proc.wait()
    except Exception as exc:
        raise RuntimeError(
            f"trtexec subprocess failed: {exc}. "
            "Ensure TensorRT is installed correctly and ONNX model is valid."
        ) from exc

    build_time_minutes = (time.perf_counter() - build_start) / 60.0

    if proc.returncode != 0:
        raise RuntimeError(
            f"trtexec exited with code {proc.returncode}. "
            "Check the log output above for details. Common causes: "
            "unsupported ONNX operator, insufficient GPU memory, or invalid precision "
            "for the target GPU."
        )

    if not Path(output_path).exists():
        raise RuntimeError(
            f"trtexec reported success but engine file not found at {output_path}. "
            "This is unexpected — check disk space and output path permissions."
        )

    engine_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    logger.info(
        f"TensorRT engine built: {output_path} "
        f"({engine_size_mb:.1f} MB, {build_time_minutes:.1f} min)"
    )

    return TRTExportResult(
        output_path=output_path,
        engine_size_mb=round(engine_size_mb, 3),
        precision=precision_lower,
        build_time_minutes=round(build_time_minutes, 3),
        estimated_jetson_latency_ms=None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Benchmarking
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_model(
    model_path: str,
    model_format: str,
    image_size: int = 640,
    n_iterations: int = 200,
    warmup_iterations: int = 20,
    batch_size: int = 1,
) -> BenchmarkResult:
    """Benchmark inference latency and throughput for ONNX or PyTorch models.

    Args:
        model_path: Path to the model file (.onnx or .pt).
        model_format: "onnx" or "pytorch".
        image_size: Input image size (square). Default 640.
        n_iterations: Number of timed inference iterations. Default 200.
        warmup_iterations: Untimed warmup runs before timing. Default 20.
        batch_size: Batch size per forward pass. Default 1.

    Returns:
        BenchmarkResult with mean/p50/p95/p99 latencies and FPS.

    Raises:
        ImportError: If required packages are missing.
        FileNotFoundError: If the model file does not exist.
        ValueError: If model_format is not "onnx" or "pytorch".
    """
    model_path = str(model_path)
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Provide the path to a valid model file."
        )

    fmt = model_format.lower()
    if fmt not in ("onnx", "pytorch"):
        raise ValueError(
            f"Unsupported model_format '{model_format}'. "
            "Only 'onnx' and 'pytorch' are supported. "
            "TensorRT engine benchmarking requires trtexec directly."
        )

    device_str = "cpu"
    latencies: list[float] = []

    if fmt == "onnx":
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for ONNX benchmarking but is not installed. "
                "Fix: pip install onnxruntime  or  pip install onnxruntime-gpu"
            ) from exc

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            sess = ort.InferenceSession(model_path, providers=providers)
            # Detect if CUDA provider was actually used
            active = sess.get_providers()
            if "CUDAExecutionProvider" in active:
                device_str = "cuda"
        except Exception:
            sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

        in_name = sess.get_inputs()[0].name
        in_shape_raw = sess.get_inputs()[0].shape
        concrete = [
            batch_size if (not isinstance(d, int) or d <= 0) and i == 0
            else (image_size if (not isinstance(d, int) or d <= 0) else d)
            for i, d in enumerate(in_shape_raw)
        ]
        dummy = np.random.rand(*concrete).astype(np.float32)

        logger.info(
            f"Benchmarking ONNX: {model_path} — "
            f"{warmup_iterations} warmup + {n_iterations} timed runs"
        )
        for _ in range(warmup_iterations):
            sess.run(None, {in_name: dummy})

        for _ in range(n_iterations):
            t0 = time.perf_counter()
            sess.run(None, {in_name: dummy})
            latencies.append((time.perf_counter() - t0) * 1000.0)

    elif fmt == "pytorch":
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "torch and ultralytics are required for PyTorch benchmarking. "
                "Fix: pip install torch ultralytics"
            ) from exc

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        model = YOLO(model_path)

        dummy_img = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)

        logger.info(
            f"Benchmarking PyTorch: {model_path} on {device_str} — "
            f"{warmup_iterations} warmup + {n_iterations} timed runs"
        )
        with torch.no_grad():
            for _ in range(warmup_iterations):
                model.predict(dummy_img, verbose=False, device=device_str)

            for _ in range(n_iterations):
                t0 = time.perf_counter()
                model.predict(dummy_img, verbose=False, device=device_str)
                latencies.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(latencies, dtype=np.float64)
    mean_ms = float(np.mean(arr))
    std_ms = float(np.std(arr))
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    # Memory estimate via psutil (optional)
    memory_mb = 0.0
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        memory_mb = proc.memory_info().rss / (1024 * 1024)
    except Exception:
        pass

    logger.info(
        f"Benchmark complete — mean={mean_ms:.2f}ms p95={p95:.2f}ms "
        f"p99={p99:.2f}ms fps={fps:.1f}"
    )

    return BenchmarkResult(
        model_path=model_path,
        format=fmt,
        device=device_str,
        mean_latency_ms=round(mean_ms, 3),
        std_latency_ms=round(std_ms, 3),
        p50_latency_ms=round(p50, 3),
        p95_latency_ms=round(p95, 3),
        p99_latency_ms=round(p99, 3),
        throughput_fps=round(fps, 2),
        memory_mb=round(memory_mb, 2),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Standalone inference script generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_inference_script(
    class_names: list[str],
    input_size: int = 640,
    conf_threshold: float = 0.25,
) -> str:
    """Generate a complete standalone Python inference script.

    The script:
    - Has no ultralytics or torch dependency — only onnxruntime, cv2, numpy.
    - Accepts argv[1] as a video path or "0" for webcam.
    - Loads model.onnx from the same directory as the script.
    - Runs ONNX inference per frame, draws bounding boxes with class + confidence.
    - Overlays live FPS in the top-left corner.
    - Saves output to output_tracked.mp4 in the working directory.

    Args:
        class_names: Ordered list of class name strings.
        input_size: Model input size (square). Default 640.
        conf_threshold: Minimum confidence score to display a detection.

    Returns:
        Complete Python source code as a string.
    """
    classes_repr = repr(class_names)
    return f'''#!/usr/bin/env python3
"""
Standalone ONNX inference script — generated by RoverDataKit.

Usage:
    python infer_onnx.py <video_path_or_0>

    <video_path_or_0>: path to a video file, or "0" for the default webcam.

Requirements:
    pip install onnxruntime-gpu  # or onnxruntime for CPU-only
    pip install opencv-python numpy

No ultralytics or torch dependency required.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# ── Configuration ──────────────────────────────────────────────────────────
CLASS_NAMES: list[str] = {classes_repr}
INPUT_SIZE: int = {input_size}
CONF_THRESHOLD: float = {conf_threshold}
MODEL_PATH: str = str(Path(__file__).parent / "model.onnx")

# Colour palette: one BGR colour per class (wraps around if more classes than colours)
_PALETTE = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (128, 255, 0), (255, 128, 0),
    (0, 128, 255), (128, 0, 255), (255, 0, 128), (0, 255, 128),
]


def get_colour(class_id: int) -> tuple[int, int, int]:
    return _PALETTE[class_id % len(_PALETTE)]


# ── Model loading ──────────────────────────────────────────────────────────

def load_session(model_path: str) -> ort.InferenceSession:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        sess = ort.InferenceSession(model_path, providers=providers)
        active = sess.get_providers()
        device = "CUDA" if "CUDAExecutionProvider" in active else "CPU"
        print(f"[INFO] Model loaded on {{device}}: {{model_path}}")
        return sess
    except Exception:
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        print(f"[INFO] Model loaded on CPU (fallback): {{model_path}}")
        return sess


# ── Pre/post processing ─────────────────────────────────────────────────────

def preprocess(frame: np.ndarray, input_size: int) -> tuple[np.ndarray, float, float, int, int]:
    """Letterbox resize → CHW float32 normalised to [0, 1].

    Returns:
        blob, scale_x, scale_y, pad_left, pad_top
    """
    orig_h, orig_w = frame.shape[:2]
    scale = min(input_size / orig_w, input_size / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    pad_left = (input_size - new_w) // 2
    pad_top = (input_size - new_h) // 2
    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    blob = canvas[:, :, ::-1].astype(np.float32) / 255.0   # BGR → RGB, normalise
    blob = np.transpose(blob, (2, 0, 1))                    # HWC → CHW
    blob = np.expand_dims(blob, axis=0)                     # add batch dim
    return blob, scale, scale, pad_left, pad_top


def postprocess(
    output: np.ndarray,
    scale_x: float,
    scale_y: float,
    pad_left: int,
    pad_top: int,
    conf_threshold: float,
) -> list[tuple[int, int, int, int, float, int]]:
    """Parse YOLOv8 output into (x1, y1, x2, y2, conf, class_id) tuples.

    YOLOv8 ONNX output shape: [1, 4+num_classes, num_anchors]
    or [1, num_anchors, 4+num_classes] depending on export version.
    """
    preds = output[0]  # remove batch dim

    # Normalise to [num_anchors, 4+num_classes]
    if preds.shape[0] < preds.shape[1]:
        preds = preds.T   # transpose if transposed export

    detections = []
    for pred in preds:
        cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]
        class_scores = pred[4:]
        class_id = int(np.argmax(class_scores))
        conf = float(class_scores[class_id])
        if conf < conf_threshold:
            continue

        # Convert from letterboxed coordinates back to original image
        x1 = int((cx - w / 2 - pad_left) / scale_x)
        y1 = int((cy - h / 2 - pad_top) / scale_y)
        x2 = int((cx + w / 2 - pad_left) / scale_x)
        y2 = int((cy + h / 2 - pad_top) / scale_y)
        detections.append((x1, y1, x2, y2, conf, class_id))

    return detections


def draw_detections(
    frame: np.ndarray,
    detections: list[tuple[int, int, int, int, float, int]],
) -> np.ndarray:
    for x1, y1, x2, y2, conf, class_id in detections:
        colour = get_colour(class_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        label = f"{{CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else class_id}} {{conf:.2f}}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        # background rectangle for label
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(
            frame, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA,
        )
    return frame


# ── Main loop ──────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python infer_onnx.py <video_path_or_0>")
        sys.exit(1)

    source = sys.argv[1]
    cap_source = 0 if source == "0" else source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {{source}}")
        sys.exit(1)

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = "output_tracked.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps_in, (orig_w, orig_h))

    session = load_session(MODEL_PATH)
    input_name = session.get_inputs()[0].name

    frame_count = 0
    fps_display = 0.0
    t_prev = time.perf_counter()

    print(f"[INFO] Running inference. Output: {{out_path}}  Press Ctrl+C to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            blob, sx, sy, pl, pt = preprocess(frame, INPUT_SIZE)
            outputs = session.run(None, {{input_name: blob}})
            detections = postprocess(outputs[0], sx, sy, pl, pt, CONF_THRESHOLD)
            frame = draw_detections(frame, detections)

            # FPS overlay
            frame_count += 1
            if frame_count % 10 == 0:
                t_now = time.perf_counter()
                fps_display = 10.0 / max(t_now - t_prev, 1e-6)
                t_prev = t_now
            cv2.putText(
                frame, f"FPS: {{fps_display:.1f}}", (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA,
            )

            writer.write(frame)
    except KeyboardInterrupt:
        print("[INFO] Interrupted.")

    cap.release()
    writer.release()
    print(f"[INFO] Done. {{frame_count}} frames processed. Saved to {{out_path}}")


if __name__ == "__main__":
    main()
'''


# ─────────────────────────────────────────────────────────────────────────────
# Jetson deployment packaging
# ─────────────────────────────────────────────────────────────────────────────

_REQUIREMENTS_TXT = """\
# Deployment requirements for RoverDataKit ONNX inference
# Install with: pip install -r requirements.txt

onnxruntime-gpu>=1.18.0   # use onnxruntime (no -gpu suffix) on CPU-only machines
opencv-python>=4.10.0
numpy>=1.26.0
"""

_README_TEMPLATE = """\
# RoverDataKit — Jetson Deployment Package

Generated by RoverDataKit edge export pipeline.

## Target device
{target_device}

## Contents

```
models/
  model.pt       — YOLOv8 PyTorch weights (fallback)
  model.onnx     — ONNX model for cross-platform deployment
  model.engine   — TensorRT engine (if included)
config/
  classes.txt    — One class name per line
  deploy_config.json — Deployment configuration
inference/
  infer_onnx.py  — Standalone inference script (no torch/ultralytics needed)
  requirements.txt
```

## Quick start

```bash
pip install -r inference/requirements.txt
python inference/infer_onnx.py /path/to/video.mp4
# or for webcam:
python inference/infer_onnx.py 0
```

Output is saved to `output_tracked.mp4` in the working directory.

## TensorRT (Jetson)

If `models/model.engine` is included, it was built for the target device listed
above. To rebuild for a different device, run:

```bash
trtexec --onnx=models/model.onnx --saveEngine=models/model.engine --fp16
```

## Configuration

Edit `config/deploy_config.json` to adjust confidence threshold, input size,
and other runtime parameters.
"""


def package_jetson_deployment(
    weights_path: str,
    onnx_path: str,
    trt_path: str | None,
    class_names: list[str],
    output_dir: str,
    target_device: str = "jetson_orin",
    conf_threshold: float = 0.25,
) -> str:
    """Assemble a self-contained Jetson deployment package and compress it.

    Creates the following structure under output_dir, then archives it:

        output_dir/
          models/model.pt, model.onnx, [model.engine]
          config/classes.txt, deploy_config.json
          inference/infer_onnx.py, requirements.txt
          README.md

    Args:
        weights_path: Path to the .pt weights file.
        onnx_path: Path to the exported .onnx file.
        trt_path: Path to the .engine file, or None if TRT was not built.
        class_names: Ordered list of class names.
        output_dir: Root directory for the package (will be created).
        target_device: Human-readable target device label.
        conf_threshold: Confidence threshold baked into the inference script.

    Returns:
        Absolute path to the generated .tar.gz archive.

    Raises:
        FileNotFoundError: If weights_path or onnx_path do not exist.
    """
    weights_path = str(weights_path)
    onnx_path = str(onnx_path)
    output_dir_path = Path(output_dir)

    for p, label in [(weights_path, "weights"), (onnx_path, "ONNX")]:
        if not Path(p).exists():
            raise FileNotFoundError(
                f"{label} file not found: {p}. "
                "Ensure the file exists before packaging."
            )

    if trt_path and not Path(trt_path).exists():
        logger.warning(
            f"TensorRT engine not found at {trt_path} — "
            "package will be created without it."
        )
        trt_path = None

    # --- Create directory tree ---
    models_dir = output_dir_path / "models"
    config_dir = output_dir_path / "config"
    inference_dir = output_dir_path / "inference"
    for d in (models_dir, config_dir, inference_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- Copy model files ---
    logger.info(f"Copying model files to {output_dir_path}")
    shutil.copy2(weights_path, models_dir / "model.pt")
    shutil.copy2(onnx_path, models_dir / "model.onnx")
    if trt_path:
        shutil.copy2(trt_path, models_dir / "model.engine")
        logger.info("TensorRT engine included in package.")

    # --- config/classes.txt ---
    classes_file = config_dir / "classes.txt"
    classes_file.write_text("\n".join(class_names) + "\n", encoding="utf-8")

    # --- config/deploy_config.json ---
    onnx_size_mb = round(Path(onnx_path).stat().st_size / (1024 * 1024), 3)
    trt_size_mb = (
        round(Path(trt_path).stat().st_size / (1024 * 1024), 3)
        if trt_path else None
    )
    deploy_config = {
        "target_device": target_device,
        "conf_threshold": conf_threshold,
        "input_size": 640,
        "class_names": class_names,
        "num_classes": len(class_names),
        "models": {
            "pytorch": "models/model.pt",
            "onnx": "models/model.onnx",
            "onnx_size_mb": onnx_size_mb,
            "tensorrt": "models/model.engine" if trt_path else None,
            "tensorrt_size_mb": trt_size_mb,
        },
        "generated_by": "RoverDataKit edge_exporter",
    }
    config_file = config_dir / "deploy_config.json"
    config_file.write_text(
        json.dumps(deploy_config, indent=2), encoding="utf-8"
    )

    # --- inference/infer_onnx.py ---
    script = generate_inference_script(
        class_names=class_names,
        input_size=640,
        conf_threshold=conf_threshold,
    )
    (inference_dir / "infer_onnx.py").write_text(script, encoding="utf-8")

    # --- inference/requirements.txt ---
    (inference_dir / "requirements.txt").write_text(
        _REQUIREMENTS_TXT, encoding="utf-8"
    )

    # --- README.md ---
    readme = _README_TEMPLATE.format(target_device=target_device)
    (output_dir_path / "README.md").write_text(readme, encoding="utf-8")

    # --- Create tarball ---
    archive_path = str(output_dir_path) + ".tar.gz"
    logger.info(f"Creating archive: {archive_path}")
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(output_dir_path, arcname=output_dir_path.name)

    archive_size_mb = Path(archive_path).stat().st_size / (1024 * 1024)
    logger.info(
        f"Deployment package ready: {archive_path} ({archive_size_mb:.1f} MB)"
    )

    return archive_path
