"""
labelox/core/utils.py — Shared utilities: logging, image ops, hashing.
"""
from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from PIL import Image


# ─── Logging ─────────────────────────────────────────────────────────────────

def setup_logging(log_file: str | Path | None = None, level: str = "DEBUG") -> None:
    """Configure Loguru. Safe to call multiple times."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
               "<level>{message}</level>",
        colorize=True,
    )
    if log_file is not None:
        logger.add(
            str(log_file),
            level=level,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            enqueue=True,
        )


# ─── Image Utilities ─────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def is_supported_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def load_image_dimensions(path: str | Path) -> tuple[int, int]:
    """Return (width, height) without loading full image into memory."""
    with Image.open(str(path)) as img:
        return img.size  # (width, height)


def generate_thumbnail(
    image_path: str | Path,
    max_size: tuple[int, int] = (256, 256),
) -> bytes:
    """Generate JPEG thumbnail bytes. Returns raw JPEG bytes."""
    with Image.open(str(image_path)) as img:
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()


def save_thumbnail(
    image_path: str | Path,
    output_path: str | Path,
    max_size: tuple[int, int] = (256, 256),
) -> str:
    """Generate and save thumbnail. Returns output path."""
    data = generate_thumbnail(image_path, max_size)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)
    return str(output_path)


def compute_blur_score(image_path: str | Path) -> float:
    """Laplacian variance — higher = sharper."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def compute_md5(file_path: str | Path, chunk_size: int = 8192) -> str:
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def validate_image(path: str | Path) -> bool:
    """Check that a file is a valid, readable image."""
    try:
        with Image.open(str(path)) as img:
            img.verify()
        return True
    except Exception:
        return False
