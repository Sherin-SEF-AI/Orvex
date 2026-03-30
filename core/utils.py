"""
core/utils.py — subprocess wrappers for FFmpeg, ffprobe, exiftool + Loguru setup.

All external tool calls go through these helpers so the rest of the codebase
never has to construct subprocess commands directly.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_file: str | Path | None = None, level: str = "DEBUG") -> None:
    """Configure Loguru for the application.

    Call once at startup (desktop main.py or FastAPI lifespan).
    Safe to call multiple times — removes previous handlers first.

    Args:
        log_file: Optional path to a rotating log file.
        level:    Minimum log level (DEBUG, INFO, WARNING, ERROR).
    """
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
            enqueue=True,  # thread-safe
        )


# ---------------------------------------------------------------------------
# External tool presence checks
# ---------------------------------------------------------------------------

class MissingToolError(RuntimeError):
    """Raised when a required external binary is not found on PATH."""


def _require_tool(name: str) -> str:
    """Return the full path to *name* or raise MissingToolError."""
    path = shutil.which(name)
    if path is None:
        raise MissingToolError(
            f"'{name}' not found on PATH. "
            f"Install it and ensure it is accessible before running RoverDataKit."
        )
    return path


def check_dependencies() -> dict[str, str]:
    """Verify all required external tools are installed.

    Returns a dict mapping tool name → full path.
    Raises MissingToolError listing every missing tool (not just the first).
    """
    required = ["ffmpeg", "ffprobe", "exiftool"]
    missing: list[str] = []
    found: dict[str, str] = {}

    for tool in required:
        try:
            found[tool] = _require_tool(tool)
        except MissingToolError:
            missing.append(tool)

    if missing:
        raise MissingToolError(
            f"Required tools not found: {', '.join(missing)}. "
            f"Install them and re-run."
        )

    return found


# ---------------------------------------------------------------------------
# ffprobe wrapper
# ---------------------------------------------------------------------------

def ffprobe(file_path: str | Path) -> dict[str, Any]:
    """Run ffprobe on *file_path* and return parsed JSON output.

    Probes streams and format. Raises on non-zero exit or parse failure.

    Returns:
        Parsed ffprobe JSON with keys 'streams' and 'format'.
    """
    _require_tool("ffprobe")
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        str(file_path),
    ]
    logger.debug("ffprobe: {}", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed on '{file_path}' (exit {result.returncode}):\n"
            f"{result.stderr.strip()}"
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"ffprobe produced unparseable output for '{file_path}': {exc}"
        ) from exc


def ffprobe_video_stream(file_path: str | Path) -> dict[str, Any] | None:
    """Return the first video stream dict from ffprobe output, or None."""
    data = ffprobe(file_path)
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            return stream
    return None


def ffprobe_duration(file_path: str | Path) -> float:
    """Return duration in seconds from the format block."""
    data = ffprobe(file_path)
    raw = data.get("format", {}).get("duration")
    if raw is None:
        raise RuntimeError(f"ffprobe found no duration for '{file_path}'")
    return float(raw)


# ---------------------------------------------------------------------------
# ffmpeg wrapper
# ---------------------------------------------------------------------------

def ffmpeg_run(
    args: list[str],
    *,
    capture_output: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run ffmpeg with the given argument list.

    Args:
        args:           Arguments passed after 'ffmpeg' (do not include 'ffmpeg').
        capture_output: If True, capture stdout/stderr instead of inheriting them.
        check:          Raise CalledProcessError on non-zero exit if True.

    Returns:
        CompletedProcess instance.
    """
    _require_tool("ffmpeg")
    cmd = ["ffmpeg", "-hide_banner"] + args
    logger.debug("ffmpeg: {}", " ".join(cmd))
    return subprocess.run(cmd, capture_output=capture_output, text=True, check=check)


def ffmpeg_extract_frames(
    input_path: str | Path,
    output_pattern: str | Path,
    fps: float,
    quality: int = 2,
    start_time: float | None = None,
    duration: float | None = None,
    progress_callback: Any | None = None,
    total_duration: float | None = None,
) -> subprocess.CompletedProcess:
    """Extract frames from a video file at the given fps.

    Args:
        input_path:        Path to input video.
        output_pattern:    Output path pattern, e.g. '/out/frame_%06d.jpg'.
        fps:               Frame rate for extraction.
        quality:           JPEG quality scale (2 = best, 31 = worst for mjpeg).
        start_time:        Optional start offset in seconds.
        duration:          Optional maximum duration in seconds.
        progress_callback: Optional callable(pct: float) called as ffmpeg runs.
        total_duration:    Total video duration in seconds (for progress calc).

    Returns:
        CompletedProcess from the ffmpeg invocation.
    """
    import re as _re

    args: list[str] = ["-progress", "pipe:1"]
    if start_time is not None:
        args += ["-ss", str(start_time)]
    args += ["-i", str(input_path)]
    if duration is not None:
        args += ["-t", str(duration)]
    args += [
        "-vf", f"fps={fps}",
        "-q:v", str(quality),
        "-vsync", "vfr",
        str(output_pattern),
    ]

    _require_tool("ffmpeg")
    cmd = ["ffmpeg", "-hide_banner", "-y"] + args
    logger.debug("ffmpeg: {}", " ".join(cmd))

    if not progress_callback or not total_duration or total_duration <= 0:
        return subprocess.run(cmd, capture_output=True, text=True, check=True)

    # Stream stdout (-progress pipe:1) for live progress
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    time_re = _re.compile(r"out_time_us=(\d+)")
    total_us = total_duration * 1_000_000
    for line in proc.stdout:
        m = time_re.search(line)
        if m:
            current_us = int(m.group(1))
            pct = min(100.0, current_us / total_us * 100)
            progress_callback(pct)
    proc.wait()
    stderr_out = proc.stderr.read() if proc.stderr else ""
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, output="", stderr=stderr_out,
        )
    return subprocess.CompletedProcess(cmd, proc.returncode, "", stderr_out)


def ffmpeg_concat_demuxer(
    input_paths: list[str | Path],
    output_path: str | Path,
    *,
    copy_streams: bool = True,
) -> subprocess.CompletedProcess:
    """Concatenate multiple video files using the concat demuxer (no re-encode).

    Writes a temporary concat list file alongside the output.

    Args:
        input_paths: Ordered list of input file paths to concatenate.
        output_path: Destination file path.
        copy_streams: If True, stream-copy (no re-encode). Default True.

    Returns:
        CompletedProcess from the ffmpeg invocation.
    """
    output_path = Path(output_path)
    list_file = output_path.with_suffix(".concat_list.txt")
    with open(list_file, "w") as f:
        for p in input_paths:
            f.write(f"file '{Path(p).resolve()}'\n")

    args = [
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
    ]
    if copy_streams:
        args += ["-c", "copy"]
    args.append(str(output_path))
    try:
        return ffmpeg_run(args)
    finally:
        list_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# exiftool wrapper
# ---------------------------------------------------------------------------

def exiftool(
    file_path: str | Path,
    *extra_args: str,
) -> dict[str, Any]:
    """Run exiftool on *file_path* and return parsed JSON output.

    Args:
        file_path:   File to probe.
        extra_args:  Additional exiftool flags, e.g. '-b', '-GyroData'.

    Returns:
        Dict of tag name → value (first element from exiftool JSON array).
    """
    _require_tool("exiftool")
    cmd = ["exiftool", "-json", str(file_path)] + list(extra_args)
    logger.debug("exiftool: {}", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"exiftool failed on '{file_path}' (exit {result.returncode}):\n"
            f"{result.stderr.strip()}"
        )
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"exiftool produced unparseable output for '{file_path}': {exc}"
        ) from exc
    if not parsed:
        raise RuntimeError(f"exiftool returned empty result for '{file_path}'")
    return parsed[0]


def exiftool_binary_tag(file_path: str | Path, tag: str) -> bytes:
    """Extract a single binary tag from a file via exiftool.

    Example: exiftool_binary_tag('video.insv', 'GyroData')

    Args:
        file_path: Input file.
        tag:       Exiftool tag name (without leading '-').

    Returns:
        Raw bytes of the tag value.
    """
    _require_tool("exiftool")
    cmd = ["exiftool", "-b", f"-{tag}", str(file_path)]
    logger.debug("exiftool binary: {}", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"exiftool failed reading tag '{tag}' from '{file_path}' "
            f"(exit {result.returncode}):\n{result.stderr.decode().strip()}"
        )
    return result.stdout


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def file_size_mb(file_path: str | Path) -> float:
    """Return file size in megabytes."""
    return Path(file_path).stat().st_size / (1024 * 1024)
