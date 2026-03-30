"""
core/export_profiles.py — Named export configuration profiles.

Profiles are stored as TOML files in a user-configured profiles directory
(default: ~/.roverdatakit/data/profiles/).

Three built-in profiles are provided:
  slam_vio   — optimised for SLAM/VIO pipelines (low fps, moderate quality)
  annotation — optimised for annotation tools (very low fps, high quality)
  archive    — high-fidelity archival (high fps, maximum quality)

Custom profiles can be created and saved via the desktop UI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import toml
from loguru import logger


@dataclass
class ExportProfile:
    name: str
    output_format: str = "euroc"      # "euroc" | "rosbag2" | "hdf5"
    frame_fps: float = 5.0
    frame_quality: int = 95
    blur_threshold: float = 100.0
    dedup_threshold: float = 0.98
    description: str = ""


# Built-in profiles bundled with the application
_BUILTIN_PROFILES: list[ExportProfile] = [
    ExportProfile(
        name="slam_vio",
        output_format="euroc",
        frame_fps=5.0,
        frame_quality=90,
        blur_threshold=100.0,
        dedup_threshold=0.98,
        description="Optimised for SLAM/VIO pipelines (EuRoC format, 5 fps)",
    ),
    ExportProfile(
        name="annotation",
        output_format="euroc",
        frame_fps=2.0,
        frame_quality=95,
        blur_threshold=150.0,
        dedup_threshold=0.95,
        description="Optimised for annotation tools (2 fps, sharp frames only)",
    ),
    ExportProfile(
        name="archive",
        output_format="euroc",
        frame_fps=10.0,
        frame_quality=100,
        blur_threshold=50.0,
        dedup_threshold=0.99,
        description="High-fidelity archival (10 fps, maximum quality)",
    ),
]


def list_profiles(profiles_dir: Path) -> list[ExportProfile]:
    """Return all profiles: built-ins first, then user profiles from profiles_dir.

    Args:
        profiles_dir: Directory containing user .toml profile files.

    Returns:
        List of ExportProfile, deduplicated by name (user profiles override built-ins).
    """
    profiles: dict[str, ExportProfile] = {p.name: p for p in _BUILTIN_PROFILES}

    if profiles_dir.exists():
        for toml_file in sorted(profiles_dir.glob("*.toml")):
            try:
                p = _load_toml(toml_file)
                profiles[p.name] = p
            except Exception as exc:
                logger.warning("Failed to load profile '{}': {}", toml_file.name, exc)

    return list(profiles.values())


def save_profile(profile: ExportProfile, profiles_dir: Path) -> None:
    """Save a profile to profiles_dir/<profile.name>.toml.

    Args:
        profile:      Profile to save.
        profiles_dir: Directory to write into (created if absent).
    """
    profiles_dir.mkdir(parents=True, exist_ok=True)
    safe_name = profile.name.replace(" ", "_").lower()
    path = profiles_dir / f"{safe_name}.toml"
    data = {
        "name": profile.name,
        "output_format": profile.output_format,
        "frame_fps": profile.frame_fps,
        "frame_quality": profile.frame_quality,
        "blur_threshold": profile.blur_threshold,
        "dedup_threshold": profile.dedup_threshold,
        "description": profile.description,
    }
    with open(path, "w", encoding="utf-8") as f:
        toml.dump(data, f)
    logger.info("Saved export profile '{}' → {}", profile.name, path)


def load_profile(name: str, profiles_dir: Path) -> ExportProfile:
    """Load a profile by name from profiles_dir or from built-ins.

    Args:
        name:         Profile name (case-insensitive).
        profiles_dir: Directory of user profiles.

    Returns:
        ExportProfile matching name.

    Raises:
        KeyError: if no profile with that name is found.
    """
    all_profiles = {p.name.lower(): p for p in list_profiles(profiles_dir)}
    key = name.lower()
    if key not in all_profiles:
        raise KeyError(
            f"Export profile '{name}' not found. "
            f"Available: {', '.join(all_profiles.keys())}"
        )
    return all_profiles[key]


def _load_toml(path: Path) -> ExportProfile:
    with open(path, encoding="utf-8") as f:
        data = toml.load(f)
    return ExportProfile(
        name=data["name"],
        output_format=data.get("output_format", "euroc"),
        frame_fps=float(data.get("frame_fps", 5.0)),
        frame_quality=int(data.get("frame_quality", 95)),
        blur_threshold=float(data.get("blur_threshold", 100.0)),
        dedup_threshold=float(data.get("dedup_threshold", 0.98)),
        description=data.get("description", ""),
    )
