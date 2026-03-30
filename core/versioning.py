"""
core/versioning.py — Dataset versioning via DVC + git-backed manifests.

All functions operate on a dataset_dir on the local filesystem.
No UI imports — this module is a pure core library.

DVC requires git — every DVC function checks for .git/ first and raises
RuntimeError with an actionable message if git is absent.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from core.models import DatasetDiff, DatasetVersion

# ── DVC / Git checks ────────────────────────────────────────────────────────


def check_dvc_installation() -> bool:
    """Return True if the ``dvc`` binary is on PATH.

    Uses :func:`shutil.which` — no subprocess is launched.
    """
    found = shutil.which("dvc") is not None
    logger.debug("check_dvc_installation: {}", "found" if found else "not found")
    return found


def check_git_initialized(dataset_dir: str) -> bool:
    """Return True if a ``.git/`` directory exists inside *dataset_dir*.

    Args:
        dataset_dir: Absolute path to the dataset root.
    """
    git_path = Path(dataset_dir) / ".git"
    result = git_path.is_dir()
    logger.debug("check_git_initialized({}): {}", dataset_dir, result)
    return result


# ── DVC repo init + remote ───────────────────────────────────────────────────


def init_dvc_repo(dataset_dir: str) -> bool:
    """Initialise DVC inside *dataset_dir*.

    Idempotent — if ``.dvc/`` already exists the function returns ``True``
    immediately without re-running ``dvc init``.

    Args:
        dataset_dir: Absolute path to the dataset root.

    Returns:
        ``True`` on success.

    Raises:
        RuntimeError: If git is not initialised in *dataset_dir*.
        RuntimeError: If ``dvc init`` exits non-zero.
    """
    ddir = Path(dataset_dir)

    if not check_git_initialized(dataset_dir):
        raise RuntimeError(
            f"Git not initialized in {dataset_dir}. Run: git init"
        )

    if (ddir / ".dvc").is_dir():
        logger.info("DVC already initialized in '{}'", dataset_dir)
        return True

    logger.info("Initializing DVC in '{}'", dataset_dir)
    result = subprocess.run(
        ["dvc", "init"],
        cwd=str(ddir),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"dvc init failed in '{dataset_dir}'.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}\n"
            "Ensure DVC is installed: pip install dvc"
        )
    logger.info("DVC initialized successfully in '{}'", dataset_dir)
    return True


def add_dataset_to_dvc(
    dataset_dir: str,
    remote_name: str = "local_remote",
    remote_path: str | None = None,
) -> dict[str, list[str]]:
    """Track the ``images/`` and ``labels/`` sub-directories with DVC.

    Steps:
    1. Run ``dvc add <dataset_dir>/images`` and ``dvc add <dataset_dir>/labels``
       for each sub-directory that exists.
    2. If *remote_path* is provided, configure it as the default DVC remote.

    Args:
        dataset_dir:  Absolute path to the dataset root.
        remote_name:  Name for the DVC remote (default ``"local_remote"``).
        remote_path:  Local filesystem path for the remote storage.
                      Pass ``None`` to skip remote configuration.

    Returns:
        ``{"dvc_files": [<list of .dvc file paths created>]}``

    Raises:
        RuntimeError: With actionable message on any subprocess failure.
    """
    ddir = Path(dataset_dir)
    dvc_files: list[str] = []

    for sub in ("images", "labels"):
        sub_path = ddir / sub
        if not sub_path.exists():
            logger.debug("add_dataset_to_dvc: '{}' does not exist, skipping", sub_path)
            continue

        logger.info("Running: dvc add {}", sub_path)
        result = subprocess.run(
            ["dvc", "add", str(sub_path)],
            cwd=str(ddir),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"dvc add failed for '{sub_path}'.\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}\n"
                "Check that DVC is initialized (run init_dvc_repo first) "
                "and that the path exists."
            )
        dvc_file = str(sub_path) + ".dvc"
        if Path(dvc_file).exists():
            dvc_files.append(dvc_file)
        logger.info("DVC tracking added for '{}'", sub_path)

    if remote_path:
        logger.info("Configuring DVC remote '{}' -> '{}'", remote_name, remote_path)
        result = subprocess.run(
            ["dvc", "remote", "add", "-d", remote_name, remote_path],
            cwd=str(ddir),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"dvc remote add failed.\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}\n"
                f"Verify remote path '{remote_path}' is accessible."
            )
        logger.info("DVC remote '{}' configured at '{}'", remote_name, remote_path)

    return {"dvc_files": dvc_files}


# ── Internal helpers ─────────────────────────────────────────────────────────


def _compute_dataset_hash(dataset_dir: str) -> str:
    """Compute a stable MD5 hash over all files in ``images/`` and ``labels/``.

    Files are walked in deterministic sorted order.  Each file's raw bytes are
    fed into a single running :class:`hashlib.md5` digest.

    Args:
        dataset_dir: Absolute path to the dataset root.

    Returns:
        Lowercase hex digest string.
    """
    ddir = Path(dataset_dir)
    hasher = hashlib.md5()

    for sub in ("images", "labels"):
        sub_path = ddir / sub
        if not sub_path.exists():
            continue
        for fpath in sorted(sub_path.rglob("*")):
            if fpath.is_file():
                hasher.update(fpath.name.encode())
                hasher.update(fpath.read_bytes())

    digest = hasher.hexdigest()
    logger.debug("_compute_dataset_hash({}): {}", dataset_dir, digest)
    return digest


def _count_frames_and_classes(
    dataset_dir: str,
) -> tuple[int, int, dict[str, int]]:
    """Count image files and parse YOLO label files for class frequencies.

    Image files recognised: ``.jpg``, ``.jpeg``, ``.png`` (case-insensitive).
    Label files recognised: ``.txt`` under ``labels/``.

    YOLO label format — one detection per line::

        <class_id> <cx> <cy> <w> <h>

    Args:
        dataset_dir: Absolute path to the dataset root.

    Returns:
        ``(file_count, total_frames, class_distribution)`` where
        *file_count* is the number of image + label files combined,
        *total_frames* is the image count, and
        *class_distribution* maps ``"<class_id>"`` to occurrence count.
    """
    ddir = Path(dataset_dir)
    image_exts = {".jpg", ".jpeg", ".png"}

    # Count images
    images_dir = ddir / "images"
    image_files: list[Path] = []
    if images_dir.exists():
        image_files = [
            p for p in images_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in image_exts
        ]
    total_frames = len(image_files)

    # Count labels and parse class IDs
    labels_dir = ddir / "labels"
    label_files: list[Path] = []
    class_distribution: dict[str, int] = {}
    if labels_dir.exists():
        label_files = [p for p in labels_dir.rglob("*.txt") if p.is_file()]
        for lf in label_files:
            try:
                for line in lf.read_text(encoding="utf-8").splitlines():
                    parts = line.strip().split()
                    if not parts:
                        continue
                    class_id = parts[0]
                    class_distribution[class_id] = class_distribution.get(class_id, 0) + 1
            except Exception as exc:
                logger.warning("Could not parse label file '{}': {}", lf, exc)

    file_count = total_frames + len(label_files)
    logger.debug(
        "_count_frames_and_classes({}): {} images, {} labels, {} classes",
        dataset_dir, total_frames, len(label_files), len(class_distribution),
    )
    return file_count, total_frames, class_distribution


# ── Version manifest directory ───────────────────────────────────────────────

_VERSIONS_DIR = ".dvc_versions"


def _versions_dir(dataset_dir: str) -> Path:
    d = Path(dataset_dir) / _VERSIONS_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _version_path(dataset_dir: str, tag: str) -> Path:
    return _versions_dir(dataset_dir) / f"{tag}.json"


# ── Public API ───────────────────────────────────────────────────────────────


def commit_dataset_version(
    dataset_dir: str,
    version_tag: str,
    message: str,
    metadata: dict[str, Any] | None = None,
) -> DatasetVersion:
    """Snapshot the current state of *dataset_dir* as a named version.

    Steps:
    1. Compute a reproducible hash of all image + label content.
    2. Count frames and class distribution.
    3. Persist a JSON manifest under ``{dataset_dir}/.dvc_versions/{tag}.json``.
    4. If DVC is initialized, attempt ``dvc push`` (non-fatal on failure).

    Args:
        dataset_dir:  Absolute path to the dataset root.
        version_tag:  Unique tag string, e.g. ``"v1.0.0"``.
        message:      Human-readable description of this version.
        metadata:     Optional extra dict stored verbatim in the manifest.

    Returns:
        :class:`~core.models.DatasetVersion` representing the new snapshot.

    Raises:
        ValueError: If *version_tag* already exists in ``.dvc_versions/``.
    """
    manifest_path = _version_path(dataset_dir, version_tag)
    if manifest_path.exists():
        raise ValueError(
            f"Version tag '{version_tag}' already exists in '{dataset_dir}'. "
            "Choose a different tag or delete the existing manifest first."
        )

    logger.info("commit_dataset_version: tag='{}' dir='{}'", version_tag, dataset_dir)

    dataset_hash = _compute_dataset_hash(dataset_dir)
    file_count, total_frames, class_distribution = _count_frames_and_classes(dataset_dir)

    # Collect .dvc file paths if DVC is initialised
    dvc_files: list[str] = []
    ddir = Path(dataset_dir)
    for sub in ("images", "labels"):
        candidate = ddir / (sub + ".dvc")
        if candidate.exists():
            dvc_files.append(str(candidate))

    version = DatasetVersion(
        tag=version_tag,
        timestamp=datetime.now(timezone.utc),
        message=message,
        metadata=metadata or {},
        file_count=file_count,
        total_frames=total_frames,
        class_distribution=class_distribution,
        dataset_hash=dataset_hash,
        dvc_files=dvc_files,
    )

    # Serialise using Pydantic v2 model_dump, datetime → ISO string
    raw = version.model_dump()
    raw["timestamp"] = version.timestamp.isoformat()
    manifest_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    logger.info("Version manifest written to '{}'", manifest_path)

    # Non-fatal DVC push
    if (ddir / ".dvc").is_dir() and check_dvc_installation():
        logger.info("Attempting dvc push for version '{}'", version_tag)
        push_result = subprocess.run(
            ["dvc", "push"],
            cwd=str(ddir),
            capture_output=True,
            text=True,
        )
        if push_result.returncode != 0:
            logger.warning(
                "dvc push failed (non-fatal): {}", push_result.stderr.strip()
            )
        else:
            logger.info("dvc push succeeded for version '{}'", version_tag)

    return version


def list_dataset_versions(dataset_dir: str) -> list[DatasetVersion]:
    """Return all saved version manifests for *dataset_dir*, sorted by timestamp.

    Args:
        dataset_dir: Absolute path to the dataset root.

    Returns:
        List of :class:`~core.models.DatasetVersion`, oldest first.
        Returns an empty list if no versions exist.
    """
    vdir = Path(dataset_dir) / _VERSIONS_DIR
    if not vdir.exists():
        return []

    versions: list[DatasetVersion] = []
    for json_file in vdir.glob("*.json"):
        try:
            raw = json.loads(json_file.read_text(encoding="utf-8"))
            # Restore datetime from ISO string
            if isinstance(raw.get("timestamp"), str):
                raw["timestamp"] = datetime.fromisoformat(raw["timestamp"])
            versions.append(DatasetVersion(**raw))
        except Exception as exc:
            logger.warning("Could not load version manifest '{}': {}", json_file, exc)

    versions.sort(key=lambda v: v.timestamp)
    logger.debug("list_dataset_versions({}): {} version(s) found", dataset_dir, len(versions))
    return versions


def diff_dataset_versions(
    dataset_dir: str,
    version_a: str,
    version_b: str,
) -> DatasetDiff:
    """Compute the diff between two saved versions.

    Args:
        dataset_dir: Absolute path to the dataset root.
        version_a:   Tag of the baseline version.
        version_b:   Tag of the comparison version.

    Returns:
        :class:`~core.models.DatasetDiff` with frame deltas and class deltas.

    Raises:
        FileNotFoundError: If either version manifest is absent.
    """
    path_a = _version_path(dataset_dir, version_a)
    path_b = _version_path(dataset_dir, version_b)

    for tag, path in ((version_a, path_a), (version_b, path_b)):
        if not path.exists():
            raise FileNotFoundError(
                f"Version '{tag}' not found in '{dataset_dir}/.dvc_versions/'. "
                f"Available: {[p.stem for p in _versions_dir(dataset_dir).glob('*.json')]}"
            )

    def _load(path: Path) -> DatasetVersion:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw.get("timestamp"), str):
            raw["timestamp"] = datetime.fromisoformat(raw["timestamp"])
        return DatasetVersion(**raw)

    va = _load(path_a)
    vb = _load(path_b)

    frames_added = max(0, vb.total_frames - va.total_frames)
    frames_removed = max(0, va.total_frames - vb.total_frames)
    total_frames_delta = vb.total_frames - va.total_frames

    # Class distribution delta — union of all class keys
    all_classes = set(va.class_distribution) | set(vb.class_distribution)
    class_distribution_delta: dict[str, float] = {
        cls: float(vb.class_distribution.get(cls, 0) - va.class_distribution.get(cls, 0))
        for cls in all_classes
    }

    # Session lists — stored in metadata if present
    sessions_a: set[str] = set(va.metadata.get("sessions", []))
    sessions_b: set[str] = set(vb.metadata.get("sessions", []))
    new_sessions = sorted(sessions_b - sessions_a)
    removed_sessions = sorted(sessions_a - sessions_b)

    diff = DatasetDiff(
        version_a=version_a,
        version_b=version_b,
        frames_added=frames_added,
        frames_removed=frames_removed,
        frames_changed=0,  # byte-level change detection not in scope
        class_distribution_delta=class_distribution_delta,
        total_frames_delta=total_frames_delta,
        new_sessions=new_sessions,
        removed_sessions=removed_sessions,
    )
    logger.info(
        "diff_dataset_versions: {} vs {}: +{} -{} frames",
        version_a, version_b, frames_added, frames_removed,
    )
    return diff


def restore_dataset_version(
    dataset_dir: str,
    version_tag: str,
) -> bool:
    """Restore the dataset to a previously committed version.

    If DVC is initialised, runs ``dvc checkout`` to restore tracked file
    content.  The manifest is used only to verify the version exists.

    **Caller responsibility:** Show a confirmation dialog before calling this
    function — it will overwrite the current working tree.

    Args:
        dataset_dir:  Absolute path to the dataset root.
        version_tag:  Tag of the version to restore.

    Returns:
        ``True`` on success.

    Raises:
        FileNotFoundError: If the version manifest does not exist.
        RuntimeError:      If ``dvc checkout`` fails.
    """
    manifest_path = _version_path(dataset_dir, version_tag)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Version '{version_tag}' not found in '{dataset_dir}/.dvc_versions/'. "
            f"Available: {[p.stem for p in _versions_dir(dataset_dir).glob('*.json')]}"
        )

    logger.warning(
        "restore_dataset_version: restoring '{}' to version '{}' — "
        "current working tree may be overwritten",
        dataset_dir, version_tag,
    )

    ddir = Path(dataset_dir)
    if (ddir / ".dvc").is_dir() and check_dvc_installation():
        logger.info("Running: dvc checkout in '{}'", dataset_dir)
        result = subprocess.run(
            ["dvc", "checkout"],
            cwd=str(ddir),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"dvc checkout failed while restoring version '{version_tag}'.\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}\n"
                "Ensure the DVC remote is configured and accessible."
            )
        logger.info("dvc checkout succeeded for version '{}'", version_tag)
    else:
        logger.warning(
            "DVC not initialized or not installed — skipping dvc checkout. "
            "Only the manifest was verified."
        )

    return True
