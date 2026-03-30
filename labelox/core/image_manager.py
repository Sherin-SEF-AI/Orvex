"""
labelox/core/image_manager.py — Image import, deduplication, thumbnails, sequences.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from loguru import logger
from sqlalchemy.orm import Session

from labelox.core.database import DBImage, DBSequence, get_session
from labelox.core.utils import (
    SUPPORTED_EXTENSIONS,
    compute_blur_score,
    compute_md5,
    is_supported_image,
    load_image_dimensions,
    save_thumbnail,
    validate_image,
)

ProgressCB = Callable[[int, int, str], None]  # (current, total, filename)


def import_images(
    source_paths: list[str | Path],
    project_id: str,
    thumb_dir: str | Path | None = None,
    db: Session | None = None,
    progress_callback: ProgressCB | None = None,
) -> list[DBImage]:
    """Import images from files or folders.

    Per image:
    1. Validate readable, not corrupted
    2. Read dimensions via PIL
    3. MD5 hash — skip if already imported
    4. Generate thumbnail
    5. Compute blur score
    6. Insert into DB

    Returns list of created DBImage objects.
    """
    close = False
    if db is None:
        db = get_session()
        close = True

    # Expand directories to individual files
    all_files: list[Path] = []
    for p in source_paths:
        p = Path(p)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and is_supported_image(f):
                    all_files.append(f)
        elif p.is_file() and is_supported_image(p):
            all_files.append(p)

    if not all_files:
        logger.warning("No supported images found in {} paths", len(source_paths))
        return []

    # Fetch existing MD5 hashes for dedup
    existing_md5s: set[str] = set()
    rows = db.query(DBImage.md5).filter(
        DBImage.project_id == project_id,
        DBImage.md5.isnot(None),
    ).all()
    for row in rows:
        existing_md5s.add(row[0])

    created: list[DBImage] = []
    total = len(all_files)
    skipped = 0

    try:
        for i, fpath in enumerate(all_files):
            fname = fpath.name
            if progress_callback:
                progress_callback(i, total, fname)

            # Validate
            if not validate_image(fpath):
                logger.warning("Skipping corrupt image: {}", fpath)
                skipped += 1
                continue

            # Dedup
            md5 = compute_md5(fpath)
            if md5 in existing_md5s:
                logger.debug("Skipping duplicate: {} (md5={})", fname, md5)
                skipped += 1
                continue
            existing_md5s.add(md5)

            # Dimensions
            try:
                width, height = load_image_dimensions(fpath)
            except Exception as exc:
                logger.warning("Cannot read dimensions of {}: {}", fpath, exc)
                skipped += 1
                continue

            # Thumbnail
            thumb_path = None
            if thumb_dir:
                img_id = str(uuid.uuid4())
                thumb_p = Path(thumb_dir) / f"{img_id}.jpg"
                try:
                    thumb_path = save_thumbnail(fpath, thumb_p)
                except Exception as exc:
                    logger.warning("Thumbnail failed for {}: {}", fname, exc)
            else:
                img_id = str(uuid.uuid4())

            # Blur
            blur = compute_blur_score(fpath)

            # Create record
            img = DBImage(
                id=img_id,
                project_id=project_id,
                file_path=str(fpath.resolve()),
                file_name=fname,
                width=width,
                height=height,
                file_size_bytes=fpath.stat().st_size,
                md5=md5,
                blur_score=blur,
                thumbnail_path=thumb_path,
            )
            db.add(img)
            created.append(img)

        db.commit()

        # Update project image count
        from labelox.core.database import DBProject
        proj = db.get(DBProject, project_id)
        if proj:
            proj.image_count = (proj.image_count or 0) + len(created)
            db.commit()

        if progress_callback:
            progress_callback(total, total, "Done")

        logger.info(
            "Imported {} images (skipped {})", len(created), skipped,
        )
        return created

    finally:
        if close:
            db.close()


def import_from_rovix_session(
    euroc_session_dir: str | Path,
    project_id: str,
    cameras: list[str] | None = None,
    db: Session | None = None,
    progress_callback: ProgressCB | None = None,
) -> list[DBImage]:
    """Import from ROVIX/Orvex EuRoC format.

    Reads cam0/data/*.jpg + cam0/timestamps.csv.
    Preserves timestamp_ns, frame_index, sequence_id.
    """
    if cameras is None:
        cameras = ["cam0"]

    euroc = Path(euroc_session_dir)
    all_files: list[tuple[Path, int | None, int]] = []

    for cam in cameras:
        cam_dir = euroc / cam / "data"
        if not cam_dir.exists():
            logger.warning("Camera dir not found: {}", cam_dir)
            continue

        # Try to read timestamps
        ts_file = euroc / cam / "timestamps.csv"
        timestamps: dict[str, int] = {}
        if ts_file.exists():
            for line in ts_file.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) >= 2:
                    fname = parts[0].strip()
                    try:
                        timestamps[fname] = int(parts[1].strip())
                    except ValueError:
                        pass

        frames = sorted(cam_dir.glob("*"))
        for idx, f in enumerate(frames):
            if f.is_file() and is_supported_image(f):
                ts_ns = timestamps.get(f.name)
                all_files.append((f, ts_ns, idx))

    if not all_files:
        logger.warning("No images found in EuRoC session: {}", euroc)
        return []

    close = False
    if db is None:
        db = get_session()
        close = True

    try:
        seq = DBSequence(project_id=project_id, name=euroc.name, frame_count=len(all_files))
        db.add(seq)
        db.flush()

        created: list[DBImage] = []
        total = len(all_files)

        for i, (fpath, ts_ns, idx) in enumerate(all_files):
            if progress_callback:
                progress_callback(i, total, fpath.name)

            try:
                w, h = load_image_dimensions(fpath)
            except Exception:
                continue

            img = DBImage(
                project_id=project_id,
                file_path=str(fpath.resolve()),
                file_name=fpath.name,
                width=w,
                height=h,
                file_size_bytes=fpath.stat().st_size,
                md5=compute_md5(fpath),
                sequence_id=seq.id,
                frame_index=idx,
                timestamp_ns=ts_ns,
            )
            db.add(img)
            created.append(img)

        db.commit()
        logger.info("Imported {} frames from EuRoC session", len(created))
        return created

    finally:
        if close:
            db.close()


def detect_sequences(
    images: list[DBImage],
    max_gap_ns: int = 1_000_000_000,  # 1 second
) -> dict[str, list[str]]:
    """Group images into sequences by filename pattern or timestamp proximity.

    Returns: {sequence_id: [image_id, ...]}
    """
    _NUMERIC_RE = re.compile(r"(\d{4,})")

    # Strategy 1: group by existing sequence_id
    by_seq: dict[str, list[DBImage]] = {}
    unsequenced: list[DBImage] = []
    for img in images:
        if img.sequence_id:
            by_seq.setdefault(img.sequence_id, []).append(img)
        else:
            unsequenced.append(img)

    # Strategy 2: group unsequenced by folder + numeric pattern
    by_folder: dict[str, list[DBImage]] = {}
    for img in unsequenced:
        parent = str(Path(img.file_path).parent)
        by_folder.setdefault(parent, []).append(img)

    for folder, imgs in by_folder.items():
        # Check if filenames have sequential numbers
        numbered: list[tuple[int, DBImage]] = []
        for img in imgs:
            m = _NUMERIC_RE.search(img.file_name)
            if m:
                numbered.append((int(m.group(1)), img))

        if len(numbered) >= 3:
            numbered.sort(key=lambda t: t[0])
            sid = str(uuid.uuid4())
            by_seq[sid] = [img for _, img in numbered]

    # Strategy 3: timestamp proximity
    for sid, imgs in list(by_seq.items()):
        ts_imgs = [(img.timestamp_ns, img) for img in imgs if img.timestamp_ns]
        if not ts_imgs:
            continue
        ts_imgs.sort()
        groups: list[list[DBImage]] = [[ts_imgs[0][1]]]
        for i in range(1, len(ts_imgs)):
            if ts_imgs[i][0] - ts_imgs[i - 1][0] > max_gap_ns:
                groups.append([])
            groups[-1].append(ts_imgs[i][1])
        if len(groups) > 1:
            del by_seq[sid]
            for g in groups:
                by_seq[str(uuid.uuid4())] = g

    return {sid: [img.id for img in imgs] for sid, imgs in by_seq.items()}
