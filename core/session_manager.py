"""
core/session_manager.py — Session CRUD with TOML persistence.

Each session is stored as a folder under the sessions root:
  <sessions_root>/<session_id>/session.toml

The TOML file holds all Session fields.  audit_results are embedded as
inline tables.  Large binary data (frames, CSVs) lives alongside the folder
but is referenced by path, not stored in TOML.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import toml
from loguru import logger
from pydantic import ValidationError

from core.models import AuditResult, DeviceType, ExtractionStatus, Session

_SESSION_FILE = "session.toml"


class SessionNotFoundError(KeyError):
    """Raised when a session ID is not found in the sessions root."""


class SessionManager:
    """Manages session lifecycle and disk persistence.

    Args:
        sessions_root: Directory under which individual session folders live.
                       Created on first access if absent.
    """

    def __init__(self, sessions_root: str | Path) -> None:
        self.sessions_root = Path(sessions_root)
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        logger.debug("SessionManager: root = {}", self.sessions_root)

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create_session(
        self,
        name: str,
        environment: str,
        location: str,
        notes: str = "",
    ) -> Session:
        """Create a new session, persist it, and return it.

        Args:
            name:        Human-readable session name.
            environment: e.g. "road", "indoor", "gravel".
            location:    e.g. "Kalamassery".
            notes:       Optional free-text notes.

        Returns:
            Newly created Session.
        """
        session_id = str(uuid.uuid4())
        session = Session(
            id=session_id,
            name=name,
            created_at=datetime.now(timezone.utc),
            environment=environment,
            location=location,
            files=[],
            audit_results=[],
            extraction_status="pending",
            notes=notes,
        )
        self._save(session)
        logger.info("Created session '{}' ({})", name, session_id)
        return session

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_session(self, session_id: str) -> Session:
        """Load and return a session by ID.

        Raises:
            SessionNotFoundError: if the session folder or TOML file is absent.
        """
        toml_path = self._toml_path(session_id)
        if not toml_path.exists():
            raise SessionNotFoundError(
                f"Session '{session_id}' not found at '{toml_path}'. "
                "It may have been deleted or the ID is incorrect."
            )
        return self._load(toml_path)

    def list_sessions(self) -> list[Session]:
        """Return all sessions sorted by created_at descending (newest first)."""
        sessions: list[Session] = []
        for toml_file in self.sessions_root.glob(f"*/{_SESSION_FILE}"):
            try:
                sessions.append(self._load(toml_file))
            except Exception as exc:
                logger.warning("Skipping malformed session at '{}': {}", toml_file, exc)
        sessions.sort(key=lambda s: s.created_at, reverse=True)
        return sessions

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update_session(self, session: Session) -> None:
        """Persist an updated session.  The session must already exist.

        Raises:
            SessionNotFoundError: if the session folder does not exist.
        """
        folder = self._session_folder(session.id)
        if not folder.exists():
            raise SessionNotFoundError(
                f"Cannot update session '{session.id}' — folder not found."
            )
        self._save(session)
        logger.debug("Updated session '{}'", session.id)

    def add_file(self, session_id: str, file_path: str | Path) -> Session:
        """Append a file path to the session's file list.

        Warns if the same file is already present (duplicate detection).

        Returns:
            Updated Session.
        """
        session = self.get_session(session_id)
        fp = str(file_path)
        if fp in session.files:
            logger.warning(
                "File '{}' is already imported in session '{}'. Skipping.", fp, session_id
            )
            return session
        session.files.append(fp)
        self.update_session(session)
        return session

    def remove_file(self, session_id: str, file_path: str | Path) -> Session:
        """Remove a file path from the session's file list.

        Returns:
            Updated Session.
        """
        session = self.get_session(session_id)
        fp = str(file_path)
        if fp not in session.files:
            logger.warning("File '{}' not found in session '{}' — nothing removed.", fp, session_id)
            return session
        session.files.remove(fp)
        self.update_session(session)
        return session

    def set_audit_results(
        self, session_id: str, results: list[AuditResult]
    ) -> Session:
        """Replace the audit results for a session."""
        session = self.get_session(session_id)
        session.audit_results = results
        self.update_session(session)
        return session

    def set_extraction_status(
        self, session_id: str, status: ExtractionStatus
    ) -> Session:
        """Update the extraction status field."""
        session = self.get_session(session_id)
        session.extraction_status = status
        self.update_session(session)
        return session

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_session(self, session_id: str, *, delete_files: bool = False) -> None:
        """Delete a session's TOML and optionally its entire folder.

        Args:
            session_id:   ID of the session to delete.
            delete_files: If True, removes the session folder and all contents.
                          If False, only removes session.toml.

        Raises:
            SessionNotFoundError: if the session does not exist.
        """
        toml_path = self._toml_path(session_id)
        if not toml_path.exists():
            raise SessionNotFoundError(
                f"Session '{session_id}' not found — cannot delete."
            )
        if delete_files:
            import shutil
            shutil.rmtree(self._session_folder(session_id))
            logger.info("Deleted session folder for '{}'", session_id)
        else:
            toml_path.unlink()
            logger.info("Deleted session.toml for '{}'", session_id)

    # ------------------------------------------------------------------
    # Folder helpers
    # ------------------------------------------------------------------

    def session_folder(self, session_id: str) -> Path:
        """Return (and create if absent) the session's data folder."""
        folder = self._session_folder(session_id)
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def extraction_output_dir(self, session_id: str, device_prefix: str = "") -> Path:
        """Return the extraction output directory for a session."""
        sub = f"extraction_{device_prefix}" if device_prefix else "extraction"
        path = self.session_folder(session_id) / sub
        path.mkdir(parents=True, exist_ok=True)
        return path

    def merge_chapter_files(self, session_id: str, files: list[str]) -> None:
        """Reorder session.files so that the given chapter files are consecutive.

        This is used after the Merge Chapters UI action to keep chapter siblings
        together in the file list, making downstream extraction treat them as one
        continuous recording.

        Args:
            session_id: Session to update.
            files:      Complete list of chapter file paths (all siblings).
        """
        session = self.get_session(session_id)
        existing = set(session.files)
        files_in_session = [f for f in files if f in existing]
        if not files_in_session:
            return

        # Place chapter files consecutively at the position of the first one
        other_files = [f for f in session.files if f not in set(files_in_session)]
        first_pos = next(
            (i for i, f in enumerate(session.files) if f in set(files_in_session)),
            len(other_files),
        )
        reordered = other_files[:first_pos] + sorted(files_in_session) + other_files[first_pos:]
        updated = session.model_copy(update={"files": reordered})
        self._save(updated)
        logger.info(
            "merge_chapter_files: reordered {} chapter(s) in session '{}'",
            len(files_in_session), session_id,
        )

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def _session_folder(self, session_id: str) -> Path:
        return self.sessions_root / session_id

    def _toml_path(self, session_id: str) -> Path:
        return self._session_folder(session_id) / _SESSION_FILE

    def _save(self, session: Session) -> None:
        folder = self._session_folder(session.id)
        folder.mkdir(parents=True, exist_ok=True)
        toml_path = folder / _SESSION_FILE

        data = _session_to_dict(session)
        with open(toml_path, "w", encoding="utf-8") as f:
            toml.dump(data, f)

    def _load(self, toml_path: Path) -> Session:
        with open(toml_path, encoding="utf-8") as f:
            data = toml.load(f)
        return _dict_to_session(data)


# ---------------------------------------------------------------------------
# TOML serialisation helpers (keep Session ↔ dict conversion here, not in
# models.py, so models.py stays a pure Pydantic layer).
# ---------------------------------------------------------------------------

def _session_to_dict(session: Session) -> dict[str, Any]:
    d: dict[str, Any] = {
        "id": session.id,
        "name": session.name,
        "created_at": session.created_at.isoformat(),
        "environment": session.environment,
        "location": session.location,
        "files": session.files,
        "extraction_status": session.extraction_status,
        "notes": session.notes,
        "audit_results": [_audit_to_dict(r) for r in session.audit_results],
    }
    return d


def _audit_to_dict(r: AuditResult) -> dict[str, Any]:
    return {
        "file_path": r.file_path,
        "device_type": r.device_type.value,
        "duration_seconds": r.duration_seconds,
        "has_imu": r.has_imu,
        "has_gps": r.has_gps,
        "imu_sample_count": r.imu_sample_count,
        "imu_rate_hz": r.imu_rate_hz,
        "gps_sample_count": r.gps_sample_count,
        "gps_rate_hz": r.gps_rate_hz,
        "video_fps": r.video_fps,
        "video_resolution_w": r.video_resolution[0],
        "video_resolution_h": r.video_resolution[1],
        "file_size_mb": r.file_size_mb,
        "issues": r.issues,
    }


def _dict_to_session(d: dict[str, Any]) -> Session:
    audit_results = [
        AuditResult(
            file_path=r["file_path"],
            device_type=DeviceType(r["device_type"]),
            duration_seconds=r["duration_seconds"],
            has_imu=r["has_imu"],
            has_gps=r["has_gps"],
            imu_sample_count=r["imu_sample_count"],
            imu_rate_hz=r["imu_rate_hz"],
            gps_sample_count=r["gps_sample_count"],
            gps_rate_hz=r["gps_rate_hz"],
            video_fps=r["video_fps"],
            video_resolution=(r["video_resolution_w"], r["video_resolution_h"]),
            file_size_mb=r["file_size_mb"],
            issues=r.get("issues", []),
        )
        for r in d.get("audit_results", [])
    ]
    return Session(
        id=d["id"],
        name=d["name"],
        created_at=datetime.fromisoformat(d["created_at"]),
        environment=d["environment"],
        location=d["location"],
        files=d.get("files", []),
        audit_results=audit_results,
        extraction_status=d.get("extraction_status", "pending"),
        notes=d.get("notes", ""),
    )
