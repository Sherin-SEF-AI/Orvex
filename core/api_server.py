"""
core/api_server.py — Standalone FastAPI REST server for RoverDataKit.

Exposes all pipeline capabilities via authenticated REST API.
Runs standalone:  python -m core.api_server
Embedded use:     start_api_server_thread(sm, host, port)

API key is generated on first run, stored as SHA-256 hash in a TOML config
file, and shown in plaintext exactly once.  Rule 21: never log the key.

All routes use prefix /api/v1/.
Public routes (no auth): /api/v1/health, /api/v1/version.
All other routes require:  Authorization: Bearer <api_key>
"""
from __future__ import annotations

import csv
import hashlib
import io
import os
import secrets
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# Optional imports — wrap so the desktop app can import this module even when
# fastapi / uvicorn are not installed, then fail loudly only at runtime.
# ---------------------------------------------------------------------------
try:
    from fastapi import Depends, FastAPI, HTTPException, Request, Security
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

try:
    import uvicorn
    _UVICORN_AVAILABLE = True
except ImportError:
    _UVICORN_AVAILABLE = False

try:
    import toml
    _TOML_AVAILABLE = True
except ImportError:
    _TOML_AVAILABLE = False

try:
    from pydantic import BaseModel as _BaseModel
    _PYDANTIC_AVAILABLE = True
except ImportError:
    _PYDANTIC_AVAILABLE = False

# ---------------------------------------------------------------------------
# API version string and feature list
# ---------------------------------------------------------------------------

_API_VERSION = "1.0.0"
_FEATURES = [
    "segmentation",
    "occupancy",
    "lanes",
    "tracking",
    "versioning",
    "experiments",
    "edge_export",
]

# ---------------------------------------------------------------------------
# Background task store
# ---------------------------------------------------------------------------

_tasks: dict[str, dict[str, Any]] = {}
_tasks_lock = threading.Lock()


def _new_task() -> str:
    """Allocate a new task entry and return its ID."""
    task_id = secrets.token_hex(8)
    with _tasks_lock:
        _tasks[task_id] = {
            "status": "running",
            "progress": 0,
            "result": None,
            "error": None,
        }
    return task_id


def _update_task(task_id: str, **kwargs: Any) -> None:
    with _tasks_lock:
        if task_id in _tasks:
            _tasks[task_id].update(kwargs)


def _get_task(task_id: str) -> dict[str, Any] | None:
    with _tasks_lock:
        return dict(_tasks.get(task_id, {})) if task_id in _tasks else None


# ---------------------------------------------------------------------------
# API key management
# ---------------------------------------------------------------------------

def create_api_key() -> str:
    """Generate a secure random API key."""
    return secrets.token_urlsafe(32)


def hash_api_key(key: str) -> str:
    """Return SHA-256 hex digest of *key* for storage."""
    return hashlib.sha256(key.encode()).hexdigest()


def verify_api_key(provided_key: str, stored_hash: str) -> bool:
    """Return True when hash of *provided_key* matches *stored_hash*."""
    return hashlib.sha256(provided_key.encode()).hexdigest() == stored_hash


def load_or_create_api_key(config_path: str) -> tuple[str | None, bool]:
    """Load API key hash from *config_path* (TOML).

    If no key exists, generate one, store only the hash, and return the
    plaintext key (shown once, never stored).

    Returns:
        (plaintext_key, True)  — new key was generated.
        (None, False)          — existing key found; plaintext not available.
    """
    if not _TOML_AVAILABLE:
        raise ImportError(
            "Package 'toml' is required for API key management. "
            "Install it: pip install toml"
        )

    path = Path(config_path)
    config: dict[str, Any] = {}

    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                config = toml.load(f)
        except Exception as exc:
            logger.warning("api_server: could not read config '{}': {}", path, exc)

    existing_hash = config.get("api_server", {}).get("api_key_hash")
    if existing_hash:
        logger.debug("api_server: loaded existing API key hash from '{}'", path)
        return None, False

    # Generate new key — store only the hash
    new_key = create_api_key()
    new_hash = hash_api_key(new_key)

    if "api_server" not in config:
        config["api_server"] = {}
    config["api_server"]["api_key_hash"] = new_hash

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            toml.dump(config, f)
        logger.info("api_server: new API key hash written to '{}'", path)
    except Exception as exc:
        logger.error("api_server: could not write config '{}': {}", path, exc)

    # Rule 21 — key shown once, never logged
    return new_key, True


# ---------------------------------------------------------------------------
# Request / response Pydantic schemas (local — no UI dependency)
# ---------------------------------------------------------------------------

if _PYDANTIC_AVAILABLE:
    from pydantic import BaseModel

    class _CreateSessionBody(BaseModel):
        name: str
        environment: str
        location: str
        notes: str = ""

    class _AuditBody(BaseModel):
        session_id: str

    class _ExtractionBody(BaseModel):
        session_id: str
        frame_fps: float = 5.0
        frame_format: str = "jpg"
        frame_quality: int = 95
        output_format: str = "euroc"
        sync_devices: bool = True
        imu_interpolation: bool = True


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(session_manager) -> "FastAPI":  # noqa: F821
    """Create and return a configured FastAPI application.

    Args:
        session_manager: A ``core.session_manager.SessionManager`` instance.
                         The app holds a reference — no global state leaks.

    Returns:
        FastAPI application instance.
    """
    if not _FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is not installed. Install it with: "
            "pip install fastapi uvicorn[standard]"
        )

    # Track uptime
    _start_time = datetime.now()

    sm = session_manager

    app = FastAPI(
        title="RoverDataKit REST API",
        version=_API_VERSION,
        description=(
            "Phase 3 REST API for RoverDataKit.\n\n"
            "Authenticate with: `Authorization: Bearer <api_key>`\n\n"
            "Public endpoints: `/api/v1/health`, `/api/v1/version`"
        ),
        docs_url="/api/v1/docs",
        redoc_url="/api/v1/redoc",
        openapi_url="/api/v1/openapi.json",
    )

    # ------------------------------------------------------------------
    # API key dependency
    # ------------------------------------------------------------------

    _bearer_scheme = HTTPBearer(auto_error=False)

    # The app stores the active key hash on its state so it can be
    # regenerated without restarting.
    app.state.api_key_hash: str | None = None

    def _require_api_key(
        credentials: HTTPAuthorizationCredentials | None = Security(_bearer_scheme),
    ) -> str:
        """FastAPI dependency — validate Bearer token."""
        stored_hash = app.state.api_key_hash
        if stored_hash is None:
            # No key configured — open access (development mode warning).
            logger.warning("api_server: no API key configured — all requests accepted")
            return "unauthenticated"
        if credentials is None:
            raise HTTPException(
                status_code=401,
                detail="Missing Authorization header. Use: Authorization: Bearer <api_key>",
            )
        if not verify_api_key(credentials.credentials, stored_hash):
            raise HTTPException(
                status_code=401,
                detail="Invalid API key. Generate a new key in the API Settings panel.",
            )
        return credentials.credentials

    # ------------------------------------------------------------------
    # Response helpers
    # ------------------------------------------------------------------

    def _ok(data: Any) -> dict:
        return {"data": data, "error": None}

    def _err(message: str) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"data": None, "error": message},
        )

    # ------------------------------------------------------------------
    # Public routes — no auth
    # ------------------------------------------------------------------

    @app.get("/api/v1/health", tags=["meta"])
    def health() -> dict:
        uptime = int((datetime.now() - _start_time).total_seconds())
        return _ok({"status": "ok", "version": _API_VERSION, "uptime_seconds": uptime})

    @app.get("/api/v1/version", tags=["meta"])
    def version() -> dict:
        return _ok({"version": _API_VERSION, "features": _FEATURES})

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    @app.get("/api/v1/sessions/", tags=["sessions"])
    def list_sessions(_auth: str = Depends(_require_api_key)) -> dict:
        sessions = sm.list_sessions()
        summaries = [
            {
                "id": s.id,
                "name": s.name,
                "environment": s.environment,
                "location": s.location,
                "created_at": s.created_at.isoformat(),
                "extraction_status": s.extraction_status,
            }
            for s in sessions
        ]
        return _ok(summaries)

    @app.post("/api/v1/sessions/", tags=["sessions"], status_code=201)
    def create_session(
        body: _CreateSessionBody,
        _auth: str = Depends(_require_api_key),
    ) -> dict:
        try:
            s = sm.create_session(
                name=body.name,
                environment=body.environment,
                location=body.location,
                notes=body.notes,
            )
            return _ok({"id": s.id})
        except Exception as exc:
            logger.error("create_session failed: {}", exc)
            return _err(str(exc))

    @app.get("/api/v1/sessions/{session_id}", tags=["sessions"])
    def get_session(
        session_id: str,
        _auth: str = Depends(_require_api_key),
    ) -> dict:
        from core.session_manager import SessionNotFoundError
        try:
            s = sm.get_session(session_id)
        except SessionNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"Session '{session_id}' not found.",
            )
        return _ok({
            "id": s.id,
            "name": s.name,
            "environment": s.environment,
            "location": s.location,
            "created_at": s.created_at.isoformat(),
            "extraction_status": s.extraction_status,
            "notes": s.notes,
            "files": s.files,
            "audit_results": [r.model_dump() for r in s.audit_results],
        })

    @app.delete("/api/v1/sessions/{session_id}", tags=["sessions"])
    def delete_session(
        session_id: str,
        _auth: str = Depends(_require_api_key),
    ) -> dict:
        from core.session_manager import SessionNotFoundError
        try:
            sm.delete_session(session_id, delete_files=False)
        except SessionNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"Session '{session_id}' not found.",
            )
        return _ok({"deleted": session_id})

    @app.get("/api/v1/sessions/{session_id}/status", tags=["sessions"])
    def session_status(
        session_id: str,
        _auth: str = Depends(_require_api_key),
    ) -> dict:
        from core.session_manager import SessionNotFoundError
        try:
            s = sm.get_session(session_id)
        except SessionNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"Session '{session_id}' not found.",
            )

        session_dir = sm.sessions_root / session_id

        # Frames
        frames_dir = session_dir / "extraction" / "cam0" / "data"
        frame_paths = list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")) \
            if frames_dir.exists() else []

        # Annotations (look for annotations.json)
        ann_file = session_dir / "annotations.json"
        ann_count = 0
        if ann_file.exists():
            try:
                import json
                with open(ann_file, encoding="utf-8") as f:
                    ann_data = json.load(f)
                ann_count = len(ann_data) if isinstance(ann_data, list) else 0
            except Exception:
                ann_count = 0

        has_depth = (session_dir / "depth").exists()
        has_seg = (session_dir / "segmentation").exists()
        has_lanes = (session_dir / "lanes").exists()
        has_tracking = (session_dir / "tracking").exists()

        return _ok({
            "session_id": session_id,
            "has_frames": len(frame_paths) > 0,
            "frame_count": len(frame_paths),
            "has_annotations": ann_count > 0,
            "annotation_count": ann_count,
            "has_depth": has_depth,
            "has_segmentation": has_seg,
            "has_lanes": has_lanes,
            "has_tracking": has_tracking,
        })

    # ------------------------------------------------------------------
    # Pipeline — audit
    # ------------------------------------------------------------------

    @app.post("/api/v1/pipeline/audit", tags=["pipeline"])
    def run_audit(
        body: _AuditBody,
        _auth: str = Depends(_require_api_key),
    ) -> dict:
        from core.session_manager import SessionNotFoundError
        try:
            sm.get_session(body.session_id)
        except SessionNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"Session '{body.session_id}' not found.",
            )

        task_id = _new_task()

        def _worker():
            try:
                from core.audit import audit_session
                results = audit_session(sm, body.session_id, progress_cb=lambda p: _update_task(task_id, progress=p))
                _update_task(
                    task_id,
                    status="done",
                    progress=100,
                    result=[r.model_dump() for r in results],
                )
                logger.info("api_server: audit task {} completed", task_id)
            except Exception as exc:
                _update_task(task_id, status="failed", error=str(exc))
                logger.error("api_server: audit task {} failed: {}", task_id, exc)

        threading.Thread(target=_worker, daemon=True, name=f"audit-{task_id}").start()
        return _ok({"task_id": task_id})

    # ------------------------------------------------------------------
    # Pipeline — extraction
    # ------------------------------------------------------------------

    @app.post("/api/v1/pipeline/extract", tags=["pipeline"])
    def run_extraction(
        body: _ExtractionBody,
        _auth: str = Depends(_require_api_key),
    ) -> dict:
        from core.session_manager import SessionNotFoundError
        try:
            sm.get_session(body.session_id)
        except SessionNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"Session '{body.session_id}' not found.",
            )

        task_id = _new_task()

        def _worker():
            try:
                from core.models import ExtractionConfig
                config = ExtractionConfig(
                    session_id=body.session_id,
                    frame_fps=body.frame_fps,
                    frame_format=body.frame_format,
                    frame_quality=body.frame_quality,
                    output_format=body.output_format,  # type: ignore[arg-type]
                    sync_devices=body.sync_devices,
                    imu_interpolation=body.imu_interpolation,
                )
                from core.extractor_gopro import extract_gopro
                session = sm.get_session(body.session_id)
                mp4_files = [f for f in session.files if f.lower().endswith(".mp4")]
                if not mp4_files:
                    raise ValueError(
                        f"Session '{body.session_id}' has no .MP4 files to extract. "
                        "Add GoPro or exported video files first."
                    )
                result = extract_gopro(mp4_files[0], config)
                _update_task(
                    task_id,
                    status="done",
                    progress=100,
                    result={
                        "session_id": result.session_id,
                        "frame_count": len(result.frame_paths),
                        "imu_sample_count": len(result.imu_samples),
                        "gps_sample_count": len(result.gps_samples),
                        "duration_seconds": result.duration_seconds,
                        "stats": result.stats,
                    },
                )
                logger.info("api_server: extraction task {} completed", task_id)
            except Exception as exc:
                _update_task(task_id, status="failed", error=str(exc))
                logger.error("api_server: extraction task {} failed: {}", task_id, traceback.format_exc())

        threading.Thread(target=_worker, daemon=True, name=f"extract-{task_id}").start()
        return _ok({"task_id": task_id})

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------

    @app.get("/api/v1/tasks/{task_id}", tags=["tasks"])
    def get_task(
        task_id: str,
        _auth: str = Depends(_require_api_key),
    ) -> dict:
        info = _get_task(task_id)
        if info is None:
            raise HTTPException(
                status_code=404,
                detail=f"Task '{task_id}' not found. It may have expired.",
            )
        return _ok({
            "task_id": task_id,
            "status": info["status"],
            "progress": info["progress"],
            "result": info["result"],
            "error": info["error"],
        })

    # ------------------------------------------------------------------
    # Frames
    # ------------------------------------------------------------------

    @app.get("/api/v1/sessions/{session_id}/frames", tags=["frames"])
    def list_frames(
        session_id: str,
        page: int = 1,
        per_page: int = 50,
        _auth: str = Depends(_require_api_key),
    ) -> dict:
        from core.session_manager import SessionNotFoundError
        try:
            sm.get_session(session_id)
        except SessionNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"Session '{session_id}' not found.",
            )

        frames_dir = sm.sessions_root / session_id / "extraction" / "cam0" / "data"
        if not frames_dir.exists():
            return _ok({"frames": [], "total": 0, "page": page, "per_page": per_page})

        all_frames = sorted(
            [p.name for p in frames_dir.iterdir() if p.suffix.lower() in (".jpg", ".png")]
        )
        total = len(all_frames)
        start = (page - 1) * per_page
        end = start + per_page
        page_frames = all_frames[start:end]

        return _ok({
            "frames": [f"{session_id}/{f}" for f in page_frames],
            "total": total,
            "page": page,
            "per_page": per_page,
        })

    @app.get("/frames/{session_id}/{frame_name}", tags=["frames"])
    def serve_frame(
        session_id: str,
        frame_name: str,
        request: Request,
    ) -> FileResponse:
        """Serve the raw image file.

        Auth is checked via query param ``key`` OR Authorization header for
        browser/direct URL access convenience.
        """
        # Lightweight auth — check header or query param
        stored_hash = app.state.api_key_hash
        if stored_hash is not None:
            auth_header = request.headers.get("authorization", "")
            query_key = request.query_params.get("key", "")
            provided = ""
            if auth_header.lower().startswith("bearer "):
                provided = auth_header[7:]
            elif query_key:
                provided = query_key
            if not provided or not verify_api_key(provided, stored_hash):
                raise HTTPException(
                    status_code=401,
                    detail="Valid API key required to fetch frames.",
                )

        frame_path = (
            sm.sessions_root
            / session_id
            / "extraction"
            / "cam0"
            / "data"
            / frame_name
        )
        if not frame_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Frame '{frame_name}' not found in session '{session_id}'.",
            )
        return FileResponse(str(frame_path))

    # ------------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------------

    @app.get("/api/v1/sessions/{session_id}/annotations", tags=["annotations"])
    def list_annotations(
        session_id: str,
        _auth: str = Depends(_require_api_key),
    ) -> dict:
        from core.session_manager import SessionNotFoundError
        try:
            sm.get_session(session_id)
        except SessionNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"Session '{session_id}' not found.",
            )

        ann_file = sm.sessions_root / session_id / "annotations.json"
        if not ann_file.exists():
            return _ok([])

        try:
            import json
            with open(ann_file, encoding="utf-8") as f:
                data = json.load(f)
            return _ok(data if isinstance(data, list) else [])
        except Exception as exc:
            logger.error("api_server: could not read annotations for '{}': {}", session_id, exc)
            return _err(f"Could not read annotations file: {exc}")

    # ------------------------------------------------------------------
    # IMU data
    # ------------------------------------------------------------------

    @app.get("/api/v1/sessions/{session_id}/imu", tags=["telemetry"])
    def get_imu(
        session_id: str,
        start_ns: int | None = None,
        end_ns: int | None = None,
        _auth: str = Depends(_require_api_key),
    ) -> dict:
        from core.session_manager import SessionNotFoundError
        try:
            sm.get_session(session_id)
        except SessionNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"Session '{session_id}' not found.",
            )

        imu_csv = sm.sessions_root / session_id / "extraction" / "imu0" / "data.csv"
        if not imu_csv.exists():
            return _ok([])

        samples: list[dict] = []
        try:
            with open(imu_csv, encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f, skipinitialspace=True)
                for row in reader:
                    # Skip comment/header rows
                    first_col = next(iter(row.values()), "")
                    if first_col.startswith("#"):
                        continue
                    try:
                        # EuRoC column names (strip #timestamp header comment)
                        ts_key = next(
                            (k for k in row if "timestamp" in k.lower()), None
                        )
                        if ts_key is None:
                            continue
                        ts = int(row[ts_key])
                        if start_ns is not None and ts < start_ns:
                            continue
                        if end_ns is not None and ts > end_ns:
                            continue
                        samples.append({
                            "timestamp_ns": ts,
                            "gyro_x": float(row.get("w_RS_S_x [rad s^-1]", row.get("gyro_x", 0.0))),
                            "gyro_y": float(row.get("w_RS_S_y [rad s^-1]", row.get("gyro_y", 0.0))),
                            "gyro_z": float(row.get("w_RS_S_z [rad s^-1]", row.get("gyro_z", 0.0))),
                            "accel_x": float(row.get("a_RS_S_x [m s^-2]", row.get("accel_x", 0.0))),
                            "accel_y": float(row.get("a_RS_S_y [m s^-2]", row.get("accel_y", 0.0))),
                            "accel_z": float(row.get("a_RS_S_z [m s^-2]", row.get("accel_z", 0.0))),
                        })
                    except (ValueError, KeyError):
                        continue
        except Exception as exc:
            logger.error("api_server: could not read IMU CSV for '{}': {}", session_id, exc)
            return _err(f"Could not read IMU data: {exc}")

        return _ok(samples)

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------

    @app.get("/api/v1/models/", tags=["models"])
    def list_models(_auth: str = Depends(_require_api_key)) -> dict:
        models_dir = Path("data") / "models"
        if not models_dir.exists():
            return _ok([])
        model_files = []
        for pt_file in sorted(models_dir.rglob("*.pt")):
            stat = pt_file.stat()
            model_files.append({
                "name": pt_file.stem,
                "path": str(pt_file),
                "size_mb": round(stat.st_size / 1_048_576, 2),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
        return _ok(model_files)

    # ------------------------------------------------------------------
    # Global exception handler
    # ------------------------------------------------------------------

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("api_server: unhandled exception on {}: {}", request.url, exc)
        return JSONResponse(
            status_code=500,
            content={"data": None, "error": str(exc), "detail": traceback.format_exc()},
        )

    logger.debug("api_server: app created (version={})", _API_VERSION)
    return app


# ---------------------------------------------------------------------------
# Server launch helpers
# ---------------------------------------------------------------------------

def start_api_server_thread(
    session_manager,
    host: str = "127.0.0.1",
    port: int = 8080,
    api_key_config_path: str = "data/config.toml",
) -> "threading.Thread":
    """Start uvicorn in a daemon thread and return the thread.

    The API key hash is loaded / generated before the thread starts.
    If a new key is created it is printed to stdout exactly once (Rule 21).

    Args:
        session_manager: SessionManager instance to expose via the API.
        host:            Bind address (default: loopback-only).
        port:            TCP port (default: 8080).
        api_key_config_path: TOML file path for API key hash storage.

    Returns:
        threading.Thread (daemon, already started).

    Raises:
        ImportError: if FastAPI or uvicorn are not installed.
    """
    if not _FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is required to run the API server. "
            "Install it: pip install fastapi uvicorn[standard]"
        )
    if not _UVICORN_AVAILABLE:
        raise ImportError(
            "uvicorn is required to run the API server. "
            "Install it: pip install uvicorn[standard]"
        )

    plaintext_key, is_new = load_or_create_api_key(api_key_config_path)

    if is_new:
        # Rule 21 — shown once in plaintext, NEVER written to logs
        print("\n" + "=" * 60)
        print("  RoverDataKit API Server — New API Key Generated")
        print("  Save this key now — it will NOT be shown again.")
        print(f"\n  API Key: {plaintext_key}\n")
        print("=" * 60 + "\n")

    app = create_app(session_manager)

    # Store the hash on the app so routes can verify keys
    if _TOML_AVAILABLE:
        cfg_path = Path(api_key_config_path)
        if cfg_path.exists():
            try:
                import toml as _toml
                with open(cfg_path, encoding="utf-8") as _f:
                    _cfg = _toml.load(_f)
                app.state.api_key_hash = _cfg.get("api_server", {}).get("api_key_hash")
            except Exception as exc:
                logger.warning("api_server: could not reload key hash: {}", exc)

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",   # keep uvicorn logs quiet; we use loguru
        access_log=False,
    )
    server = uvicorn.Server(config)

    def _run() -> None:
        logger.info("api_server: starting on http://{}:{}", host, port)
        server.run()
        logger.info("api_server: server stopped")

    thread = threading.Thread(target=_run, daemon=True, name="api-server")
    thread.start()
    return thread


def run_server(
    session_manager,
    host: str = "0.0.0.0",
    port: int = 8080,
    api_key_config_path: str = "data/config.toml",
) -> None:
    """Blocking server start — used by ``__main__`` entry point.

    Unlike ``start_api_server_thread`` this call does not return until the
    server is killed.
    """
    if not _FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is not installed. Install it: pip install fastapi uvicorn[standard]"
        )
    if not _UVICORN_AVAILABLE:
        raise ImportError(
            "uvicorn is not installed. Install it: pip install uvicorn[standard]"
        )

    plaintext_key, is_new = load_or_create_api_key(api_key_config_path)
    if is_new:
        print("\n" + "=" * 60)
        print("  RoverDataKit API Server — New API Key Generated")
        print("  Save this key now — it will NOT be shown again.")
        print(f"\n  API Key: {plaintext_key}\n")
        print("=" * 60 + "\n")

    app = create_app(session_manager)

    if _TOML_AVAILABLE:
        cfg_path = Path(api_key_config_path)
        if cfg_path.exists():
            try:
                import toml as _toml
                with open(cfg_path, encoding="utf-8") as _f:
                    _cfg = _toml.load(_f)
                app.state.api_key_hash = _cfg.get("api_server", {}).get("api_key_hash")
            except Exception as exc:
                logger.warning("api_server: could not reload key hash: {}", exc)

    uvicorn.run(app, host=host, port=port, log_level="info")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path

    _sessions_root = _Path.home() / ".roverdatakit" / "data" / "sessions"
    _config_path = str(_Path.home() / ".roverdatakit" / "data" / "config.toml")

    try:
        from core.session_manager import SessionManager
    except ImportError:
        # Allow running from the project root without installing the package
        sys.path.insert(0, str(_Path(__file__).parent.parent))
        from core.session_manager import SessionManager

    _sm = SessionManager(_sessions_root)
    _host = os.environ.get("ROVER_API_HOST", "127.0.0.1")
    _port = int(os.environ.get("ROVER_API_PORT", "8080"))
    _config = os.environ.get("ROVER_API_CONFIG", _config_path)

    logger.info("Starting RoverDataKit API server on http://{}:{}", _host, _port)
    run_server(_sm, host=_host, port=_port, api_key_config_path=_config)
