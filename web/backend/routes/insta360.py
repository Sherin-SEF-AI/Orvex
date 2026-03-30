"""
web/backend/routes/insta360.py — Insta360 X4 360° processing endpoints.

Routes at /insta360/:
  POST /scan               — detect INSV pairs in a folder
  POST /validate           — pre-flight validation for a pair
  POST /process            — start full pipeline (returns task_id)
  GET  /tasks/{task_id}    — task status + per-stage progress
  GET  /{session_id}/gps   — GPS samples
  GET  /{session_id}/imu   — IMU samples (paginated)
  GET  /{session_id}/frames/{view}            — paginated frame list
  GET  /{session_id}/frames/{view}/{name}     — serve frame image
  GET  /{session_id}/preview/equirect         — preview frame JPEG
  GET  /{session_id}/manifest                 — dataset manifest JSON
  GET  /health             — ffmpeg v360 + exiftool capability check
"""
from __future__ import annotations

import csv
import json
import threading
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from core.models import INSVPair, Insta360ProcessingConfig

router = APIRouter(prefix="/insta360", tags=["insta360"])

# In-process task dict (mirrors api_server.py pattern)
_tasks: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ScanRequest(BaseModel):
    folder_path: str


class ValidateRequest(BaseModel):
    pair: dict           # INSVPair as dict
    output_dir: str


class ProcessRequest(BaseModel):
    pair: dict                    # INSVPair as dict
    config: dict                  # Insta360ProcessingConfig as dict
    output_dir: str
    session_id: str = ""


# ---------------------------------------------------------------------------
# POST /scan
# ---------------------------------------------------------------------------

@router.post("/scan", summary="Detect INSV file pairs in a folder")
def scan_folder(body: ScanRequest) -> dict:
    folder = Path(body.folder_path)
    if not folder.exists():
        raise HTTPException(status_code=400, detail=f"Folder not found: {body.folder_path}")
    try:
        from core.insv_telemetry import find_insv_pairs
        pairs = find_insv_pairs(str(folder))
        return {"data": [p.model_dump() for p in pairs], "error": None}
    except Exception as exc:
        return {"data": [], "error": str(exc)}


# ---------------------------------------------------------------------------
# POST /validate
# ---------------------------------------------------------------------------

@router.post("/validate", summary="Pre-flight validation for an INSV pair")
def validate_pair(body: ValidateRequest) -> dict:
    try:
        from core.insta360_processor import validate_insv_pair
        pair = INSVPair(**body.pair)
        issues = validate_insv_pair(pair, body.output_dir)
        return {"data": {"valid": len(issues) == 0, "issues": issues}, "error": None}
    except Exception as exc:
        return {"data": {"valid": False, "issues": [str(exc)]}, "error": str(exc)}


# ---------------------------------------------------------------------------
# POST /process  (async pipeline)
# ---------------------------------------------------------------------------

@router.post("/process", summary="Start Insta360 processing pipeline")
def start_process(body: ProcessRequest) -> dict:
    task_id = str(uuid.uuid4())
    _tasks[task_id] = {"status": "running", "progress": 0, "stage": "", "result": None, "error": None}

    def _run() -> None:
        try:
            from core.insta360_processor import run_full_insta360_pipeline
            pair = INSVPair(**body.pair)
            config = Insta360ProcessingConfig(**body.config)
            session_id = body.session_id or f"insta360_{pair.base_name}"

            Path(body.output_dir).mkdir(parents=True, exist_ok=True)

            def progress_cb(stage: str, pct: int) -> None:
                _tasks[task_id]["stage"] = stage
                _tasks[task_id]["progress"] = pct

            result = run_full_insta360_pipeline(
                insv_pair=pair,
                config=config,
                output_dir=body.output_dir,
                session_id=session_id,
                progress_callback=progress_cb,
            )
            _tasks[task_id]["status"] = "done"
            _tasks[task_id]["progress"] = 100
            _tasks[task_id]["result"] = result.model_dump()
        except Exception as exc:
            _tasks[task_id]["status"] = "failed"
            _tasks[task_id]["error"] = str(exc)

    threading.Thread(target=_run, daemon=True).start()
    return {"data": {"task_id": task_id}, "error": None}


# ---------------------------------------------------------------------------
# GET /tasks/{task_id}
# ---------------------------------------------------------------------------

@router.get("/tasks/{task_id}", summary="Get pipeline task status")
def get_task(task_id: str) -> dict:
    task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return {"data": task, "error": None}


# ---------------------------------------------------------------------------
# GET /{session_id}/gps
# ---------------------------------------------------------------------------

@router.get("/{session_id}/gps", summary="Get GPS samples for a processed session")
def get_gps(session_id: str, request: Request) -> dict:
    sm = request.app.state.session_manager
    dataset_dir = _find_dataset_dir(sm, session_id)
    gps_csv = dataset_dir / "gps" / "data.csv"
    if not gps_csv.exists():
        return {"data": [], "error": None}
    samples = []
    try:
        with gps_csv.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                samples.append({
                    "timestamp_ns": int(row["timestamp_ns"]),
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                    "altitude_m": float(row.get("altitude_m", 0)),
                    "speed_mps": float(row.get("speed_mps", 0)),
                    "fix_type": int(row.get("fix_type", 3)),
                })
    except Exception as exc:
        return {"data": [], "error": str(exc)}
    return {"data": samples, "error": None}


# ---------------------------------------------------------------------------
# GET /{session_id}/imu  (paginated)
# ---------------------------------------------------------------------------

@router.get("/{session_id}/imu", summary="Get IMU samples (paginated)")
def get_imu(session_id: str, request: Request, page: int = 1, per_page: int = 1000) -> dict:
    sm = request.app.state.session_manager
    dataset_dir = _find_dataset_dir(sm, session_id)
    imu_csv = dataset_dir / "imu0" / "data.csv"
    if not imu_csv.exists():
        return {"data": [], "error": None}
    samples = []
    try:
        with imu_csv.open() as f:
            # EuRoC format: skip header comment line
            lines = [l for l in f if not l.startswith("#")]
        reader = csv.DictReader(lines)
        all_samples = list(reader)
        start = (page - 1) * per_page
        end = start + per_page
        for row in all_samples[start:end]:
            try:
                row_keys = list(row.keys())
                samples.append({
                    "timestamp_ns": int(row.get(row_keys[0], 0)),
                    "gyro_x": float(row.get(row_keys[1], 0)),
                    "gyro_y": float(row.get(row_keys[2], 0)),
                    "gyro_z": float(row.get(row_keys[3], 0)),
                    "accel_x": float(row.get(row_keys[4], 0)),
                    "accel_y": float(row.get(row_keys[5], 0)),
                    "accel_z": float(row.get(row_keys[6], 0)),
                })
            except (IndexError, ValueError):
                pass
    except Exception as exc:
        return {"data": [], "error": str(exc)}
    return {"data": samples, "error": None}


# ---------------------------------------------------------------------------
# GET /{session_id}/frames/{view}  (paginated)
# ---------------------------------------------------------------------------

@router.get("/{session_id}/frames/{view}", summary="List frames for a perspective view")
def list_frames(
    session_id: str,
    view: str,
    request: Request,
    page: int = 1,
    per_page: int = 50,
) -> dict:
    if view not in ("front", "right", "rear", "left"):
        raise HTTPException(status_code=400, detail=f"Invalid view: {view}. Use front|right|rear|left")
    sm = request.app.state.session_manager
    dataset_dir = _find_dataset_dir(sm, session_id)
    cam_map = {"front": "cam0", "right": "cam1", "rear": "cam2", "left": "cam3"}
    cam_dir = dataset_dir / cam_map[view] / "data"
    if not cam_dir.exists():
        return {"data": {"frames": [], "total": 0}, "error": None}
    frames = sorted(f.name for f in cam_dir.iterdir() if f.suffix in (".jpg", ".png"))
    total = len(frames)
    start = (page - 1) * per_page
    return {"data": {"frames": frames[start:start + per_page], "total": total, "page": page}, "error": None}


# ---------------------------------------------------------------------------
# GET /{session_id}/frames/{view}/{name}  (serve image)
# ---------------------------------------------------------------------------

@router.get("/{session_id}/frames/{view}/{name}", summary="Serve a perspective frame image")
def get_frame(session_id: str, view: str, name: str, request: Request) -> FileResponse:
    sm = request.app.state.session_manager
    dataset_dir = _find_dataset_dir(sm, session_id)
    cam_map = {"front": "cam0", "right": "cam1", "rear": "cam2", "left": "cam3"}
    cam_dir = dataset_dir / cam_map.get(view, "cam0") / "data"
    frame_path = cam_dir / name
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail=f"Frame not found: {name}")
    media_type = "image/jpeg" if name.endswith(".jpg") else "image/png"
    return FileResponse(str(frame_path), media_type=media_type)


# ---------------------------------------------------------------------------
# GET /{session_id}/preview/equirect
# ---------------------------------------------------------------------------

@router.get("/{session_id}/preview/equirect", summary="Preview equirectangular frame at 5s")
def get_equirect_preview(session_id: str, request: Request, t: float = 5.0) -> FileResponse:
    sm = request.app.state.session_manager
    dataset_dir = _find_dataset_dir(sm, session_id)
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="No manifest found — run processing first")
    manifest = json.loads(manifest_path.read_text())
    equirect_path = manifest.get("source_files", {}).get("equirectangular")
    if not equirect_path or not Path(equirect_path).exists():
        raise HTTPException(status_code=404, detail="Equirectangular video not found")
    try:
        from core.insta360_processor import get_equirect_preview_frame
        preview_path = str(dataset_dir / "preview_equirect.jpg")
        get_equirect_preview_frame(equirect_path, timestamp_seconds=t, output_path=preview_path)
        return FileResponse(preview_path, media_type="image/jpeg")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# GET /{session_id}/manifest
# ---------------------------------------------------------------------------

@router.get("/{session_id}/manifest", summary="Get dataset manifest JSON")
def get_manifest(session_id: str, request: Request) -> dict:
    sm = request.app.state.session_manager
    dataset_dir = _find_dataset_dir(sm, session_id)
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        return {"data": None, "error": "No manifest found — run processing first"}
    try:
        return {"data": json.loads(manifest_path.read_text()), "error": None}
    except Exception as exc:
        return {"data": None, "error": str(exc)}


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@router.get("/health", summary="Check ffmpeg v360 and exiftool availability")
def health() -> dict:
    result: dict = {}
    try:
        from core.insta360_processor import verify_ffmpeg_v360_support
        result["ffmpeg"] = verify_ffmpeg_v360_support()
    except Exception as exc:
        result["ffmpeg"] = {"v360_available": False, "minimum_met": False, "error": str(exc)}
    try:
        import subprocess
        r = subprocess.run(["exiftool", "-ver"], capture_output=True, text=True, timeout=5)
        result["exiftool"] = {"available": r.returncode == 0, "version": r.stdout.strip()}
    except Exception:
        result["exiftool"] = {"available": False, "version": None}
    all_ok = result.get("ffmpeg", {}).get("minimum_met", False) and result.get("exiftool", {}).get("available", False)
    return {"data": result, "error": None if all_ok else "Missing or outdated dependencies"}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _find_dataset_dir(sm, session_id: str) -> Path:
    """Resolve dataset directory for a processed Insta360 session."""
    # Try session folder first
    try:
        base = sm.session_folder(session_id)
        if base.exists():
            return base
    except Exception:
        pass
    # Fallback: look for insta360_output subfolder by session_id prefix
    data_root = Path.home() / ".roverdatakit" / "data"
    candidate = data_root / session_id
    if candidate.exists():
        return candidate
    # Return a non-existent path — callers handle missing files gracefully
    return Path(session_id)
