"""
labelox/web/backend/websocket.py — WebSocket endpoints for task progress + collaboration.
"""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from labelox.web.backend.tasks import TaskStatus, get_task

router = APIRouter()


@router.websocket("/ws/tasks/{task_id}")
async def task_progress(websocket: WebSocket, task_id: str):
    """Stream task progress updates to the client."""
    await websocket.accept()
    try:
        while True:
            task = get_task(task_id)
            if task is None:
                await websocket.send_json({"error": "Task not found"})
                break

            await websocket.send_json({
                "task_id": task.id,
                "status": task.status.value,
                "progress": task.progress,
                "total": task.total,
                "message": task.message,
                "error": task.error,
            })

            if task.status in (TaskStatus.SUCCESS, TaskStatus.FAILURE):
                break

            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass


# ─── Collaboration (multi-user annotation sync) ─────────────────────────────

_project_connections: dict[str, list[WebSocket]] = {}


@router.websocket("/ws/collaborate/{project_id}")
async def collaborate(websocket: WebSocket, project_id: str):
    """Real-time annotation sync between collaborators."""
    await websocket.accept()

    if project_id not in _project_connections:
        _project_connections[project_id] = []
    _project_connections[project_id].append(websocket)

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            # Broadcast to all other connections in same project
            for ws in _project_connections.get(project_id, []):
                if ws != websocket:
                    try:
                        await ws.send_text(data)
                    except Exception:
                        pass
    except WebSocketDisconnect:
        conns = _project_connections.get(project_id, [])
        if websocket in conns:
            conns.remove(websocket)
