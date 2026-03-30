"""
web/backend/websocket.py — WebSocket progress streaming.

The frontend connects to ws://localhost:8000/ws/tasks/{task_id}.
This handler subscribes to the Redis pub/sub channel task:{task_id}
and forwards every JSON message to the connected browser client.

Protocol (JSON):
  {"task_id": str, "status": "running"|"done"|"failed",
   "progress": int (0-100, -1 = message only), "message": str}
"""
from __future__ import annotations

import asyncio
import json

import redis.asyncio as aioredis
from fastapi import WebSocket, WebSocketDisconnect

REDIS_URL = "redis://localhost:6379/0"


async def task_progress_ws(websocket: WebSocket, task_id: str) -> None:
    """Stream task progress from Redis pub/sub to the WebSocket client."""
    await websocket.accept()

    r = aioredis.from_url(REDIS_URL, decode_responses=True)
    pubsub = r.pubsub()
    await pubsub.subscribe(f"task:{task_id}")

    try:
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            data = json.loads(message["data"])
            try:
                await websocket.send_json(data)
            except WebSocketDisconnect:
                break
            # Stop streaming once task reaches a terminal state
            if data.get("status") in ("done", "failed"):
                break
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    finally:
        await pubsub.unsubscribe(f"task:{task_id}")
        await r.aclose()
        try:
            await websocket.close()
        except Exception:
            pass
