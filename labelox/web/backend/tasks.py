"""
labelox/web/backend/tasks.py — Background task registry.

In production, these would be Celery tasks. For now, we use a simple
in-process thread-pool approach with task status tracking.
"""
from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass
class TaskRecord:
    id: str
    name: str
    status: TaskStatus = TaskStatus.PENDING
    progress: int = 0
    total: int = 0
    message: str = ""
    result: Any = None
    error: str | None = None


_tasks: dict[str, TaskRecord] = {}
_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=4)


def get_task(task_id: str) -> TaskRecord | None:
    return _tasks.get(task_id)


def list_tasks() -> list[TaskRecord]:
    return list(_tasks.values())


def submit_task(name: str, fn: Callable, *args, **kwargs) -> str:
    """Submit a function to run in the background. Returns task_id."""
    task_id = str(uuid.uuid4())
    record = TaskRecord(id=task_id, name=name)
    _tasks[task_id] = record

    def _progress_cb(current: int, total: int) -> None:
        record.progress = current
        record.total = total

    def _status_cb(msg: str) -> None:
        record.message = msg

    def _run() -> None:
        record.status = TaskStatus.RUNNING
        try:
            result = fn(
                *args,
                progress_callback=_progress_cb,
                status_callback=_status_cb,
                **kwargs,
            )
            record.status = TaskStatus.SUCCESS
            record.result = result
        except TypeError:
            # Function doesn't accept callbacks
            try:
                result = fn(*args, **kwargs)
                record.status = TaskStatus.SUCCESS
                record.result = result
            except Exception as exc:
                record.status = TaskStatus.FAILURE
                record.error = str(exc)
        except Exception as exc:
            record.status = TaskStatus.FAILURE
            record.error = str(exc)

    _executor.submit(_run)
    return task_id
