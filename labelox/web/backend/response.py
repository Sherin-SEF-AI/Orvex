"""
labelox/web/backend/response.py — Standard API response helpers.
"""
from __future__ import annotations

from typing import Any

from fastapi.responses import JSONResponse


def ok(data: Any = None) -> dict:
    return {"data": data, "error": None}


def err(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"data": None, "error": message},
    )
