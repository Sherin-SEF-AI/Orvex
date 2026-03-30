"""
labelox/web/backend/auth.py — Simple JWT auth for web mode.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_SECRET = "labelox-dev-secret-change-in-production"
_BEARER = HTTPBearer(auto_error=False)


def _b64url_encode(data: bytes) -> str:
    import base64
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    import base64
    s += "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s)


def create_token(user_id: str, name: str, role: str = "annotator") -> str:
    header = _b64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload_data = {
        "sub": user_id,
        "name": name,
        "role": role,
        "iat": int(time.time()),
        "exp": int(time.time()) + 86400 * 7,  # 7 days
    }
    payload = _b64url_encode(json.dumps(payload_data).encode())
    sig_input = f"{header}.{payload}".encode()
    sig = _b64url_encode(hmac.new(_SECRET.encode(), sig_input, hashlib.sha256).digest())
    return f"{header}.{payload}.{sig}"


def decode_token(token: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid token format")
    header_b, payload_b, sig_b = parts
    sig_input = f"{header_b}.{payload_b}".encode()
    expected = hmac.new(_SECRET.encode(), sig_input, hashlib.sha256).digest()
    actual = _b64url_decode(sig_b)
    if not hmac.compare_digest(expected, actual):
        raise ValueError("Invalid signature")
    payload = json.loads(_b64url_decode(payload_b))
    if payload.get("exp", 0) < time.time():
        raise ValueError("Token expired")
    return payload


async def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_BEARER),
) -> dict[str, Any]:
    """FastAPI dependency — returns user dict or raises 401."""
    if creds is None:
        # Allow anonymous in dev mode
        return {"sub": "anonymous", "name": "Anonymous", "role": "admin"}
    try:
        return decode_token(creds.credentials)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc))


def require_role(role: str):
    """Dependency factory for role-based access."""
    async def _check(user: dict = Depends(get_current_user)):
        if user.get("role") not in (role, "admin"):
            raise HTTPException(status_code=403, detail=f"Requires role: {role}")
        return user
    return _check
