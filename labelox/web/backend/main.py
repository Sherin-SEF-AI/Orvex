"""
labelox/web/backend/main.py — FastAPI application entry point.

Usage: uvicorn labelox.web.backend.main:app --reload
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from labelox.web.backend.response import err, ok


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise DB on startup."""
    from labelox.core.database import init_db
    init_db("sqlite:///labelox_web.db")
    yield


app = FastAPI(
    title="LABELOX API",
    version="0.1.0",
    description="AI-powered annotation platform API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return err(str(exc), 500)


# ─── Routes ──────────────────────────────────────────────────────────────────

from labelox.web.backend.routes.projects import router as projects_router
from labelox.web.backend.routes.images import router as images_router
from labelox.web.backend.routes.annotations import router as annotations_router
from labelox.web.backend.routes.auto_annotate import router as auto_annotate_router
from labelox.web.backend.routes.sam import router as sam_router
from labelox.web.backend.routes.export import router as export_router
from labelox.web.backend.routes.review import router as review_router
from labelox.web.backend.routes.stats import router as stats_router
from labelox.web.backend.routes.users import router as users_router
from labelox.web.backend.websocket import router as ws_router

app.include_router(projects_router, prefix="/api/v1/projects", tags=["projects"])
app.include_router(images_router, prefix="/api/v1/images", tags=["images"])
app.include_router(annotations_router, prefix="/api/v1/annotations", tags=["annotations"])
app.include_router(auto_annotate_router, prefix="/api/v1/auto-annotate", tags=["auto-annotate"])
app.include_router(sam_router, prefix="/api/v1/sam", tags=["sam"])
app.include_router(export_router, prefix="/api/v1/export", tags=["export"])
app.include_router(review_router, prefix="/api/v1/review", tags=["review"])
app.include_router(stats_router, prefix="/api/v1/stats", tags=["stats"])
app.include_router(users_router, prefix="/api/v1/users", tags=["users"])
app.include_router(ws_router)


@app.get("/api/v1/health")
async def health():
    return ok({"status": "healthy"})
