"""
labelox/core/database.py — SQLAlchemy ORM models and database management.

SQLite for desktop, PostgreSQL for web. Configurable via engine URL.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
    func,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)


# ─── Base ────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return str(uuid.uuid4())


# ─── ORM Models ──────────────────────────────────────────────────────────────

class DBProject(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    settings_json: Mapped[str] = mapped_column(Text, default="{}")
    label_classes_json: Mapped[str] = mapped_column(Text, default="[]")
    annotation_types_json: Mapped[str] = mapped_column(Text, default='["bbox"]')
    image_count: Mapped[int] = mapped_column(Integer, default=0)
    annotated_count: Mapped[int] = mapped_column(Integer, default=0)
    reviewed_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow,
    )

    images: Mapped[list[DBImage]] = relationship(
        back_populates="project", cascade="all, delete-orphan",
    )

    @property
    def settings(self) -> dict:
        return json.loads(self.settings_json)

    @settings.setter
    def settings(self, val: dict) -> None:
        self.settings_json = json.dumps(val)

    @property
    def label_classes(self) -> list[dict]:
        return json.loads(self.label_classes_json)

    @label_classes.setter
    def label_classes(self, val: list[dict]) -> None:
        self.label_classes_json = json.dumps(val)


class DBImage(Base):
    __tablename__ = "images"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="CASCADE"), index=True,
    )
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), default="unlabeled", index=True)
    assigned_to: Mapped[str | None] = mapped_column(String(255), nullable=True)
    auto_annotation_status: Mapped[str] = mapped_column(String(20), default="pending")
    review_decision: Mapped[str | None] = mapped_column(String(20), nullable=True)
    review_comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    sequence_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    frame_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    timestamp_ns: Mapped[int | None] = mapped_column(Integer, nullable=True)
    thumbnail_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    blur_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    scene_class: Mapped[str | None] = mapped_column(String(50), nullable=True)
    md5: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow,
    )

    project: Mapped[DBProject] = relationship(back_populates="images")
    annotations: Mapped[list[DBAnnotation]] = relationship(
        back_populates="image", cascade="all, delete-orphan",
    )


class DBAnnotation(Base):
    __tablename__ = "annotations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    image_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("images.id", ondelete="CASCADE"), index=True,
    )
    label_id: Mapped[str] = mapped_column(String(36), nullable=False)
    label_name: Mapped[str] = mapped_column(String(255), nullable=False)
    annotation_type: Mapped[str] = mapped_column(String(20), nullable=False)
    data_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_auto: Mapped[bool] = mapped_column(Boolean, default=False)
    is_reviewed: Mapped[bool] = mapped_column(Boolean, default=False)
    track_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    attributes_json: Mapped[str] = mapped_column(Text, default="{}")
    created_by: Mapped[str] = mapped_column(String(255), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow,
    )
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)

    image: Mapped[DBImage] = relationship(back_populates="annotations")


class DBAnnotator(Base):
    __tablename__ = "annotators"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    role: Mapped[str] = mapped_column(String(20), default="annotator")
    color: Mapped[str] = mapped_column(String(7), default="#e94560")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class DBSequence(Base):
    __tablename__ = "sequences"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="CASCADE"), index=True,
    )
    name: Mapped[str] = mapped_column(String(255), default="")
    frame_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class DBReviewHistory(Base):
    __tablename__ = "review_history"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    image_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("images.id", ondelete="CASCADE"), index=True,
    )
    reviewer_name: Mapped[str] = mapped_column(String(255), nullable=False)
    decision: Mapped[str] = mapped_column(String(20), nullable=False)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class DBAnnotationSession(Base):
    __tablename__ = "annotation_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    image_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("images.id", ondelete="CASCADE"), index=True,
    )
    annotator_name: Mapped[str] = mapped_column(String(255), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)


class DBExportHistory(Base):
    __tablename__ = "export_history"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="CASCADE"), index=True,
    )
    format: Mapped[str] = mapped_column(String(20), nullable=False)
    output_dir: Mapped[str] = mapped_column(Text, nullable=False)
    total_images: Mapped[int] = mapped_column(Integer, default=0)
    total_annotations: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# ─── Engine / Session Factory ────────────────────────────────────────────────

_engine = None
_SessionLocal = None


def _sqlite_wal_pragma(dbapi_conn, connection_record):
    """Enable WAL mode for better concurrent read performance."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def init_db(url: str = "sqlite:///labelox.db") -> Any:
    """Create engine, apply pragmas, create all tables. Returns engine."""
    global _engine, _SessionLocal

    kwargs: dict[str, Any] = {}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}

    _engine = create_engine(url, echo=False, **kwargs)

    if url.startswith("sqlite"):
        event.listen(_engine, "connect", _sqlite_wal_pragma)

    Base.metadata.create_all(_engine)
    _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)

    logger.info("Database initialised: {}", url)
    return _engine


def get_session() -> Session:
    """Return a new database session. Caller must close it."""
    if _SessionLocal is None:
        raise RuntimeError("Database not initialised — call init_db() first.")
    return _SessionLocal()


def get_engine():
    """Return the current engine or None."""
    return _engine


# ─── CRUD helpers ────────────────────────────────────────────────────────────

def create_project(
    name: str,
    description: str = "",
    label_classes: list[dict] | None = None,
    settings: dict | None = None,
    db: Session | None = None,
) -> DBProject:
    """Create a new project and return it."""
    close = False
    if db is None:
        db = get_session()
        close = True
    try:
        proj = DBProject(
            name=name,
            description=description,
            label_classes_json=json.dumps(label_classes or []),
            settings_json=json.dumps(settings or {}),
        )
        db.add(proj)
        db.commit()
        db.refresh(proj)
        return proj
    finally:
        if close:
            db.close()


def get_project(project_id: str, db: Session | None = None) -> DBProject | None:
    close = False
    if db is None:
        db = get_session()
        close = True
    try:
        return db.get(DBProject, project_id)
    finally:
        if close:
            db.close()


def list_projects(db: Session | None = None) -> list[DBProject]:
    close = False
    if db is None:
        db = get_session()
        close = True
    try:
        return list(db.query(DBProject).order_by(DBProject.updated_at.desc()).all())
    finally:
        if close:
            db.close()


def delete_project(project_id: str, db: Session | None = None) -> bool:
    close = False
    if db is None:
        db = get_session()
        close = True
    try:
        proj = db.get(DBProject, project_id)
        if proj is None:
            return False
        db.delete(proj)
        db.commit()
        return True
    finally:
        if close:
            db.close()


def get_images(
    project_id: str,
    status: str | None = None,
    limit: int = 100,
    offset: int = 0,
    db: Session | None = None,
) -> list[DBImage]:
    """Fetch images for a project with optional status filter."""
    close = False
    if db is None:
        db = get_session()
        close = True
    try:
        q = db.query(DBImage).filter(DBImage.project_id == project_id)
        if status:
            q = q.filter(DBImage.status == status)
        return list(q.order_by(DBImage.frame_index, DBImage.file_name).offset(offset).limit(limit).all())
    finally:
        if close:
            db.close()


def get_next_unlabeled_image(
    project_id: str,
    db: Session | None = None,
) -> DBImage | None:
    """Get the next unlabeled image in a project."""
    close = False
    if db is None:
        db = get_session()
        close = True
    try:
        return (
            db.query(DBImage)
            .filter(DBImage.project_id == project_id, DBImage.status == "unlabeled")
            .order_by(DBImage.frame_index, DBImage.file_name)
            .first()
        )
    finally:
        if close:
            db.close()


def get_annotations_for_image(
    image_id: str,
    db: Session | None = None,
) -> list[DBAnnotation]:
    close = False
    if db is None:
        db = get_session()
        close = True
    try:
        return list(
            db.query(DBAnnotation)
            .filter(DBAnnotation.image_id == image_id)
            .order_by(DBAnnotation.created_at)
            .all()
        )
    finally:
        if close:
            db.close()


def bulk_insert_annotations(
    annotations: list[DBAnnotation],
    db: Session | None = None,
) -> int:
    """Insert multiple annotations. Returns count inserted."""
    close = False
    if db is None:
        db = get_session()
        close = True
    try:
        db.add_all(annotations)
        db.commit()
        return len(annotations)
    finally:
        if close:
            db.close()


def get_project_stats(project_id: str, db: Session | None = None) -> dict:
    """Single-query stats aggregation for a project."""
    close = False
    if db is None:
        db = get_session()
        close = True
    try:
        total = db.query(func.count(DBImage.id)).filter(
            DBImage.project_id == project_id,
        ).scalar() or 0
        annotated = db.query(func.count(DBImage.id)).filter(
            DBImage.project_id == project_id,
            DBImage.status == "annotated",
        ).scalar() or 0
        reviewed = db.query(func.count(DBImage.id)).filter(
            DBImage.project_id == project_id,
            DBImage.status == "reviewed",
        ).scalar() or 0
        rejected = db.query(func.count(DBImage.id)).filter(
            DBImage.project_id == project_id,
            DBImage.status == "rejected",
        ).scalar() or 0
        total_ann = db.query(func.count(DBAnnotation.id)).join(DBImage).filter(
            DBImage.project_id == project_id,
        ).scalar() or 0

        return {
            "total_images": total,
            "annotated_images": annotated,
            "reviewed_images": reviewed,
            "rejected_images": rejected,
            "total_annotations": total_ann,
            "completion_percent": (annotated + reviewed) / total * 100 if total else 0,
        }
    finally:
        if close:
            db.close()
