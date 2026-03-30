"""
labelox/core/models.py — All Pydantic v2 data models for LABELOX.

Every field typed. No Optional unless truly nullable.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ─── Enums ───────────────────────────────────────────────────────────────────

class AnnotationType(str, Enum):
    BBOX = "bbox"
    MASK = "mask"
    POLYGON = "polygon"
    POLYLINE = "polyline"
    KEYPOINT = "keypoint"
    CUBOID_3D = "cuboid_3d"
    CLASSIFICATION = "classification"


class ImageStatus(str, Enum):
    UNLABELED = "unlabeled"
    IN_PROGRESS = "in_progress"
    ANNOTATED = "annotated"
    REVIEWED = "reviewed"
    REJECTED = "rejected"
    SKIPPED = "skipped"


class ReviewDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_WORK = "needs_work"


class ExportFormat(str, Enum):
    YOLO = "yolo"
    COCO = "coco"
    CVAT_XML = "cvat_xml"
    PASCAL_VOC = "pascal_voc"
    CSV = "csv"


class AutoAnnotateStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


# ─── Label Schema ────────────────────────────────────────────────────────────

class LabelAttribute(BaseModel):
    name: str
    type: str  # "select" | "text" | "number" | "checkbox"
    options: list[str] = []
    default: str | None = None
    required: bool = False


class LabelClass(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    color: str  # hex e.g. "#e94560"
    annotation_types: list[AnnotationType] = [AnnotationType.BBOX]
    hotkey: str | None = None
    supercategory: str | None = None
    attributes: list[LabelAttribute] = []
    is_active: bool = True


# ─── Geometry Primitives ─────────────────────────────────────────────────────

class Point(BaseModel):
    x: float  # normalised 0-1
    y: float  # normalised 0-1


class BBoxAnnotation(BaseModel):
    x: float       # normalised top-left x
    y: float       # normalised top-left y
    width: float   # normalised width
    height: float  # normalised height


class MaskAnnotation(BaseModel):
    rle: dict  # COCO RLE {counts, size}
    polygon_points: list[list[float]] | None = None  # fallback polygon


class PolylineAnnotation(BaseModel):
    points: list[Point]
    is_closed: bool = False  # True => polygon


class KeypointPoint(BaseModel):
    x: float
    y: float
    label: str
    visibility: int = 2  # 0=not labeled, 1=occluded, 2=visible


class KeypointAnnotation(BaseModel):
    points: list[KeypointPoint]
    skeleton: list[list[int]] = []  # pairs of keypoint indices


class Cuboid3DAnnotation(BaseModel):
    front_top_left: Point
    front_top_right: Point
    front_bottom_left: Point
    front_bottom_right: Point
    back_top_left: Point
    back_top_right: Point
    back_bottom_left: Point
    back_bottom_right: Point
    depth_estimate_m: float | None = None


# ─── Annotation ──────────────────────────────────────────────────────────────

class Annotation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    image_id: str
    label_id: str
    label_name: str
    annotation_type: AnnotationType
    # Only one populated based on type:
    bbox: BBoxAnnotation | None = None
    mask: MaskAnnotation | None = None
    polyline: PolylineAnnotation | None = None
    keypoints: KeypointAnnotation | None = None
    cuboid_3d: Cuboid3DAnnotation | None = None
    classification: str | None = None
    # Common fields:
    confidence: float | None = None
    is_auto: bool = False
    is_reviewed: bool = False
    track_id: int | None = None
    attributes: dict[str, Any] = {}
    created_by: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    comment: str | None = None


# ─── Image Record ────────────────────────────────────────────────────────────

class ImageRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    file_path: str
    file_name: str
    width: int
    height: int
    file_size_bytes: int
    status: ImageStatus = ImageStatus.UNLABELED
    assigned_to: str | None = None
    annotations: list[Annotation] = []
    auto_annotation_status: AutoAnnotateStatus = AutoAnnotateStatus.PENDING
    review_decision: ReviewDecision | None = None
    review_comment: str | None = None
    reviewed_by: str | None = None
    sequence_id: str | None = None
    frame_index: int | None = None
    timestamp_ns: int | None = None
    thumbnail_path: str | None = None
    blur_score: float | None = None
    scene_class: str | None = None
    md5: str | None = None
    metadata: dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ─── Project ─────────────────────────────────────────────────────────────────

class ProjectSettings(BaseModel):
    allow_overlapping: bool = True
    min_bbox_size_px: int = 5
    require_attributes: bool = False
    auto_annotate_on_import: bool = False
    auto_annotate_model: str = "yolov8n.pt"
    auto_annotate_conf: float = 0.25
    require_review: bool = True
    review_sample_percent: float = 100.0
    next_image_key: str = "d"
    prev_image_key: str = "a"
    save_key: str = "s"


class Project(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str = ""
    label_classes: list[LabelClass] = []
    annotation_types: list[AnnotationType] = [AnnotationType.BBOX]
    image_count: int = 0
    annotated_count: int = 0
    reviewed_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    settings: ProjectSettings = Field(default_factory=ProjectSettings)


# ─── Team ────────────────────────────────────────────────────────────────────

class Annotator(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    role: str = "annotator"  # "annotator" | "reviewer" | "admin"
    color: str = "#e94560"
    is_active: bool = True


# ─── Export ───────────────────────────────────────────────────────────────────

class ExportConfig(BaseModel):
    project_id: str
    format: ExportFormat
    output_dir: str
    include_images: bool = True
    include_auto_annotations: bool = True
    only_reviewed: bool = False
    only_approved: bool = False
    split_train_val: bool = True
    train_ratio: float = 0.8
    class_filter: list[str] = []
    status_filter: list[ImageStatus] = []


class ExportResult(BaseModel):
    format: ExportFormat
    output_dir: str
    total_images: int
    total_annotations: int
    train_images: int = 0
    val_images: int = 0
    files_created: list[str] = []
    export_time_seconds: float = 0.0


# ─── Stats ────────────────────────────────────────────────────────────────────

class AnnotatorStats(BaseModel):
    annotator_name: str
    images_annotated: int = 0
    annotations_created: int = 0
    avg_time_per_image_seconds: float = 0.0
    accuracy_percent: float | None = None


class ProjectStats(BaseModel):
    project_id: str
    total_images: int = 0
    annotated_images: int = 0
    reviewed_images: int = 0
    approved_images: int = 0
    rejected_images: int = 0
    total_annotations: int = 0
    annotations_per_class: dict[str, int] = {}
    auto_annotation_percent: float = 0.0
    completion_percent: float = 0.0
    annotators: list[AnnotatorStats] = []


# ─── Review History ──────────────────────────────────────────────────────────

class ReviewRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    image_id: str
    reviewer_name: str
    decision: ReviewDecision
    comment: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ─── Annotation Session (time tracking) ─────────────────────────────────────

class AnnotationSession(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    image_id: str
    annotator_name: str
    started_at: datetime
    ended_at: datetime | None = None
    duration_seconds: float = 0.0
