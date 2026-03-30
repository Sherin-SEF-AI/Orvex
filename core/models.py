from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class DeviceType(str, Enum):
    GOPRO = "gopro"
    INSTA360 = "insta360"
    SENSOR_LOGGER = "sensor_logger"


class RecordingQuality(str, Enum):
    EXCELLENT = "excellent"   # no issues, full GPS, full IMU
    GOOD      = "good"        # minor issues (e.g. low GPS rate)
    FAIR      = "fair"        # degraded (HyperSmooth, partial GPS)
    POOR      = "poor"        # major issues (no GPS, IMU gaps)


class IMUSample(BaseModel):
    timestamp_ns: int
    accel_x: float  # m/s²
    accel_y: float
    accel_z: float
    gyro_x: float   # rad/s
    gyro_y: float
    gyro_z: float


class GPSSample(BaseModel):
    timestamp_ns: int
    latitude: float
    longitude: float
    altitude_m: float
    speed_mps: float
    fix_type: int   # 0=no fix, 2=2D, 3=3D
    dop: float = 0.0  # dilution of precision


class AuditResult(BaseModel):
    file_path: str
    device_type: DeviceType
    duration_seconds: float
    has_imu: bool
    has_gps: bool
    imu_sample_count: int
    imu_rate_hz: float
    gps_sample_count: int
    gps_rate_hz: float
    video_fps: float
    video_resolution: tuple[int, int]
    file_size_mb: float
    issues: list[str]           # blocking problems
    warnings: list[str] = []    # non-blocking notes
    chapter_files: list[str] = []   # sibling chapter paths detected
    quality: RecordingQuality = RecordingQuality.GOOD


ExtractionStatus = Literal["pending", "running", "done", "failed"]
CalibrationSessionType = Literal["imu_static", "camera_intrinsic", "camera_imu_extrinsic"]
OutputFormat = Literal["euroc", "custom"]


class Session(BaseModel):
    id: str              # UUID
    name: str
    created_at: datetime
    environment: str     # "road", "indoor", "gravel", etc.
    location: str        # e.g. "Kalamassery"
    files: list[str]     # paths to raw files
    audit_results: list[AuditResult]
    extraction_status: ExtractionStatus
    notes: str


class CalibrationSession(BaseModel):
    id: str
    camera_device: DeviceType
    session_type: CalibrationSessionType
    file_path: str
    status: ExtractionStatus
    results: dict
    reprojection_error_px: float | None


class ExtractionConfig(BaseModel):
    session_id: str
    frame_fps: float = 5.0
    frame_format: str = "jpg"
    frame_quality: int = 95
    output_format: OutputFormat = "euroc"
    sync_devices: bool = True
    imu_interpolation: bool = True
    blur_threshold: float = 100.0   # Laplacian variance below = blurry
    dedup_threshold: float = 0.98   # SSIM above = duplicate


class FrameMetadata(BaseModel):
    frame_path: str
    timestamp_ns: int
    blur_score: float           # Laplacian variance (higher = sharper)
    is_blurry: bool
    is_duplicate: bool
    imu_at_frame: IMUSample | None = None   # nearest IMU sample
    gps_at_frame: GPSSample | None = None   # nearest GPS sample


class CalibrationResult(BaseModel):
    session_id: str
    step: CalibrationSessionType
    completed_at: datetime
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    dist_coeffs: list[float] = []       # k1,k2,p1,p2,k3
    T_cam_imu: list[list[float]] = []   # 4x4 extrinsic matrix
    reprojection_error_px: float | None = None
    imu_noise_params: dict = {}


class DatasetManifest(BaseModel):
    created_at: datetime
    format: str
    session_ids: list[str]
    total_frames: int
    total_imu_samples: int
    total_gps_samples: int
    duration_seconds: float
    calibration: CalibrationResult | None = None
    warnings: list[str] = []


class ExtractedSession(BaseModel):
    """Returned by extractor functions (extract_gopro, extract_insta360, etc.)."""
    session_id: str
    device_type: DeviceType
    imu_samples: list[IMUSample]
    gps_samples: list[GPSSample]
    frame_paths: list[str]
    frame_timestamps_ns: list[int]
    duration_seconds: float
    stats: dict  # extractor-specific summary counts / rates
    frame_metadata: list[FrameMetadata] = []


# ---------------------------------------------------------------------------
# Phase 2 models
# ---------------------------------------------------------------------------

# ── Auto-labeling ──────────────────────────────────────────────────────────

class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]   # [x1, y1, x2, y2] absolute pixels
    bbox_xywhn: list[float]  # [cx, cy, w, h] normalized 0-1


class FrameAnnotation(BaseModel):
    frame_path: str
    detections: list[Detection]
    inference_time_ms: float
    model_version: str


# ── Uncertainty / Active Learning ──────────────────────────────────────────

class UncertaintyScore(BaseModel):
    frame_path: str
    score: float                      # 0=certain, 1=maximally uncertain
    method: str
    num_detections: int
    high_uncertainty_detections: int  # detections with conf < 0.5


# ── Depth Estimation ───────────────────────────────────────────────────────

class DepthResult(BaseModel):
    frame_path: str
    depth_raw_path: str
    depth_color_path: str | None = None
    min_depth: float
    max_depth: float
    mean_depth: float
    inference_time_ms: float
    is_metric: bool = False  # always False unless GPS scale estimation applied


# ── SLAM Validation ────────────────────────────────────────────────────────

class TrajectoryPoint(BaseModel):
    timestamp_ns: int
    tx: float
    ty: float
    tz: float
    qx: float
    qy: float
    qz: float
    qw: float


class TrajectoryMetrics(BaseModel):
    total_distance_m: float
    duration_seconds: float
    avg_speed_mps: float
    loop_closure_drift_m: float | None = None
    loop_closure_drift_percent: float | None = None
    ate_rmse: float | None = None   # absolute trajectory error vs ground truth
    rpe_rmse: float | None = None   # relative pose error vs ground truth
    tracking_lost_count: int
    keyframe_count: int
    map_point_count: int


class SLAMResult(BaseModel):
    session_id: str
    mode: str
    trajectory: list[TrajectoryPoint]
    metrics: TrajectoryMetrics
    config_used: str
    tracking_log: list[str]
    success: bool
    failure_reason: str | None = None


# ── 3D Reconstruction ──────────────────────────────────────────────────────

class ColmapResult(BaseModel):
    session_id: str
    num_images_total: int
    num_images_registered: int
    num_points3d: int
    mean_reprojection_error: float
    points: list[list[float]]       # Nx3 XYZ
    colors: list[list[int]]         # Nx3 RGB 0-255
    camera_poses: list[TrajectoryPoint]
    ply_path: str | None = None


# ── Augmentation ───────────────────────────────────────────────────────────

class AugmentationConfig(BaseModel):
    horizontal_flip: bool = True
    vertical_flip: bool = False
    random_rotate_90: bool = True
    brightness_contrast: bool = True
    hue_saturation: bool = True
    gaussian_noise: bool = True
    motion_blur: bool = True
    jpeg_compression: bool = True
    mosaic: bool = True
    rain_simulation: bool = False
    fog_simulation: bool = False
    multiplier: int = 3


class AugmentationResult(BaseModel):
    original_count: int
    augmented_count: int
    output_dir: str
    per_transform_counts: dict[str, int]


# ── Training ───────────────────────────────────────────────────────────────

class EpochMetrics(BaseModel):
    epoch: int
    box_loss: float
    cls_loss: float
    dfl_loss: float
    precision: float
    recall: float
    map50: float
    map50_95: float
    lr: float
    epoch_time_seconds: float


class TrainingConfig(BaseModel):
    dataset_dir: str
    model_variant: str = "yolov8n"
    pretrained_weights: str = "yolov8n.pt"
    epochs: int = 100
    batch_size: int = 16
    image_size: int = 640
    learning_rate: float = 0.01
    device: str = "auto"
    project_name: str = "rover_detection"
    run_name: str = Field(default="run1")


class TrainingRun(BaseModel):
    run_id: str
    config: TrainingConfig
    best_weights_path: str
    best_epoch: int
    final_map50: float
    final_map50_95: float
    epoch_metrics: list[EpochMetrics]
    training_time_minutes: float
    status: str  # "running", "done", "cancelled", "failed"


# ── Analytics ─────────────────────────────────────────────────────────────

class SceneDiversityReport(BaseModel):
    lighting_distribution: dict[str, float]   # {"bright": %, "normal": %, "dark": %}
    brightness_stats: dict[str, float]         # mean, std, min, max
    edge_density_stats: dict[str, float]       # mean, std, min, max
    estimated_scene_types: dict[str, float]    # urban/residential/highway percentages


class GeoCoverageReport(BaseModel):
    total_distance_km: float
    bounding_box: dict[str, float]   # min_lat, max_lat, min_lon, max_lon
    unique_grid_cells: int
    coverage_map_path: str           # path to folium HTML heatmap
    avg_speed_mps: float
    stationary_time_percent: float


__all__ = [
    "DeviceType",
    "RecordingQuality",
    "IMUSample",
    "GPSSample",
    "AuditResult",
    "ExtractionStatus",
    "CalibrationSessionType",
    "OutputFormat",
    "Session",
    "CalibrationSession",
    "ExtractionConfig",
    "FrameMetadata",
    "CalibrationResult",
    "DatasetManifest",
    "ExtractedSession",
    # Phase 2
    "Detection",
    "FrameAnnotation",
    "UncertaintyScore",
    "DepthResult",
    "TrajectoryPoint",
    "TrajectoryMetrics",
    "SLAMResult",
    "ColmapResult",
    "AugmentationConfig",
    "AugmentationResult",
    "EpochMetrics",
    "TrainingConfig",
    "TrainingRun",
    "SceneDiversityReport",
    "GeoCoverageReport",
    # Phase 3 (inference / review / continuous learning)
    "ModelRegistry",
    "InferenceRequest",
    "InferenceResult",
    "ReviewStatus",
    "AnnotationReview",
    "LearningTrigger",
    "ModelComparison",
    # Phase 3 (perception + infra)
    "SegmentationResult",
    "SegmentationStats",
    "OccupancyConfig",
    "OccupancyFrame",
    "LaneCurve",
    "LaneDepartureStatus",
    "LaneFrame",
    "LaneConfig",
    "TrackHistory",
    "TrackingResult",
    "TrackingStats",
    "DatasetVersion",
    "DatasetDiff",
    "MLflowRun",
    "RunComparison",
    "ONNXExportResult",
    "TRTExportResult",
    "BenchmarkResult",
    # Insta360 X4 360° processing
    "Insta360ProcessingConfig",
    "INSVPair",
    "PerspectiveView",
    "Insta360ExtractionResult",
    "Insta360Session",
]


# ── Phase 3: Inference Server ──────────────────────────────────────────────

class ModelRegistry(BaseModel):
    model_id: str
    name: str
    weights_path: str
    model_variant: str          # yolov8n / yolov8s / etc.
    training_run_id: str | None = None
    created_at: datetime
    metrics: dict               # {"map50": float, "map50_95": float, "best_epoch": int}
    is_active: bool = False


class InferenceRequest(BaseModel):
    image_path: str | None = None
    image_base64: str | None = None   # base64-encoded JPEG/PNG
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    model_id: str | None = None       # None → use active model


class InferenceResult(BaseModel):
    model_id: str
    image_path: str | None = None
    detections: list[Detection]
    inference_time_ms: float
    model_variant: str


# ── Phase 3: Annotation Review ─────────────────────────────────────────────

class ReviewStatus(str, Enum):
    pending   = "pending"
    accepted  = "accepted"
    corrected = "corrected"
    rejected  = "rejected"


class AnnotationReview(BaseModel):
    frame_path: str
    original_detections: list[Detection]
    corrected_detections: list[Detection]
    status: ReviewStatus = ReviewStatus.pending
    reviewed_at: datetime | None = None
    reviewer_notes: str = ""


# ── Phase 3: Continuous Learning ───────────────────────────────────────────

class LearningTrigger(BaseModel):
    session_id: str
    trigger_type: str           # "correction_threshold" | "manual" | "scheduled"
    corrections_count: int
    triggered_at: datetime
    resulting_run_id: str | None = None


class ModelComparison(BaseModel):
    baseline_model_id: str
    candidate_model_id: str
    val_dir: str
    baseline_map50: float
    candidate_map50: float
    improved: bool
    delta_map50: float


# ---------------------------------------------------------------------------
# Phase 3 new models
# ---------------------------------------------------------------------------

# ── Segmentation ────────────────────────────────────────────────────────────

class SegmentationResult(BaseModel):
    frame_path: str
    mask_path: str
    overlay_path: str
    class_pixel_percent: dict[str, float]
    road_area_percent: float
    sky_area_percent: float
    inference_time_ms: float
    is_valid_rover_frame: bool


class SegmentationStats(BaseModel):
    per_class_mean_percent: dict[str, float]
    per_class_std_percent: dict[str, float]
    frames_with_road: int
    frames_without_road: int
    mean_road_coverage: float
    invalid_frames: list[str]


# ── Occupancy Grid ──────────────────────────────────────────────────────────

class OccupancyConfig(BaseModel):
    grid_resolution_m: float = 0.1
    grid_width_m: float = 20.0
    grid_height_m: float = 30.0
    camera_height_m: float = 1.0
    max_depth_m: float = 30.0
    temporal_fusion_window: int = 5
    decay_factor: float = 0.95


class OccupancyFrame(BaseModel):
    frame_path: str
    grid_path: str
    visualization_path: str
    timestamp_ns: int
    occupied_cells: int
    free_cells: int
    unknown_cells: int
    occupancy_percent: float


# ── Lane Detection ──────────────────────────────────────────────────────────

class LaneCurve(BaseModel):
    lane_type: str          # "ego_left", "ego_right", "adjacent_left", "adjacent_right"
    points: list[tuple[float, float]]  # (x, y) image coordinates
    polynomial: list[float]            # cubic coefficients [a, b, c, d]
    confidence: float
    marking_type: str       # "solid", "dashed", "double", "unknown", "none"


class LaneDepartureStatus(BaseModel):
    status: str             # "centered" | "drifting_left" | "drifting_right" | "no_lane"
    lateral_offset_percent: float
    confidence: float


class LaneFrame(BaseModel):
    frame_path: str
    overlay_path: str
    lanes: list[LaneCurve]
    departure: LaneDepartureStatus
    detection_method: str   # "ufld" | "classical" | "none"
    inference_time_ms: float


class LaneConfig(BaseModel):
    use_ufld: bool = True
    ufld_conf_threshold: float = 0.5
    classical_fallback: bool = True
    roi_top_percent: float = 0.55
    camera_height_m: float = 1.0
    camera_pitch_deg: float = 0.0


# ── Tracking ────────────────────────────────────────────────────────────────

class TrackHistory(BaseModel):
    track_id: int
    class_name: str
    first_frame: int
    last_frame: int
    duration_frames: int
    center_trajectory: list[tuple[float, float]]
    mean_confidence: float
    is_static: bool


class TrackingResult(BaseModel):
    session_id: str
    total_tracks: int
    active_tracks_per_frame: list[int]
    track_histories: dict[int, TrackHistory]
    class_track_counts: dict[str, int]


class TrackingStats(BaseModel):
    total_unique_objects: int
    mean_track_duration_seconds: float
    max_simultaneous_tracks: int
    static_objects_percent: float
    objects_per_class: dict[str, int]
    mean_objects_per_frame: float
    track_fragmentation_rate: float
    high_density_frames: list[int]


# ── Dataset Versioning ──────────────────────────────────────────────────────

class DatasetVersion(BaseModel):
    tag: str
    timestamp: datetime
    message: str
    metadata: dict
    file_count: int
    total_frames: int
    class_distribution: dict[str, int]
    dataset_hash: str
    dvc_files: list[str]


class DatasetDiff(BaseModel):
    version_a: str
    version_b: str
    frames_added: int
    frames_removed: int
    frames_changed: int
    class_distribution_delta: dict[str, float]
    total_frames_delta: int
    new_sessions: list[str]
    removed_sessions: list[str]


# ── Experiment Tracking ─────────────────────────────────────────────────────

class MLflowRun(BaseModel):
    run_id: str
    run_name: str
    status: str
    start_time: datetime
    params: dict
    metrics: dict
    tags: dict
    artifact_uri: str


class RunComparison(BaseModel):
    runs: list[MLflowRun]
    best_run_id: str
    metric_comparison: dict
    param_differences: dict


# ── Edge Export ─────────────────────────────────────────────────────────────

class ONNXExportResult(BaseModel):
    output_path: str
    model_size_mb: float
    input_shape: list[int]
    output_shapes: list[list[int]]
    onnx_opset: int
    simplified: bool
    test_latency_ms: float
    verification_passed: bool


class TRTExportResult(BaseModel):
    output_path: str
    engine_size_mb: float
    precision: str
    build_time_minutes: float
    estimated_jetson_latency_ms: float | None = None


class BenchmarkResult(BaseModel):
    model_path: str
    format: str
    device: str
    mean_latency_ms: float
    std_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_fps: float
    memory_mb: float


# ── Insta360 X4 360° Processing ─────────────────────────────────────────────

class Insta360ProcessingConfig(BaseModel):
    # Stitching
    output_width: int = 7680
    output_height: int = 3840
    stitch_crf: int = 10
    fisheye_fov: float = 210.0
    # Perspective split
    perspective_width: int = 2160
    perspective_height: int = 2160
    h_fov: float = 110.0
    v_fov: float = 110.0
    perspective_crf: int = 15
    views: list[str] = ["front", "right", "rear", "left"]
    use_gpu: bool = False
    # Frame extraction
    frame_fps: float = 5.0
    frame_format: str = "jpg"
    frame_quality: int = 95
    # Pitch/roll correction (if camera not mounted level)
    pitch_correction_deg: float = 0.0
    roll_correction_deg: float = 0.0
    # Storage management
    keep_equirect_video: bool = True
    keep_perspective_videos: bool = False
    output_format: str = "euroc"


class INSVPair(BaseModel):
    back_path: str          # _00_ file (telemetry + back lens)
    front_path: str | None  # _10_ file (front lens) — None if missing
    base_name: str          # e.g. VID_20250301_091532_001
    duration_seconds: float
    file_size_mb: float
    has_gps: bool
    has_imu: bool
    issues: list[str]


class PerspectiveView(BaseModel):
    view_name: str          # "front" | "right" | "rear" | "left"
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    video_path: str | None
    frame_dir: str | None
    frame_count: int
    mean_blur_score: float
    is_usable: bool


class Insta360ExtractionResult(BaseModel):
    session_id: str
    insv_pair: INSVPair
    equirect_path: str | None
    perspective_views: dict[str, PerspectiveView]
    imu_samples: int
    imu_rate_hz: float
    gps_samples: int
    gps_rate_hz: float
    total_frames_per_view: int
    dataset_dir: str
    manifest_path: str
    processing_time_minutes: float
    disk_usage_gb: float
    issues: list[str]
    warnings: list[str]


class Insta360Session(BaseModel):
    id: str
    name: str
    created_at: datetime
    insv_pairs: list[INSVPair]
    processing_status: str  # "pending"|"stitching"|"splitting"|"extracting"|"done"|"failed"
    config: Insta360ProcessingConfig
    results: list[Insta360ExtractionResult]
    total_duration_seconds: float
    location: str
    notes: str
