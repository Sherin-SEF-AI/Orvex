# Orvex

**Production-grade data pipeline for autonomous rover dataset collection, processing, and perception.**

Orvex handles the full lifecycle of multi-sensor data — from raw GoPro/Insta360/Android recordings through telemetry extraction, calibration, synchronization, dataset assembly, auto-labeling, training, 3D reconstruction, SLAM validation, and edge deployment. Built for Indian road conditions.

**Author:** Sherin Joseph Roy
**Repository:** [https://github.com/Sherin-SEF-AI/Orvex.git](https://github.com/Sherin-SEF-AI/Orvex.git)
**License:** MIT
**Python:** 3.11+

---

## Demos

### IMU Telemetry Graph

https://github.com/Sherin-SEF-AI/Orvex/raw/main/media/imu-graph.mp4

### Depth Estimation

https://github.com/Sherin-SEF-AI/Orvex/raw/main/media/depthestimation.mp4

### Frame Extraction

https://github.com/Sherin-SEF-AI/Orvex/raw/main/media/frame-extraction.mp4

---

## Table of Contents

1. [Demos](#demos)
2. [Architecture](#architecture)
3. [Supported Devices](#supported-devices)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Desktop Application](#desktop-application)
7. [Web Application](#web-application)
8. [Complete Feature Reference](#complete-feature-reference)
9. [A-to-Z Usage Guide](#a-to-z-usage-guide)
10. [Calibration Workflow](#calibration-workflow)
11. [AI Models](#ai-models)
12. [Export Formats](#export-formats)
13. [API Reference](#api-reference)
14. [Testing](#testing)
15. [Project Structure](#project-structure)
16. [System Requirements](#system-requirements)

---

## Architecture

Orvex is a dual-interface application: a PyQt6 desktop app and a FastAPI + React web app. Both share identical business logic through the `core/` module — zero code duplication.

```
                  +------------------+
                  |     core/        |   33 pure-Python modules
                  |  (business logic)|   No UI imports
                  +--------+---------+
                           |
              +------------+------------+
              |                         |
    +---------+----------+   +----------+---------+
    |   desktop/         |   |   web/             |
    |   PyQt6 app        |   |   FastAPI + React  |
    |   27 widget panels |   |   96+ API endpoints|
    |   Custom dark theme|   |   WebSocket progress|
    +--------------------+   +--------------------+
```

The core layer is device-aware, format-aware, and self-validating. Every function that touches a file operates on real data — no mock objects, no dummy returns. Errors are actionable: if HyperSmooth is enabled on a GoPro file, the error message tells the user exactly how to disable it.

---

## Supported Devices

| Device | Format | IMU Rate | GPS | Chapter Handling |
|--------|--------|----------|-----|-----------------|
| **GoPro Hero 11** | `.MP4` (GPMF telemetry) | 200 Hz | Yes (GPS5 stream) | Auto-detects GH01/GH02/GH03 splits at 4 GB boundaries |
| **Insta360 X4** | `.insv` | Via exiftool | Optional (GPS remote or app) | Front/back pair detection |
| **Android Sensor Logger** | `.csv` / `.json` | Variable | Via phone GPS | Auto-detects timestamp format (ISO 8601, Unix epoch, seconds_elapsed) |

---

## Installation

### Prerequisites

| Dependency | Purpose | Install |
|------------|---------|---------|
| Python 3.11+ | Runtime | `sudo apt install python3.11` |
| FFmpeg | Frame extraction, video probing | `sudo apt install ffmpeg` |
| exiftool | Insta360 metadata extraction | `sudo apt install libimage-exiftool-perl` |
| Git | Dataset versioning (DVC) | `sudo apt install git` |

**Optional external tools** (required only for specific features):

| Tool | Feature | Install |
|------|---------|---------|
| COLMAP | 3D reconstruction | `sudo apt install colmap` |
| ORBSLAM3 | SLAM validation | Build from source |
| OpenImuCameraCalibrator | Camera-IMU extrinsic calibration | Build from source |
| DVC | Dataset versioning | `pip install dvc` |
| MLflow | Experiment tracking | `pip install mlflow` |
| NVIDIA TensorRT | Edge deployment | NVIDIA SDK |

### Setup

```bash
git clone https://github.com/Sherin-SEF-AI/Orvex.git
cd Orvex

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install core dependencies
pip install -r requirements.txt

# Install AI/ML dependencies (for auto-labeling, depth, segmentation)
pip install -e ".[ai]"

# Install web dependencies (for FastAPI backend)
pip install -e ".[web]"

# Verify
python -c "from core.models import DeviceType; print('OK')"
ffmpeg -version | head -1
```

### Verify FFmpeg

Orvex calls FFmpeg via subprocess. If FFmpeg is missing, you will see a clear error at startup.

```bash
which ffmpeg    # Must return a path
which exiftool  # Required for Insta360
```

---

## Quick Start

### Desktop App

```bash
source venv/bin/activate
PYTHONPATH=. python -m desktop.main
```

The Orvex window opens with a dark theme. The left sidebar contains 28 tools organized into 6 sections. Start by creating a session, importing files, and working through the pipeline.

### Web App

```bash
# Terminal 1: API server
source venv/bin/activate
PYTHONPATH=. uvicorn web.backend.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: React frontend
cd web/frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser. The API docs are at `http://localhost:8000/docs`.

---

## Desktop Application

### Window Layout

```
+-------------------------------------------------------------+
|  [*] Orvex                              [_] [O] [X]         |
+-------------+-----------------------------------------------+
|             |                                                |
| DATA        |   Active panel content                         |
| PIPELINE    |   (one of 28 stacked widgets)                  |
|  Sessions   |                                                |
|  Audit      |                                                |
|  Extract    |                                                |
|  Sensor Log |                                                |
|  Calibrate  |                                                |
|  Telemetry  |                                                |
|  Frames     |                                                |
|  Dataset    |                                                |
|             |                                                |
| INTELLIGENCE|                                                |
|  Auto-Label |                                                |
|  Active Lrn |                                                |
|  Analytics  |                                                |
|  Augment    |                                                |
|  Train      |                                                |
|  Depth      |-----------------------------------------------+
|  3D Reconst |  Log Panel (collapsible)                       |
|  SLAM Valid |  Real-time output from background tasks        |
|             +-----------------------------------------------+
+-------------+ Status: progress bar | session name            |
+-------------------------------------------------------------+
```

### Sidebar Sections

**Data Pipeline** — Raw file ingestion through dataset assembly. Sessions, Audit, Extract, Sensor Logger, Calibrate, Telemetry, Frames, Dataset.

**Intelligence** — AI-powered analysis and training. Auto-Label, Active Learning, Analytics, Augment, Train, Depth, 3D Reconstruct, SLAM Validate.

**Deployment** — Model serving and feedback loops. Inference, Review, Auto-Retrain.

**Perception** — Specialized computer vision tasks. Segment, Occupancy, Lanes, Track.

**Infrastructure** — DevOps and experiment management. Versions, Experiments, Edge Export, API.

**360 Camera** — Insta360 X4-specific processing. Insta360.

### Theme

Orvex uses a custom dark theme designed for extended use:

| Element | Hex | Usage |
|---------|-----|-------|
| Background | `#1a1a2e` | Main window |
| Panel | `#16213e` | Sidebar, cards |
| Accent | `#0f3460` | Interactive elements |
| Highlight | `#e94560` | Active selection, warnings |
| Text | `#e0e0e0` | Primary content |
| Muted | `#666680` | Secondary text, disabled |
| Success | `#27ae60` | Completed operations |
| Warning | `#f39c12` | Attention required |

All icons are vector-drawn at runtime using QPainter — no external icon files or image assets.

---

## Web Application

### Backend (FastAPI)

23 route modules providing 96+ endpoints. All responses use the envelope format:

```json
{
  "data": { ... },
  "error": null
}
```

Long-running operations (extraction, training, auto-labeling) return a `task_id` immediately. Clients connect to `ws://host:8000/ws/tasks/{task_id}` to receive real-time progress updates:

```json
{"status": "running", "progress": 45, "total": 100, "message": "Extracting frames..."}
```

### Frontend (React + Vite)

25 pages matching every desktop panel:

Sessions, Audit, Extraction, Calibration, Frames, Dataset, AutoLabel, ActiveLearning, Analytics, Augmentation, Training, DepthEstimation, Reconstruction, SlamValidation, Inference, AnnotationReview, ContinuousLearning, Segmentation, OccupancyGrid, LaneDetection, Tracking, Versioning, Experiments, EdgeExport, Insta360.

---

## Complete Feature Reference

### Phase 1: Data Pipeline

| Feature | Module | Description |
|---------|--------|-------------|
| **Session Manager** | `session_manager.py` | Create, list, delete sessions. Each session is a TOML file containing metadata (name, environment, location, device list, notes). Duplicate file detection on import. |
| **File Auditor** | `audit.py` | Analyzes every imported file. Detects device type from filename/metadata. Extracts duration, resolution, FPS, IMU sample rate, GPS fix quality. Flags issues: HyperSmooth enabled, no GPS fix, gaps in IMU stream. |
| **GoPro Extractor** | `extractor_gopro.py` | Parses GPMF telemetry from Hero 11 MP4 files. Extracts accelerometer (200 Hz), gyroscope, GPS5 stream. Handles chapter-split files (GH01xxxx through GH09xxxx) as continuous recordings. Outputs EuRoC-compatible directory structure. |
| **Insta360 Extractor** | `extractor_insta360.py` | Extracts equirectangular frames and IMU from INSV binary format via exiftool subprocess. Detects front/back file pairs. |
| **Sensor Logger Parser** | `extractor_sensorlogger.py` | Parses Android Sensor Logger CSV/JSON exports. Auto-detects column names and timestamp formats. Normalizes all timestamps to nanosecond Unix epoch. |
| **Multi-Device Sync** | `synchronizer.py` | Three synchronization methods: (1) GPS time anchor using GoPro UTC timestamps as master clock, (2) IMU cross-correlation of acceleration magnitude signals between devices, (3) manual offset input. Warns if computed lag exceeds 2 seconds. |
| **Calibration Pipeline** | `calibration.py` | Four-step guided workflow: IMU static noise characterization (Allan deviation), camera intrinsic calibration (OpenCV), camera-IMU extrinsic calibration (OpenImuCameraCalibrator subprocess), validation checks. Each step is resumable. |
| **Telemetry Viewer** | `telemetry_plot.py` | Real-time PyQtGraph plots of accelerometer and gyroscope data (6 axes). GPS track displayed on a Folium map rendered in QWebEngineView. Scrub bar synced to frame browser. |
| **Frame Browser** | `frame_browser.py` | Grid view of extracted frames with lazy thumbnail loading. Blur detection (Laplacian variance). Duplicate detection (SSIM > 0.98). Export selected frames as ZIP. |
| **Dataset Builder** | `dataset_builder.py` | Assembles final datasets from one or more sessions. Three output formats: EuRoC (directory structure), ROS bag (rosbag2 sqlite3), HDF5 (single archive). Generates dataset manifest with calibration references. |

### Phase 2: Intelligence

| Feature | Module | Description |
|---------|--------|-------------|
| **Auto-Label** | `autolabel.py` | YOLOv8 inference on extracted frames. 13 rover-specific classes (person, bicycle, car, motorcycle, autorickshaw, bus, truck, traffic_light, stop_sign, dog, cow, pothole_region, speed_bump). Model caching per device. Batch inference. Exports to CVAT XML 1.1 or YOLO .txt format. |
| **Active Learning** | `active_learning.py` | Entropy-based uncertainty sampling. Ranks unlabeled images by model uncertainty to prioritize annotation effort. |
| **Analytics** | `road_analytics.py` | Dataset statistics: class distributions, scene diversity, geographic coverage, temporal coverage. |
| **Data Augmentation** | `augmentor.py` | Albumentations pipeline: horizontal flip, random rotate, brightness/contrast, hue/saturation, Gaussian noise, motion blur. YOLOv8-style mosaic augmentation built from scratch. |
| **Training** | `trainer.py` | Launches YOLOv8 training as a subprocess (killable without crashing the app). Per-epoch metric parsing. Generates YAML dataset configuration. MLflow experiment logging. |
| **Depth Estimation** | `depth_estimator.py` | Depth-Anything-v2 monocular depth estimation. Three model sizes (small, base, large). Affine-invariant relative depth maps. HuggingFace transformers integration. |
| **3D Reconstruction** | `reconstructor.py` | COLMAP Structure-from-Motion pipeline. SIFT feature extraction with GPU support. Sequential matcher for video sequences, exhaustive matcher for small sets. Dense mapping. Exports colored PLY point clouds via Open3D. |
| **SLAM Validation** | `slam_validator.py` | Runs ORBSLAM3 on EuRoC-formatted datasets. Parses TUM-format trajectories. Umeyama alignment with GPS ground truth. Computes ATE (Absolute Trajectory Error) and RPE (Relative Pose Error) via the evo library. |

### Phase 3: Deployment and Perception

| Feature | Module | Description |
|---------|--------|-------------|
| **Inference Server** | `inference_server.py` | Real-time model inference API for live deployment. |
| **Annotation Review** | `annotation_review.py` | Human-in-the-loop QA workflow. Approve, reject, or correct auto-generated annotations. Tracks annotator accuracy. |
| **Continuous Learning** | `continuous_learning.py` | Automated retraining loop: deploy model, collect predictions, incorporate human corrections, retrain on expanded dataset. |
| **Semantic Segmentation** | `segmentation.py` | SegFormer (Cityscapes-finetuned) with 21 classes including road, sidewalk, building, vegetation, person, car, autorickshaw, pothole_region, unpaved_road. Per-pixel masks with colorized overlays. |
| **Occupancy Grid** | `occupancy.py` | Bird's-eye-view occupancy mapping. Back-projects depth maps to 3D point clouds, projects to BEV grid. Fuses detection bounding boxes as obstacles with class-specific radii (person 0.5 m, car 2.0 m). Temporal fusion over sliding window. |
| **Lane Detection** | `lane_detector.py` | Dual-mode: classical CV pipeline (Canny edge detection, Hough transform, polynomial fitting with Savitzky-Golay smoothing) always available; optional Ultra Fast Lane Detection (UFLD) when user provides weights. |
| **Multi-Object Tracking** | `tracker.py` | ByteTrack implementation from scratch. 8-state Kalman filter with constant-velocity model. Two-stage IoU matching: high-confidence detections matched to active tracks, low-confidence detections matched to lost tracks. Hungarian assignment via scipy. MOT Challenge CSV export. Track heatmap generation. |
| **Edge Export** | `edge_exporter.py` | YOLOv8 model export pipeline: PyTorch to ONNX (with optimization), ONNX to TensorRT (for NVIDIA Jetson). Includes model benchmarking and standalone inference script generation. |
| **Dataset Versioning** | `versioning.py` | Git + DVC-backed dataset version control. Initialize, commit, diff, and restore dataset versions. |
| **Experiment Tracking** | `experiment_tracker.py` | MLflow integration with SQLite backend. Log metrics per training run, compare runs, launch MLflow UI. |

---

## A-to-Z Usage Guide

This guide walks through the complete workflow from raw field recordings to a deployed perception model.

### Step 1: Create a Session

Open Orvex and click **Sessions** in the sidebar.

1. Click **New Session**
2. Enter a name (e.g., `kalamassery_road_001`)
3. Set the environment tag: `road`, `indoor`, `gravel`, `highway`, or custom
4. Set the location (e.g., `Kalamassery, Kochi`)
5. Add notes about recording conditions

A session is a logical container for one data collection run. It groups raw files from multiple devices and tracks processing state.

### Step 2: Import Raw Files

Still in the **Sessions** panel:

1. Click **Add Files** or drag and drop files into the session
2. Select your GoPro `.MP4` files, Insta360 `.insv` files, and/or Sensor Logger `.csv` exports
3. Orvex detects the device type automatically from file metadata
4. Chapter-split GoPro files (GH010001.MP4, GH020001.MP4) are grouped as a single continuous recording

### Step 3: Audit Files

Switch to the **Audit** tab. Auditing runs automatically on import, but you can re-run it manually.

The audit checks:
- Video codec, resolution, FPS
- IMU stream presence and sample rate
- GPS fix quality (flags if DOP > 5 or fix_type < 3)
- HyperSmooth status (warns if enabled — it corrupts raw IMU correlation)
- SuperView status (non-standard aspect ratio)
- Timestamp gaps in IMU stream (flags gaps > 100 ms)

Review the issues column. Common actions:
- **HyperSmooth detected**: Re-record with Settings > Stabilization > Off
- **No GPS fix**: Record outdoors with clear sky, wait for GPS lock before starting
- **IMU gaps**: Check for SD card write speed issues

### Step 4: Extract Telemetry

Switch to the **Extract** tab.

1. Select the session to extract
2. Configure extraction parameters:
   - **Frame FPS**: How many frames per second to extract (default: 5)
   - **Frame format**: JPG (default) or PNG
   - **Frame quality**: 1-100 for JPG (default: 95)
   - **IMU interpolation**: Interpolate gyro onto accel timestamps (recommended)
3. Click **Extract**

Orvex extracts:
- Video frames via FFmpeg at the specified rate
- IMU data (accelerometer + gyroscope) at native rate (200 Hz for GoPro)
- GPS data with timestamps
- Output follows EuRoC format:
  ```
  session_output/
  +-- cam0/
  |   +-- data/
  |       +-- 1706000000000000000.jpg
  |       +-- 1706000000200000000.jpg
  |       +-- ...
  +-- imu0/
  |   +-- data.csv    (timestamp_ns, gyro_xyz, accel_xyz)
  +-- gps.csv         (timestamp_ns, lat, lon, alt, speed, fix)
  ```

### Step 5: Synchronize Devices (Multi-Device Sessions)

If your session has files from multiple devices, switch to the **Telemetry** tab.

The telemetry viewer shows overlaid accelerometer magnitude plots from all devices. Use synchronization to align them:

- **GPS anchor** (recommended when all devices have GPS): Uses GoPro GPS timestamps as the master clock. Other devices are offset to match.
- **Cross-correlation** (when GPS is unavailable): Correlates acceleration magnitude signals to find the time offset. Both devices must have experienced the same motion event (e.g., a sharp tap).
- **Manual offset**: Enter the offset in milliseconds if you know it.

After synchronization, the overlay plot should show aligned signals. If the computed lag is > 2 seconds, Orvex warns you — this likely indicates a sync failure.

### Step 6: Calibrate Camera (One-Time per Device)

Switch to the **Calibrate** tab. This is a multi-step guided workflow.

**Step 6a: IMU Static Noise Characterization**
1. Record your GoPro sitting flat on a stable table for at least 4 hours
2. Import this recording and select it as the static calibration file
3. Orvex extracts IMU data and computes Allan deviation to derive:
   - Accelerometer noise density (m/s^2/sqrt(Hz))
   - Accelerometer random walk (m/s^2/sqrt(s))
   - Gyroscope noise density (rad/s/sqrt(Hz))
   - Gyroscope random walk (rad/s/sqrt(s))
4. Output: `step1_imu_noise_params.json`

**Step 6b: Camera Intrinsic Calibration**
1. Print an AprilTag or chessboard calibration pattern
2. Record a 2-3 minute video moving the camera in front of the pattern, covering all corners and angles
3. Import the calibration video
4. Orvex extracts frames, detects the calibration pattern, and computes:
   - Focal lengths (fx, fy)
   - Principal point (cx, cy)
   - Distortion coefficients (k1, k2, p1, p2)
   - Reprojection error (must be < 0.5 px; at least 15 poses required)
5. Output: `step2_camera_intrinsics.json`

**Step 6c: Camera-IMU Extrinsic Calibration**
1. Record a video shaking the camera aggressively in front of a fixed calibration board
2. Orvex invokes OpenImuCameraCalibrator as a subprocess (this takes 20-40 minutes)
3. Real-time stdout from the subprocess is streamed to the log panel
4. Output: `step3_camera_imu_extrinsics.json` containing the 4x4 transformation matrix T_cam_imu

**Step 6d: Validation**
Orvex checks that all steps are complete and results are within tolerance:
- Reprojection error < 0.5 px
- Translation magnitude < 10 cm
- Rotation magnitude < 0.5 degrees

Calibration results are saved to `data/calibration/<session_id>/` and are reusable across sessions for the same device.

### Step 7: Browse Frames

Switch to the **Frames** tab.

- View extracted frames as a thumbnail grid
- Each frame shows:
  - Timestamp
  - Blur score (Laplacian variance) — blurry frames are flagged
  - Duplicate status (SSIM > 0.98 with adjacent frame)
- Click a frame to see full resolution with overlaid metadata
- Select frames for export or annotation

### Step 8: Build Dataset

Switch to the **Dataset** tab.

1. Select one or more sessions to include
2. Choose output format:
   - **EuRoC** (default): Directory structure with cam0/, imu0/, gps.csv
   - **ROS bag**: rosbag2 sqlite3 format (requires rosbag2_py)
   - **HDF5**: Single .h5 archive (requires h5py)
3. Optionally attach calibration data
4. Click **Build**
5. Orvex generates a dataset manifest (JSON) with session metadata, device info, durations, and calibration references
6. Integrity check runs automatically: verifies frame count matches, no timestamp gaps

### Step 9: Auto-Label with YOLOv8

Switch to the **Auto-Label** tab.

1. Select a model variant: `yolov8n.pt` (fast), `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt` (accurate)
2. Set confidence threshold (default: 0.25)
3. Click **Run**
4. YOLOv8 processes all extracted frames and generates bounding box annotations for 13 classes:

| Class | COCO Mapping | Notes |
|-------|-------------|-------|
| person | COCO class 0 | Pedestrians |
| bicycle | COCO class 1 | |
| car | COCO class 2 | |
| motorcycle | COCO class 3 | |
| autorickshaw | No COCO equivalent | Requires fine-tuned model |
| bus | COCO class 5 | |
| truck | COCO class 7 | |
| traffic_light | COCO class 9 | |
| stop_sign | COCO class 11 | |
| dog | COCO class 16 | |
| cow | COCO class 19 | |
| pothole_region | No COCO equivalent | Requires fine-tuned model |
| speed_bump | No COCO equivalent | Requires fine-tuned model |

5. Annotations are exported to CVAT XML 1.1 or YOLO .txt format

### Step 10: Run Depth Estimation

Switch to the **Depth** tab.

1. Select model size: small (fast), base (balanced), large (accurate)
2. Click **Run**
3. Depth-Anything-v2 produces relative depth maps for each frame
4. Depth maps are used downstream by the occupancy grid generator

### Step 11: Run Semantic Segmentation

Switch to the **Segment** tab.

1. Click **Run** to process frames with SegFormer (Cityscapes-finetuned)
2. Produces per-pixel class masks for 21 classes:
   road, sidewalk, building, wall, fence, pole, traffic_light, traffic_sign, vegetation, terrain, sky, person, rider, car, truck, bus, motorcycle, bicycle, autorickshaw, pothole_region, unpaved_road
3. Colorized overlay images are generated

### Step 12: Run Lane Detection

Switch to the **Lanes** tab.

1. Choose mode:
   - **Classical CV** (always available): Canny + Hough + polynomial fitting
   - **UFLD** (requires weights): Ultra Fast Lane Detection neural network
2. Click **Run**
3. Detected lanes are drawn as polylines on frames

### Step 13: Run Multi-Object Tracking

Switch to the **Track** tab.

1. Auto-label results from Step 9 are used as detections
2. ByteTrack assigns persistent track IDs across frames using:
   - Kalman filter prediction (constant-velocity model)
   - Two-stage IoU matching (high-conf to active, low-conf to lost)
   - Hungarian assignment
3. Output: MOT Challenge CSV format, track statistics, heatmap visualization
4. Track metrics: total tracks, average length, fragmentation rate

### Step 14: Generate Occupancy Grid

Switch to the **Occupancy** tab.

1. Requires depth maps (Step 10) and detections (Step 9)
2. Produces bird's-eye-view occupancy grids:
   - Depth maps are back-projected to 3D point clouds
   - Points are projected to a top-down BEV grid
   - Detection bounding boxes mark occupied cells with class-specific radii
   - Temporal fusion smooths the grid over a sliding window

### Step 15: 3D Reconstruction

Switch to the **3D Reconstruct** tab.

1. Requires COLMAP installed on the system
2. Click **Run** to execute the SfM pipeline:
   - SIFT feature extraction (GPU-accelerated if available)
   - Sequential matching (for video) or exhaustive matching (< 100 frames)
   - Incremental mapper
3. Output: colored PLY point cloud, camera poses, reprojection statistics
4. Warns if < 80% of frames were successfully registered

### Step 16: Validate with SLAM

Switch to the **SLAM Validate** tab.

1. Requires ORBSLAM3 installed on the system
2. Select the EuRoC-format dataset and calibration files
3. Click **Run** to launch ORBSLAM3 as a subprocess
4. Orvex parses the TUM-format trajectory output and computes:
   - ATE (Absolute Trajectory Error) against GPS ground truth
   - RPE (Relative Pose Error) at 1-second intervals
   - Loop closure drift percentage
5. Trajectory alignment uses Umeyama (scale + rotation + translation)

### Step 17: Augment and Train

**Augment** (Augment tab):
1. Configure augmentation pipeline: flip, rotate, brightness, contrast, noise, blur, mosaic
2. Set augmentation factor (e.g., 3x = generate 3 augmented copies per image)
3. Run to expand the training set

**Train** (Train tab):
1. Select base model (yolov8n/s/m/l/x)
2. Set epochs, batch size, image size, learning rate
3. Click **Start Training** — runs as a subprocess (will not freeze the UI)
4. Per-epoch metrics (loss, mAP@50, mAP@50-95) are displayed live
5. Training run is logged to MLflow (Experiments tab)

### Step 18: Export for Edge Deployment

Switch to the **Edge Export** tab.

1. Select the trained model (.pt file)
2. Export chain: PyTorch -> ONNX -> TensorRT
3. Configure input resolution, batch size, precision (FP16 for Jetson)
4. Run benchmarks on the exported model
5. Generate a standalone Python inference script for deployment

### Step 19: Review Annotations

Switch to the **Review** tab.

1. Browse auto-generated and manually corrected annotations
2. Approve, reject, or edit each annotation
3. Annotator accuracy is tracked per reviewer
4. Rejected annotations feed back into the active learning loop

### Step 20: Version Your Dataset

Switch to the **Versions** tab.

1. Initialize DVC in your dataset directory
2. Commit a named version (e.g., `v1.0-initial`)
3. As you add data and retrain, commit new versions
4. Diff between versions to see what changed
5. Restore any previous version

### Step 21: Continuous Learning

Switch to the **Auto-Retrain** tab.

This closes the loop:
1. Deploy a model (via inference server or edge export)
2. Collect new predictions in the field
3. Import predictions back as annotations
4. Human reviewers correct errors (Review tab)
5. Retrain on the expanded, corrected dataset
6. Repeat

---

## Calibration Workflow

```
Step 1: IMU Static (4+ hours)
  Input:  Static GoPro recording on flat surface
  Method: Allan deviation at tau=1/rate and tau=1s
  Output: accel/gyro noise density and random walk
  File:   step1_imu_noise_params.json

Step 2: Camera Intrinsic
  Input:  Video of chessboard/AprilTag pattern
  Method: cv2.calibrateCamera with >=15 detected poses
  Output: fx, fy, cx, cy, distortion coefficients
  File:   step2_camera_intrinsics.json
  Check:  Reprojection error < 0.5 px

Step 3: Camera-IMU Extrinsic (20-40 min)
  Input:  Shaking video in front of fixed board + Step 1 + Step 2
  Method: OpenImuCameraCalibrator subprocess
  Output: 4x4 T_cam_imu transformation matrix
  File:   step3_camera_imu_extrinsics.json
  Check:  Translation < 5 mm, rotation < 0.5 degrees

Step 4: Validation
  Checks: All steps complete, all tolerances met
```

---

## AI Models

| Model | Task | Source | Classes | Notes |
|-------|------|--------|---------|-------|
| YOLOv8 (n/s/m/l/x) | Object detection | ultralytics | 13 rover classes | Auto-downloaded on first use |
| SegFormer | Semantic segmentation | HuggingFace (nvidia/segformer-b0-finetuned-cityscapes-1024-1024) | 21 classes | Extended with autorickshaw, pothole, unpaved_road |
| Depth-Anything-v2 | Monocular depth | HuggingFace | Dense depth map | Small/base/large variants |
| UFLD | Lane detection | User-provided weights | Lane polylines | Optional; classical CV fallback always available |

All models are cached after first load. GPU is used automatically when available (CUDA).

---

## Export Formats

### Dataset Formats

| Format | Structure | Compatible With |
|--------|-----------|----------------|
| **EuRoC** | `cam0/data/*.jpg` + `imu0/data.csv` + `gps.csv` | ORBSLAM3, VINS-Mono, Kalibr |
| **ROS bag** | rosbag2 SQLite3 | ROS 2 ecosystem |
| **HDF5** | Single `.h5` archive | Custom processing pipelines |

### Annotation Formats

| Format | Exporter | Use Case |
|--------|----------|----------|
| **CVAT XML 1.1** | `autolabel.py` | Import into CVAT for manual refinement |
| **YOLO .txt** | `autolabel.py` | Direct YOLOv8 training |

### IMU CSV Header (EuRoC-compatible)

```
#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]
```

All timestamps are stored as 64-bit integer nanoseconds. No floats for timestamps anywhere in the pipeline.

---

## API Reference

The web backend exposes 96+ endpoints across 23 route modules. Full interactive documentation is available at `http://localhost:8000/docs` when the server is running.

### Route Groups

| Prefix | Module | Key Endpoints |
|--------|--------|---------------|
| `/sessions` | sessions.py | `POST /` create, `GET /` list, `GET /{id}` get, `DELETE /{id}` delete |
| `/audit` | audit.py | `POST /{session_id}` run audit |
| `/extraction` | extraction.py | `POST /{session_id}` start extraction |
| `/calibration` | calibration.py | `POST /{session_id}/step/{n}` run calibration step |
| `/dataset` | dataset.py | `POST /{session_id}/build` assemble dataset |
| `/autolabel` | autolabel.py | `POST /{session_id}` run YOLOv8 |
| `/depth` | depth.py | `POST /{session_id}` run depth estimation |
| `/segment` | segmentation.py | `POST /{session_id}` run segmentation |
| `/tracking` | tracking.py | `POST /{session_id}` run ByteTrack |
| `/training` | training.py | `POST /start` launch training, `GET /status/{task_id}` |
| `/review` | annotation_review.py | `POST /{session_id}/reviews` save reviews |
| `/versions` | versioning.py | `POST /init` init DVC, `GET /versions` list, `POST /restore` restore |
| `/experiments` | experiments.py | `GET /runs` list runs, `GET /compare` compare |
| `/edge` | edge_export.py | `POST /export` export to ONNX/TRT |
| `/inference` | inference.py | `POST /predict` run inference |

### WebSocket

```
ws://host:8000/ws/tasks/{task_id}
```

Streams JSON messages with `status`, `progress`, `total`, and `message` fields until the task completes.

---

## Testing

```bash
# Run all tests
PYTHONPATH=. pytest tests/ -v

# Run a specific test module
PYTHONPATH=. pytest tests/test_audit.py -v

# Run with coverage
PYTHONPATH=. pytest tests/ --cov=core --cov-report=term-missing
```

Tests that require fixture files (real GoPro MP4, Insta360 INSV, Sensor Logger CSV) are automatically skipped when fixtures are absent. Place test files in `tests/fixtures/`:

```
tests/fixtures/
  sample_hero11.MP4          # GoPro Hero 11 recording
  sample_x4.insv             # Insta360 X4 recording
  sample_sensorlogger/       # Sensor Logger CSV folder
```

---

## Project Structure

```
Orvex/
|-- core/                          # Shared business logic (33 modules)
|   |-- models.py                  # Pydantic v2 data models
|   |-- session_manager.py         # Session CRUD + TOML persistence
|   |-- audit.py                   # File analysis and device detection
|   |-- extractor_gopro.py         # GoPro GPMF telemetry extraction
|   |-- extractor_insta360.py      # Insta360 INSV extraction
|   |-- extractor_sensorlogger.py  # Android Sensor Logger parsing
|   |-- synchronizer.py            # Multi-device temporal alignment
|   |-- calibration.py             # 4-step calibration pipeline
|   |-- dataset_builder.py         # EuRoC / ROS bag / HDF5 assembly
|   |-- autolabel.py               # YOLOv8 auto-labeling
|   |-- segmentation.py            # SegFormer semantic segmentation
|   |-- depth_estimator.py         # Depth-Anything-v2
|   |-- lane_detector.py           # Classical CV + UFLD lane detection
|   |-- tracker.py                 # ByteTrack multi-object tracking
|   |-- occupancy.py               # BEV occupancy grid
|   |-- reconstructor.py           # COLMAP 3D reconstruction
|   |-- slam_validator.py          # ORBSLAM3 validation + evo metrics
|   |-- trainer.py                 # YOLOv8 training subprocess
|   |-- augmentor.py               # Albumentations + mosaic augmentation
|   |-- active_learning.py         # Uncertainty-based sampling
|   |-- continuous_learning.py     # Retraining feedback loop
|   |-- annotation_review.py       # Human-in-the-loop QA
|   |-- edge_exporter.py           # ONNX + TensorRT export
|   |-- versioning.py              # DVC dataset versioning
|   |-- experiment_tracker.py      # MLflow experiment logging
|   |-- inference_server.py        # Real-time inference API
|   |-- road_analytics.py          # Dataset analytics
|   |-- export_profiles.py         # Named export configurations
|   |-- insv_telemetry.py          # INSV binary telemetry parsing
|   |-- insta360_processor.py      # Insta360 processing pipeline
|   |-- api_server.py              # FastAPI integration layer
|   |-- utils.py                   # FFmpeg/ffprobe subprocess wrappers
|   +-- __init__.py
|
|-- desktop/                       # PyQt6 desktop application
|   |-- main.py                    # Entry point
|   |-- app.py                     # QApplication factory
|   |-- mainwindow.py              # Main window (27 stacked panels)
|   |-- theme.py                   # Dark theme colors + stylesheet
|   |-- workers.py                 # QThread background workers
|   +-- widgets/                   # 28 specialized UI panels
|       |-- session_panel.py
|       |-- audit_widget.py
|       |-- extraction_widget.py
|       |-- calibration_widget.py
|       |-- telemetry_plot.py
|       |-- frame_browser.py
|       |-- dataset_widget.py
|       |-- autolabel_widget.py
|       |-- training_widget.py
|       |-- depth_widget.py
|       |-- slam_widget.py
|       |-- reconstruction_widget.py
|       |-- tracking_widget.py
|       |-- segmentation_widget.py
|       |-- occupancy_widget.py
|       |-- lane_widget.py
|       |-- review_widget.py
|       +-- ... (11 more)
|
|-- web/
|   |-- backend/
|   |   |-- main.py               # FastAPI app (23 routers, 96+ endpoints)
|   |   +-- routes/                # One file per route group
|   +-- frontend/
|       |-- src/
|       |   |-- pages/             # 25 React pages
|       |   +-- components/        # Shared UI components
|       |-- package.json
|       +-- vite.config.js
|
|-- tests/                         # pytest test suite (32 test files)
|   |-- conftest.py                # Shared fixtures and skip markers
|   |-- test_audit.py
|   |-- test_session_manager.py
|   +-- ... (30 more)
|
|-- data/                          # Default data directory
|   |-- sessions/
|   |-- calibration/
|   +-- exports/
|
|-- requirements.txt               # Core Python dependencies
|-- requirements-web.txt           # Web backend dependencies
|-- pyproject.toml                 # Package metadata and entry points
+-- README.md
```

---

## System Requirements

### Minimum

| Component | Requirement |
|-----------|-------------|
| OS | Ubuntu 22.04+ / Debian 12+ / Fedora 38+ |
| Python | 3.11 or higher |
| RAM | 8 GB |
| Storage | 50 GB free (datasets can be large) |
| FFmpeg | Any recent version |

### Recommended (for AI features)

| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA with CUDA 11.8+ (RTX 3060 or better) |
| VRAM | 6 GB+ (8 GB for Depth-Anything-v2 large) |
| RAM | 32 GB |
| Storage | 500 GB SSD |

### Optional External Tools

| Tool | Version | Required For |
|------|---------|-------------|
| COLMAP | 3.8+ | 3D reconstruction |
| ORBSLAM3 | Latest | SLAM validation |
| OpenImuCameraCalibrator | Latest | Camera-IMU extrinsic calibration (Step 3) |
| DVC | 3.0+ | Dataset versioning |
| MLflow | 2.10+ | Experiment tracking |
| NVIDIA TensorRT | 8.6+ | Edge model export |

---

## License

MIT License. See LICENSE file for details.

---

**Orvex** is built for the real challenges of autonomous navigation in Indian road conditions — unstructured roads, mixed traffic, variable surfaces, and limited GPS coverage. Every module is designed around these constraints.
