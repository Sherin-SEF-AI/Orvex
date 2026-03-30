"""
core/road_analytics.py — Scene diversity, class distribution, and geographic
coverage analytics for rover datasets.

Generates a self-contained HTML report with embedded charts.

No UI imports — pure Python business logic.

Dependencies:
    pip install folium Jinja2 matplotlib
    (numpy, opencv-python already in requirements.txt)
"""
from __future__ import annotations

import base64
import io
import math
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from loguru import logger

from core.models import FrameAnnotation, GPSSample, GeoCoverageReport, SceneDiversityReport

ProgressCB = Callable[[int], None]
StatusCB = Callable[[str], None]

# 10-metre grid cell size in degrees (approximate)
_GRID_DEG = 10.0 / 111_320.0


# ---------------------------------------------------------------------------
# Class distribution
# ---------------------------------------------------------------------------

def compute_class_distribution(annotations: list[FrameAnnotation]) -> dict:
    """Compute per-class detection statistics across all annotations.

    Returns dict with keys:
        per_class: dict[str, {count, percent, avg_confidence,
                               avg_bbox_area_percent, frames_present,
                               co_occurrence: dict[str,int]}]
        total_detections: int
        total_frames: int
    """
    total = 0
    frames_present: dict[str, int] = {}
    counts: dict[str, int] = {}
    conf_sums: dict[str, float] = {}
    area_sums: dict[str, float] = {}
    co_occur: dict[str, dict[str, int]] = {}

    for ann in annotations:
        names_in_frame = {d.class_name for d in ann.detections}
        for det in ann.detections:
            total += 1
            cn = det.class_name
            counts[cn] = counts.get(cn, 0) + 1
            conf_sums[cn] = conf_sums.get(cn, 0.0) + det.confidence

            x1, y1, x2, y2 = det.bbox_xyxy
            area_pct = (x2 - x1) * (y2 - y1)  # in pixels² — normalised later
            area_sums[cn] = area_sums.get(cn, 0.0) + area_pct

        for cn in names_in_frame:
            frames_present[cn] = frames_present.get(cn, 0) + 1
            co = co_occur.setdefault(cn, {})
            for other in names_in_frame:
                if other != cn:
                    co[other] = co.get(other, 0) + 1

    per_class: dict[str, dict] = {}
    for cn, cnt in counts.items():
        per_class[cn] = {
            "count": cnt,
            "percent": cnt / total * 100.0 if total else 0.0,
            "avg_confidence": conf_sums[cn] / cnt if cnt else 0.0,
            "avg_bbox_area_percent": area_sums[cn] / cnt if cnt else 0.0,
            "frames_present": frames_present.get(cn, 0),
            "co_occurrence": co_occur.get(cn, {}),
        }

    return {
        "per_class": per_class,
        "total_detections": total,
        "total_frames": len(annotations),
    }


# ---------------------------------------------------------------------------
# Scene diversity
# ---------------------------------------------------------------------------

def compute_scene_diversity(
    frame_paths: list[str],
    sample_n: int = 500,
    progress_callback: ProgressCB | None = None,
    status_callback: StatusCB | None = None,
) -> SceneDiversityReport:
    """Analyse lighting, edge density, and scene type from sampled frames.

    Args:
        frame_paths:  All frame paths from an extracted session.
        sample_n:     Maximum number of frames to sample.
        progress_callback: 0-100.
        status_callback:   Status strings.

    Returns:
        SceneDiversityReport.
    """
    if not frame_paths:
        return SceneDiversityReport(
            lighting_distribution={"bright": 0.0, "normal": 0.0, "dark": 0.0},
            brightness_stats={},
            edge_density_stats={},
            estimated_scene_types={},
        )

    step = max(1, len(frame_paths) // sample_n)
    sampled = frame_paths[::step][:sample_n]
    n = len(sampled)

    brightnesses: list[float] = []
    edge_densities: list[float] = []
    sky_ratios: list[float] = []
    road_ratios: list[float] = []

    for i, fp in enumerate(sampled):
        img = cv2.imread(fp)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Brightness
        brightness = float(np.mean(gray))
        brightnesses.append(brightness)

        # Edge density
        edges = cv2.Canny(gray, 100, 200)
        edge_density = float(edges.sum() / 255) / (h * w)
        edge_densities.append(edge_density)

        # Sky ratio: top 20% rows, HSV hue in [90,130] + low saturation
        top = img[: h // 5, :, :]
        hsv_top = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
        sky_mask = (
            (hsv_top[:, :, 0] >= 90) & (hsv_top[:, :, 0] <= 130)
            & (hsv_top[:, :, 1] < 60)
        )
        sky_ratio = float(sky_mask.mean())
        sky_ratios.append(sky_ratio)

        # Road ratio: bottom 30% rows, gray asphalt tones [60,120]
        bot = gray[int(h * 0.7):, :]
        road_mask = (bot >= 60) & (bot <= 120)
        road_ratios.append(float(road_mask.mean()))

        if progress_callback and i % max(1, n // 20) == 0:
            progress_callback(int(i / n * 100))

    if status_callback:
        status_callback("Computing diversity statistics…")

    bright_arr = np.array(brightnesses) if brightnesses else np.array([128.0])
    edge_arr = np.array(edge_densities) if edge_densities else np.array([0.0])

    # Lighting distribution
    bright_pct = float(np.mean(bright_arr > 150)) * 100
    dark_pct = float(np.mean(bright_arr < 80)) * 100
    normal_pct = 100.0 - bright_pct - dark_pct

    # Scene type heuristics
    sky_mean = float(np.mean(sky_ratios)) if sky_ratios else 0.0
    road_mean = float(np.mean(road_ratios)) if road_ratios else 0.0
    # Urban: high edge density + low sky
    # Highway: high road ratio + medium edge
    # Residential: medium all
    edge_mean = float(edge_arr.mean())
    urban = min(100.0, edge_mean * 1000)  # scale to %
    highway = min(100.0, road_mean * 150)
    residential = max(0.0, 100.0 - urban - highway)

    return SceneDiversityReport(
        lighting_distribution={
            "bright": round(bright_pct, 1),
            "normal": round(max(0.0, normal_pct), 1),
            "dark": round(dark_pct, 1),
        },
        brightness_stats={
            "mean": round(float(bright_arr.mean()), 2),
            "std":  round(float(bright_arr.std()),  2),
            "min":  round(float(bright_arr.min()),  2),
            "max":  round(float(bright_arr.max()),  2),
        },
        edge_density_stats={
            "mean": round(float(edge_arr.mean()), 6),
            "std":  round(float(edge_arr.std()),  6),
            "min":  round(float(edge_arr.min()),  6),
            "max":  round(float(edge_arr.max()),  6),
        },
        estimated_scene_types={
            "urban":        round(min(urban, 100.0), 1),
            "highway":      round(min(highway, 100.0), 1),
            "residential":  round(max(residential, 0.0), 1),
        },
    )


# ---------------------------------------------------------------------------
# Geographic coverage
# ---------------------------------------------------------------------------

def compute_geographic_coverage(
    gps_samples: list[GPSSample],
    output_dir: str,
) -> GeoCoverageReport:
    """Compute GPS coverage statistics and generate a folium heatmap.

    Args:
        gps_samples: From ExtractedSession.gps_samples.
        output_dir:  Directory to write coverage_map.html.

    Returns:
        GeoCoverageReport.

    Raises:
        ValueError: if no GPS samples provided.
    """
    if not gps_samples:
        raise ValueError(
            "No GPS samples provided for geographic coverage analysis. "
            "Ensure extraction was run and the recording has GPS data."
        )

    lats = [s.latitude for s in gps_samples]
    lons = [s.longitude for s in gps_samples]
    speeds = [s.speed_mps for s in gps_samples]
    timestamps = [s.timestamp_ns for s in gps_samples]

    # Total distance
    total_dist_km = _compute_total_distance_km(lats, lons)

    # Bounding box
    bbox = {
        "min_lat": float(np.min(lats)),
        "max_lat": float(np.max(lats)),
        "min_lon": float(np.min(lons)),
        "max_lon": float(np.max(lons)),
    }

    # Unique 10m grid cells
    grid_cells: set[tuple[int, int]] = set()
    for lat, lon in zip(lats, lons):
        grid_cells.add((int(lat / _GRID_DEG), int(lon / _GRID_DEG)))

    # Speed stats
    speed_arr = np.array(speeds)
    avg_speed = float(speed_arr.mean())

    # Stationary time: frames where speed < 0.5 m/s
    stationary_frames = int(np.sum(speed_arr < 0.5))
    stationary_pct = stationary_frames / len(speeds) * 100.0 if speeds else 0.0

    # Generate folium heatmap
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    map_path = out_dir / "coverage_map.html"
    _generate_folium_map(lats, lons, speeds, str(map_path))

    logger.info(
        "road_analytics: GPS coverage — {:.2f} km, {} grid cells, {:.1f}% stationary",
        total_dist_km, len(grid_cells), stationary_pct,
    )
    return GeoCoverageReport(
        total_distance_km=round(total_dist_km, 3),
        bounding_box=bbox,
        unique_grid_cells=len(grid_cells),
        coverage_map_path=str(map_path.resolve()),
        avg_speed_mps=round(avg_speed, 3),
        stationary_time_percent=round(stationary_pct, 1),
    )


def _compute_total_distance_km(lats: list[float], lons: list[float]) -> float:
    """Haversine sum over consecutive GPS points."""
    R = 6371.0  # Earth radius km
    total = 0.0
    for i in range(1, len(lats)):
        dlat = math.radians(lats[i] - lats[i - 1])
        dlon = math.radians(lons[i] - lons[i - 1])
        a = (math.sin(dlat / 2) ** 2
             + math.cos(math.radians(lats[i - 1]))
             * math.cos(math.radians(lats[i]))
             * math.sin(dlon / 2) ** 2)
        total += 2 * R * math.asin(math.sqrt(a))
    return total


def _generate_folium_map(
    lats: list[float],
    lons: list[float],
    speeds: list[float],
    output_path: str,
) -> None:
    """Generate folium HTML heatmap of GPS track with speed color coding."""
    try:
        import folium
        from folium.plugins import HeatMap
    except ImportError as exc:
        logger.warning(
            "folium not installed — skipping GPS map generation. "
            "Install with: pip install folium"
        )
        Path(output_path).write_text(
            "<html><body><p>folium not installed — GPS map unavailable.</p></body></html>"
        )
        return

    center_lat = float(np.mean(lats))
    center_lon = float(np.mean(lons))

    m = folium.Map(location=[center_lat, center_lon], zoom_start=15)

    # Speed-colored polyline segments
    speed_arr = np.array(speeds)
    sp_max = float(speed_arr.max()) if speed_arr.max() > 0 else 1.0

    for i in range(1, len(lats)):
        speed_norm = speeds[i] / sp_max
        # Interpolate green (slow) → red (fast)
        r = int(speed_norm * 255)
        g = int((1 - speed_norm) * 255)
        color = f"#{r:02x}{g:02x}00"
        folium.PolyLine(
            [(lats[i - 1], lons[i - 1]), (lats[i], lons[i])],
            color=color,
            weight=2,
            opacity=0.7,
        ).add_to(m)

    # Start / end markers
    folium.Marker(
        [lats[0], lons[0]],
        popup="Start",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(m)
    folium.Marker(
        [lats[-1], lons[-1]],
        popup="End",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(m)

    # Heatmap layer
    heat_data = [[lat, lon, spd / sp_max] for lat, lon, spd in zip(lats, lons, speeds)]
    HeatMap(heat_data, radius=8).add_to(m)

    m.save(output_path)


# ---------------------------------------------------------------------------
# Temporal coverage
# ---------------------------------------------------------------------------

def compute_temporal_coverage(
    frame_timestamps: list[int],
    imu_timestamps: list[int],
) -> dict:
    """Check for gaps in frame and IMU streams.

    Returns dict with:
        frame_gaps:               list of (start_ns, end_ns, gap_s) for gaps > 0.5s
        imu_gaps:                 list of (start_ns, end_ns, gap_s) for gaps > 50ms
        total_gap_time_seconds:   float
        data_completeness_percent: float
    """
    frame_gaps = _find_gaps(frame_timestamps, threshold_ns=int(0.5e9))
    imu_gaps = _find_gaps(imu_timestamps, threshold_ns=int(50e6))

    total_duration_ns = 0
    total_gap_ns = 0

    if frame_timestamps:
        total_duration_ns = frame_timestamps[-1] - frame_timestamps[0]
        total_gap_ns = sum(g[2] for g in frame_gaps) * int(1e9)

    completeness = (
        (1.0 - total_gap_ns / total_duration_ns) * 100.0
        if total_duration_ns > 0
        else 100.0
    )

    return {
        "frame_gaps": [(s, e, round(g, 3)) for s, e, g in frame_gaps],
        "imu_gaps":   [(s, e, round(g, 3)) for s, e, g in imu_gaps],
        "total_gap_time_seconds": round(total_gap_ns / 1e9, 3),
        "data_completeness_percent": round(max(0.0, completeness), 1),
    }


def _find_gaps(
    timestamps: list[int],
    threshold_ns: int,
) -> list[tuple[int, int, float]]:
    """Return (start_ns, end_ns, gap_seconds) for all gaps above threshold."""
    gaps: list[tuple[int, int, float]] = []
    for i in range(1, len(timestamps)):
        delta = timestamps[i] - timestamps[i - 1]
        if delta > threshold_ns:
            gaps.append((timestamps[i - 1], timestamps[i], delta / 1e9))
    return gaps


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def generate_dataset_report(
    session_ids: list[str],
    annotations: list[FrameAnnotation],
    gps_samples: list[GPSSample],
    output_dir: str | None = None,
) -> str:
    """Generate a self-contained HTML report for a dataset.

    Embeds all charts as base64 PNG images.

    Args:
        session_ids: Session IDs included in the report.
        annotations: All FrameAnnotations for the dataset.
        gps_samples: GPS samples for geographic coverage.
        output_dir:  Where to write the HTML file. Uses a temp dir if None.

    Returns:
        Absolute path to the written HTML file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- Compute analytics ---
    cls_dist = compute_class_distribution(annotations)

    # Chart 1: Class distribution bar
    chart_cls = _bar_chart_base64(
        labels=list(cls_dist["per_class"].keys()),
        values=[v["count"] for v in cls_dist["per_class"].values()],
        title="Detections per Class",
        xlabel="Class",
        ylabel="Count",
    )

    # Coverage map placeholder
    coverage_map_html = "<p>GPS data not available.</p>"
    geo_stats: dict = {}
    if gps_samples:
        import tempfile
        tmp_out = output_dir or tempfile.mkdtemp()
        geo = compute_geographic_coverage(gps_samples, tmp_out)
        geo_stats = {
            "distance_km": geo.total_distance_km,
            "unique_cells": geo.unique_grid_cells,
            "avg_speed": geo.avg_speed_mps,
            "stationary": geo.stationary_time_percent,
        }
        if Path(geo.coverage_map_path).exists():
            coverage_map_html = Path(geo.coverage_map_path).read_text()

    # --- HTML assembly ---
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_frames = len(annotations)
    total_dets = cls_dist["total_detections"]
    n_empty = sum(1 for a in annotations if not a.detections)

    html = textwrap.dedent(f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8"/>
          <title>RoverDataKit Dataset Report</title>
          <style>
            body {{ font-family: sans-serif; background:#1a1a2e; color:#e0e0e0; margin:2rem; }}
            h1,h2 {{ color:#e94560; }}
            table {{ border-collapse:collapse; width:100%; }}
            th,td {{ padding:8px 12px; border:1px solid #0f3460; text-align:left; }}
            th {{ background:#16213e; }}
            tr:nth-child(even) {{ background:#16213e55; }}
            img {{ max-width:100%; border-radius:4px; }}
            .stat {{ display:inline-block; margin:1rem; padding:1rem 2rem;
                     background:#16213e; border-radius:6px; text-align:center; }}
            .stat .val {{ font-size:2rem; color:#e94560; font-weight:bold; }}
            .stat .lbl {{ font-size:0.85rem; color:#666680; }}
            .map-container {{ width:100%; height:500px; border:none; overflow:hidden; }}
          </style>
        </head>
        <body>
          <h1>RoverDataKit — Dataset Report</h1>
          <p>Generated: {generated_at}</p>
          <p>Sessions: {', '.join(session_ids) or 'N/A'}</p>

          <div>
            <div class="stat"><div class="val">{n_frames}</div><div class="lbl">Total Frames</div></div>
            <div class="stat"><div class="val">{total_dets}</div><div class="lbl">Total Detections</div></div>
            <div class="stat"><div class="val">{n_empty}</div><div class="lbl">Empty Frames</div></div>
            <div class="stat"><div class="val">{len(cls_dist['per_class'])}</div><div class="lbl">Classes Detected</div></div>
          </div>

          <h2>Class Distribution</h2>
          <img src="data:image/png;base64,{chart_cls}" alt="Class distribution"/>

          <h2>Per-Class Statistics</h2>
          <table>
            <tr><th>Class</th><th>Count</th><th>%</th><th>Avg Conf</th><th>Frames Present</th></tr>
    """)

    for cls, stats in sorted(
        cls_dist["per_class"].items(),
        key=lambda x: x[1]["count"],
        reverse=True,
    ):
        html += (
            f"    <tr><td>{cls}</td><td>{stats['count']}</td>"
            f"<td>{stats['percent']:.1f}%</td>"
            f"<td>{stats['avg_confidence']:.3f}</td>"
            f"<td>{stats['frames_present']}</td></tr>\n"
        )

    html += "  </table>\n"

    if geo_stats:
        html += textwrap.dedent(f"""\
          <h2>Geographic Coverage</h2>
          <div>
            <div class="stat"><div class="val">{geo_stats['distance_km']:.2f} km</div><div class="lbl">Total Distance</div></div>
            <div class="stat"><div class="val">{geo_stats['unique_cells']}</div><div class="lbl">10m Grid Cells</div></div>
            <div class="stat"><div class="val">{geo_stats['avg_speed']:.1f} m/s</div><div class="lbl">Avg Speed</div></div>
            <div class="stat"><div class="val">{geo_stats['stationary']:.1f}%</div><div class="lbl">Stationary Time</div></div>
          </div>
          <h3>GPS Track Map</h3>
          <div class="map-container">
            <iframe srcdoc="{_escape_html(coverage_map_html)}"
                    style="width:100%;height:500px;border:none;"></iframe>
          </div>
        """)

    html += "</body>\n</html>"

    # Write output
    if output_dir:
        out_path = Path(output_dir) / "dataset_report.html"
    else:
        import tempfile
        out_path = Path(tempfile.mkdtemp()) / "dataset_report.html"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    logger.info("road_analytics: HTML report written to {}", out_path)
    return str(out_path.resolve())


def _bar_chart_base64(
    labels: list[str],
    values: list[float],
    title: str,
    xlabel: str,
    ylabel: str,
) -> str:
    """Render a matplotlib bar chart and return base64-encoded PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.bar(labels, values, color="#e94560")
    ax.set_title(title, color="#e0e0e0")
    ax.set_xlabel(xlabel, color="#e0e0e0")
    ax.set_ylabel(ylabel, color="#e0e0e0")
    ax.tick_params(colors="#e0e0e0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#0f3460")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, facecolor="#1a1a2e")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _escape_html(html: str) -> str:
    """Escape HTML for embedding in srcdoc attribute."""
    return html.replace("&", "&amp;").replace('"', "&quot;")
