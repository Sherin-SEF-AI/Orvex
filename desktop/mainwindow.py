"""
desktop/mainwindow.py — Main application window.

Layout:
  ┌────────────────────────────────────────────────────────────────────┐
  │  Brand header (48px) + MenuBar                                     │
  ├──────────────┬─────────────────────────────────────────────────────┤
  │              │                                                      │
  │  Sidebar     │  QStackedWidget (27 feature pages)                  │
  │  200px       │                                                      │
  │              ├─────────────────────────────────────────────────────┤
  │  Sectioned:  │  Log panel (collapsible, bottom)                    │
  │  DATA PIPE   │                                                      │
  │  INTELLIG.   │                                                      │
  │  DEPLOY      │                                                      │
  └──────────────┴─────────────────────────────────────────────────────┘
"""
from __future__ import annotations

from datetime import datetime, timezone

import math

from PyQt6.QtCore import Qt, QSize, QPointF, QRectF, pyqtSignal
from PyQt6.QtGui import QAction, QBrush, QColor, QFont, QIcon, QPainter, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.session_manager import SessionManager
from desktop.theme import (
    BG, PANEL, ACCENT, HI, TEXT, MUTED,
    SUCCESS, WARNING, BORDER, HOVER, CARD, INPUT,
    GLOBAL_QSS, section_header_style,
)
from desktop.widgets.session_panel import SessionPanel
from desktop.widgets.audit_widget import AuditWidget
from desktop.widgets.extraction_widget import ExtractionWidget
from desktop.widgets.sensorlogger_widget import SensorLoggerWidget
from desktop.widgets.calibration_widget import CalibrationWidget
from desktop.widgets.telemetry_plot import TelemetryPlot
from desktop.widgets.frame_browser import FrameBrowser
from desktop.widgets.dataset_widget import DatasetWidget
# Phase 2 widgets
from desktop.widgets.autolabel_widget import AutoLabelWidget
from desktop.widgets.active_learning_widget import ActiveLearningWidget
from desktop.widgets.analytics_widget import AnalyticsWidget
from desktop.widgets.augmentation_widget import AugmentationWidget
from desktop.widgets.training_widget import TrainingWidget
from desktop.widgets.depth_widget import DepthWidget
from desktop.widgets.slam_widget import SlamWidget
from desktop.widgets.reconstruction_widget import ReconstructionWidget
# Phase 3 widgets (inference / review / continuous learning)
from desktop.widgets.inference_widget import InferenceWidget
from desktop.widgets.annotation_review_widget import AnnotationReviewWidget
from desktop.widgets.continuous_learning_widget import ContinuousLearningWidget
# Phase 3 widgets (perception + infra)
from desktop.widgets.segmentation_widget import SegmentationWidget
from desktop.widgets.occupancy_widget import OccupancyWidget
from desktop.widgets.lane_widget import LaneWidget
from desktop.widgets.tracking_widget import TrackingWidget
from desktop.widgets.versioning_widget import VersioningWidget
from desktop.widgets.experiment_widget import ExperimentWidget
from desktop.widgets.edge_export_widget import EdgeExportWidget
from desktop.widgets.api_settings_widget import ApiSettingsWidget
from desktop.widgets.insta360_widget import Insta360Widget

# Alias for legacy references inside this file
HIGHLIGHT = HI

_VERSION = "3.0.0"

# Sectioned sidebar definition: (section_title, [(label, icon_key), ...])
SIDEBAR_SECTIONS = [
    ("DATA PIPELINE", [
        ("Sessions",       "folder"),
        ("Audit",          "magnifier"),
        ("Extract",        "gear"),
        ("Sensor Logger",  "phone"),
        ("Calibrate",      "ruler"),
        ("Telemetry",      "chart_line"),
        ("Frames",         "grid_frames"),
        ("Dataset",        "disk"),
    ]),
    ("INTELLIGENCE", [
        ("Auto-Label",      "robot"),
        ("Active Learning", "target"),
        ("Analytics",       "bar_chart"),
        ("Augment",         "shuffle"),
        ("Train",           "dumbbell"),
        ("Depth",           "layers"),
        ("3D Reconstruct",  "cube"),
        ("SLAM Validate",   "satellite"),
    ]),
    ("DEPLOYMENT", [
        ("Inference",      "brain"),
        ("Review",         "pencil"),
        ("Auto-Retrain",   "loop_arrows"),
    ]),
    ("PERCEPTION", [
        ("Segment",        "mask"),
        ("Occupancy",      "grid_bev"),
        ("Lanes",          "road"),
        ("Track",          "crosshair"),
    ]),
    ("INFRA", [
        ("Versions",       "box_tag"),
        ("Experiments",    "flask"),
        ("Edge Export",    "chip"),
        ("API",            "plug"),
    ]),
    ("360° CAMERA", [
        ("Insta360",       "circle_360"),
    ]),
]

# Color per sidebar section
_SECTION_COLORS: dict[str, str] = {
    "DATA PIPELINE": TEXT,
    "INTELLIGENCE":  HI,
    "DEPLOYMENT":    WARNING,
    "PERCEPTION":    SUCCESS,
    "INFRA":         MUTED,
    "360° CAMERA":   "#4a9eff",
}


def _make_icon(shape_key: str, color: str, size: int = 24) -> QIcon:
    """Draw a unique vector icon for each sidebar item using QPainter."""
    try:
        from PyQt6.QtWidgets import QApplication
        dpr = QApplication.primaryScreen().devicePixelRatio() if QApplication.primaryScreen() else 1.0
    except Exception:
        dpr = 1.0
    px_size = int(size * dpr)
    pm = QPixmap(px_size, px_size)
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.GlobalColor.transparent)

    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    pen = QPen(QColor(color), 1.6)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)

    s = float(size)
    m = s * 0.15  # margin

    if shape_key == "folder":
        # folder body + tab
        p.drawRoundedRect(QRectF(m, m + 4, s - 2*m, s - 2*m - 4), 2, 2)
        p.drawLine(QPointF(m, m + 4), QPointF(m + 6, m + 4))
        p.drawLine(QPointF(m + 6, m + 4), QPointF(m + 8, m))
        p.drawLine(QPointF(m + 8, m), QPointF(m + 14, m))
        p.drawLine(QPointF(m + 14, m), QPointF(m + 14, m + 4))

    elif shape_key == "magnifier":
        r = (s - 2*m) * 0.35
        cx, cy = s * 0.42, s * 0.42
        p.drawEllipse(QPointF(cx, cy), r, r)
        p.drawLine(QPointF(cx + r*0.7, cy + r*0.7), QPointF(s - m, s - m))

    elif shape_key == "gear":
        cx, cy = s/2, s/2
        r_out, r_in = (s - 2*m) * 0.45, (s - 2*m) * 0.25
        p.drawEllipse(QPointF(cx, cy), r_in, r_in)
        for i in range(6):
            a = math.radians(i * 60)
            p.drawLine(QPointF(cx + r_in*math.cos(a), cy + r_in*math.sin(a)),
                       QPointF(cx + r_out*math.cos(a), cy + r_out*math.sin(a)))

    elif shape_key == "ruler":
        p.save()
        p.translate(s/2, s/2)
        p.rotate(45)
        w, h = s * 0.22, s * 0.6
        p.drawRect(QRectF(-w/2, -h/2, w, h))
        for i in range(3):
            y = -h/2 + h * (i+1) / 4
            p.drawLine(QPointF(-w/2, y), QPointF(-w/2 + w*0.4, y))
        p.restore()

    elif shape_key == "chart_line":
        pts = [QPointF(m, s - m), QPointF(s*0.3, s*0.55),
               QPointF(s*0.5, s*0.65), QPointF(s*0.7, s*0.3), QPointF(s - m, m + 2)]
        for i in range(len(pts) - 1):
            p.drawLine(pts[i], pts[i+1])

    elif shape_key == "grid_frames":
        g = 1.5
        w = (s - 2*m - g) / 2
        for r in range(2):
            for c in range(2):
                x = m + c * (w + g)
                y = m + r * (w + g)
                p.drawRect(QRectF(x, y, w, w))

    elif shape_key == "disk":
        # cylinder: top ellipse + sides + bottom ellipse
        x0, y0, w_d, h_e = m, m + 2, s - 2*m, 4.0
        h_body = s - 2*m - 4
        p.drawEllipse(QRectF(x0, y0, w_d, h_e))
        p.drawLine(QPointF(x0, y0 + h_e/2), QPointF(x0, y0 + h_body))
        p.drawLine(QPointF(x0 + w_d, y0 + h_e/2), QPointF(x0 + w_d, y0 + h_body))
        p.drawArc(QRectF(x0, y0 + h_body - h_e/2, w_d, h_e), 180*16, 180*16)

    elif shape_key == "robot":
        # head
        p.drawRoundedRect(QRectF(m + 2, m + 5, s - 2*m - 4, s - 2*m - 5), 3, 3)
        # eyes
        p.setBrush(QColor(color))
        p.drawEllipse(QPointF(s * 0.38, s * 0.52), 1.5, 1.5)
        p.drawEllipse(QPointF(s * 0.62, s * 0.52), 1.5, 1.5)
        p.setBrush(Qt.BrushStyle.NoBrush)
        # antenna
        p.drawLine(QPointF(s/2, m + 5), QPointF(s/2, m))
        p.setBrush(QColor(color))
        p.drawEllipse(QPointF(s/2, m), 1.5, 1.5)
        p.setBrush(Qt.BrushStyle.NoBrush)

    elif shape_key == "target":
        cx, cy = s/2, s/2
        for r_frac in (0.42, 0.28, 0.14):
            r = (s - 2*m) * r_frac
            p.drawEllipse(QPointF(cx, cy), r, r)
        p.setBrush(QColor(color))
        p.drawEllipse(QPointF(cx, cy), 1.5, 1.5)
        p.setBrush(Qt.BrushStyle.NoBrush)

    elif shape_key == "bar_chart":
        bw = (s - 2*m) / 5
        heights = [0.4, 0.65, 1.0]
        for i, h_frac in enumerate(heights):
            x = m + (i * 2) * bw
            h = (s - 2*m) * h_frac
            p.drawRect(QRectF(x, s - m - h, bw, h))

    elif shape_key == "shuffle":
        y1, y2 = s * 0.35, s * 0.65
        p.drawLine(QPointF(m, y1), QPointF(s - m, y2))
        p.drawLine(QPointF(m, y2), QPointF(s - m, y1))
        # arrowheads
        for ey in (y1, y2):
            p.drawLine(QPointF(s - m, ey), QPointF(s - m - 3, ey - 2.5))
            p.drawLine(QPointF(s - m, ey), QPointF(s - m - 3, ey + 2.5))

    elif shape_key == "dumbbell":
        cy = s / 2
        r = 3.5
        p.drawEllipse(QPointF(m + r, cy), r, r)
        p.drawEllipse(QPointF(s - m - r, cy), r, r)
        p.drawLine(QPointF(m + 2*r, cy), QPointF(s - m - 2*r, cy))

    elif shape_key == "layers":
        cx = s / 2
        for i, y in enumerate([s * 0.3, s * 0.5, s * 0.7]):
            rx, ry = (s - 2*m) * 0.45, 3.0
            p.drawEllipse(QPointF(cx, y), rx, ry)

    elif shape_key == "cube":
        # isometric wireframe
        cx, cy = s/2, s/2
        d = (s - 2*m) * 0.38
        top = QPointF(cx, cy - d)
        bot = QPointF(cx, cy + d)
        left = QPointF(cx - d, cy)
        right = QPointF(cx + d, cy)
        mid = QPointF(cx, cy)
        p.drawLine(top, right)
        p.drawLine(top, left)
        p.drawLine(right, bot)
        p.drawLine(left, bot)
        # depth lines
        back = QPointF(cx, cy - d * 0.3)
        p.drawLine(left, mid)
        p.drawLine(right, mid)
        p.drawLine(mid, bot)

    elif shape_key == "satellite":
        cx, cy = s/2, s/2
        r = (s - 2*m) * 0.2
        p.drawEllipse(QPointF(cx, cy), r, r)
        # solar panels
        pw = (s - 2*m) * 0.25
        ph = (s - 2*m) * 0.15
        p.drawRect(QRectF(m, cy - ph/2, pw, ph))
        p.drawRect(QRectF(s - m - pw, cy - ph/2, pw, ph))
        p.drawLine(QPointF(m + pw, cy), QPointF(cx - r, cy))
        p.drawLine(QPointF(s - m - pw, cy), QPointF(cx + r, cy))

    elif shape_key == "brain":
        cx, cy = s/2, s/2
        r = (s - 2*m) * 0.4
        p.drawEllipse(QPointF(cx, cy), r, r)
        # wavy internal line
        p.drawLine(QPointF(cx - r*0.5, cy), QPointF(cx - r*0.15, cy - r*0.4))
        p.drawLine(QPointF(cx - r*0.15, cy - r*0.4), QPointF(cx + r*0.15, cy + r*0.4))
        p.drawLine(QPointF(cx + r*0.15, cy + r*0.4), QPointF(cx + r*0.5, cy))

    elif shape_key == "pencil":
        p.save()
        p.translate(s/2, s/2)
        p.rotate(45)
        w, h = 4.0, s * 0.55
        p.drawRect(QRectF(-w/2, -h/2, w, h))
        # tip triangle
        tip = QPolygonF([QPointF(-w/2, h/2), QPointF(w/2, h/2), QPointF(0, h/2 + 3)])
        p.drawPolygon(tip)
        p.restore()

    elif shape_key == "loop_arrows":
        cx, cy = s/2, s/2
        r = (s - 2*m) * 0.35
        p.drawArc(QRectF(cx - r, cy - r, 2*r, 2*r), 30*16, 300*16)
        # arrowhead at end of arc
        a_end = math.radians(30)
        ex, ey = cx + r*math.cos(a_end), cy - r*math.sin(a_end)
        p.drawLine(QPointF(ex, ey), QPointF(ex + 3, ey - 3))
        p.drawLine(QPointF(ex, ey), QPointF(ex - 1, ey - 4))

    elif shape_key == "mask":
        cx, cy = s/2, s/2
        rx, ry = (s - 2*m) * 0.44, (s - 2*m) * 0.32
        p.drawEllipse(QPointF(cx, cy), rx, ry)
        # eye holes
        p.setBrush(QColor(PANEL))
        p.drawEllipse(QPointF(cx - rx*0.35, cy - 1), rx*0.2, ry*0.25)
        p.drawEllipse(QPointF(cx + rx*0.35, cy - 1), rx*0.2, ry*0.25)
        p.setBrush(Qt.BrushStyle.NoBrush)

    elif shape_key == "grid_bev":
        step = (s - 2*m) / 2
        p.setBrush(QColor(color))
        for r in range(3):
            for c in range(3):
                p.drawEllipse(QPointF(m + c*step, m + r*step), 1.5, 1.5)
        p.setBrush(Qt.BrushStyle.NoBrush)

    elif shape_key == "road":
        # two converging lines
        p.drawLine(QPointF(m, s - m), QPointF(s*0.38, m))
        p.drawLine(QPointF(s - m, s - m), QPointF(s*0.62, m))
        # dashed center
        pen_d = QPen(QColor(color), 1.2, Qt.PenStyle.DashLine)
        p.setPen(pen_d)
        p.drawLine(QPointF(s/2, s - m - 2), QPointF(s/2, m + 2))

    elif shape_key == "crosshair":
        cx, cy = s/2, s/2
        r = (s - 2*m) * 0.35
        p.drawEllipse(QPointF(cx, cy), r, r)
        p.drawLine(QPointF(cx, cy - r - 2), QPointF(cx, cy + r + 2))
        p.drawLine(QPointF(cx - r - 2, cy), QPointF(cx + r + 2, cy))

    elif shape_key == "box_tag":
        p.drawRect(QRectF(m, m + 3, s - 2*m, s - 2*m - 3))
        # tag triangle on top-right
        tr = s - m
        p.drawLine(QPointF(tr - 6, m + 3), QPointF(tr, m + 9))

    elif shape_key == "flask":
        neck_w = s * 0.15
        cx = s / 2
        # neck
        p.drawLine(QPointF(cx - neck_w, m), QPointF(cx - neck_w, s * 0.4))
        p.drawLine(QPointF(cx + neck_w, m), QPointF(cx + neck_w, s * 0.4))
        # body
        p.drawLine(QPointF(cx - neck_w, s * 0.4), QPointF(m, s - m))
        p.drawLine(QPointF(cx + neck_w, s * 0.4), QPointF(s - m, s - m))
        p.drawLine(QPointF(m, s - m), QPointF(s - m, s - m))

    elif shape_key == "chip":
        r = QRectF(m + 3, m + 3, s - 2*m - 6, s - 2*m - 6)
        p.drawRect(r)
        # pins on each side
        pin_len = 3
        for i in range(3):
            t = r.top() + r.height() * (i + 1) / 4
            p.drawLine(QPointF(r.left(), t), QPointF(r.left() - pin_len, t))
            p.drawLine(QPointF(r.right(), t), QPointF(r.right() + pin_len, t))
        for i in range(3):
            t = r.left() + r.width() * (i + 1) / 4
            p.drawLine(QPointF(t, r.top()), QPointF(t, r.top() - pin_len))
            p.drawLine(QPointF(t, r.bottom()), QPointF(t, r.bottom() + pin_len))

    elif shape_key == "plug":
        cx, cy = s/2, s/2
        r = (s - 2*m) * 0.28
        p.drawEllipse(QPointF(cx, cy), r, r)
        # two prongs going down
        p.drawLine(QPointF(cx - 3, cy + r), QPointF(cx - 3, s - m))
        p.drawLine(QPointF(cx + 3, cy + r), QPointF(cx + 3, s - m))

    elif shape_key == "circle_360":
        cx, cy = s/2, s/2
        r = (s - 2*m) * 0.38
        p.drawEllipse(QPointF(cx, cy), r, r)
        # arc arrow
        pen2 = QPen(QColor(color), 2.0)
        pen2.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen2)
        p.drawArc(QRectF(cx - r*0.6, cy - r*0.6, r*1.2, r*1.2), 60*16, 270*16)
        # arrowhead
        p.setPen(pen)
        a_end = math.radians(60)
        ex, ey = cx + r*0.6*math.cos(a_end), cy - r*0.6*math.sin(a_end)
        p.drawLine(QPointF(ex, ey), QPointF(ex + 3, ey + 1))
        p.drawLine(QPointF(ex, ey), QPointF(ex - 1, ey + 3))

    elif shape_key == "phone":
        # Phone body (rounded rect) + screen + home dot
        body = QRectF(m + 4, m, s - 2*m - 8, s - 2*m)
        p.drawRoundedRect(body, 3, 3)
        screen = QRectF(m + 6, m + 3, s - 2*m - 12, s - 2*m - 8)
        p.drawRect(screen)
        cx = s / 2
        p.setBrush(QColor(color))
        p.drawEllipse(QPointF(cx, s - m - 2), 1.5, 1.5)
        p.setBrush(Qt.BrushStyle.NoBrush)

    p.end()
    return QIcon(pm)


def _make_orvex_logo(size: int = 64) -> QIcon:
    """Draw the Orvex logo: two interlocking orbital arcs with a bright center node."""
    pm = QPixmap(size, size)
    pm.fill(QColor(0, 0, 0, 0))
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    cx, cy = size / 2, size / 2
    m = size * 0.08  # margin

    # Outer ring — subtle border
    ring_r = (size - 2 * m) / 2
    ring_pen = QPen(QColor(HI), size * 0.04)
    ring_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(ring_pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(QPointF(cx, cy), ring_r, ring_r)

    # Orbital arc 1 — tilted ellipse (upper-left to lower-right)
    arc_pen = QPen(QColor("#4a9eff"), size * 0.05)
    arc_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(arc_pen)
    p.save()
    p.translate(cx, cy)
    p.rotate(-30)
    arc_rx = ring_r * 0.65
    arc_ry = ring_r * 0.35
    p.drawArc(QRectF(-arc_rx, -arc_ry, arc_rx * 2, arc_ry * 2), 30 * 16, 180 * 16)
    p.restore()

    # Orbital arc 2 — opposite tilt (upper-right to lower-left)
    arc_pen2 = QPen(QColor(SUCCESS), size * 0.045)
    arc_pen2.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(arc_pen2)
    p.save()
    p.translate(cx, cy)
    p.rotate(30)
    p.drawArc(QRectF(-arc_rx, -arc_ry, arc_rx * 2, arc_ry * 2), 200 * 16, 180 * 16)
    p.restore()

    # Center node — bright gradient dot
    node_r = size * 0.1
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(HI))
    p.drawEllipse(QPointF(cx, cy), node_r, node_r)
    # Inner highlight
    p.setBrush(QColor("#ff6b80"))
    p.drawEllipse(QPointF(cx - node_r * 0.2, cy - node_r * 0.2),
                  node_r * 0.45, node_r * 0.45)

    # Small orbiting dots
    import math as _m
    dot_r = size * 0.035
    for angle_deg, color in [(45, "#4a9eff"), (225, SUCCESS)]:
        rad = _m.radians(angle_deg)
        dx = cx + ring_r * 0.72 * _m.cos(rad)
        dy = cy + ring_r * 0.72 * _m.sin(rad)
        p.setBrush(QColor(color))
        p.drawEllipse(QPointF(dx, dy), dot_r, dot_r)

    p.end()
    return QIcon(pm)


# Sidebar style: minimal overrides on top of GLOBAL_QSS
_SIDEBAR_STYLE = f"""
QListWidget {{
    background: {PANEL};
    border: none;
    border-right: 1px solid {BORDER};
    outline: none;
}}
QListWidget::item {{
    padding: 10px 16px;
    border: none;
    border-bottom: none;
}}
QListWidget::item:selected {{
    background: rgba(15, 52, 96, 0.85);
    color: white;
    border-left: 3px solid {HIGHLIGHT};
    padding-left: 13px;
    font-weight: bold;
}}
QListWidget::item:hover:!selected {{
    background: {HOVER};
    border-left: 3px solid {BORDER};
    padding-left: 13px;
}}
"""


class MainWindow(QMainWindow):
    session_changed = pyqtSignal(str)   # session_id or "" for deselect

    def __init__(self, session_manager: SessionManager, parent=None) -> None:
        super().__init__(parent)
        self._sm = session_manager
        self._current_session_id: str = ""
        # Sidebar row → stack index (skips section header rows)
        self._sidebar_index_map: dict[int, int] = {}
        self.setWindowTitle("Orvex")
        self.setWindowIcon(_make_orvex_logo(64))
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window
        )
        self.setMinimumSize(800, 500)
        self.resize(1400, 900)
        self._drag_pos = None
        self._build_ui()
        self._build_menu()
        self._apply_stylesheet()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Custom title bar (replaces OS decoration) ─────────────────
        self._title_bar = QWidget()
        self._title_bar.setFixedHeight(38)
        self._title_bar.setStyleSheet(
            f"background: {PANEL}; border-bottom: 1px solid {BORDER};"
        )
        h_layout = QHBoxLayout(self._title_bar)
        h_layout.setContentsMargins(10, 0, 6, 0)
        h_layout.setSpacing(8)

        logo_label = QLabel()
        logo_icon = _make_orvex_logo(32)
        logo_label.setPixmap(logo_icon.pixmap(QSize(22, 22)))
        logo_label.setStyleSheet("background: transparent;")
        h_layout.addWidget(logo_label)

        brand = QLabel("Orvex")
        brand.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        brand.setStyleSheet(f"color: {TEXT}; background: transparent;")
        h_layout.addWidget(brand)

        h_layout.addStretch()

        # Window control buttons
        btn_style = (
            "QPushButton {{ background: transparent; color: {clr};"
            " border: none; font-size: 14px; padding: 4px 10px;"
            " border-radius: 3px; }}"
            "QPushButton:hover {{ background: {hover}; }}"
        )
        min_btn = QPushButton("─")
        min_btn.setFixedSize(36, 28)
        min_btn.setStyleSheet(btn_style.format(clr=MUTED, hover=HOVER))
        min_btn.clicked.connect(self.showMinimized)

        max_btn = QPushButton("☐")
        max_btn.setFixedSize(36, 28)
        max_btn.setStyleSheet(btn_style.format(clr=MUTED, hover=HOVER))
        max_btn.clicked.connect(self._toggle_maximize)

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(36, 28)
        close_btn.setStyleSheet(btn_style.format(clr=MUTED, hover=HI))
        close_btn.clicked.connect(self.close)

        h_layout.addWidget(min_btn)
        h_layout.addWidget(max_btn)
        h_layout.addWidget(close_btn)

        self._title_bar.installEventFilter(self)
        outer.addWidget(self._title_bar)

        # ── Body (sidebar + stacked content) ──────────────────────────
        from PyQt6.QtWidgets import QSplitter
        body_splitter = QSplitter(Qt.Orientation.Horizontal)
        body_splitter.setHandleWidth(1)
        body_splitter.setStyleSheet(f"QSplitter::handle {{ background: {BORDER}; }}")

        # Sidebar — resizable, sectioned
        self._sidebar = QListWidget()
        self._sidebar.setMinimumWidth(140)
        self._sidebar.setMaximumWidth(320)
        self._sidebar.setStyleSheet(_SIDEBAR_STYLE)
        self._sidebar.setIconSize(QSize(24, 24))

        stack_index = 0
        for section_title, items in SIDEBAR_SECTIONS:
            # Section header row (non-selectable label)
            header_item = QListWidgetItem(f"  {section_title}")
            header_item.setFlags(Qt.ItemFlag.NoItemFlags)
            header_item.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            header_item.setForeground(QBrush(QColor(MUTED)))
            header_item.setSizeHint(QSize(0, 28))
            self._sidebar.addItem(header_item)

            # Nav items
            section_color = _SECTION_COLORS.get(section_title, TEXT)
            for label, icon_key in items:
                icon = _make_icon(icon_key, section_color)
                item = QListWidgetItem(f"  {label}")
                item.setIcon(icon)
                item.setSizeHint(QSize(0, 40))
                item.setFont(QFont("Segoe UI", 11))
                self._sidebar.addItem(item)
                actual_row = self._sidebar.count() - 1
                self._sidebar_index_map[actual_row] = stack_index
                stack_index += 1

        self._sidebar.currentRowChanged.connect(self._on_sidebar_changed)
        body_splitter.addWidget(self._sidebar)

        # Stacked widget — same order as flattened SIDEBAR_SECTIONS items
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self._stack = QStackedWidget()

        # Phase 1 widgets (stack indices 0-6)
        self._session_panel = SessionPanel(self._sm)
        self._session_panel.session_selected.connect(self._on_session_selected)
        self._stack.addWidget(self._session_panel)          # 0 Sessions

        self._audit_widget = AuditWidget(self._sm)
        self._stack.addWidget(self._audit_widget)            # 1 Audit

        self._extraction_widget = ExtractionWidget(self._sm)
        self._stack.addWidget(self._extraction_widget)       # 2 Extract

        self._sensorlogger_widget = SensorLoggerWidget(self._sm)
        self._stack.addWidget(self._sensorlogger_widget)     # 3 Sensor Logger

        self._calibration_widget = CalibrationWidget(self._sm)
        self._stack.addWidget(self._calibration_widget)      # 4 Calibrate

        self._telemetry_widget = TelemetryPlot(self._sm)
        self._stack.addWidget(self._telemetry_widget)        # 5 Telemetry

        self._frame_browser = FrameBrowser(self._sm)
        self._stack.addWidget(self._frame_browser)           # 6 Frames

        self._dataset_widget = DatasetWidget(self._sm)
        self._stack.addWidget(self._dataset_widget)          # 7 Dataset

        # Phase 2 widgets (stack indices 8-15)
        self._autolabel_widget = AutoLabelWidget(self._sm)
        self._stack.addWidget(self._autolabel_widget)        # 8 Auto-Label

        self._active_learning_widget = ActiveLearningWidget(self._sm)
        self._stack.addWidget(self._active_learning_widget)  # 9 Active Learning

        self._analytics_widget = AnalyticsWidget(self._sm)
        self._stack.addWidget(self._analytics_widget)        # 10 Analytics

        self._augmentation_widget = AugmentationWidget(self._sm)
        self._stack.addWidget(self._augmentation_widget)     # 11 Augment

        self._training_widget = TrainingWidget(self._sm)
        self._stack.addWidget(self._training_widget)         # 12 Train

        self._depth_widget = DepthWidget(self._sm)
        self._stack.addWidget(self._depth_widget)            # 13 Depth

        self._reconstruction_widget = ReconstructionWidget(self._sm)
        self._stack.addWidget(self._reconstruction_widget)   # 14 3D Reconstruct

        self._slam_widget = SlamWidget(self._sm)
        self._stack.addWidget(self._slam_widget)             # 15 SLAM Validate

        # Phase 3 widgets (stack indices 16-18)
        self._inference_widget = InferenceWidget(self._sm)
        self._stack.addWidget(self._inference_widget)        # 16 Inference

        self._review_widget = AnnotationReviewWidget(self._sm)
        self._stack.addWidget(self._review_widget)           # 17 Review

        self._cl_widget = ContinuousLearningWidget(self._sm)
        self._stack.addWidget(self._cl_widget)               # 18 Auto-Retrain

        # Phase 3 perception + infra widgets (stack indices 19-26)
        self._segmentation_widget = SegmentationWidget(self._sm)
        self._stack.addWidget(self._segmentation_widget)     # 19 Segment

        self._occupancy_widget = OccupancyWidget(self._sm)
        self._stack.addWidget(self._occupancy_widget)        # 20 Occupancy

        self._lane_widget = LaneWidget(self._sm)
        self._stack.addWidget(self._lane_widget)             # 21 Lanes

        self._tracking_widget = TrackingWidget(self._sm)
        self._stack.addWidget(self._tracking_widget)         # 22 Track

        self._versioning_widget = VersioningWidget()
        self._stack.addWidget(self._versioning_widget)       # 23 Versions

        self._experiment_widget = ExperimentWidget()
        self._stack.addWidget(self._experiment_widget)       # 24 Experiments

        self._edge_export_widget = EdgeExportWidget(self._sm)
        self._stack.addWidget(self._edge_export_widget)      # 25 Edge Export

        self._api_widget = ApiSettingsWidget(self._sm)
        self._stack.addWidget(self._api_widget)              # 26 API

        # 360° CAMERA widgets (stack index 27)
        self._insta360_widget = Insta360Widget(self._sm)
        self._stack.addWidget(self._insta360_widget)         # 27 Insta360

        right_layout.addWidget(self._stack, stretch=1)

        # ── Log panel (collapsible) ────────────────────────────────────
        self._log_panel = QWidget()
        self._log_panel.setStyleSheet(f"background: {CARD}; border-top: 1px solid {BORDER};")
        log_layout = QVBoxLayout(self._log_panel)
        log_layout.setContentsMargins(4, 2, 4, 2)

        log_header = QHBoxLayout()
        log_label = QLabel("LOG")
        log_label.setStyleSheet(section_header_style())
        log_header.addWidget(log_label)
        log_header.addStretch()
        self._toggle_log_btn = QToolButton()
        self._toggle_log_btn.setText("▾")
        self._toggle_log_btn.setStyleSheet(
            f"color: {MUTED}; border: none; background: transparent; font-size: 13px;"
        )
        self._toggle_log_btn.clicked.connect(self._toggle_log)
        log_header.addWidget(self._toggle_log_btn)
        log_layout.addLayout(log_header)

        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMinimumHeight(60)
        self._log_text.setStyleSheet(
            f"background: {CARD}; color: {TEXT}; font-family: monospace; font-size: 11px;"
            f" border: none; border-radius: 0;"
        )
        log_layout.addWidget(self._log_text)
        right_layout.addWidget(self._log_panel)

        body_splitter.addWidget(right_panel)
        body_splitter.setStretchFactor(0, 0)   # sidebar: don't stretch
        body_splitter.setStretchFactor(1, 1)   # content: takes all space
        body_splitter.setSizes([200, 1200])
        outer.addWidget(body_splitter, stretch=1)

        # ── Status bar ─────────────────────────────────────────────────
        self._status_bar = QStatusBar()
        self._progress_bar = QProgressBar()
        self._progress_bar.setMaximumWidth(200)
        self._progress_bar.setVisible(False)
        self._status_bar.addPermanentWidget(self._progress_bar)

        self._session_name_label = QLabel("")
        self._session_name_label.setStyleSheet(
            f"color: {TEXT}; font-weight: bold; font-size: 11px; padding: 0 8px;"
        )
        self._status_bar.addPermanentWidget(self._session_name_label)

        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready.")

        # Default to Sessions page (sidebar row 1 — first real item after header)
        self._sidebar.setCurrentRow(1)

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        mb = self.menuBar()
        # GLOBAL_QSS already styles QMenuBar/QMenu — no inline override needed

        file_menu = mb.addMenu("File")
        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        session_menu = mb.addMenu("Sessions")
        new_session = QAction("New Session…", self)
        new_session.setShortcut("Ctrl+N")
        new_session.triggered.connect(self._new_session)
        session_menu.addAction(new_session)

        help_menu = mb.addMenu("Help")
        about = QAction("About Orvex", self)
        about.triggered.connect(self._show_about)
        help_menu.addAction(about)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_sidebar_changed(self, row: int) -> None:
        stack_idx = self._sidebar_index_map.get(row)
        if stack_idx is not None:
            self._stack.setCurrentIndex(stack_idx)

    def _on_session_selected(self, session_id: str) -> None:
        self._current_session_id = session_id
        self.session_changed.emit(session_id)

        # Propagate to all widgets
        widgets_with_session = [
            self._audit_widget,
            self._extraction_widget,
            self._calibration_widget,
            self._telemetry_widget,
            self._frame_browser,
            self._dataset_widget,
            self._autolabel_widget,
            self._active_learning_widget,
            self._analytics_widget,
            self._augmentation_widget,
            self._training_widget,
            self._depth_widget,
            self._slam_widget,
            self._reconstruction_widget,
            # Phase 3 deployment
            self._inference_widget,
            self._review_widget,
            self._cl_widget,
            # Phase 3 perception
            self._segmentation_widget,
            self._occupancy_widget,
            self._lane_widget,
            self._tracking_widget,
            # Previously missing
            self._session_panel,
            self._api_widget,
            self._insta360_widget,
        ]
        for w in widgets_with_session:
            if hasattr(w, "on_session_changed"):
                w.on_session_changed(session_id)

        if session_id:
            try:
                session = self._sm.get_session(session_id)
                self._status_bar.showMessage(
                    f"{session.environment}  •  {session.location}"
                )
                self._session_name_label.setText(session.name)
            except Exception:
                pass
        else:
            self._session_name_label.setText("")

        self._log(f"Session selected: {session_id}")

    def show_continuous_learning(self, session_id: str) -> None:
        """Called by AnnotationReviewWidget 'Trigger retraining' button."""
        self._cl_widget.show_for_session(session_id)
        # row 21 = Auto-Retrain (after DEPLOYMENT header at row 18)
        self._sidebar.setCurrentRow(21)
        self._stack.setCurrentIndex(18)

    def _toggle_log(self) -> None:
        visible = self._log_text.isVisible()
        self._log_text.setVisible(not visible)
        self._toggle_log_btn.setText("▾" if not visible else "▸")

    def _new_session(self) -> None:
        self._sidebar.setCurrentRow(1)   # Sessions row (row 1, after header)

    def _show_about(self) -> None:
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(
            self, "About Orvex",
            f"Orvex v{_VERSION}\n\n"
            "Production-grade data pipeline for autonomous rover datasets.\n\n"
            "Phase 1: Data Pipeline\n"
            " • Session management\n"
            " • GoPro / Insta360 / SensorLogger file audit\n"
            " • Telemetry extraction (IMU, GPS, frames)\n"
            " • Camera calibration workflow\n"
            " • EuRoC dataset builder\n\n"
            "Phase 2: Intelligence & Validation\n"
            " • YOLOv8 Auto-labeling\n"
            " • Depth-Anything-v2 depth estimation\n"
            " • ORBSLAM3 trajectory validation\n"
            " • COLMAP 3D reconstruction\n"
            " • Active learning frame selection\n"
            " • Scene analytics & GPS coverage\n"
            " • Data augmentation\n"
            " • YOLOv8 model fine-tuning\n\n"
            "Phase 3: Deployment & Continuous Learning\n"
            " • Model registry + live inference server\n"
            " • Human-in-the-loop annotation review\n"
            " • Auto-retrain on corrections\n",
        )

    # ------------------------------------------------------------------
    # Log helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        """Append an HTML-formatted, color-coded log entry."""
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        lower = msg.lower()
        if "error" in lower or "fail" in lower:
            color = HI
        elif "warn" in lower:
            color = WARNING
        elif any(k in lower for k in ("✓", "done", "success", "complete", "ready")):
            color = SUCCESS
        else:
            color = MUTED
        ts_html = f"<span style='color:{MUTED}'>[{ts}]</span>"
        msg_html = f"<span style='color:{color}'>{msg}</span>"
        self._log_text.append(f"{ts_html} {msg_html}")

    # ------------------------------------------------------------------
    # Stylesheet
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Custom title bar — drag & maximize
    # ------------------------------------------------------------------

    def _toggle_maximize(self) -> None:
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def eventFilter(self, obj, event) -> bool:
        from PyQt6.QtCore import QEvent
        from PyQt6.QtGui import QMouseEvent

        if obj is self._title_bar:
            if event.type() == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton:
                    self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                    return True
            elif event.type() == QEvent.Type.MouseMove:
                if self._drag_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
                    if self.isMaximized():
                        self.showNormal()
                        self._drag_pos.setX(int(self.width() * 0.5))
                    self.move(event.globalPosition().toPoint() - self._drag_pos)
                    return True
            elif event.type() == QEvent.Type.MouseButtonRelease:
                self._drag_pos = None
                return True
            elif event.type() == QEvent.Type.MouseButtonDblClick:
                self._toggle_maximize()
                return True
        return super().eventFilter(obj, event)

    def _apply_stylesheet(self) -> None:
        self.setStyleSheet(GLOBAL_QSS)
