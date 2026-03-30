"""
labelox/desktop/theme.py — Central theme for LABELOX.

Same dark palette as Orvex/RoverDataKit. Annotation-specific additions.
"""
from __future__ import annotations

# ── Core palette ─────────────────────────────────────────────────────────────
BG      = "#1a1a2e"
PANEL   = "#16213e"
ACCENT  = "#0f3460"
HI      = "#e94560"
TEXT    = "#e0e0e0"
MUTED   = "#666680"

# ── Extended palette ─────────────────────────────────────────────────────────
BORDER  = "#1e2a4a"
HOVER   = "#1a3a6a"
CARD    = "#0d1b2e"
INPUT   = "#111827"
SUCCESS = "#27ae60"
WARNING = "#f39c12"

# ── Annotation-specific ─────────────────────────────────────────────────────
CANVAS_BG = "#0a0a1a"
TOOL_ACTIVE = HI

ANNOTATION_COLORS = [
    "#e94560", "#4ecca3", "#4a9eff", "#f5a623", "#9b59b6",
    "#1abc9c", "#e67e22", "#3498db", "#e74c3c", "#2ecc71",
    "#f1c40f", "#8e44ad", "#16a085", "#d35400", "#2980b9",
    "#c0392b", "#27ae60", "#f39c12", "#7f8c8d", "#1a5276",
]

# ─── Global QSS ─────────────────────────────────────────────────────────────
GLOBAL_QSS = f"""
QWidget {{
    background: {BG};
    color: {TEXT};
    font-family: "Segoe UI", "Inter", "Helvetica Neue", Arial, sans-serif;
    font-size: 12px;
    selection-background-color: {HOVER};
    selection-color: {TEXT};
}}

QMainWindow {{
    background: {BG};
}}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
QPushButton {{
    background: {ACCENT};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 6px 14px;
    font-weight: 500;
    min-height: 22px;
}}
QPushButton:hover {{
    background: {HOVER};
    border-color: {MUTED};
}}
QPushButton:pressed {{
    background: {PANEL};
}}
QPushButton:disabled {{
    background: {CARD};
    color: {MUTED};
    border-color: {CARD};
}}
QPushButton#PrimaryBtn {{
    background: {ACCENT};
    border-color: #1a4a80;
}}
QPushButton#PrimaryBtn:hover {{
    background: {HOVER};
}}
QPushButton#DangerBtn {{
    background: #6b1a2a;
    border-color: {HI};
}}
QPushButton#DangerBtn:hover {{
    background: {HI};
    color: white;
}}
QPushButton#SuccessBtn {{
    background: #1a4a30;
    border-color: {SUCCESS};
}}
QPushButton#SuccessBtn:hover {{
    background: {SUCCESS};
    color: white;
}}
QPushButton#ToolBtn {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 3px;
    padding: 4px 8px;
    min-width: 28px;
    min-height: 28px;
}}
QPushButton#ToolBtn:hover {{
    background: {HOVER};
    border-color: {BORDER};
}}
QPushButton#ToolBtn:checked {{
    background: {ACCENT};
    border-color: {HI};
}}

/* ── Inputs ──────────────────────────────────────────────────────────────── */
QLineEdit, QTextEdit, QPlainTextEdit {{
    background: {INPUT};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 4px 8px;
}}
QLineEdit:focus, QTextEdit:focus {{
    border-color: {ACCENT};
}}

QComboBox {{
    background: {INPUT};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 4px 8px;
    min-height: 22px;
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox QAbstractItemView {{
    background: {PANEL};
    color: {TEXT};
    border: 1px solid {BORDER};
    selection-background-color: {HOVER};
}}

QSpinBox, QDoubleSpinBox {{
    background: {INPUT};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 2px 6px;
}}

/* ── Sliders ─────────────────────────────────────────────────────────────── */
QSlider::groove:horizontal {{
    height: 4px;
    background: {BORDER};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT};
    border: 1px solid {BORDER};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{
    background: {HI};
}}

/* ── Scroll bars ─────────────────────────────────────────────────────────── */
QScrollBar:vertical {{
    background: {BG};
    width: 10px;
    border: none;
}}
QScrollBar::handle:vertical {{
    background: {BORDER};
    min-height: 30px;
    border-radius: 5px;
}}
QScrollBar::handle:vertical:hover {{
    background: {MUTED};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: {BG};
    height: 10px;
    border: none;
}}
QScrollBar::handle:horizontal {{
    background: {BORDER};
    min-width: 30px;
    border-radius: 5px;
}}

/* ── Tables / Lists ──────────────────────────────────────────────────────── */
QTableWidget, QTableView {{
    background: {CARD};
    alternate-background-color: {BG};
    gridline-color: {BORDER};
    border: 1px solid {BORDER};
    border-radius: 3px;
}}
QHeaderView::section {{
    background: {PANEL};
    color: {TEXT};
    border: 1px solid {BORDER};
    padding: 4px 8px;
    font-weight: 600;
}}
QListWidget {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 3px;
}}
QListWidget::item {{
    padding: 4px 8px;
    border-bottom: 1px solid {BORDER};
}}
QListWidget::item:selected {{
    background: {HOVER};
}}
QListWidget::item:hover {{
    background: {ACCENT};
}}

/* ── Group boxes ─────────────────────────────────────────────────────────── */
QGroupBox {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {MUTED};
}}

/* ── Tabs ────────────────────────────────────────────────────────────────── */
QTabWidget::pane {{
    background: {BG};
    border: 1px solid {BORDER};
    border-top: none;
}}
QTabBar::tab {{
    background: {PANEL};
    color: {MUTED};
    border: 1px solid {BORDER};
    border-bottom: none;
    padding: 6px 16px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}}
QTabBar::tab:selected {{
    background: {BG};
    color: {TEXT};
}}
QTabBar::tab:hover {{
    color: {TEXT};
}}

/* ── Splitters ───────────────────────────────────────────────────────────── */
QSplitter::handle {{
    background: {BORDER};
}}
QSplitter::handle:horizontal {{
    width: 2px;
}}
QSplitter::handle:vertical {{
    height: 2px;
}}

/* ── Progress bar ────────────────────────────────────────────────────────── */
QProgressBar {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 3px;
    text-align: center;
    color: {TEXT};
    min-height: 16px;
}}
QProgressBar::chunk {{
    background: {ACCENT};
    border-radius: 2px;
}}

/* ── Menu bar ────────────────────────────────────────────────────────────── */
QMenuBar {{
    background: {PANEL};
    color: {TEXT};
    border-bottom: 1px solid {BORDER};
}}
QMenuBar::item:selected {{
    background: {HOVER};
}}
QMenu {{
    background: {PANEL};
    color: {TEXT};
    border: 1px solid {BORDER};
}}
QMenu::item:selected {{
    background: {HOVER};
}}
QMenu::separator {{
    height: 1px;
    background: {BORDER};
    margin: 4px 8px;
}}

/* ── Status bar ──────────────────────────────────────────────────────────── */
QStatusBar {{
    background: {PANEL};
    color: {MUTED};
    border-top: 1px solid {BORDER};
}}

/* ── Tooltips ────────────────────────────────────────────────────────────── */
QToolTip {{
    background: {PANEL};
    color: {TEXT};
    border: 1px solid {BORDER};
    padding: 4px 8px;
}}

/* ── Scroll area ─────────────────────────────────────────────────────────── */
QScrollArea {{
    border: none;
    background: transparent;
}}
"""


# ── Helper Functions ─────────────────────────────────────────────────────────

def card_style() -> str:
    return f"background:{CARD}; border:1px solid {BORDER}; border-radius:4px; padding:8px;"


def section_header_style() -> str:
    return f"color:{MUTED}; font-size:10px; font-weight:700; letter-spacing:1px; padding:4px 8px;"


def badge_style(color: str = HI) -> str:
    return f"background:{color}; color:white; border-radius:8px; padding:2px 8px; font-size:10px; font-weight:600;"
