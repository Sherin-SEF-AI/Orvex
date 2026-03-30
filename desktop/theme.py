"""
desktop/theme.py — Central theme module for RoverDataKit.

Single source of truth for all visual constants and QSS stylesheet.
Import from here in every widget; do not redeclare colors locally.

Usage:
    from desktop.theme import (
        BG, PANEL, ACCENT, HI, TEXT, MUTED,
        BORDER, HOVER, CARD, INPUT, SUCCESS, WARNING,
        GLOBAL_QSS,
        card_style, section_header_style, badge_style, apply_plot_theme,
    )
"""
from __future__ import annotations

# ── Core palette ─────────────────────────────────────────────────────────────
BG      = "#1a1a2e"   # main background
PANEL   = "#16213e"   # sidebar / panels
ACCENT  = "#0f3460"   # primary interactive
HI      = "#e94560"   # highlight / danger / active
TEXT    = "#e0e0e0"   # primary text
MUTED   = "#666680"   # secondary / disabled text

# ── Extended palette ──────────────────────────────────────────────────────────
BORDER  = "#1e2a4a"   # subtle borders
HOVER   = "#1a3a6a"   # hover state for dark controls
CARD    = "#0d1b2e"   # stat cards — slightly darker than BG
INPUT   = "#111827"   # input field backgrounds
SUCCESS = "#27ae60"
WARNING = "#f39c12"

# ── pyqtgraph pass-throughs ───────────────────────────────────────────────────
PYQTGRAPH_BG    = BG
PYQTGRAPH_PANEL = PANEL

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL QSS — applied once at QMainWindow level; cascades to all children
# ─────────────────────────────────────────────────────────────────────────────
GLOBAL_QSS = f"""

/* ── Base ─────────────────────────────────────────────────────────────────── */
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

/* ── Status bar ───────────────────────────────────────────────────────────── */
QStatusBar {{
    background: {PANEL};
    color: {MUTED};
    border-top: 1px solid {BORDER};
    font-size: 11px;
    padding: 0 8px;
}}
QStatusBar::item {{
    border: none;
}}

/* ── Menu bar + menus ─────────────────────────────────────────────────────── */
QMenuBar {{
    background: {PANEL};
    color: {TEXT};
    border-bottom: 1px solid {BORDER};
    padding: 2px 0;
}}
QMenuBar::item {{
    padding: 4px 10px;
    background: transparent;
}}
QMenuBar::item:selected {{
    background: {ACCENT};
}}
QMenuBar::item:pressed {{
    background: {HI};
}}
QMenu {{
    background: {PANEL};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 0;
}}
QMenu::item {{
    padding: 6px 24px 6px 12px;
}}
QMenu::item:selected {{
    background: {HOVER};
}}
QMenu::separator {{
    height: 1px;
    background: {BORDER};
    margin: 3px 8px;
}}

/* ── Buttons ──────────────────────────────────────────────────────────────── */
QPushButton {{
    background: {ACCENT};
    color: {TEXT};
    border: none;
    padding: 5px 14px;
    border-radius: 5px;
    font-size: 12px;
    min-height: 26px;
}}
QPushButton:hover {{
    background: {HOVER};
    border: 1px solid {ACCENT};
}}
QPushButton:pressed {{
    background: {ACCENT};
    border: 1px solid {HI};
}}
QPushButton:disabled {{
    background: #1e2040;
    color: #3a3a55;
    border: 1px solid #1e2040;
}}

/* Danger variant — btn.setObjectName("DangerBtn") */
QPushButton#DangerBtn {{
    background: {HI};
    color: #ffffff;
    font-weight: bold;
}}
QPushButton#DangerBtn:hover {{
    background: #c0304e;
    border: none;
}}
QPushButton#DangerBtn:pressed {{
    background: #a0203e;
}}
QPushButton#DangerBtn:disabled {{
    background: #3a1a22;
    color: #553344;
}}

/* Primary/success variant — btn.setObjectName("PrimaryBtn") */
QPushButton#PrimaryBtn {{
    background: {SUCCESS};
    color: #ffffff;
    font-weight: bold;
}}
QPushButton#PrimaryBtn:hover {{
    background: #22994e;
    border: none;
}}
QPushButton#PrimaryBtn:pressed {{
    background: #1a7a3e;
}}
QPushButton#PrimaryBtn:disabled {{
    background: #1a3a28;
    color: #335544;
}}

/* Warning variant — btn.setObjectName("WarningBtn") */
QPushButton#WarningBtn {{
    background: {WARNING};
    color: #1a1a2e;
    font-weight: bold;
}}
QPushButton#WarningBtn:hover {{
    background: #e08c10;
    border: none;
}}
QPushButton#WarningBtn:disabled {{
    background: #3a2a10;
    color: #554433;
}}

/* Tool buttons (Browse, Clear, small actions) */
QToolButton {{
    background: {ACCENT};
    color: {TEXT};
    border: none;
    border-radius: 4px;
    padding: 3px 8px;
    font-size: 11px;
}}
QToolButton:hover {{
    background: {HOVER};
    border: 1px solid {BORDER};
}}
QToolButton:pressed {{
    background: {ACCENT};
}}

/* ── Input fields ─────────────────────────────────────────────────────────── */
QLineEdit {{
    background: {INPUT};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 4px 8px;
    selection-background-color: {ACCENT};
    min-height: 24px;
}}
QLineEdit:focus {{
    border: 1px solid {ACCENT};
}}
QLineEdit:disabled {{
    color: {MUTED};
    border-color: #1e2040;
    background: #0d1020;
}}

QTextEdit {{
    background: {INPUT};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 4px;
    selection-background-color: {ACCENT};
}}
QTextEdit:focus {{
    border: 1px solid {ACCENT};
}}

QPlainTextEdit {{
    background: {INPUT};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 4px;
}}
QPlainTextEdit:focus {{
    border: 1px solid {ACCENT};
}}

/* ── ComboBox ─────────────────────────────────────────────────────────────── */
QComboBox {{
    background: {INPUT};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 4px 8px;
    min-height: 24px;
    min-width: 80px;
}}
QComboBox:hover {{
    border-color: {ACCENT};
}}
QComboBox:focus {{
    border-color: {ACCENT};
}}
QComboBox:on {{
    border-color: {ACCENT};
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
    subcontrol-origin: padding;
    subcontrol-position: right center;
}}
QComboBox::down-arrow {{
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {MUTED};
    margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    background: {PANEL};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    selection-background-color: {HOVER};
    outline: none;
}}

/* ── SpinBox ──────────────────────────────────────────────────────────────── */
QSpinBox, QDoubleSpinBox {{
    background: {INPUT};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 4px 6px;
    min-height: 24px;
}}
QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {ACCENT};
}}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    width: 16px;
    border: none;
    background: {BORDER};
    border-radius: 0;
}}
QSpinBox::up-button, QDoubleSpinBox::up-button {{
    border-top-right-radius: 5px;
}}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    border-bottom-right-radius: 5px;
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background: {HOVER};
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    width: 0; height: 0;
    border-left: 3px solid transparent;
    border-right: 3px solid transparent;
    border-bottom: 4px solid {MUTED};
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    width: 0; height: 0;
    border-left: 3px solid transparent;
    border-right: 3px solid transparent;
    border-top: 4px solid {MUTED};
}}

/* ── CheckBox ─────────────────────────────────────────────────────────────── */
QCheckBox {{
    color: {TEXT};
    spacing: 8px;
    background: transparent;
}}
QCheckBox::indicator {{
    width: 15px;
    height: 15px;
    border: 1px solid {BORDER};
    border-radius: 3px;
    background: {INPUT};
}}
QCheckBox::indicator:hover {{
    border-color: {ACCENT};
    background: {PANEL};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
}}
QCheckBox::indicator:checked:hover {{
    background: {HOVER};
    border-color: {HI};
}}
QCheckBox:disabled {{
    color: {MUTED};
}}
QCheckBox::indicator:disabled {{
    border-color: #1e2040;
    background: #0d1020;
}}

/* ── Slider ───────────────────────────────────────────────────────────────── */
QSlider::groove:horizontal {{
    height: 4px;
    background: {BORDER};
    border-radius: 2px;
    margin: 0;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 2px;
    height: 4px;
}}
QSlider::handle:horizontal {{
    background: {TEXT};
    border: 2px solid {ACCENT};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{
    background: {HI};
    border-color: {HI};
}}
QSlider::groove:vertical {{
    width: 4px;
    background: {BORDER};
    border-radius: 2px;
}}
QSlider::sub-page:vertical {{
    background: {ACCENT};
    border-radius: 2px;
}}
QSlider::handle:vertical {{
    background: {TEXT};
    border: 2px solid {ACCENT};
    width: 14px;
    height: 14px;
    margin: 0 -5px;
    border-radius: 7px;
}}

/* ── ScrollBar ────────────────────────────────────────────────────────────── */
QScrollBar:vertical {{
    background: transparent;
    width: 6px;
    margin: 0;
    border: none;
}}
QScrollBar::handle:vertical {{
    background: {ACCENT};
    border-radius: 3px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: {HI};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
    background: none;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: transparent;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 6px;
    margin: 0;
    border: none;
}}
QScrollBar::handle:horizontal {{
    background: {ACCENT};
    border-radius: 3px;
    min-width: 24px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {HI};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
    background: none;
}}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: transparent;
}}

/* ── GroupBox ─────────────────────────────────────────────────────────────── */
QGroupBox {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 6px;
    margin-top: 14px;
    padding: 10px 8px 8px 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    top: -1px;
    padding: 0 6px;
    color: {MUTED};
    font-size: 10px;
    font-weight: bold;
    letter-spacing: 0.8px;
    background: {PANEL};
}}

/* ── Tables ───────────────────────────────────────────────────────────────── */
QTableView, QTableWidget {{
    background: {PANEL};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 5px;
    gridline-color: {BORDER};
    selection-background-color: {HOVER};
    alternate-background-color: {CARD};
    outline: none;
}}
QTableView::item, QTableWidget::item {{
    padding: 4px 6px;
    border: none;
}}
QTableView::item:selected, QTableWidget::item:selected {{
    background: {HOVER};
    color: {TEXT};
}}
QTableView::item:hover, QTableWidget::item:hover {{
    background: #122038;
}}
QHeaderView::section {{
    background: {ACCENT};
    color: {TEXT};
    border: none;
    border-right: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
    padding: 5px 8px;
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 0.3px;
}}
QHeaderView::section:hover {{
    background: {HOVER};
}}
QHeaderView::section:last {{
    border-right: none;
}}
QHeaderView {{
    background: {ACCENT};
}}

/* ── ListWidget ───────────────────────────────────────────────────────────── */
QListWidget {{
    background: {PANEL};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 5px;
    outline: none;
}}
QListWidget::item {{
    padding: 4px 8px;
    border-bottom: 1px solid {BORDER};
}}
QListWidget::item:selected {{
    background: {HOVER};
    color: {TEXT};
    border-left: 2px solid {HI};
}}
QListWidget::item:hover:!selected {{
    background: #122038;
}}

/* ── Splitter ─────────────────────────────────────────────────────────────── */
QSplitter::handle {{
    background: {BORDER};
}}
QSplitter::handle:horizontal {{
    width: 2px;
}}
QSplitter::handle:vertical {{
    height: 2px;
}}
QSplitter::handle:hover {{
    background: {ACCENT};
}}

/* ── ProgressBar ──────────────────────────────────────────────────────────── */
QProgressBar {{
    background: #0e0e1a;
    border: none;
    border-radius: 5px;
    text-align: center;
    color: {TEXT};
    font-size: 10px;
    min-height: 10px;
}}
QProgressBar::chunk {{
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 {ACCENT}, stop:1 {HI}
    );
    border-radius: 5px;
}}

/* ── TabWidget ────────────────────────────────────────────────────────────── */
QTabWidget::pane {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 0 5px 5px 5px;
    top: -1px;
}}
QTabBar::tab {{
    background: {BG};
    color: {MUTED};
    padding: 6px 16px;
    border: 1px solid {BORDER};
    border-bottom: none;
    border-radius: 5px 5px 0 0;
    margin-right: 2px;
    font-size: 11px;
}}
QTabBar::tab:selected {{
    background: {PANEL};
    color: {TEXT};
    border-bottom: 1px solid {PANEL};
}}
QTabBar::tab:hover:!selected {{
    background: {HOVER};
    color: {TEXT};
}}

/* ── ToolTip ──────────────────────────────────────────────────────────────── */
QToolTip {{
    background: {CARD};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 11px;
}}

/* ── ScrollArea ───────────────────────────────────────────────────────────── */
QScrollArea {{
    border: none;
    background: transparent;
}}
QScrollArea > QWidget > QWidget {{
    background: transparent;
}}

/* ── Label ────────────────────────────────────────────────────────────────── */
QLabel {{
    background: transparent;
    color: {TEXT};
}}

/* ── Frame ────────────────────────────────────────────────────────────────── */
QFrame[frameShape="4"],
QFrame[frameShape="5"] {{
    background: {BORDER};
    border: none;
    max-height: 1px;
    max-width: 1px;
}}

"""


# ─────────────────────────────────────────────────────────────────────────────
# Helper style generators
# ─────────────────────────────────────────────────────────────────────────────

def card_style(border_top_color: str | None = None) -> str:
    """QSS for a stat/info card widget.

    Set via widget.setObjectName("StatCard") + widget.setStyleSheet(card_style(...)).

    Args:
        border_top_color: optional top accent line color (e.g. ACCENT or HI).
    """
    top = f"border-top: 2px solid {border_top_color};" if border_top_color else ""
    return (
        f"#StatCard {{"
        f"  background: {CARD};"
        f"  border: 1px solid {BORDER};"
        f"  border-radius: 6px;"
        f"  {top}"
        f"}}"
    )


def section_header_style() -> str:
    """QSS for an uppercase section label (sidebar divider, group header)."""
    return (
        f"color: {MUTED};"
        f" font-size: 9px;"
        f" font-weight: bold;"
        f" letter-spacing: 1.2px;"
        f" padding: 6px 14px 2px 14px;"
        f" background: transparent;"
    )


def badge_style(color: str) -> str:
    """QSS for a small inline status badge label.

    Args:
        color: the badge text and border color.
    """
    return (
        f"color: {color};"
        f" border: 1px solid {color};"
        f" border-radius: 3px;"
        f" padding: 1px 6px;"
        f" font-size: 9px;"
        f" font-weight: bold;"
        f" letter-spacing: 0.5px;"
        f" background: transparent;"
    )


# ─────────────────────────────────────────────────────────────────────────────
# pyqtgraph theme helper
# ─────────────────────────────────────────────────────────────────────────────

def apply_plot_theme(plot_widget) -> None:
    """Apply RoverDataKit theme colors to a pyqtgraph PlotWidget.

    Call immediately after creating the widget, before adding data curves.
    Safe to call even if pyqtgraph is not installed (silently no-ops).

    Args:
        plot_widget: a pyqtgraph.PlotWidget instance.
    """
    try:
        import pyqtgraph as pg
        pi = plot_widget.getPlotItem()
        axis_pen  = pg.mkPen(color=BORDER,  width=1)
        label_pen = pg.mkPen(color=MUTED,   width=1)
        for axis_name in ("left", "bottom", "right", "top"):
            ax = pi.getAxis(axis_name)
            if ax is not None:
                ax.setPen(axis_pen)
                ax.setTextPen(label_pen)
        pi.getViewBox().setBackgroundColor(CARD)
        plot_widget.showGrid(x=True, y=True, alpha=0.15)
    except Exception:
        pass
