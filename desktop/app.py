"""
desktop/app.py — QApplication setup and dark palette configuration.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Force dark window decorations on Linux (GNOME/GTK)
os.environ.setdefault("GTK_THEME", "Adwaita:dark")

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtWidgets import QApplication

# Must be imported (or AA_ShareOpenGLContexts set) BEFORE QApplication is created
try:
    import PyQt6.QtWebEngineWidgets  # noqa: F401
except ImportError:
    pass

from core.utils import setup_logging, MissingToolError, check_dependencies
from core.session_manager import SessionManager


_DEFAULT_DATA_DIR = Path.home() / ".roverdatakit" / "data"
_DEFAULT_SESSIONS_DIR = _DEFAULT_DATA_DIR / "sessions"
_DEFAULT_LOG_FILE = Path.home() / ".roverdatakit" / "roverdatakit.log"


def create_app(argv: list[str] | None = None) -> tuple[QApplication, SessionManager]:
    """Initialise QApplication, logging, and SessionManager.

    Returns:
        (app, session_manager) ready for MainWindow creation.
    """
    if argv is None:
        argv = sys.argv

    setup_logging(log_file=_DEFAULT_LOG_FILE)

    app = QApplication(argv)
    app.setApplicationName("Orvex")
    app.setOrganizationName("Orvex")
    app.setStyle("Fusion")
    # Force dark title bar on Linux (Qt 6.5+)
    try:
        app.styleHints().setColorScheme(Qt.ColorScheme.Dark)
    except AttributeError:
        pass  # Qt < 6.5 fallback — GTK_THEME env var handles it
    _apply_dark_palette(app)

    # Dependency check — show a warning dialog but don't abort startup
    try:
        check_dependencies()
    except MissingToolError as exc:
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.warning(
            None,
            "Missing Dependencies",
            str(exc) + "\n\nSome features will not work until these are installed.",
        )

    _DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    sm = SessionManager(_DEFAULT_SESSIONS_DIR)

    return app, sm


def _apply_dark_palette(app: QApplication) -> None:
    palette = QPalette()
    bg = QColor("#1a1a2e")
    panel = QColor("#16213e")
    accent = QColor("#0f3460")
    text = QColor("#e0e0e0")
    muted = QColor("#888888")
    highlight = QColor("#e94560")

    palette.setColor(QPalette.ColorRole.Window, bg)
    palette.setColor(QPalette.ColorRole.WindowText, text)
    palette.setColor(QPalette.ColorRole.Base, panel)
    palette.setColor(QPalette.ColorRole.AlternateBase, bg)
    palette.setColor(QPalette.ColorRole.ToolTipBase, panel)
    palette.setColor(QPalette.ColorRole.ToolTipText, text)
    palette.setColor(QPalette.ColorRole.Text, text)
    palette.setColor(QPalette.ColorRole.Button, accent)
    palette.setColor(QPalette.ColorRole.ButtonText, text)
    palette.setColor(QPalette.ColorRole.BrightText, highlight)
    palette.setColor(QPalette.ColorRole.Highlight, accent)
    palette.setColor(QPalette.ColorRole.HighlightedText, text)
    palette.setColor(QPalette.ColorRole.PlaceholderText, muted)

    app.setPalette(palette)
