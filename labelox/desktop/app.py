"""
labelox/desktop/app.py — QApplication setup for LABELOX.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication

from labelox.core.database import init_db
from labelox.core.utils import setup_logging

_DEFAULT_DATA_DIR = Path.home() / ".labelox"
_DEFAULT_LOG_FILE = _DEFAULT_DATA_DIR / "labelox.log"
_DEFAULT_DB_PATH = _DEFAULT_DATA_DIR / "labelox.db"


def _apply_dark_palette(app: QApplication) -> None:
    """Fallback QPalette for widgets that ignore QSS."""
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window, QColor("#1a1a2e"))
    p.setColor(QPalette.ColorRole.WindowText, QColor("#e0e0e0"))
    p.setColor(QPalette.ColorRole.Base, QColor("#111827"))
    p.setColor(QPalette.ColorRole.AlternateBase, QColor("#16213e"))
    p.setColor(QPalette.ColorRole.Text, QColor("#e0e0e0"))
    p.setColor(QPalette.ColorRole.Button, QColor("#0f3460"))
    p.setColor(QPalette.ColorRole.ButtonText, QColor("#e0e0e0"))
    p.setColor(QPalette.ColorRole.Highlight, QColor("#1a3a6a"))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor("#e0e0e0"))
    p.setColor(QPalette.ColorRole.ToolTipBase, QColor("#16213e"))
    p.setColor(QPalette.ColorRole.ToolTipText, QColor("#e0e0e0"))
    app.setPalette(p)


def create_app(argv: list[str] | None = None) -> tuple[QApplication, str]:
    """Create and configure the LABELOX QApplication.

    Returns (app, db_url).
    """
    if argv is None:
        argv = sys.argv

    # Force GTK dark on Linux
    os.environ.setdefault("GTK_THEME", "Adwaita:dark")

    # Ensure data dirs exist
    _DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (_DEFAULT_DATA_DIR / "projects").mkdir(exist_ok=True)
    (_DEFAULT_DATA_DIR / "models").mkdir(exist_ok=True)

    setup_logging(log_file=_DEFAULT_LOG_FILE)

    app = QApplication(argv)
    app.setApplicationName("LABELOX")
    app.setOrganizationName("LABELOX")
    app.setStyle("Fusion")

    try:
        app.styleHints().setColorScheme(Qt.ColorScheme.Dark)
    except AttributeError:
        pass

    _apply_dark_palette(app)

    # Initialise database
    db_url = f"sqlite:///{_DEFAULT_DB_PATH}"
    init_db(db_url)

    return app, db_url
