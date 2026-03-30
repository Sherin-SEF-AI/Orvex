"""
labelox/desktop/main.py — LABELOX desktop entry point.

Usage: python -m labelox.desktop.main
"""
from __future__ import annotations

import sys


def _force_dark_titlebar(window) -> None:
    """Attempt to force dark window decoration on Linux/X11."""
    try:
        import ctypes
        from PyQt6.QtGui import QGuiApplication
        native = window.windowHandle()
        if native is None:
            return
        wid = int(native.winId())
        display = QGuiApplication.instance().nativeInterface()
        # X11 _GTK_THEME_VARIANT hint
        if hasattr(display, "display"):
            pass  # Fallback: frameless window handles this
    except Exception:
        pass


def main() -> None:
    from labelox.desktop.app import create_app
    from labelox.desktop.mainwindow import LabeloxMainWindow
    from labelox.desktop.theme import GLOBAL_QSS

    app, db_url = create_app()

    window = LabeloxMainWindow()
    window.setStyleSheet(GLOBAL_QSS)
    window.resize(1400, 900)
    window.show()

    _force_dark_titlebar(window)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
