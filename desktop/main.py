"""
desktop/main.py — Entry point for the RoverDataKit desktop application.

Usage:
    python -m desktop.main
    # or
    python desktop/main.py
"""
from __future__ import annotations

import sys

from desktop.app import create_app
from desktop.mainwindow import MainWindow


def _force_dark_titlebar(window) -> None:
    """Set _GTK_THEME_VARIANT=dark on the X11 window so GNOME renders a dark title bar."""
    try:
        import ctypes
        import ctypes.util

        display_lib = ctypes.util.find_library("X11")
        if not display_lib:
            return
        xlib = ctypes.cdll.LoadLibrary(display_lib)

        wid = int(window.winId())
        display = xlib.XOpenDisplay(None)
        if not display:
            return

        atom_name = xlib.XInternAtom(display, b"_GTK_THEME_VARIANT", False)
        atom_utf8 = xlib.XInternAtom(display, b"UTF8_STRING", False)

        value = b"dark"
        xlib.XChangeProperty(
            display, wid, atom_name, atom_utf8,
            8, 0,  # PropModeReplace
            value, len(value),
        )
        xlib.XFlush(display)
        xlib.XCloseDisplay(display)
    except Exception:
        pass


def main() -> int:
    app, session_manager = create_app(sys.argv)
    window = MainWindow(session_manager)
    window.show()
    _force_dark_titlebar(window)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
