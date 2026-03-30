"""
desktop/widgets/api_settings_widget.py — API server settings and control panel.

Allows the user to start/stop the embedded REST API server, regenerate the
API key, and monitor recent requests.

Rule 21: API key is shown exactly once on generation.
         The plaintext key is never stored; only the hash goes to disk.
"""
from __future__ import annotations

import threading
import webbrowser
from datetime import datetime
from typing import Any

from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QClipboard, QGuiApplication
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from desktop.theme import (
    ACCENT,
    BG,
    BORDER,
    CARD,
    HI,
    INPUT,
    MUTED,
    PANEL,
    SUCCESS,
    TEXT,
    WARNING,
)

# ---------------------------------------------------------------------------
# Colours used only here
# ---------------------------------------------------------------------------
_GREEN  = SUCCESS     # running state
_GRAY   = MUTED       # stopped state
_RED    = HI          # error / danger
_ORANGE = WARNING


class ApiSettingsWidget(QWidget):
    """Control panel for the embedded RoverDataKit REST API server.

    Integrates with ``core.api_server.start_api_server_thread``.

    Args:
        sm:     SessionManager instance to pass to the API server.
        parent: Qt parent widget.
    """

    def __init__(self, sm: Any = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sm = sm
        self._server_thread: threading.Thread | None = None
        self._server_running = False
        self._log_lines: list[str] = []
        self._max_log_lines = 50

        self._build_ui()
        self._refresh_status()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        root.addWidget(self._build_server_group())
        root.addWidget(self._build_key_group())
        root.addWidget(self._build_log_group())
        root.addStretch(1)

    # ── Server group ──────────────────────────────────────────────────

    def _build_server_group(self) -> QGroupBox:
        box = QGroupBox("API Server")
        layout = QVBoxLayout(box)
        layout.setSpacing(8)

        # Status badge row
        status_row = QHBoxLayout()
        self._status_label = QLabel("● Stopped")
        self._status_label.setStyleSheet(f"color: {_GRAY}; font-weight: bold; font-size: 13px;")
        status_row.addWidget(self._status_label)
        status_row.addStretch()
        layout.addLayout(status_row)

        # Host row
        host_row = QHBoxLayout()
        host_row.addWidget(QLabel("Host:"))
        self._host_edit = QLineEdit("127.0.0.1")
        self._host_edit.setPlaceholderText("127.0.0.1")
        self._host_edit.setToolTip(
            "Bind address for the API server.\n"
            "Use 127.0.0.1 to restrict to localhost only.\n"
            "Use 0.0.0.0 to allow LAN access (ensure firewall rules are correct)."
        )
        self._host_edit.setMaximumWidth(160)
        host_row.addWidget(self._host_edit)
        host_row.addStretch()
        layout.addLayout(host_row)

        # Port row
        port_row = QHBoxLayout()
        port_row.addWidget(QLabel("Port:"))
        self._port_spin = QSpinBox()
        self._port_spin.setRange(1024, 65535)
        self._port_spin.setValue(8080)
        self._port_spin.setToolTip("TCP port for the API server (1024–65535).")
        self._port_spin.setMaximumWidth(100)
        port_row.addWidget(self._port_spin)
        port_row.addStretch()
        layout.addLayout(port_row)

        # Buttons row
        btn_row = QHBoxLayout()
        self._start_btn = QPushButton("Start Server")
        self._start_btn.setObjectName("PrimaryBtn")
        self._start_btn.clicked.connect(self._on_start)

        self._stop_btn = QPushButton("Stop Server")
        self._stop_btn.setObjectName("DangerBtn")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)

        self._docs_btn = QPushButton("Open API Docs")
        self._docs_btn.setEnabled(False)
        self._docs_btn.clicked.connect(self._on_open_docs)
        self._docs_btn.setToolTip("Opens the interactive Swagger UI in your default browser.")

        btn_row.addWidget(self._start_btn)
        btn_row.addWidget(self._stop_btn)
        btn_row.addWidget(self._docs_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        return box

    # ── API key group ─────────────────────────────────────────────────

    def _build_key_group(self) -> QGroupBox:
        box = QGroupBox("API Key")
        layout = QVBoxLayout(box)
        layout.setSpacing(8)

        # Key display row
        key_row = QHBoxLayout()
        self._key_display = QLineEdit()
        self._key_display.setReadOnly(True)
        self._key_display.setEchoMode(QLineEdit.EchoMode.Password)
        self._key_display.setPlaceholderText("No key loaded — start the server to generate")
        self._key_display.setToolTip(
            "The API key is shown once when first generated.\n"
            "To retrieve it again, regenerate using the button below.\n"
            "The previous key will immediately stop working."
        )
        key_row.addWidget(self._key_display)

        self._copy_btn = QPushButton("Copy")
        self._copy_btn.setMaximumWidth(60)
        self._copy_btn.setEnabled(False)
        self._copy_btn.setToolTip("Copy the API key to clipboard.")
        self._copy_btn.clicked.connect(self._on_copy_key)
        key_row.addWidget(self._copy_btn)
        layout.addLayout(key_row)

        # Key note
        note = QLabel(
            "The API key is stored as a one-way hash — the plaintext is only available "
            "immediately after generation."
        )
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {MUTED}; font-size: 10px;")
        layout.addWidget(note)

        # Regenerate button
        regen_row = QHBoxLayout()
        self._regen_btn = QPushButton("Regenerate Key")
        self._regen_btn.setObjectName("WarningBtn")
        self._regen_btn.setToolTip(
            "Generate a new API key.\n"
            "WARNING: The current key will immediately stop working.\n"
            "Update all clients before regenerating."
        )
        self._regen_btn.clicked.connect(self._on_regen_key)
        regen_row.addWidget(self._regen_btn)
        regen_row.addStretch()
        layout.addLayout(regen_row)

        return box

    # ── Request log group ─────────────────────────────────────────────

    def _build_log_group(self) -> QGroupBox:
        box = QGroupBox("Request Log (last 50)")
        layout = QVBoxLayout(box)
        layout.setSpacing(6)

        self._log_edit = QTextEdit()
        self._log_edit.setReadOnly(True)
        self._log_edit.setFont(
            self._log_edit.font()  # inherits theme monospace via QSS
        )
        self._log_edit.setStyleSheet(
            f"font-family: 'Consolas', 'Courier New', monospace; font-size: 11px;"
        )
        self._log_edit.setMinimumHeight(140)
        self._log_edit.setPlaceholderText("No requests logged yet.")
        layout.addWidget(self._log_edit)

        clear_row = QHBoxLayout()
        self._clear_log_btn = QPushButton("Clear")
        self._clear_log_btn.setMaximumWidth(70)
        self._clear_log_btn.clicked.connect(self._on_clear_log)
        clear_row.addStretch()
        clear_row.addWidget(self._clear_log_btn)
        layout.addLayout(clear_row)

        return box

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_start(self) -> None:
        if self._server_running:
            return

        host = self._host_edit.text().strip() or "127.0.0.1"
        port = self._port_spin.value()

        try:
            from core.api_server import start_api_server_thread, load_or_create_api_key
        except ImportError as exc:
            QMessageBox.critical(
                self,
                "Missing Dependency",
                f"Could not import core.api_server:\n\n{exc}\n\n"
                "Make sure FastAPI and uvicorn are installed:\n"
                "  pip install fastapi uvicorn[standard]",
            )
            return

        config_path = _resolve_config_path()

        try:
            plaintext_key, is_new = load_or_create_api_key(config_path)
        except Exception as exc:
            QMessageBox.critical(self, "API Key Error", f"Could not load/create API key:\n\n{exc}")
            return

        if is_new and plaintext_key:
            self._show_new_key(plaintext_key)

        try:
            thread = start_api_server_thread(
                self._sm,
                host=host,
                port=port,
                api_key_config_path=config_path,
            )
            self._server_thread = thread
            self._server_running = True
            self._add_log(f"[{_ts()}] Server started on http://{host}:{port}")
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Server Start Failed",
                f"Could not start the API server:\n\n{exc}",
            )
            return

        self._refresh_status()

    @pyqtSlot()
    def _on_stop(self) -> None:
        # uvicorn threads are daemon threads and have no clean public stop API
        # when embedded via uvicorn.Server in a thread.  Inform the user.
        QMessageBox.information(
            self,
            "Stop API Server",
            "The embedded API server runs in a daemon thread.\n\n"
            "To stop it, restart the application.\n\n"
            "The server will automatically stop when RoverDataKit exits.",
        )

    @pyqtSlot()
    def _on_open_docs(self) -> None:
        port = self._port_spin.value()
        url = f"http://localhost:{port}/api/v1/docs"
        webbrowser.open(url)
        self._add_log(f"[{_ts()}] Opened docs: {url}")

    @pyqtSlot()
    def _on_copy_key(self) -> None:
        key = self._key_display.text()
        if not key:
            return
        clipboard = QGuiApplication.clipboard()
        if clipboard:
            clipboard.setText(key)
        self._add_log(f"[{_ts()}] API key copied to clipboard.")

    @pyqtSlot()
    def _on_regen_key(self) -> None:
        reply = QMessageBox.warning(
            self,
            "Regenerate API Key",
            "Regenerating the API key will immediately invalidate the current key.\n\n"
            "All clients using the old key will receive 401 Unauthorized errors.\n\n"
            "Proceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            from core.api_server import create_api_key, hash_api_key
            import toml as _toml
        except ImportError as exc:
            QMessageBox.critical(self, "Import Error", str(exc))
            return

        config_path = _resolve_config_path()
        from pathlib import Path
        cfg_file = Path(config_path)
        config: dict = {}
        if cfg_file.exists():
            try:
                with open(cfg_file, encoding="utf-8") as f:
                    config = _toml.load(f)
            except Exception:
                pass

        new_key = create_api_key()
        new_hash = hash_api_key(new_key)
        if "api_server" not in config:
            config["api_server"] = {}
        config["api_server"]["api_key_hash"] = new_hash

        try:
            cfg_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cfg_file, "w", encoding="utf-8") as f:
                _toml.dump(config, f)
        except Exception as exc:
            QMessageBox.critical(self, "Write Error", f"Could not save new key hash:\n{exc}")
            return

        self._show_new_key(new_key)
        self._add_log(f"[{_ts()}] API key regenerated — update all clients.")

    @pyqtSlot()
    def _on_clear_log(self) -> None:
        self._log_lines.clear()
        self._log_edit.clear()

    # ------------------------------------------------------------------
    # Public slot — called by mainwindow when session selection changes
    # ------------------------------------------------------------------

    def on_session_changed(self, session_id: str | None) -> None:  # noqa: ARG002
        """No-op — API settings are session-independent."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _refresh_status(self) -> None:
        if self._server_running:
            port = self._port_spin.value()
            self._status_label.setText(f"● Running on port {port}")
            self._status_label.setStyleSheet(
                f"color: {_GREEN}; font-weight: bold; font-size: 13px;"
            )
            self._start_btn.setEnabled(False)
            self._stop_btn.setEnabled(True)
            self._docs_btn.setEnabled(True)
            self._host_edit.setEnabled(False)
            self._port_spin.setEnabled(False)
        else:
            self._status_label.setText("● Stopped")
            self._status_label.setStyleSheet(
                f"color: {_GRAY}; font-weight: bold; font-size: 13px;"
            )
            self._start_btn.setEnabled(True)
            self._stop_btn.setEnabled(False)
            self._docs_btn.setEnabled(False)
            self._host_edit.setEnabled(True)
            self._port_spin.setEnabled(True)

    def _show_new_key(self, plaintext_key: str) -> None:
        """Display a new key in the key field and show a one-time dialog."""
        self._key_display.setEchoMode(QLineEdit.EchoMode.Normal)
        self._key_display.setText(plaintext_key)
        self._copy_btn.setEnabled(True)

        dialog = _KeyDisplayDialog(plaintext_key, self)
        dialog.exec()

        # After the dialog is dismissed, mask the key again
        self._key_display.setEchoMode(QLineEdit.EchoMode.Password)

    def _add_log(self, line: str) -> None:
        self._log_lines.append(line)
        if len(self._log_lines) > self._max_log_lines:
            self._log_lines = self._log_lines[-self._max_log_lines:]
        self._log_edit.setPlainText("\n".join(self._log_lines))
        # Scroll to bottom
        sb = self._log_edit.verticalScrollBar()
        if sb:
            sb.setValue(sb.maximum())


# ---------------------------------------------------------------------------
# One-time key display dialog
# ---------------------------------------------------------------------------

class _KeyDisplayDialog(QMessageBox):
    """Modal dialog that shows a new API key with a copy button.

    Follows Rule 21: key shown once, copy-to-clipboard provided.
    """

    def __init__(self, api_key: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._key = api_key
        self.setWindowTitle("New API Key Generated")
        self.setIcon(QMessageBox.Icon.Information)
        self.setText(
            "<b>Your new API key has been generated.</b><br><br>"
            "Save it now — it will <b>not</b> be shown again.<br>"
            "Only the SHA-256 hash is stored on disk."
        )
        self.setInformativeText(
            f"<code style='font-size:13px; letter-spacing:1px;'>{api_key}</code>"
        )
        self.setTextFormat(Qt.TextFormat.RichText)

        copy_btn = self.addButton("Copy to Clipboard", QMessageBox.ButtonRole.ActionRole)
        copy_btn.clicked.connect(self._copy)
        self.addButton("Close", QMessageBox.ButtonRole.AcceptRole)

    @pyqtSlot()
    def _copy(self) -> None:
        clipboard = QGuiApplication.clipboard()
        if clipboard:
            clipboard.setText(self._key)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _ts() -> str:
    """Return current time as HH:MM:SS string."""
    return datetime.now().strftime("%H:%M:%S")


def _resolve_config_path() -> str:
    """Return the canonical config path for the API key hash."""
    from pathlib import Path
    return str(Path.home() / ".roverdatakit" / "data" / "config.toml")
