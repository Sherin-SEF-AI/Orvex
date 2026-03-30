"""
desktop/widgets/session_panel.py — Session browser sidebar.

Lists sessions with status badges. Emits session_selected(str) when the
user clicks a session row. Provides [+ New] button and delete action.
Includes a search filter to narrow sessions by name/location.
"""
from __future__ import annotations

from datetime import datetime, timezone

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.session_manager import SessionManager, SessionNotFoundError
from desktop.theme import BG, PANEL, ACCENT, HI, TEXT, MUTED, SUCCESS, WARNING, BORDER, HOVER, CARD, badge_style

_BG        = BG
_PANEL     = PANEL
_ACCENT    = ACCENT
_HIGHLIGHT = HI
_TEXT      = TEXT
_MUTED     = MUTED

_STATUS_COLORS = {
    "pending": _MUTED,
    "running": "#f39c12",
    "done":    "#27ae60",
    "failed":  _HIGHLIGHT,
}

_STATUS_ICONS = {
    "pending": "○",
    "running": "◉",
    "done":    "●",
    "failed":  "✕",
}

_QUALITY_COLORS = {
    "excellent": "#2ecc71",
    "good":      "#3498db",
    "fair":      "#f39c12",
    "poor":      "#e94560",
}

def _ago(dt: datetime) -> str:
    """Return human-readable time since dt."""
    try:
        now = datetime.now(timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = now - dt
        s = int(delta.total_seconds())
        if s < 60:    return "just now"
        if s < 3600:  return f"{s // 60}m ago"
        if s < 86400: return f"{s // 3600}h ago"
        return f"{s // 86400}d ago"
    except Exception:
        return ""


class SessionPanel(QWidget):
    """Sidebar listing all sessions with search filter."""

    session_selected = pyqtSignal(str)  # session_id

    def __init__(self, session_manager: SessionManager, parent=None) -> None:
        super().__init__(parent)
        self._sm = session_manager
        self._all_sessions: list = []
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setSpacing(6)

        # ── Header ─────────────────────────────────────────────────────
        header_row = QHBoxLayout()
        header = QLabel("Sessions")
        header.setFont(QFont("sans-serif", 11, QFont.Weight.Bold))
        self._count_lbl = QLabel("0")
        self._count_lbl.setFont(QFont("monospace", 9))
        self._count_lbl.setStyleSheet(
            f"color: {_TEXT}; background: {_ACCENT};"
            f" border-radius: 8px; padding: 1px 7px; font-size: 10px;"
        )
        header_row.addWidget(header)
        header_row.addStretch()
        header_row.addWidget(self._count_lbl)
        layout.addLayout(header_row)

        # ── Search box ─────────────────────────────────────────────────
        self._search = QLineEdit()
        self._search.setPlaceholderText("🔍  Search sessions…")
        self._search.setFixedHeight(30)
        self._search.textChanged.connect(self._filter_sessions)
        layout.addWidget(self._search)

        # ── Session list ───────────────────────────────────────────────
        self._list = QListWidget()
        self._list.itemClicked.connect(self._on_item_clicked)
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._context_menu)
        layout.addWidget(self._list, stretch=1)

        # ── New session button ─────────────────────────────────────────
        new_btn = QPushButton("＋  New Session")
        new_btn.setFixedHeight(34)
        new_btn.clicked.connect(self.new_session_dialog)
        layout.addWidget(new_btn)

    def refresh(self) -> None:
        """Reload session list from disk."""
        self._all_sessions = self._sm.list_sessions()
        self._count_lbl.setText(str(len(self._all_sessions)))
        self._apply_filter(self._search.text() if hasattr(self, "_search") else "")

    def _filter_sessions(self, text: str) -> None:
        self._apply_filter(text)

    def _apply_filter(self, text: str) -> None:
        self._list.clear()
        query = text.strip().lower()
        filtered = [
            s for s in self._all_sessions
            if not query
            or query in s.name.lower()
            or query in s.location.lower()
            or query in s.environment.lower()
        ]

        for s in filtered:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, s.id)
            color = _STATUS_COLORS.get(s.extraction_status, _MUTED)
            icon  = _STATUS_ICONS.get(s.extraction_status, "○")
            # Compute total duration from audit results
            total_s = sum(r.duration_seconds for r in s.audit_results) if s.audit_results else 0.0
            quality = s.audit_results[0].quality.value if s.audit_results and hasattr(s.audit_results[0], 'quality') else None
            label = _SessionItemWidget(
                s.name, s.environment, s.location,
                color, icon, len(s.files),
                quality=quality,
                duration_s=total_s,
                created_ago=_ago(s.created_at),
            )
            item.setSizeHint(label.sizeHint())
            self._list.addItem(item)
            self._list.setItemWidget(item, label)

        if not filtered:
            if query:
                msg = f'No sessions match "{text}"'
            else:
                msg = "No sessions — click ＋ New"
            placeholder = QListWidgetItem(msg)
            placeholder.setForeground(QColor(_MUTED))
            placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
            placeholder.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._list.addItem(placeholder)

    def new_session_dialog(self) -> None:
        dlg = _NewSessionDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            name, env, loc, notes = dlg.values()
            if not name.strip():
                QMessageBox.warning(self, "Validation", "Session name cannot be empty.")
                return
            self._sm.create_session(name.strip(), env.strip(), loc.strip(), notes.strip())
            self.refresh()

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        session_id = item.data(Qt.ItemDataRole.UserRole)
        if session_id:
            self.session_selected.emit(session_id)

    def _context_menu(self, pos) -> None:
        item = self._list.itemAt(pos)
        if item is None:
            return
        session_id = item.data(Qt.ItemDataRole.UserRole)
        if not session_id:
            return

        menu = QMenu(self)
        rename_action = menu.addAction("✏  Rename Session…")
        menu.addSeparator()
        delete_action = menu.addAction("🗑  Delete Session…")
        action = menu.exec(self._list.mapToGlobal(pos))
        if action == delete_action:
            self._delete_session(session_id)

    def _delete_session(self, session_id: str) -> None:
        try:
            s = self._sm.get_session(session_id)
        except SessionNotFoundError:
            self.refresh()
            return
        reply = QMessageBox.question(
            self,
            "Delete Session",
            f"Delete session '<b>{s.name}</b>'?\n\n"
            "This removes the TOML record only.\n"
            "Raw files and extraction output are NOT deleted.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._sm.delete_session(session_id)
            self.session_selected.emit("")
            self.refresh()


class _SessionItemWidget(QWidget):
    def __init__(
        self,
        name: str,
        env: str,
        location: str,
        status_color: str,
        status_icon: str,
        file_count: int,
        quality: str | None = None,
        duration_s: float = 0.0,
        created_ago: str = "",
        parent=None,
    ):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 7, 10, 7)
        layout.setSpacing(3)

        # ── Name row ──────────────────────────────────────────────────
        name_row = QHBoxLayout()
        name_row.setSpacing(6)

        name_label = QLabel(name)
        name_label.setFont(QFont("sans-serif", 11, QFont.Weight.Bold))

        status_dot = QLabel(status_icon)
        status_dot.setFont(QFont("monospace", 11))
        status_dot.setStyleSheet(f"color: {status_color}; font-weight: bold;")
        status_dot.setToolTip(f"Extraction: {status_icon}")

        name_row.addWidget(name_label, stretch=1)
        name_row.addWidget(status_dot)
        layout.addLayout(name_row)

        # ── Meta row ──────────────────────────────────────────────────
        meta_row = QHBoxLayout()
        meta_row.setSpacing(6)

        env_lbl = QLabel(env)
        env_lbl.setFont(QFont("sans-serif", 8))
        env_lbl.setStyleSheet(badge_style(MUTED))

        loc_lbl = QLabel(f"📍 {location}")
        loc_lbl.setFont(QFont("sans-serif", 8))

        meta_row.addWidget(env_lbl)
        meta_row.addWidget(loc_lbl)
        meta_row.addStretch()
        layout.addLayout(meta_row)

        # ── Stats row ─────────────────────────────────────────────────
        stats_row = QHBoxLayout()
        stats_row.setSpacing(8)

        if quality:
            q_color = _QUALITY_COLORS.get(quality, _MUTED)
            q_lbl = QLabel(quality.upper())
            q_lbl.setFont(QFont("monospace", 7, QFont.Weight.Bold))
            q_lbl.setStyleSheet(
                f"color: {q_color}; border: 1px solid {q_color};"
                f" border-radius: 3px; padding: 0px 5px;"
            )
            q_lbl.setToolTip(f"Recording quality: {quality}")
            stats_row.addWidget(q_lbl)

        if duration_s > 0:
            mins = int(duration_s) // 60
            secs = int(duration_s) % 60
            dur_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
            dur_lbl = QLabel(f"⏱ {dur_str}")
            dur_lbl.setFont(QFont("monospace", 8))
            stats_row.addWidget(dur_lbl)

        stats_row.addStretch()

        files_lbl = QLabel(f"📄 {file_count}")
        files_lbl.setFont(QFont("monospace", 8))
        stats_row.addWidget(files_lbl)

        if created_ago:
            ago_lbl = QLabel(created_ago)
            ago_lbl.setFont(QFont("sans-serif", 8))
            stats_row.addWidget(ago_lbl)

        layout.addLayout(stats_row)


class _NewSessionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Session")
        self.setModal(True)
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Header
        hdr = QLabel("Create New Session")
        hdr.setFont(QFont("sans-serif", 13, QFont.Weight.Bold))
        layout.addWidget(hdr)

        form = QFormLayout()
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._name = QLineEdit()
        self._name.setPlaceholderText("e.g. Kalamassery Road Run 01")

        self._env = QLineEdit()
        self._env.setPlaceholderText("road / indoor / gravel")

        self._loc = QLineEdit()
        self._loc.setPlaceholderText("e.g. Kalamassery")

        self._notes = QLineEdit()
        self._notes.setPlaceholderText("Optional notes")

        for lbl_text, widget in [
            ("Name *", self._name),
            ("Environment", self._env),
            ("Location", self._loc),
            ("Notes", self._notes),
        ]:
            form.addRow(QLabel(lbl_text), widget)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Create Session")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> tuple[str, str, str, str]:
        return (
            self._name.text(),
            self._env.text() or "unknown",
            self._loc.text() or "unknown",
            self._notes.text(),
        )
