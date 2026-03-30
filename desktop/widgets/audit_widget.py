"""
desktop/widgets/audit_widget.py — File audit results tab.

Layout:
  ┌─────────────────────────────────────────────────────────┐
  │  [Run Audit]  [Export JSON]          session: <name>    │
  ├──────────────────────────────────────────────────────────┤
  │  File  | Device | Duration | IMU Hz | GPS Hz | FPS | …  │  ← QTableView
  ├──────────────────────────────────────────────────────────┤
  │  Issues panel (expandable) for selected row             │
  └──────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PyQt6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    Qt,
    pyqtSlot,
)
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableView,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.models import AuditResult
from core.session_manager import SessionManager
from desktop.theme import BG, PANEL, ACCENT, HI, TEXT, MUTED, SUCCESS, WARNING, BORDER, HOVER, CARD
from desktop.workers import AuditWorker
from desktop.widgets.timing_bar import TimingBar

_COLUMNS = [
    "File",
    "Device",
    "Duration",
    "IMU Rate",
    "GPS Rate",
    "FPS",
    "Resolution",
    "Size",
    "IMU",
    "GPS",
    "Issues",
]

_STATUS_OK   = SUCCESS
_STATUS_WARN = WARNING
_STATUS_ERR  = HI
_PANEL  = PANEL
_ACCENT = ACCENT
_TEXT   = TEXT
_MUTED  = MUTED
_BG     = BG


# ---------------------------------------------------------------------------
# Table model
# ---------------------------------------------------------------------------

class AuditTableModel(QAbstractTableModel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._rows: list[AuditResult] = []

    def set_results(self, results: list[AuditResult]) -> None:
        self.beginResetModel()
        self._rows = results
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(_COLUMNS)

    def headerData(self, section: int, orientation: Qt.Orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return _COLUMNS[section]
        if role == Qt.ItemDataRole.TextAlignmentRole and orientation == Qt.Orientation.Horizontal:
            return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        return None

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        r = self._rows[index.row()]
        col = index.column()

        if role == Qt.ItemDataRole.DisplayRole:
            return self._display(r, col)

        if role == Qt.ItemDataRole.ForegroundRole:
            return self._foreground(r, col)

        if role == Qt.ItemDataRole.BackgroundRole:
            # Teal tint for rows that belong to a chapter sequence
            if r.chapter_files:
                return QColor("#102030")
            # Red tint for rows with issues
            if r.issues:
                return QColor("#201a20")
            return QColor(_PANEL)

        if role == Qt.ItemDataRole.TextAlignmentRole:
            if col in (2, 3, 4, 5, 7):
                return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            if col in (8, 9, 10):
                return Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter

        if role == Qt.ItemDataRole.UserRole:
            return r  # full AuditResult for the issues panel

        return None

    def _display(self, r: AuditResult, col: int) -> str:
        match col:
            case 0:  return Path(r.file_path).name
            case 1:  return r.device_type.value.upper()
            case 2:  return f"{r.duration_seconds:.1f} s"
            case 3:  return f"{r.imu_rate_hz:.0f} Hz" if r.has_imu else "—"
            case 4:  return f"{r.gps_rate_hz:.1f} Hz" if r.has_gps else "—"
            case 5:  return f"{r.video_fps:.1f}" if r.video_fps > 0 else "—"
            case 6:
                if r.video_resolution[0] and r.video_resolution[1]:
                    return f"{r.video_resolution[0]}×{r.video_resolution[1]}"
                return "—"
            case 7:  return f"{r.file_size_mb:.1f} MB"
            case 8:  return "✓  IMU" if r.has_imu else "✗  IMU"
            case 9:  return "✓  GPS" if r.has_gps else "✗  GPS"
            case 10:
                n = len(r.issues)
                return f"⚠  {n}" if n else "—"
            case _:  return ""

    def _foreground(self, r: AuditResult, col: int) -> QColor:
        if col == 8:   # IMU
            return QColor(_STATUS_OK if r.has_imu else _STATUS_ERR)
        if col == 9:   # GPS
            return QColor(_STATUS_OK if r.has_gps else _STATUS_WARN)
        if col == 10 and r.issues:  # Issues
            return QColor(_STATUS_WARN)
        return QColor(_TEXT)

    def result_at(self, row: int) -> AuditResult | None:
        if 0 <= row < len(self._rows):
            return self._rows[row]
        return None


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class AuditWidget(QWidget):
    def __init__(self, session_manager: SessionManager, parent=None) -> None:
        super().__init__(parent)
        self._sm = session_manager
        self._session_id: str = ""
        self._worker: AuditWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # --- Toolbar ---
        toolbar = QHBoxLayout()

        self._add_btn = QPushButton("📁  Add Files…")
        self._add_btn.setEnabled(False)
        self._add_btn.setFixedHeight(32)
        self._add_btn.clicked.connect(self._add_files)
        toolbar.addWidget(self._add_btn)

        self._run_btn = QPushButton("▶  Run Audit")
        self._run_btn.setEnabled(False)
        self._run_btn.setFixedHeight(32)
        self._run_btn.setObjectName("DangerBtn")
        self._run_btn.clicked.connect(self._run_audit)
        toolbar.addWidget(self._run_btn)

        self._export_btn = QPushButton("⬇  Export JSON")
        self._export_btn.setEnabled(False)
        self._export_btn.setFixedHeight(32)
        self._export_btn.clicked.connect(self._export_json)
        toolbar.addWidget(self._export_btn)

        self._merge_btn = QPushButton("⛓  Merge Chapters")
        self._merge_btn.setEnabled(False)
        self._merge_btn.setFixedHeight(32)
        self._merge_btn.setToolTip(
            "Select rows that share a chapter group, then click to merge them into one session entry"
        )
        self._merge_btn.clicked.connect(self._merge_chapters)
        toolbar.addWidget(self._merge_btn)

        toolbar.addStretch()

        self._session_label = QLabel("No session selected")
        toolbar.addWidget(self._session_label)

        root.addLayout(toolbar)

        # --- Timing bar (shown while audit runs) ---
        self._timing_bar = TimingBar(label="Audit", parent=self)
        root.addWidget(self._timing_bar)

        # --- Files list bar (shows added files, appears after first file is added) ---
        self._files_label = QLabel("")
        self._files_label.setWordWrap(True)
        self._files_label.setVisible(False)
        root.addWidget(self._files_label)

        # --- Summary bar ---
        summary_container = QWidget()
        summary_container.setObjectName("SummaryBar")
        summary_container.setFixedHeight(30)
        summary_container.setStyleSheet(
            f"#SummaryBar {{ background: {_PANEL}; border-radius: 4px;"
            f" border: 1px solid {_ACCENT}; }}"
        )
        summary_layout = QHBoxLayout(summary_container)
        summary_layout.setContentsMargins(10, 0, 10, 0)
        summary_layout.setSpacing(16)
        self._summary_label = QLabel("")
        summary_layout.addWidget(self._summary_label)
        summary_layout.addStretch()
        root.addWidget(summary_container)

        # --- Vertical splitter: table + issues panel ---
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Table
        self._model = AuditTableModel()
        self._table = QTableView()
        self._table.setModel(self._model)
        self._table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
        self._table.setAlternatingRowColors(True)
        self._table.setSortingEnabled(False)
        self._table.horizontalHeader().setStretchLastSection(False)
        # File column stretches; all others resize to content
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in range(1, len(_COLUMNS)):
            self._table.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeMode.ResizeToContents
            )
        self._table.setMinimumHeight(120)
        self._table.verticalHeader().setVisible(False)
        self._table.selectionModel().currentRowChanged.connect(self._on_row_changed)
        self._table.selectionModel().selectionChanged.connect(self._on_selection_changed)
        splitter.addWidget(self._table)

        # Issues panel
        issues_container = QWidget()
        issues_container.setMaximumHeight(160)
        issues_layout = QVBoxLayout(issues_container)
        issues_layout.setContentsMargins(0, 4, 0, 0)
        issues_layout.setSpacing(2)

        issues_hdr = QLabel("Issues")
        issues_hdr.setFont(QFont("sans-serif", 9, QFont.Weight.Bold))
        issues_layout.addWidget(issues_hdr)

        self._issues_view = QTextEdit()
        self._issues_view.setReadOnly(True)
        self._issues_view.setFont(QFont("monospace", 9))
        issues_layout.addWidget(self._issues_view)

        splitter.addWidget(issues_container)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        root.addWidget(splitter)

        # --- Placeholder (shown when no session / no results) ---
        self._placeholder = QLabel("Select a session and click  ▶ Run Audit.")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._placeholder)
        self._placeholder.setVisible(True)
        splitter.setVisible(False)
        self._splitter = splitter

    # ------------------------------------------------------------------
    # Session change
    # ------------------------------------------------------------------

    @pyqtSlot(str)
    def on_session_changed(self, session_id: str) -> None:
        self._session_id = session_id
        self._worker = None
        self._timing_bar.reset()

        if not session_id:
            self._session_label.setText("No session selected")
            self._add_btn.setEnabled(False)
            self._run_btn.setEnabled(False)
            self._export_btn.setEnabled(False)
            self._model.set_results([])
            self._files_label.setVisible(False)
            self._show_placeholder("Select a session and click  ▶ Run Audit.")
            return

        try:
            s = self._sm.get_session(session_id)
        except Exception:
            return

        self._session_label.setText(f"{s.name}  [{s.environment} · {s.location}]")
        self._add_btn.setEnabled(True)
        self._run_btn.setEnabled(bool(s.files))
        self._refresh_files_label(s.files)

        if s.audit_results:
            self._load_results(s.audit_results)
        else:
            self._model.set_results([])
            self._show_placeholder(
                f"{len(s.files)} file(s) ready — click  ▶ Run Audit."
                if s.files else
                "No files yet — click  📁 Add Files… to add videos."
            )

    # ------------------------------------------------------------------
    # File upload
    # ------------------------------------------------------------------

    def _add_files(self) -> None:
        """Open a file picker and add selected files to the current session."""
        if not self._session_id:
            return

        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Video / Sensor Files",
            str(Path.home()),
            "Supported files (*.mp4 *.MP4 *.insv *.INSV *.csv *.CSV *.json *.JSON);;"
            "GoPro MP4 (*.mp4 *.MP4);;"
            "Insta360 INSV (*.insv *.INSV);;"
            "Sensor Logger CSV/JSON (*.csv *.CSV *.json *.JSON);;"
            "All files (*)",
        )
        if not paths:
            return

        added, skipped, errors = [], [], []
        for p in paths:
            try:
                before = len(self._sm.get_session(self._session_id).files)
                self._sm.add_file(self._session_id, p)
                after = len(self._sm.get_session(self._session_id).files)
                if after > before:
                    added.append(Path(p).name)
                else:
                    skipped.append(Path(p).name)
            except FileNotFoundError as exc:
                errors.append(str(exc))
            except Exception as exc:
                errors.append(f"{Path(p).name}: {exc}")

        # Report outcome
        msg_parts = []
        if added:
            msg_parts.append(f"Added {len(added)} file(s):\n" + "\n".join(f"  • {n}" for n in added))
        if skipped:
            msg_parts.append(f"Already in session (skipped):\n" + "\n".join(f"  • {n}" for n in skipped))
        if errors:
            msg_parts.append("Errors:\n" + "\n".join(f"  ✗ {e}" for e in errors))

        if errors:
            QMessageBox.warning(self, "Add Files", "\n\n".join(msg_parts))
        elif msg_parts:
            mw = self._main_window()
            if mw:
                mw.log("\n".join(msg_parts))

        # Refresh UI
        try:
            s = self._sm.get_session(self._session_id)
            self._run_btn.setEnabled(bool(s.files))
            self._refresh_files_label(s.files)
            if not s.audit_results:
                self._show_placeholder(
                    f"{len(s.files)} file(s) ready — click  ▶ Run Audit."
                )
        except Exception:
            pass

    def _refresh_files_label(self, files: list[str]) -> None:
        if not files:
            self._files_label.setVisible(False)
            return
        names = "  |  ".join(Path(f).name for f in files)
        self._files_label.setText(f"Files ({len(files)}):  {names}")
        self._files_label.setVisible(True)

    # ------------------------------------------------------------------
    # Audit execution
    # ------------------------------------------------------------------

    def _run_audit(self) -> None:
        if not self._session_id:
            return
        if self._worker and self._worker.isRunning():
            return

        self._run_btn.setEnabled(False)
        self._export_btn.setEnabled(False)
        self._show_placeholder("Running audit…")
        self._timing_bar.start()

        self._worker = AuditWorker(self._session_id, self._sm, parent=self)
        self._worker.progress.connect(self._on_progress)
        self._worker.status.connect(self._on_status)
        self._worker.result.connect(self._on_audit_done)
        self._worker.error.connect(self._on_audit_error)
        self._worker.timing.connect(self._timing_bar.on_timing)
        self._worker.progress.connect(self._timing_bar.on_progress)
        self._worker.start()

    @pyqtSlot(int)
    def _on_progress(self, pct: int) -> None:
        # Bubble up to MainWindow if available
        mw = self._main_window()
        if mw:
            mw.set_progress(pct)

    @pyqtSlot(str)
    def _on_status(self, msg: str) -> None:
        mw = self._main_window()
        if mw:
            mw.log(msg)
            mw.set_status(msg)

    @pyqtSlot(object)
    def _on_audit_done(self, results: object) -> None:
        self._run_btn.setEnabled(True)
        self._timing_bar.stop(success=True)
        self._load_results(results)  # type: ignore[arg-type]
        mw = self._main_window()
        if mw:
            mw.set_progress(100)

    @pyqtSlot(str)
    def _on_audit_error(self, msg: str) -> None:
        self._run_btn.setEnabled(True)
        self._timing_bar.stop(success=False)
        self._show_placeholder(f"Audit failed:\n{msg}")
        QMessageBox.critical(self, "Audit Error", msg)
        mw = self._main_window()
        if mw:
            mw.set_progress(100)

    # ------------------------------------------------------------------
    # Results display
    # ------------------------------------------------------------------

    def _load_results(self, results: list[AuditResult]) -> None:
        self._model.set_results(results)
        self._placeholder.setVisible(False)
        self._splitter.setVisible(True)
        self._export_btn.setEnabled(True)

        # Summary line
        total_dur = sum(r.duration_seconds for r in results)
        total_issues = sum(len(r.issues) for r in results)
        imu_ok = sum(1 for r in results if r.has_imu)
        gps_ok = sum(1 for r in results if r.has_gps)
        issue_color = "#f39c12" if total_issues else "#27ae60"
        self._summary_label.setText(
            f"{len(results)} file(s)  ·  "
            f"{total_dur:.1f} s total  ·  "
            f"IMU: {imu_ok}/{len(results)}  ·  "
            f"GPS: {gps_ok}/{len(results)}  ·  "
            f"{'⚠  ' if total_issues else '✓  '}{total_issues} issue(s)"
        )
        self._summary_label.setStyleSheet(
            f"color: {issue_color}; font-size: 11px; background: transparent;"
        )

    def _show_placeholder(self, msg: str) -> None:
        self._placeholder.setText(msg)
        self._placeholder.setVisible(True)
        self._splitter.setVisible(False)
        self._summary_label.setText("")

    # ------------------------------------------------------------------
    # Row selection → issues panel
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_row_changed(self) -> None:
        idx = self._table.currentIndex()
        if not idx.isValid():
            self._issues_view.clear()
            return
        r = self._model.result_at(idx.row())
        if r is None:
            self._issues_view.clear()
            return
        if r.issues:
            self._issues_view.setPlainText(
                "\n".join(f"⚠  {issue}" for issue in r.issues)
            )
        else:
            self._issues_view.setPlainText("No issues detected.")

    @pyqtSlot()
    def _on_selection_changed(self) -> None:
        """Enable Merge Chapters button when selected rows share a chapter group."""
        rows = [idx.row() for idx in self._table.selectionModel().selectedRows()]
        if len(rows) < 2:
            self._merge_btn.setEnabled(False)
            return
        results = [self._model.result_at(r) for r in rows if self._model.result_at(r)]
        # All selected rows must have chapter_files and share at least one common path
        if not all(r.chapter_files for r in results):
            self._merge_btn.setEnabled(False)
            return
        # Check they share the same chapter group (same set of chapter_files)
        sets = [frozenset(r.chapter_files) for r in results]
        self._merge_btn.setEnabled(len(set(sets)) == 1)

    # ------------------------------------------------------------------
    # Merge Chapters
    # ------------------------------------------------------------------

    def _merge_chapters(self) -> None:
        rows = [idx.row() for idx in self._table.selectionModel().selectedRows()]
        results = [self._model.result_at(r) for r in rows if self._model.result_at(r)]
        if not results:
            return
        files = list({f for r in results for f in ([r.file_path] + r.chapter_files)})
        try:
            self._sm.merge_chapter_files(self._session_id, files)
            QMessageBox.information(
                self, "Chapters Merged",
                f"Merged {len(files)} chapter file(s) into one session entry."
            )
            # Refresh session
            self.on_session_changed(self._session_id)
        except Exception as exc:
            QMessageBox.critical(self, "Merge Error", str(exc))

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export_json(self) -> None:
        if not self._session_id:
            return
        try:
            s = self._sm.get_session(self._session_id)
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Audit Report",
            f"audit_{s.name.replace(' ', '_')}.json",
            "JSON Files (*.json)",
        )
        if not path:
            return

        export = {
            "session": s.name,
            "session_id": s.id,
            "files": [
                {
                    "file": Path(r.file_path).name,
                    "device": r.device_type.value,
                    "duration_s": r.duration_seconds,
                    "has_imu": r.has_imu,
                    "has_gps": r.has_gps,
                    "imu_rate_hz": r.imu_rate_hz,
                    "gps_rate_hz": r.gps_rate_hz,
                    "video_fps": r.video_fps,
                    "video_resolution": list(r.video_resolution),
                    "file_size_mb": r.file_size_mb,
                    "issues": r.issues,
                }
                for r in s.audit_results
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(export, f, indent=2)
        QMessageBox.information(self, "Exported", f"Audit report saved to:\n{path}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _main_window(self):
        """Walk up to MainWindow to access log/progress API."""
        w = self.parent()
        while w is not None:
            if hasattr(w, "log") and hasattr(w, "set_progress"):
                return w
            w = w.parent() if hasattr(w, "parent") else None
        return None


