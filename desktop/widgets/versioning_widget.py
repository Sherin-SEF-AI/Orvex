"""
desktop/widgets/versioning_widget.py — Dataset versioning UI tab.

Layout:
  ┌──────────────────────────────────────────────────────────────┐
  │  DVC Status bar (badges + Init / Configure Remote buttons)  │
  ├──────────────────────────────────────────────────────────────┤
  │  QSplitter (horizontal)                                      │
  │   Left: version list (QListWidget)                           │
  │   Right: version details (tag / timestamp / message /        │
  │          class table / Restore button)                       │
  ├──────────────────────────────────────────────────────────────┤
  │  Commit New Version (tag + message + Commit btn)             │
  ├──────────────────────────────────────────────────────────────┤
  │  Compare Versions (A + B combos + diff table)                │
  └──────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

from pathlib import Path

from loguru import logger
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.models import DatasetDiff, DatasetVersion
from core.versioning import (
    add_dataset_to_dvc,
    check_dvc_installation,
    check_git_initialized,
    commit_dataset_version,
    diff_dataset_versions,
    init_dvc_repo,
    list_dataset_versions,
    restore_dataset_version,
)
from desktop.theme import (
    ACCENT, BORDER, HI, MUTED, PANEL, SUCCESS, TEXT, WARNING,
    badge_style,
)


class VersioningWidget(QWidget):
    """Dataset versioning tab — wraps core/versioning.py in a PyQt6 UI.

    The widget is independent of the session system; it operates on any
    *dataset directory* that contains ``images/`` and/or ``labels/``
    sub-directories.  The dataset path is set either via the Browse button
    or through :meth:`on_session_changed`.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._dataset_dir: Path | None = None
        self._versions: list[DatasetVersion] = []
        self._build_ui()
        self._refresh_status()

    # ──────────────────────────────────────────────────────────────────────
    # UI construction
    # ──────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        root.addWidget(self._build_status_bar())
        root.addWidget(self._build_main_splitter(), stretch=1)
        root.addWidget(self._build_commit_box())
        root.addWidget(self._build_diff_box())

    # ── DVC status bar ──────────────────────────────────────────────────

    def _build_status_bar(self) -> QGroupBox:
        box = QGroupBox("DVC Status")
        layout = QHBoxLayout(box)
        layout.setSpacing(10)

        self._dvc_badge = QLabel()
        layout.addWidget(self._dvc_badge)

        self._git_badge = QLabel()
        layout.addWidget(self._git_badge)

        layout.addSpacing(12)

        self._init_btn = QPushButton("Initialize DVC")
        self._init_btn.setFixedHeight(26)
        self._init_btn.setToolTip(
            "Run dvc init inside the current dataset directory"
        )
        self._init_btn.clicked.connect(self._on_init_dvc)
        layout.addWidget(self._init_btn)

        self._remote_btn = QPushButton("Configure Remote")
        self._remote_btn.setFixedHeight(26)
        self._remote_btn.setToolTip(
            "Pick a local folder to use as the DVC remote storage location"
        )
        self._remote_btn.clicked.connect(self._on_configure_remote)
        layout.addWidget(self._remote_btn)

        layout.addSpacing(20)
        layout.addWidget(QLabel("Dataset path:"))

        self._path_label = QLabel("(none)")
        self._path_label.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        self._path_label.setMaximumWidth(320)
        self._path_label.setToolTip("Current dataset directory")
        layout.addWidget(self._path_label)

        browse_btn = QPushButton("Browse\u2026")
        browse_btn.setFixedHeight(26)
        browse_btn.clicked.connect(self._on_browse_dataset)
        layout.addWidget(browse_btn)

        layout.addStretch()
        return box

    # ── Main splitter: version list + details ───────────────────────────

    def _build_main_splitter(self) -> QSplitter:
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left: version list ──────────────────────────────────────────
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        left_layout.addWidget(self._section_label("VERSIONS"))

        self._version_list = QListWidget()
        self._version_list.setMinimumWidth(220)
        self._version_list.currentItemChanged.connect(self._on_version_selected)
        left_layout.addWidget(self._version_list)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setFixedHeight(24)
        refresh_btn.clicked.connect(self._refresh_versions)
        left_layout.addWidget(refresh_btn)

        splitter.addWidget(left)

        # ── Right: version details ──────────────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.setSpacing(6)

        right_layout.addWidget(self._section_label("VERSION DETAILS"))

        info_box = QGroupBox("")
        info_layout = QVBoxLayout(info_box)
        info_layout.setSpacing(4)

        self._detail_tag = self._info_label("Tag:", "—")
        self._detail_ts = self._info_label("Timestamp:", "—")
        self._detail_msg = self._info_label("Message:", "—")
        self._detail_frames = self._info_label("Total frames:", "—")
        self._detail_hash = self._info_label("Hash:", "—")

        for row in (
            self._detail_tag,
            self._detail_ts,
            self._detail_msg,
            self._detail_frames,
            self._detail_hash,
        ):
            info_layout.addWidget(row)

        right_layout.addWidget(info_box)

        right_layout.addWidget(self._section_label("CLASS DISTRIBUTION"))

        self._class_table = QTableWidget(0, 3)
        self._class_table.setHorizontalHeaderLabels(["Class", "Count", "%"])
        self._class_table.horizontalHeader().setStretchLastSection(True)
        self._class_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self._class_table.setAlternatingRowColors(True)
        self._class_table.setMaximumHeight(160)
        right_layout.addWidget(self._class_table)

        self._restore_btn = QPushButton("Restore this version")
        self._restore_btn.setObjectName("DangerBtn")
        self._restore_btn.setEnabled(False)
        self._restore_btn.setToolTip(
            "Overwrite the current working tree with this version via dvc checkout"
        )
        self._restore_btn.clicked.connect(self._on_restore)
        right_layout.addWidget(self._restore_btn)

        right_layout.addStretch()
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        return splitter

    # ── Commit box ──────────────────────────────────────────────────────

    def _build_commit_box(self) -> QGroupBox:
        box = QGroupBox("Commit New Version")
        layout = QHBoxLayout(box)
        layout.setSpacing(8)

        layout.addWidget(QLabel("Tag:"))
        self._tag_edit = QLineEdit()
        self._tag_edit.setPlaceholderText("v1.0.0")
        self._tag_edit.setMaximumWidth(140)
        layout.addWidget(self._tag_edit)

        layout.addWidget(QLabel("Message:"))
        self._msg_edit = QLineEdit()
        self._msg_edit.setPlaceholderText("Added highway session data")
        layout.addWidget(self._msg_edit, stretch=1)

        self._commit_btn = QPushButton("Commit")
        self._commit_btn.setObjectName("PrimaryBtn")
        self._commit_btn.setFixedHeight(28)
        self._commit_btn.clicked.connect(self._on_commit)
        layout.addWidget(self._commit_btn)

        return box

    # ── Diff box ────────────────────────────────────────────────────────

    def _build_diff_box(self) -> QGroupBox:
        box = QGroupBox("Compare Versions")
        layout = QVBoxLayout(box)
        layout.setSpacing(6)

        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Version A:"))
        self._diff_combo_a = QComboBox()
        self._diff_combo_a.setMinimumWidth(120)
        selector_row.addWidget(self._diff_combo_a)

        selector_row.addSpacing(12)
        selector_row.addWidget(QLabel("Version B:"))
        self._diff_combo_b = QComboBox()
        self._diff_combo_b.setMinimumWidth(120)
        selector_row.addWidget(self._diff_combo_b)

        self._compare_btn = QPushButton("Compare")
        self._compare_btn.setFixedHeight(26)
        self._compare_btn.clicked.connect(self._on_compare)
        selector_row.addWidget(self._compare_btn)
        selector_row.addStretch()
        layout.addLayout(selector_row)

        self._diff_table = QTableWidget(0, 4)
        self._diff_table.setHorizontalHeaderLabels(["Class", "A count", "B count", "Delta"])
        self._diff_table.horizontalHeader().setStretchLastSection(True)
        self._diff_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._diff_table.setAlternatingRowColors(True)
        self._diff_table.setMaximumHeight(180)
        layout.addWidget(self._diff_table)

        self._diff_summary = QLabel("")
        self._diff_summary.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        layout.addWidget(self._diff_summary)

        return box

    # ──────────────────────────────────────────────────────────────────────
    # Slot: session changed from main window
    # ──────────────────────────────────────────────────────────────────────

    @pyqtSlot(str)
    def on_session_changed(self, session_id: str) -> None:
        """Attempt to auto-detect the dataset directory for *session_id*.

        Searches for a ``dataset/`` sub-directory inside the session folder
        (resolved via the sessions root if a SessionManager is reachable through
        the widget hierarchy).  If no dataset directory is found, the user is
        prompted to browse for one.

        Args:
            session_id: UUID of the newly selected session.
        """
        sm = self._find_session_manager()
        if sm is not None:
            try:
                session_folder = sm.session_folder(session_id)
                candidate = Path(session_folder) / "dataset"
                if candidate.is_dir():
                    logger.info(
                        "on_session_changed: auto-detected dataset dir '{}'", candidate
                    )
                    self._set_dataset_dir(candidate)
                    return
            except Exception as exc:
                logger.debug("on_session_changed: could not resolve session folder: {}", exc)

        logger.debug(
            "on_session_changed: no dataset/ found for session '{}', prompting user",
            session_id,
        )
        self._on_browse_dataset()

    # ──────────────────────────────────────────────────────────────────────
    # Internal: dataset directory management
    # ──────────────────────────────────────────────────────────────────────

    def _set_dataset_dir(self, path: Path) -> None:
        self._dataset_dir = path
        self._path_label.setText(str(path))
        self._path_label.setStyleSheet(f"color: {TEXT}; font-size: 11px;")
        self._path_label.setToolTip(str(path))
        logger.info("Dataset dir set to '{}'", path)
        self._refresh_status()
        self._refresh_versions()

    # ──────────────────────────────────────────────────────────────────────
    # Status refresh
    # ──────────────────────────────────────────────────────────────────────

    def _refresh_status(self) -> None:
        """Update DVC + git status badges based on current dataset directory."""
        dvc_ok = check_dvc_installation()
        if dvc_ok:
            self._dvc_badge.setText("\u2713 DVC Ready")
            self._dvc_badge.setStyleSheet(badge_style(SUCCESS))
        else:
            self._dvc_badge.setText("\u26a0 DVC not installed")
            self._dvc_badge.setStyleSheet(badge_style(WARNING))

        if self._dataset_dir is not None:
            git_ok = check_git_initialized(str(self._dataset_dir))
        else:
            git_ok = False

        if git_ok:
            self._git_badge.setText("\u2713 Git initialized")
            self._git_badge.setStyleSheet(badge_style(SUCCESS))
        else:
            self._git_badge.setText("\u26a0 No git repo")
            self._git_badge.setStyleSheet(badge_style(HI))

        has_dir = self._dataset_dir is not None
        self._init_btn.setEnabled(has_dir and dvc_ok)
        self._remote_btn.setEnabled(has_dir and dvc_ok)
        self._commit_btn.setEnabled(has_dir)
        self._compare_btn.setEnabled(has_dir)

    # ──────────────────────────────────────────────────────────────────────
    # Version list refresh
    # ──────────────────────────────────────────────────────────────────────

    def _refresh_versions(self) -> None:
        if self._dataset_dir is None:
            self._versions = []
            self._version_list.clear()
            self._refresh_diff_combos()
            return

        try:
            self._versions = list_dataset_versions(str(self._dataset_dir))
        except Exception as exc:
            logger.error("list_dataset_versions failed: {}", exc)
            self._versions = []

        self._version_list.clear()
        for v in reversed(self._versions):  # newest first in the list
            ts_str = v.timestamp.strftime("%Y-%m-%d %H:%M")
            text = f"{v.tag}  |  {ts_str}  |  {v.total_frames} frames"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, v.tag)
            item.setToolTip(v.message)
            self._version_list.addItem(item)

        self._refresh_diff_combos()
        self._clear_details()

    def _refresh_diff_combos(self) -> None:
        for combo in (self._diff_combo_a, self._diff_combo_b):
            combo.blockSignals(True)
            combo.clear()
            for v in self._versions:
                combo.addItem(v.tag)
            combo.blockSignals(False)

    # ──────────────────────────────────────────────────────────────────────
    # Version details panel
    # ──────────────────────────────────────────────────────────────────────

    def _on_version_selected(
        self, current: QListWidgetItem | None, _prev: QListWidgetItem | None
    ) -> None:
        if current is None:
            self._clear_details()
            return

        tag = current.data(Qt.ItemDataRole.UserRole)
        version = next((v for v in self._versions if v.tag == tag), None)
        if version is None:
            self._clear_details()
            return

        self._populate_details(version)

    def _populate_details(self, v: DatasetVersion) -> None:
        self._set_info_value(self._detail_tag, v.tag)
        self._set_info_value(
            self._detail_ts, v.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
        )
        self._set_info_value(self._detail_msg, v.message)
        self._set_info_value(self._detail_frames, str(v.total_frames))
        self._set_info_value(self._detail_hash, v.dataset_hash[:16] + "\u2026")

        # Class distribution table
        dist = v.class_distribution
        total_instances = sum(dist.values()) or 1
        self._class_table.setRowCount(len(dist))
        for row, (cls_id, count) in enumerate(sorted(dist.items())):
            pct = count / total_instances * 100
            self._class_table.setItem(row, 0, _table_item(cls_id))
            self._class_table.setItem(row, 1, _table_item(str(count), Qt.AlignmentFlag.AlignRight))
            self._class_table.setItem(row, 2, _table_item(f"{pct:.1f}%", Qt.AlignmentFlag.AlignRight))
        self._class_table.resizeColumnsToContents()

        self._restore_btn.setEnabled(True)
        self._restore_btn.setProperty("_version_tag", v.tag)

    def _clear_details(self) -> None:
        for lbl in (
            self._detail_tag, self._detail_ts, self._detail_msg,
            self._detail_frames, self._detail_hash,
        ):
            self._set_info_value(lbl, "\u2014")
        self._class_table.setRowCount(0)
        self._restore_btn.setEnabled(False)

    # ──────────────────────────────────────────────────────────────────────
    # Button handlers
    # ──────────────────────────────────────────────────────────────────────

    def _on_browse_dataset(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select dataset directory (must contain images/ or labels/)",
            str(self._dataset_dir or Path.home()),
        )
        if path:
            self._set_dataset_dir(Path(path))

    def _on_init_dvc(self) -> None:
        if self._dataset_dir is None:
            QMessageBox.warning(self, "No Dataset", "Set a dataset directory first.")
            return

        if not check_git_initialized(str(self._dataset_dir)):
            QMessageBox.critical(
                self,
                "Git Required",
                f"Git is not initialized in:\n{self._dataset_dir}\n\n"
                "Run the following command in a terminal, then try again:\n"
                f"  cd {self._dataset_dir} && git init",
            )
            return

        try:
            init_dvc_repo(str(self._dataset_dir))
            QMessageBox.information(
                self, "DVC Initialized", "DVC repository initialized successfully."
            )
            logger.info("DVC initialized in '{}'", self._dataset_dir)
        except RuntimeError as exc:
            QMessageBox.critical(self, "DVC Init Failed", str(exc))
            logger.error("DVC init failed: {}", exc)

        self._refresh_status()

    def _on_configure_remote(self) -> None:
        if self._dataset_dir is None:
            QMessageBox.warning(self, "No Dataset", "Set a dataset directory first.")
            return

        remote_path = QFileDialog.getExistingDirectory(
            self,
            "Select local DVC remote storage folder",
            str(Path.home()),
        )
        if not remote_path:
            return

        try:
            add_dataset_to_dvc(
                str(self._dataset_dir),
                remote_name="local_remote",
                remote_path=remote_path,
            )
            QMessageBox.information(
                self,
                "Remote Configured",
                f"DVC remote configured at:\n{remote_path}",
            )
            logger.info("DVC remote set to '{}'", remote_path)
        except RuntimeError as exc:
            QMessageBox.critical(self, "Remote Config Failed", str(exc))
            logger.error("DVC remote config failed: {}", exc)

    def _on_commit(self) -> None:
        if self._dataset_dir is None:
            QMessageBox.warning(self, "No Dataset", "Set a dataset directory first.")
            return

        tag = self._tag_edit.text().strip()
        message = self._msg_edit.text().strip()

        if not tag:
            QMessageBox.warning(
                self, "Tag Required", "Enter a version tag (e.g. v1.0.0)."
            )
            return
        if not message:
            QMessageBox.warning(
                self, "Message Required", "Enter a commit message describing this version."
            )
            return

        try:
            version = commit_dataset_version(
                str(self._dataset_dir),
                version_tag=tag,
                message=message,
            )
            QMessageBox.information(
                self,
                "Version Committed",
                f"Version '{version.tag}' committed.\n"
                f"Frames: {version.total_frames}\n"
                f"Hash: {version.dataset_hash[:16]}\u2026",
            )
            self._tag_edit.clear()
            self._msg_edit.clear()
            self._refresh_versions()
            logger.info("Committed dataset version '{}'", tag)
        except ValueError as exc:
            QMessageBox.warning(self, "Duplicate Tag", str(exc))
        except Exception as exc:
            QMessageBox.critical(self, "Commit Failed", str(exc))
            logger.error("commit_dataset_version failed: {}", exc)

    def _on_restore(self) -> None:
        tag = self._restore_btn.property("_version_tag")
        if not tag:
            return

        reply = QMessageBox.question(
            self,
            "Confirm Restore",
            f"Restore dataset to version '{tag}'?\n\n"
            "This will overwrite the current working tree.\n"
            "Any uncommitted changes will be lost.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            restore_dataset_version(str(self._dataset_dir), tag)
            QMessageBox.information(
                self, "Restore Complete", f"Dataset restored to version '{tag}'."
            )
            self._refresh_versions()
            logger.info("Restored dataset to version '{}'", tag)
        except FileNotFoundError as exc:
            QMessageBox.critical(self, "Version Not Found", str(exc))
        except RuntimeError as exc:
            QMessageBox.critical(self, "Restore Failed", str(exc))
            logger.error("restore_dataset_version failed: {}", exc)

    def _on_compare(self) -> None:
        if self._dataset_dir is None:
            QMessageBox.warning(self, "No Dataset", "Set a dataset directory first.")
            return

        tag_a = self._diff_combo_a.currentText()
        tag_b = self._diff_combo_b.currentText()

        if not tag_a or not tag_b:
            QMessageBox.warning(
                self, "No Versions", "Commit at least two versions before comparing."
            )
            return
        if tag_a == tag_b:
            QMessageBox.warning(
                self, "Same Version", "Select two different versions to compare."
            )
            return

        try:
            diff = diff_dataset_versions(str(self._dataset_dir), tag_a, tag_b)
            self._populate_diff(diff)
        except FileNotFoundError as exc:
            QMessageBox.critical(self, "Version Not Found", str(exc))
        except Exception as exc:
            QMessageBox.critical(self, "Compare Failed", str(exc))
            logger.error("diff_dataset_versions failed: {}", exc)

    # ──────────────────────────────────────────────────────────────────────
    # Diff table population
    # ──────────────────────────────────────────────────────────────────────

    def _populate_diff(self, diff: DatasetDiff) -> None:
        # Build union of class keys from both versions
        va_obj = next((v for v in self._versions if v.tag == diff.version_a), None)
        vb_obj = next((v for v in self._versions if v.tag == diff.version_b), None)

        all_classes = set(diff.class_distribution_delta.keys())

        self._diff_table.setRowCount(len(all_classes))
        for row, cls_id in enumerate(sorted(all_classes)):
            a_count = va_obj.class_distribution.get(cls_id, 0) if va_obj else 0
            b_count = vb_obj.class_distribution.get(cls_id, 0) if vb_obj else 0
            delta = int(diff.class_distribution_delta.get(cls_id, 0.0))

            delta_item = _table_item(
                f"{'+' if delta >= 0 else ''}{delta}",
                Qt.AlignmentFlag.AlignRight,
            )
            if delta > 0:
                delta_item.setForeground(QColor(SUCCESS))
            elif delta < 0:
                delta_item.setForeground(QColor(HI))

            self._diff_table.setItem(row, 0, _table_item(cls_id))
            self._diff_table.setItem(row, 1, _table_item(str(a_count), Qt.AlignmentFlag.AlignRight))
            self._diff_table.setItem(row, 2, _table_item(str(b_count), Qt.AlignmentFlag.AlignRight))
            self._diff_table.setItem(row, 3, delta_item)

        self._diff_table.resizeColumnsToContents()

        summary_parts = [
            f"{diff.version_a} \u2192 {diff.version_b}",
            f"+{diff.frames_added} frames added",
            f"-{diff.frames_removed} frames removed",
        ]
        if diff.new_sessions:
            summary_parts.append(f"New sessions: {', '.join(diff.new_sessions)}")
        if diff.removed_sessions:
            summary_parts.append(f"Removed sessions: {', '.join(diff.removed_sessions)}")

        self._diff_summary.setText("  \u00b7  ".join(summary_parts))

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _find_session_manager(self):
        """Walk widget hierarchy to find an object with a session_folder method."""
        p = self.parent()
        while p is not None:
            if hasattr(p, "session_manager"):
                return p.session_manager
            p = p.parent()
        return None

    def _main_window(self):
        p = self.parent()
        while p is not None:
            if hasattr(p, "log") and hasattr(p, "set_progress"):
                return p
            p = p.parent()
        return None

    @staticmethod
    def _section_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {MUTED}; font-size: 9px; font-weight: bold;"
            f" letter-spacing: 1.2px; padding: 4px 0 2px 0;"
            f" background: transparent;"
        )
        return lbl

    @staticmethod
    def _info_label(title: str, value: str) -> QWidget:
        """Create a two-column title/value row widget."""
        row = QWidget()
        row.setStyleSheet("background: transparent;")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        title_lbl = QLabel(title)
        title_lbl.setFixedWidth(90)
        title_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px; background: transparent;")
        layout.addWidget(title_lbl)

        value_lbl = QLabel(value)
        value_lbl.setStyleSheet(f"color: {TEXT}; font-size: 11px; background: transparent;")
        value_lbl.setObjectName("value")
        layout.addWidget(value_lbl, stretch=1)
        return row

    @staticmethod
    def _set_info_value(row_widget: QWidget, value: str) -> None:
        lbl = row_widget.findChild(QLabel, "value")
        if lbl is not None:
            lbl.setText(value)


# ──────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ──────────────────────────────────────────────────────────────────────────────


def _table_item(
    text: str,
    alignment: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignLeft,
) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setTextAlignment(int(alignment) | int(Qt.AlignmentFlag.AlignVCenter))
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    return item
