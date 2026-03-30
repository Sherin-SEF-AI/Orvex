"""
desktop/widgets/sensorlogger_widget.py — Android Sensor Logger CSV/ZIP import widget.

Handles CSV/ZIP file import, auto-extraction of ZIPs, per-file audit,
batch extraction to EuRoC format, and results display.
"""
from __future__ import annotations

import csv
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from desktop.theme import (
    BG, PANEL, ACCENT, HI, TEXT, MUTED, SUCCESS, WARNING, BORDER, CARD,
)
from desktop.workers import SensorLoggerWorker
from desktop.widgets.timing_bar import TimingBar


class SensorLoggerWidget(QWidget):
    """UI for importing Android Sensor Logger CSV/ZIP files and extracting telemetry."""

    def __init__(self, session_manager, parent=None) -> None:
        super().__init__(parent)
        self._sm = session_manager
        self._worker: SensorLoggerWorker | None = None
        self._csv_paths: list[str] = []
        self._temp_dirs: list[str] = []
        self._audit_results: list = []
        self._extracted_sessions: list = []
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        cfg_w = self._build_config_panel()
        tabs = self._build_center_tabs()
        right_w = self._build_right_panel()

        splitter.addWidget(cfg_w)
        splitter.addWidget(tabs)
        splitter.addWidget(right_w)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([300, 700, 260])

        root.addWidget(splitter)

    # ── Left config panel ─────────────────────────────────────────────

    def _build_config_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(260)
        scroll.setMaximumWidth(380)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            f"QScrollArea {{ background: {PANEL}; border: none; }}"
            f"QScrollArea > QWidget > QWidget {{ background: {PANEL}; }}"
        )

        cfg_w = QWidget()
        cfg_layout = QVBoxLayout(cfg_w)
        cfg_layout.setContentsMargins(8, 8, 8, 8)
        cfg_layout.setSpacing(6)

        # ── Import group ──────────────────────────────────────────────
        import_group = QGroupBox("Import")
        ig = QVBoxLayout(import_group)
        ig.setSpacing(4)

        btn_row = QHBoxLayout()
        self._add_csv_btn = QPushButton("Add CSV Files…")
        self._add_csv_btn.clicked.connect(self._add_csv_files)
        self._add_zip_btn = QPushButton("Add ZIP Files…")
        self._add_zip_btn.clicked.connect(self._add_zip_files)
        btn_row.addWidget(self._add_csv_btn)
        btn_row.addWidget(self._add_zip_btn)
        ig.addLayout(btn_row)

        self._file_list = QListWidget()
        self._file_list.setMinimumHeight(100)
        self._file_list.setMaximumHeight(200)
        self._file_list.currentRowChanged.connect(self._on_file_selected)
        ig.addWidget(self._file_list)

        rm_row = QHBoxLayout()
        self._remove_btn = QPushButton("Remove")
        self._remove_btn.clicked.connect(self._remove_selected)
        self._clear_btn = QPushButton("Clear All")
        self._clear_btn.clicked.connect(self._clear_all)
        rm_row.addWidget(self._remove_btn)
        rm_row.addWidget(self._clear_btn)
        ig.addLayout(rm_row)

        self._count_label = QLabel("0 CSV file(s) loaded")
        self._count_label.setStyleSheet(f"color: {MUTED}; font-size: 10px;")
        ig.addWidget(self._count_label)

        cfg_layout.addWidget(import_group)

        # ── Output group ──────────────────────────────────────────────
        output_group = QGroupBox("Output")
        og = QVBoxLayout(output_group)
        og.setSpacing(4)
        lbl = QLabel("Output directory")
        lbl.setStyleSheet(f"color: {MUTED}; font-size: 10px;")
        og.addWidget(lbl)
        out_row = QHBoxLayout()
        self._output_edit = QLineEdit()
        self._output_edit.setPlaceholderText("Select output directory…")
        self._output_edit.textChanged.connect(self._update_run_btn_state)
        out_browse = QPushButton("…")
        out_browse.setFixedWidth(28)
        out_browse.clicked.connect(self._browse_output)
        out_row.addWidget(self._output_edit, stretch=1)
        out_row.addWidget(out_browse)
        og.addLayout(out_row)
        cfg_layout.addWidget(output_group)

        # ── Action row ────────────────────────────────────────────────
        action_row = QHBoxLayout()
        self._run_btn = QPushButton("Run Extraction")
        self._run_btn.setObjectName("DangerBtn")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._run_extraction)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel)
        action_row.addWidget(self._run_btn)
        action_row.addWidget(self._cancel_btn)
        cfg_layout.addLayout(action_row)

        self._timing_bar = TimingBar("Sensor Logger")
        cfg_layout.addWidget(self._timing_bar)

        cfg_layout.addStretch()

        scroll.setWidget(cfg_w)
        return scroll

    # ── Center tabs ───────────────────────────────────────────────────

    def _build_center_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        tabs.addTab(self._build_audit_tab(), "Audit Results")
        tabs.addTab(self._build_summary_tab(), "Summary")
        tabs.addTab(self._build_log_tab(), "Log")
        return tabs

    def _build_audit_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)

        self._audit_table = QTableWidget(0, 8)
        self._audit_table.setHorizontalHeaderLabels([
            "File", "Duration", "IMU Hz", "GPS Hz",
            "Has IMU", "Has GPS", "Size (MB)", "Issues",
        ])
        self._audit_table.horizontalHeader().setStretchLastSection(True)
        self._audit_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._audit_table.setAlternatingRowColors(True)
        self._audit_table.setStyleSheet(
            f"QTableWidget {{ background: {BG}; color: {TEXT};"
            f" gridline-color: {BORDER}; font-size: 11px; }}"
            f"QHeaderView::section {{ background: {PANEL}; color: {TEXT};"
            f" padding: 4px; border: 1px solid {BORDER}; }}"
            f"QTableWidget::item:alternate {{ background: {CARD}; }}"
        )
        layout.addWidget(self._audit_table)
        return tab

    def _build_summary_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self._summary_labels: dict[str, QLabel] = {}
        for key, label_text in [
            ("files", "Files Processed"),
            ("imu_total", "Total IMU Samples"),
            ("gps_total", "Total GPS Samples"),
            ("duration", "Total Duration"),
            ("errors", "Errors"),
        ]:
            row = QHBoxLayout()
            name_lbl = QLabel(label_text)
            name_lbl.setStyleSheet(f"color: {MUTED}; font-size: 12px;")
            name_lbl.setFixedWidth(160)
            val_lbl = QLabel("—")
            val_lbl.setStyleSheet(f"color: {TEXT}; font-size: 12px; font-weight: bold;")
            row.addWidget(name_lbl)
            row.addWidget(val_lbl)
            row.addStretch()
            layout.addLayout(row)
            self._summary_labels[key] = val_lbl

        layout.addStretch()

        self._no_results_lbl = QLabel("Run extraction to see summary.")
        self._no_results_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._no_results_lbl.setStyleSheet(f"color: {MUTED}; font-size: 12px;")
        layout.addWidget(self._no_results_lbl)

        return tab

    def _build_log_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet(
            f"background: {CARD}; color: {TEXT};"
            f" font-family: monospace; font-size: 11px; border: none;"
        )
        layout.addWidget(self._log)
        return tab

    # ── Right panel ───────────────────────────────────────────────────

    def _build_right_panel(self) -> QWidget:
        right_w = QWidget()
        right_w.setMinimumWidth(220)
        right_w.setMaximumWidth(320)
        right_layout = QVBoxLayout(right_w)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(6)

        # File info group
        info_group = QGroupBox("Selected File")
        ig = QVBoxLayout(info_group)
        ig.setSpacing(4)

        self._info_labels: dict[str, QLabel] = {}
        for key, label_text in [
            ("path", "Path"),
            ("size", "Size"),
            ("columns", "Columns"),
            ("rows", "Rows (approx)"),
            ("timestamp_col", "Timestamp Column"),
        ]:
            name_lbl = QLabel(label_text)
            name_lbl.setStyleSheet(f"color: {MUTED}; font-size: 10px; margin-top: 2px;")
            ig.addWidget(name_lbl)
            val_lbl = QLabel("—")
            val_lbl.setStyleSheet(f"color: {TEXT}; font-size: 11px;")
            val_lbl.setWordWrap(True)
            ig.addWidget(val_lbl)
            self._info_labels[key] = val_lbl

        right_layout.addWidget(info_group)

        # Extraction status group
        status_group = QGroupBox("Extraction Status")
        sg = QVBoxLayout(status_group)
        self._status_label = QLabel("Idle")
        self._status_label.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        self._status_label.setWordWrap(True)
        sg.addWidget(self._status_label)
        right_layout.addWidget(status_group)

        right_layout.addStretch()
        return right_w

    # ------------------------------------------------------------------
    # File import logic
    # ------------------------------------------------------------------

    def _add_csv_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Sensor Logger CSV Files", "",
            "CSV Files (*.csv);;All Files (*)",
        )
        added = 0
        for p in paths:
            if p not in self._csv_paths:
                self._csv_paths.append(p)
                item = QListWidgetItem(Path(p).name)
                item.setToolTip(p)
                self._file_list.addItem(item)
                added += 1
        if added:
            self._log_line(f"Added {added} CSV file(s).")
        self._update_counts()

    def _add_zip_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select ZIP Archives", "",
            "ZIP Archives (*.zip);;All Files (*)",
        )
        total_added = 0
        for zip_path in paths:
            try:
                tmp = tempfile.mkdtemp(prefix="rdk_sl_")
                self._temp_dirs.append(tmp)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(tmp)
                    csv_names = [
                        n for n in zf.namelist()
                        if n.lower().endswith(".csv") and not n.startswith("__MACOSX")
                    ]
                if not csv_names:
                    QMessageBox.warning(
                        self, "No CSVs in ZIP",
                        f"No CSV files found inside '{Path(zip_path).name}'.",
                    )
                    continue
                for name in csv_names:
                    full = os.path.join(tmp, name)
                    if full not in self._csv_paths and os.path.isfile(full):
                        self._csv_paths.append(full)
                        display = f"{Path(zip_path).name}/{Path(name).name}"
                        item = QListWidgetItem(display)
                        item.setToolTip(full)
                        self._file_list.addItem(item)
                        total_added += 1
            except zipfile.BadZipFile:
                QMessageBox.warning(
                    self, "Invalid ZIP",
                    f"'{Path(zip_path).name}' is not a valid ZIP file.",
                )
            except Exception as exc:
                self._log_line(f"ERROR extracting {Path(zip_path).name}: {exc}")
        if total_added:
            self._log_line(f"Added {total_added} CSV file(s) from ZIP archive(s).")
        self._update_counts()

    def _remove_selected(self) -> None:
        row = self._file_list.currentRow()
        if row < 0:
            return
        item = self._file_list.takeItem(row)
        if item:
            path = item.toolTip()
            if path in self._csv_paths:
                self._csv_paths.remove(path)
        self._update_counts()

    def _clear_all(self) -> None:
        self._csv_paths.clear()
        self._file_list.clear()
        for d in self._temp_dirs:
            shutil.rmtree(d, ignore_errors=True)
        self._temp_dirs.clear()
        self._audit_results.clear()
        self._extracted_sessions.clear()
        self._audit_table.setRowCount(0)
        self._reset_summary()
        self._reset_file_info()
        self._update_counts()
        self._log_line("Cleared all files.")

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", str(Path.home()),
        )
        if path:
            self._output_edit.setText(path)

    def _update_counts(self) -> None:
        n = len(self._csv_paths)
        self._count_label.setText(f"{n} CSV file(s) loaded")
        self._update_run_btn_state()

    def _update_run_btn_state(self) -> None:
        has_files = len(self._csv_paths) > 0
        has_output = bool(self._output_edit.text().strip())
        running = self._worker is not None and self._worker.isRunning()
        self._run_btn.setEnabled(has_files and has_output and not running)

    # ------------------------------------------------------------------
    # File info (right panel)
    # ------------------------------------------------------------------

    def _on_file_selected(self, row: int) -> None:
        if row < 0 or row >= len(self._csv_paths):
            self._reset_file_info()
            return
        path = self._csv_paths[row]
        p = Path(path)

        self._info_labels["path"].setText(p.name)
        self._info_labels["path"].setToolTip(str(p))

        try:
            size_mb = p.stat().st_size / (1024 * 1024)
            self._info_labels["size"].setText(f"{size_mb:.2f} MB")
        except OSError:
            self._info_labels["size"].setText("—")

        # Read header to detect columns
        try:
            with open(path, "r", newline="", errors="replace") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                # Count approximate rows (read up to 10k lines)
                row_count = 0
                for _ in reader:
                    row_count += 1
                    if row_count >= 10000:
                        break
            if header:
                self._info_labels["columns"].setText(", ".join(header[:8]))
                if len(header) > 8:
                    self._info_labels["columns"].setToolTip(", ".join(header))
                # Detect timestamp column
                ts_aliases = {
                    "time", "loggingtime", "seconds_elapsed",
                    "timestamp", "timestamp_ns",
                }
                ts_col = "—"
                for col in header:
                    if col.strip().lower().replace(" ", "") in ts_aliases:
                        ts_col = col.strip()
                        break
                self._info_labels["timestamp_col"].setText(ts_col)
                suffix = "+" if row_count >= 10000 else ""
                self._info_labels["rows"].setText(f"{row_count}{suffix}")
            else:
                self._info_labels["columns"].setText("(empty file)")
                self._info_labels["timestamp_col"].setText("—")
                self._info_labels["rows"].setText("0")
        except Exception:
            self._info_labels["columns"].setText("(read error)")
            self._info_labels["timestamp_col"].setText("—")
            self._info_labels["rows"].setText("—")

    def _reset_file_info(self) -> None:
        for lbl in self._info_labels.values():
            lbl.setText("—")
            lbl.setToolTip("")

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _run_extraction(self) -> None:
        output_dir = self._output_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(
                self, "No Output Directory",
                "Select an output directory first.",
            )
            return
        if not self._csv_paths:
            return

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._timing_bar.start()
        self._audit_table.setRowCount(0)
        self._reset_summary()
        self._log_line(f"Starting extraction of {len(self._csv_paths)} file(s)…")

        self._worker = SensorLoggerWorker(list(self._csv_paths), output_dir)
        self._worker.progress.connect(self._timing_bar.on_progress)
        self._worker.timing.connect(self._timing_bar.on_timing)
        self._worker.status.connect(self._on_status)
        self._worker.result.connect(self._on_result)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _cancel(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(2000)
        self._timing_bar.stop(success=False)
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._status_label.setText("Cancelled")
        self._status_label.setStyleSheet(f"color: {WARNING}; font-size: 11px;")
        self._log_line("Extraction cancelled.")

    @pyqtSlot(str)
    def _on_status(self, msg: str) -> None:
        self._status_label.setText(msg)
        self._status_label.setStyleSheet(f"color: {TEXT}; font-size: 11px;")
        self._log_line(msg)

    @pyqtSlot(object)
    def _on_result(self, result: dict) -> None:
        self._timing_bar.stop(success=True)
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._worker = None

        audit_results = result.get("audit_results", [])
        extracted = result.get("extracted_sessions", [])
        errors = result.get("errors", [])

        self._audit_results = audit_results
        self._extracted_sessions = extracted

        # Populate audit table
        self._audit_table.setRowCount(len(audit_results))
        for i, ar in enumerate(audit_results):
            self._audit_table.setItem(i, 0, QTableWidgetItem(Path(ar.file_path).name))
            self._audit_table.setItem(i, 1, QTableWidgetItem(f"{ar.duration_seconds:.1f}s"))
            self._audit_table.setItem(i, 2, QTableWidgetItem(f"{ar.imu_rate_hz:.0f}"))
            self._audit_table.setItem(i, 3, QTableWidgetItem(f"{ar.gps_rate_hz:.0f}"))

            imu_item = QTableWidgetItem("Yes" if ar.has_imu else "No")
            imu_item.setForeground(QColor(SUCCESS if ar.has_imu else HI))
            self._audit_table.setItem(i, 4, imu_item)

            gps_item = QTableWidgetItem("Yes" if ar.has_gps else "No")
            gps_item.setForeground(QColor(SUCCESS if ar.has_gps else MUTED))
            self._audit_table.setItem(i, 5, gps_item)

            self._audit_table.setItem(i, 6, QTableWidgetItem(f"{ar.file_size_mb:.1f}"))

            issues_text = "; ".join(ar.issues) if ar.issues else "None"
            issues_item = QTableWidgetItem(issues_text)
            if ar.issues:
                issues_item.setForeground(QColor(WARNING))
            self._audit_table.setItem(i, 7, issues_item)

        self._audit_table.resizeColumnsToContents()

        # Populate summary
        total_imu = sum(s.stats.get("imu_count", 0) for s in extracted if hasattr(s, "stats"))
        total_gps = sum(s.stats.get("gps_count", 0) for s in extracted if hasattr(s, "stats"))
        total_dur = sum(s.duration_seconds for s in extracted)
        self._summary_labels["files"].setText(f"{len(extracted)} / {len(self._csv_paths)}")
        self._summary_labels["imu_total"].setText(f"{total_imu:,}")
        self._summary_labels["gps_total"].setText(f"{total_gps:,}")
        mins, secs = divmod(int(total_dur), 60)
        self._summary_labels["duration"].setText(f"{mins}m {secs:02d}s")
        self._summary_labels["errors"].setText(str(len(errors)))
        if errors:
            self._summary_labels["errors"].setStyleSheet(
                f"color: {HI}; font-size: 12px; font-weight: bold;"
            )
        self._no_results_lbl.setVisible(False)

        # Log errors
        for err in errors:
            self._log_line(
                f"ERROR [{err['stage']}] {Path(err['file']).name}: {err['error']}"
            )

        self._status_label.setText(f"Done — {len(extracted)} extracted")
        self._status_label.setStyleSheet(f"color: {SUCCESS}; font-size: 11px;")
        self._log_line(
            f"Extraction complete: {len(extracted)} succeeded, {len(errors)} errors."
        )

    @pyqtSlot(str)
    def _on_error(self, msg: str) -> None:
        self._timing_bar.stop(success=False)
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._worker = None
        self._status_label.setText("Failed")
        self._status_label.setStyleSheet(f"color: {HI}; font-size: 11px;")
        self._log_line(f"FATAL ERROR: {msg}")
        QMessageBox.critical(self, "Extraction Error", msg)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reset_summary(self) -> None:
        for lbl in self._summary_labels.values():
            lbl.setText("—")
            lbl.setStyleSheet(f"color: {TEXT}; font-size: 12px; font-weight: bold;")
        self._no_results_lbl.setVisible(True)

    def _log_line(self, msg: str) -> None:
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        self._log.append(f"[{ts}] {msg}")

    def closeEvent(self, event) -> None:
        for d in self._temp_dirs:
            shutil.rmtree(d, ignore_errors=True)
        super().closeEvent(event)
