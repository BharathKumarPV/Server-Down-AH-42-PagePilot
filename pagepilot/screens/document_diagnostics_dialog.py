"""
PagePilot – Document Diagnostics & Health Inspector Dialog.

Displays real, non-fabricated ingestion metadata for an attached PDF:
- Filename, path, SHA-256 hash
- Page-level breakdown (total, text-extracted, scanned/image)
- Chunking, ChromaDB embedding, and SQLite FTS5 index counts
- Status badge (Ready, Partially Indexed, Scanned - OCR Required, Failed)
- Error details if any
- Safe, duplicate-free Re-index action
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from pagepilot.db.database import DocumentDB
from pagepilot.pdf.ingestion_service import IngestionService
from pagepilot.pdf.models import (
    DOC_STATUS_EMBEDDING,
    DOC_STATUS_EXTRACTED,
    DOC_STATUS_EXTRACTING,
    DOC_STATUS_FAILED,
    DOC_STATUS_IMPORTED,
    DOC_STATUS_INDEXING,
    DOC_STATUS_PARTIALLY_INDEXED,
    DOC_STATUS_READY,
    DOC_STATUS_SCANNED_OCR_REQUIRED,
    Document,
)
from pagepilot.theme import Colors, Fonts, Radius, Spacing
from pagepilot.ui.components import Badge, Card, Divider, PrimaryButton, SecondaryButton


class _ReindexWorker(QThread):
    progress = Signal(str, int, int)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, ingestion_service: IngestionService, doc_id: int, parent=None):
        super().__init__(parent)
        self.ingestion_service = ingestion_service
        self.doc_id = doc_id

    def run(self):
        try:
            def cb(step: str, current: int, total: int):
                self.progress.emit(step, current, total)

            doc = self.ingestion_service.reindex_document(self.doc_id, progress_callback=cb)
            self.finished.emit(doc)
        except Exception as exc:
            self.error.emit(str(exc))


class StatCard(QFrame):
    """Clean stat card displaying a metric name and numeric value."""

    def __init__(self, label: str, value: str, subtext: str = "", parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f"StatCard {{"
            f"  background-color: {Colors.BG_SURFACE};"
            f"  border: 1px solid {Colors.BORDER};"
            f"  border-radius: {Radius.MD}px;"
            f"  padding: {Spacing.MD}px;"
            f"}}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)

        lbl = QLabel(label.upper())
        lbl.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        lbl.setStyleSheet(f"color: {Colors.TEXT_MUTED}; letter-spacing: 0.5px;")

        self._val_label = QLabel(value)
        self._val_label.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        self._val_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY};")

        layout.addWidget(lbl)
        layout.addWidget(self._val_label)

        if subtext:
            sub = QLabel(subtext)
            sub.setFont(QFont("Segoe UI", 8))
            sub.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
            layout.addWidget(sub)

    def update_value(self, val: str):
        self._val_label.setText(val)


class DocumentDiagnosticsDialog(QDialog):
    """
    Diagnostics dialog showing non-fabricated document metrics and re-indexing controls.
    """

    reindexed = Signal(int)  # doc_id

    def __init__(
        self,
        doc_id: int,
        doc_db: DocumentDB,
        ingestion_service: Optional[IngestionService] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.doc_id = doc_id
        self.doc_db = doc_db
        self.ingestion_service = ingestion_service
        self._worker: Optional[_ReindexWorker] = None

        self.setWindowTitle("Document Diagnostics & Ingestion Status")
        self.setMinimumSize(620, 540)
        self.resize(680, 600)
        self.setStyleSheet(f"background-color: {Colors.BG_BASE}; color: {Colors.TEXT_PRIMARY};")

        self._init_ui()
        self._load_data()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(Spacing.LG, Spacing.LG, Spacing.LG, Spacing.LG)
        main_layout.setSpacing(Spacing.MD)

        # Header Section
        header_layout = QHBoxLayout()
        header_layout.setSpacing(Spacing.MD)

        doc_icon = QLabel("📄")
        doc_icon.setFont(QFont("Segoe UI", 24))
        header_layout.addWidget(doc_icon)

        title_vbox = QVBoxLayout()
        title_vbox.setSpacing(2)
        self._title_label = QLabel("Loading...")
        self._title_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self._title_label.setWordWrap(True)

        self._hash_label = QLabel("")
        self._hash_label.setFont(QFont("Consolas", 9))
        self._hash_label.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
        self._hash_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        title_vbox.addWidget(self._title_label)
        title_vbox.addWidget(self._hash_label)
        header_layout.addLayout(title_vbox, 1)

        self._status_badge = Badge("Unknown", "neutral")
        header_layout.addWidget(self._status_badge, 0, Qt.AlignmentFlag.AlignTop)
        main_layout.addLayout(header_layout)

        # Error banner (hidden unless error exists)
        self._error_banner = QFrame()
        self._error_banner.setStyleSheet(
            f"background-color: #FEF2F2; border: 1px solid #FCA5A5; border-radius: {Radius.SM}px; padding: 10px;"
        )
        eb_layout = QHBoxLayout(self._error_banner)
        eb_layout.setContentsMargins(8, 6, 8, 6)
        self._error_text = QLabel()
        self._error_text.setStyleSheet("color: #991B1B; font-size: 11px;")
        self._error_text.setWordWrap(True)
        eb_layout.addWidget(self._error_text)
        self._error_banner.hide()
        main_layout.addWidget(self._error_banner)

        # Grid of stat cards
        stats_group = QWidget()
        self._grid = QGridLayout(stats_group)
        self._grid.setSpacing(Spacing.SM)
        self._grid.setContentsMargins(0, 0, 0, 0)

        self._card_total_pages = StatCard("Total Pages", "0")
        self._card_text_pages = StatCard("Text Pages", "0", "Extractable by PyMuPDF")
        self._card_scanned_pages = StatCard("Scanned / Image Pages", "0", "Requires OCR")
        self._card_chars = StatCard("Extracted Characters", "0")
        self._card_chunks = StatCard("Generated Chunks", "0", "Token-split chunks")
        self._card_embedded = StatCard("ChromaDB Vectors", "0", "Embedded chunks")
        self._card_indexed = StatCard("Indexed Pages", "0", "Indexed in FTS5/Vectors")

        self._grid.addWidget(self._card_total_pages, 0, 0)
        self._grid.addWidget(self._card_text_pages, 0, 1)
        self._grid.addWidget(self._card_scanned_pages, 0, 2)
        self._grid.addWidget(self._card_chars, 1, 0)
        self._grid.addWidget(self._card_chunks, 1, 1)
        self._grid.addWidget(self._card_embedded, 1, 2)
        self._grid.addWidget(self._card_indexed, 2, 0, 1, 3)

        main_layout.addWidget(stats_group)

        # File path detail
        path_box = QFrame()
        path_box.setStyleSheet(
            f"background-color: {Colors.BG_SURFACE}; border: 1px solid {Colors.BORDER}; "
            f"border-radius: {Radius.SM}px; padding: 8px;"
        )
        pb_layout = QVBoxLayout(path_box)
        pb_layout.setContentsMargins(8, 6, 8, 6)
        pb_layout.setSpacing(2)

        p_title = QLabel("SOURCE FILE LOCATION")
        p_title.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        p_title.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
        self._path_label = QLabel("")
        self._path_label.setFont(QFont("Segoe UI", 9))
        self._path_label.setWordWrap(True)
        self._path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        pb_layout.addWidget(p_title)
        pb_layout.addWidget(self._path_label)
        main_layout.addWidget(path_box)

        # Progress bar (hidden during idle)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setFixedHeight(6)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setStyleSheet(
            f"QProgressBar {{ background-color: {Colors.BG_SURFACE}; border: none; border-radius: 3px; }}"
            f"QProgressBar::chunk {{ background-color: {Colors.ACCENT}; border-radius: 3px; }}"
        )
        self._progress_bar.hide()
        main_layout.addWidget(self._progress_bar)

        self._progress_status = QLabel("")
        self._progress_status.setFont(QFont("Segoe UI", 9))
        self._progress_status.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        self._progress_status.hide()
        main_layout.addWidget(self._progress_status)

        main_layout.addStretch(1)

        # Footer Actions
        footer = QHBoxLayout()
        footer.setSpacing(Spacing.SM)

        self._reindex_btn = SecondaryButton("🔄 Re-index Document")
        self._reindex_btn.setToolTip("Safely re-extract, chunk, and embed this document without duplicate vectors")
        self._reindex_btn.clicked.connect(self._on_reindex_clicked)
        footer.addWidget(self._reindex_btn)

        footer.addStretch(1)

        close_btn = SecondaryButton("Close")
        close_btn.clicked.connect(self.accept)
        footer.addWidget(close_btn)

        main_layout.addLayout(footer)

    def _load_data(self):
        diag = self.doc_db.get_document_diagnostics(self.doc_id)
        if not diag:
            self._title_label.setText(f"Document ID {self.doc_id} not found.")
            return

        self._title_label.setText(diag["filename"])
        sha = diag["sha256_hash"] or ""
        short_sha = f"SHA-256: {sha[:16]}...{sha[-8:]}" if len(sha) > 24 else f"SHA-256: {sha}"
        self._hash_label.setText(short_sha)
        self._hash_label.setToolTip(sha)

        self._path_label.setText(diag["file_path"])

        # Update cards
        self._card_total_pages.update_value(str(diag["total_pages"]))
        self._card_text_pages.update_value(str(diag["extracted_text_pages"]))
        self._card_scanned_pages.update_value(str(diag["scanned_pages"]))
        self._card_chars.update_value(f"{diag['total_characters']:,}")
        self._card_chunks.update_value(str(diag["generated_chunks"]))
        self._card_embedded.update_value(str(diag["embedded_chunks"]))
        self._card_indexed.update_value(f"{diag['indexed_pages']} / {diag['total_pages']} pages")

        # Status badge
        status = diag["status"]
        if status == DOC_STATUS_READY:
            self._status_badge.set_text("Ready")
            self._status_badge.set_variant("success")
        elif status == DOC_STATUS_PARTIALLY_INDEXED:
            self._status_badge.set_text("Partially Indexed")
            self._status_badge.set_variant("warning")
        elif status == DOC_STATUS_SCANNED_OCR_REQUIRED:
            self._status_badge.set_text("Scanned — OCR Required")
            self._status_badge.set_variant("warning")
        elif status in (DOC_STATUS_EXTRACTING, DOC_STATUS_EMBEDDING, DOC_STATUS_INDEXING):
            self._status_badge.set_text(status.capitalize())
            self._status_badge.set_variant("accent")
        elif status == DOC_STATUS_FAILED:
            self._status_badge.set_text("Failed")
            self._status_badge.set_variant("error")
        else:
            self._status_badge.set_text(status.capitalize())
            self._status_badge.set_variant("neutral")

        # Error banner
        err = diag["error_message"]
        if err:
            self._error_text.setText(f"Last Error: {err}")
            self._error_banner.show()
        else:
            self._error_banner.hide()

    def _on_reindex_clicked(self):
        if not self.ingestion_service:
            QMessageBox.information(
                self,
                "Service Unavailable",
                "Ingestion service is not initialized in this window.",
            )
            return

        reply = QMessageBox.question(
            self,
            "Confirm Re-indexing",
            f"Are you sure you want to re-index '{self._title_label.text()}'?\n\n"
            "This will delete existing embeddings and rebuild the document search index from scratch.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._reindex_btn.setEnabled(False)
        self._progress_bar.show()
        self._progress_bar.setValue(0)
        self._progress_status.show()
        self._progress_status.setText("Preparing re-index...")

        self._worker = _ReindexWorker(self.ingestion_service, self.doc_id, self)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.error.connect(self._on_worker_error)
        self._worker.start()

    def _on_worker_progress(self, step: str, current: int, total: int):
        self._progress_status.setText(f"{step} ({current}/{total})")
        if total > 0:
            pct = int((current / total) * 100)
            self._progress_bar.setValue(min(100, max(0, pct)))

    def _on_worker_finished(self, doc: Document):
        self._progress_bar.hide()
        self._progress_status.hide()
        self._reindex_btn.setEnabled(True)
        self._load_data()
        self.reindexed.emit(self.doc_id)
        QMessageBox.information(
            self,
            "Re-indexing Complete",
            f"Successfully re-indexed '{doc.filename}'.\n"
            f"Status: {doc.status}\n"
            f"Chunks: {doc.chunk_count} | Embedded: {doc.embedded_chunks_count}",
        )

    def _on_worker_error(self, err_msg: str):
        self._progress_bar.hide()
        self._progress_status.hide()
        self._reindex_btn.setEnabled(True)
        self._load_data()
        QMessageBox.critical(
            self,
            "Re-indexing Failed",
            f"Failed to re-index document:\n{err_msg}",
        )

