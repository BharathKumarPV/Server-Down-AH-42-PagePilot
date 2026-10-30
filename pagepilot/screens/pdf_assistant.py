"""
PagePilot – PDF Assistant Screen (Stage 4).

Full offline PDF intelligence and hybrid Retrieval-Augmented Generation:
- Multi-document library management with SHA-256 deduplication
- Asynchronous document ingestion and indexing without freezing the UI
- Progress reporting for text extraction, chunking, and embedding
- Scope selection: search across all documents or focus on a single PDF
- Real-time token streaming with Qwen2.5 3B grounded strictly on PDF context
- Page-level citation pills with 1-based page numbers
- Document summarization
- Piper TTS integration for "Read Aloud"
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QFont, QCursor, QClipboard
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QProgressBar, QPushButton, QScrollArea, QSizePolicy,
    QSplitter, QTextBrowser, QTextEdit, QVBoxLayout, QWidget,
)

from pagepilot.ai.ollama_client import OllamaClient
from pagepilot.ai.rag_service import RAGService, RAGWorker
from pagepilot.config import (
    DOCS_DB_PATH,
    EMBED_MODEL,
    MODEL_NAME,
    OLLAMA_HOST,
    PIPER_VOICE,
    ensure_chroma_dir,
)
from pagepilot.db.database import DocumentDB
from pagepilot.pdf.citation import open_pdf_document
from pagepilot.pdf.document_manager import DocumentManager
from pagepilot.pdf.ingestion_service import IngestionService
from pagepilot.pdf.models import Citation, Document
from pagepilot.retrieval.embedding_client import EmbeddingClient
from pagepilot.retrieval.hybrid_retriever import HybridRetriever
from pagepilot.retrieval.keyword_search import KeywordSearch
from pagepilot.retrieval.vector_store import VectorStore
from pagepilot.screens.chat import _clean_markdown_for_speech, _md_to_html
from pagepilot.theme import Colors, Fonts, Radius, Spacing
from pagepilot.ui.components import Badge, Card, Divider, IconButton, PrimaryButton, SecondaryButton
from pagepilot.voice.tts import TtsThread, is_voice_installed


# ──────────────────────────────────────────────────────────────────────────────
# Ingestion Worker Thread
# ──────────────────────────────────────────────────────────────────────────────

class _IngestionWorker(QThread):
    progress = Signal(str, int, int)      # stage, current, total
    doc_completed = Signal(object)       # Document
    error = Signal(str)                  # error message

    def __init__(self, service: IngestionService, file_paths: list[str], parent=None):
        super().__init__(parent)
        self.service = service
        self.file_paths = file_paths

    def run(self):
        for path in self.file_paths:
            try:
                def cb(stage, cur, tot):
                    self.progress.emit(stage, cur, tot)

                doc = self.service.ingest_pdf(path, progress_callback=cb)
                self.doc_completed.emit(doc)
            except Exception as exc:
                self.error.emit(str(exc))


# ──────────────────────────────────────────────────────────────────────────────
# Citation Pill Widget
# ──────────────────────────────────────────────────────────────────────────────

class CitationPill(QPushButton):
    """Clickable pill for a citation reference [file.pdf, p. N]."""

    def __init__(self, citation: Citation, file_path: str = "", parent=None):
        label = f"📄 {citation.filename}, p. {citation.page_number}"
        super().__init__(label, parent)
        self.citation = citation
        self.file_path = file_path

        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setToolTip(f"Open {citation.filename} at page {citation.page_number}\n\nPreview:\n{citation.chunk_text[:200]}…")
        self.setFixedHeight(26)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.BG_ELEVATED};
                color: {Colors.ACCENT};
                border: 1px solid {Colors.BORDER};
                border-radius: 13px;
                padding: 0 10px;
                font-size: {Fonts.SIZE_XS}px;
                font-family: {Fonts.FAMILY_PRIMARY};
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {Colors.ACCENT_MUTED};
                border-color: {Colors.ACCENT};
            }}
        """)
        self.clicked.connect(self._on_click)

    def _on_click(self):
        if self.file_path and os.path.exists(self.file_path):
            open_pdf_document(self.file_path, self.citation.page_number)


# ──────────────────────────────────────────────────────────────────────────────
# QA Message Widget
# ──────────────────────────────────────────────────────────────────────────────

class QAMessageWidget(QFrame):
    """Renders a Q&A exchange or answer in the PDF assistant with citations."""

    speak_requested = Signal(str)
    stop_speech_requested = Signal()

    def __init__(self, role: str, text: str = "", citations: list[Citation] | None = None, doc_paths: dict[str, str] | None = None, parent=None):
        super().__init__(parent)
        self.role = role
        self._raw_text = text
        self._citations = citations or []
        self._doc_paths = doc_paths or {}
        self._is_speaking = False

        self.setObjectName("QAMessageWidget")
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(Spacing.MD, Spacing.SM, Spacing.MD, Spacing.SM)
        layout.setSpacing(Spacing.XS)

        # Header with role and actions
        header = QWidget()
        header.setStyleSheet("background: transparent;")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(Spacing.SM)

        role_icon = "👤" if self.role == "user" else "📄"
        role_title = "You" if self.role == "user" else "Document Assistant"
        role_lbl = QLabel(f"{role_icon}  {role_title}")
        role_lbl.setFont(Fonts.make(Fonts.SIZE_SM, QFont.Weight.Bold))
        role_lbl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent; border: none;")
        h_layout.addWidget(role_lbl)
        h_layout.addStretch()

        if self.role == "assistant":
            # TTS button
            self._tts_btn = IconButton("🔊", "Read aloud", size=28)
            self._tts_btn.clicked.connect(self._on_tts_clicked)
            h_layout.addWidget(self._tts_btn)

            # Copy button
            copy_btn = IconButton("📋", "Copy answer", size=28)
            copy_btn.clicked.connect(self._on_copy)
            h_layout.addWidget(copy_btn)

        layout.addWidget(header)

        # Content area
        if self.role == "user":
            user_lbl = QLabel(self._raw_text)
            user_lbl.setWordWrap(True)
            user_lbl.setStyleSheet(
                f"color: {Colors.TEXT_PRIMARY}; font-size: {Fonts.SIZE_BASE}px; "
                f"background: {Colors.BG_SURFACE_ALT}; padding: 10px 14px; "
                f"border-radius: {Radius.MD}px; border: 1px solid {Colors.BORDER};"
            )
            layout.addWidget(user_lbl)
        else:
            self._browser = QTextBrowser()
            self._browser.setOpenExternalLinks(True)
            self._browser.setStyleSheet(f"""
                QTextBrowser {{
                    background-color: {Colors.BG_SURFACE};
                    border: 1px solid {Colors.BORDER};
                    border-radius: {Radius.MD}px;
                    padding: 10px 14px;
                    color: {Colors.TEXT_PRIMARY};
                }}
            """)
            self._browser.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self._browser.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self._browser.document().setDocumentMargin(0)
            self._browser.setHtml(_md_to_html(self._raw_text))
            self._adjust_height()
            layout.addWidget(self._browser)

            # Citations container
            self._citations_box = QWidget()
            self._citations_box.setStyleSheet("background: transparent;")
            self._cit_layout = QHBoxLayout(self._citations_box)
            self._cit_layout.setContentsMargins(0, 4, 0, 0)
            self._cit_layout.setSpacing(Spacing.SM)
            self._refresh_citations()
            layout.addWidget(self._citations_box)

    def set_text(self, text: str):
        self._raw_text = text
        if self.role == "assistant":
            self._browser.setHtml(_md_to_html(text))
            self._adjust_height()

    def set_citations(self, citations: list[Citation], doc_paths: dict[str, str]):
        self._citations = citations
        self._doc_paths = doc_paths
        self._refresh_citations()

    def _refresh_citations(self):
        # Clear existing pills
        while self._cit_layout.count() > 0:
            item = self._cit_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self._citations:
            self._citations_box.hide()
            return

        lbl = QLabel("Sources:")
        lbl.setStyleSheet(f"color: {Colors.TEXT_DISABLED}; font-size: {Fonts.SIZE_XS}px; font-weight: 600;")
        self._cit_layout.addWidget(lbl)

        for cit in self._citations:
            fpath = self._doc_paths.get(cit.filename, "")
            pill = CitationPill(cit, file_path=fpath)
            self._cit_layout.addWidget(pill)

        self._cit_layout.addStretch()
        self._citations_box.show()

    def _adjust_height(self):
        doc = self._browser.document()
        doc.setTextWidth(max(300, self._browser.width()))
        h = int(doc.size().height()) + 24
        self._browser.setFixedHeight(max(50, h))

    def _on_copy(self):
        cb = QApplication.clipboard()
        if cb:
            cb.setText(self._raw_text, QClipboard.Mode.Clipboard)

    def _on_tts_clicked(self):
        if self._is_speaking:
            self.stop_speech_requested.emit()
            self.set_speaking(False)
        else:
            clean = _clean_markdown_for_speech(self._raw_text)
            if clean:
                self.set_speaking(True)
                self.speak_requested.emit(clean)

    def set_speaking(self, speaking: bool):
        self._is_speaking = speaking
        if hasattr(self, "_tts_btn"):
            if speaking:
                self._tts_btn.setText("⏹")
                self._tts_btn.setToolTip("Stop speaking")
            else:
                self._tts_btn.setText("🔊")
                self._tts_btn.setToolTip("Read aloud")


# ──────────────────────────────────────────────────────────────────────────────
# PDF Assistant Screen
# ──────────────────────────────────────────────────────────────────────────────

class PdfScreen(QWidget):
    """
    Main PDF Intelligence Screen (Stage 4).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("PdfScreen")
        self.setStyleSheet(f"background-color: {Colors.BG_APP};")

        # Backend dependencies
        self._db = DocumentDB(DOCS_DB_PATH)
        ensure_chroma_dir()
        self._vector_store = VectorStore()
        self._embedding_client = EmbeddingClient(model=EMBED_MODEL)
        self._keyword_search = KeywordSearch(self._db)
        self._hybrid_retriever = HybridRetriever(
            vector_store=self._vector_store,
            keyword_search=self._keyword_search,
            embedding_client=self._embedding_client,
        )
        self._ollama_client = OllamaClient(host=OLLAMA_HOST, model=MODEL_NAME)
        self._rag_service = RAGService(
            ollama_client=self._ollama_client,
            hybrid_retriever=self._hybrid_retriever,
            db=self._db,
        )
        self._ingestion_service = IngestionService(
            db=self._db,
            vector_store=self._vector_store,
            embedding_client=self._embedding_client,
        )
        self._doc_manager = DocumentManager(self._db, self._vector_store)

        # State
        self._selected_doc_id: Optional[int] = None
        self._ingestion_worker: Optional[_IngestionWorker] = None
        self._rag_worker: Optional[RAGWorker] = None
        self._tts_thread: Optional[TtsThread] = None
        self._active_speaking_widget: Optional[QAMessageWidget] = None
        self._current_assistant_widget: Optional[QAMessageWidget] = None

        self._build_ui()
        self._load_document_list()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header
        root.addWidget(self._build_header())

        # Main splitter (Library on left, Q&A on right)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: {Colors.BORDER};
                width: 1px;
            }}
        """)

        left_panel = self._build_library_panel()
        right_panel = self._build_qa_panel()

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        root.addWidget(splitter, 1)

    # ── Header ───────────────────────────────────────────────────────
    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setFixedHeight(56)
        header.setObjectName("PdfHeader")
        header.setStyleSheet(f"""
            QWidget#PdfHeader {{
                background-color: {Colors.BG_SURFACE};
                border-bottom: 1px solid {Colors.BORDER};
            }}
        """)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(Spacing.LG, 0, Spacing.LG, 0)
        layout.setSpacing(Spacing.MD)

        title = QLabel("📄  PDF Intelligence & Q&A")
        title.setFont(Fonts.make(Fonts.SIZE_MD, QFont.Weight.DemiBold))
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent; border: none;")
        layout.addWidget(title)

        layout.addWidget(Badge("Offline RAG", "accent"))
        layout.addWidget(Badge("Hybrid RRF", "info"))

        layout.addStretch()

        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_SM}px;")
        layout.addWidget(self._status_label)

        return header

    # ── Left Library Panel ───────────────────────────────────────────
    def _build_library_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(320)
        panel.setStyleSheet(f"background-color: {Colors.BG_SURFACE};")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(Spacing.MD, Spacing.MD, Spacing.MD, Spacing.MD)
        layout.setSpacing(Spacing.SM)

        # Import action button
        self._import_btn = PrimaryButton("📂  Import PDF Document(s)")
        self._import_btn.clicked.connect(self._on_import_clicked)
        layout.addWidget(self._import_btn)

        # Progress bar (hidden by default)
        self._progress_card = QFrame()
        self._progress_card.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.BG_ELEVATED};
                border: 1px solid {Colors.BORDER};
                border-radius: {Radius.MD}px;
                padding: 8px;
            }}
        """)
        p_layout = QVBoxLayout(self._progress_card)
        p_layout.setContentsMargins(8, 8, 8, 8)
        p_layout.setSpacing(4)

        self._progress_label = QLabel("Indexing…")
        self._progress_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: {Fonts.SIZE_XS}px; font-weight: 600;")
        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedHeight(6)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {Colors.BG_APP};
                border-radius: 3px;
                border: none;
            }}
            QProgressBar::chunk {{
                background-color: {Colors.ACCENT};
                border-radius: 3px;
            }}
        """)
        p_layout.addWidget(self._progress_label)
        p_layout.addWidget(self._progress_bar)
        self._progress_card.hide()
        layout.addWidget(self._progress_card)

        # Section header & Scope selector
        layout.addSpacing(Spacing.XS)
        lbl = QLabel("DOCUMENT LIBRARY")
        lbl.setStyleSheet(f"color: {Colors.TEXT_DISABLED}; font-size: {Fonts.SIZE_XS}px; font-weight: 700; letter-spacing: 1px;")
        layout.addWidget(lbl)

        # "All Documents" filter button
        self._all_docs_btn = QPushButton("📚  All Documents (Entire Library)")
        self._all_docs_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._all_docs_btn.setFixedHeight(34)
        self._all_docs_btn.setStyleSheet(self._filter_btn_style(active=True))
        self._all_docs_btn.clicked.connect(self._select_all_docs)
        layout.addWidget(self._all_docs_btn)

        # Scrollable list of documents
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background: transparent; border: none;")

        self._doc_list_widget = QWidget()
        self._doc_list_layout = QVBoxLayout(self._doc_list_widget)
        self._doc_list_layout.setContentsMargins(0, 0, 0, 0)
        self._doc_list_layout.setSpacing(Spacing.XS)
        self._doc_list_layout.addStretch()

        scroll.setWidget(self._doc_list_widget)
        layout.addWidget(scroll, 1)

        return panel

    def _filter_btn_style(self, active: bool) -> str:
        if active:
            return f"""
                QPushButton {{
                    background-color: {Colors.ACCENT_MUTED};
                    color: {Colors.ACCENT};
                    border: 1px solid {Colors.ACCENT};
                    border-radius: {Radius.MD}px;
                    padding: 6px 12px;
                    font-size: {Fonts.SIZE_SM}px;
                    font-weight: 600;
                    text-align: left;
                }}
            """
        return f"""
            QPushButton {{
                background-color: transparent;
                color: {Colors.TEXT_SECONDARY};
                border: 1px solid {Colors.BORDER};
                border-radius: {Radius.MD}px;
                padding: 6px 12px;
                font-size: {Fonts.SIZE_SM}px;
                text-align: left;
            }}
            QPushButton:hover {{
                background-color: {Colors.BG_ELEVATED};
                color: {Colors.TEXT_PRIMARY};
            }}
        """

    # ── Right Q&A Area ───────────────────────────────────────────────
    def _build_qa_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Scope Action Toolbar
        toolbar = QWidget()
        toolbar.setFixedHeight(48)
        toolbar.setStyleSheet(f"background-color: {Colors.BG_SURFACE_ALT}; border-bottom: 1px solid {Colors.BORDER};")
        t_layout = QHBoxLayout(toolbar)
        t_layout.setContentsMargins(Spacing.LG, 0, Spacing.LG, 0)
        t_layout.setSpacing(Spacing.MD)

        self._scope_label = QLabel("Scope: All Documents")
        self._scope_label.setFont(Fonts.make(Fonts.SIZE_SM, QFont.Weight.Medium))
        self._scope_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY};")
        t_layout.addWidget(self._scope_label)

        t_layout.addStretch()

        self._summarize_btn = SecondaryButton("✨  Summarize Document")
        self._summarize_btn.setFixedHeight(30)
        self._summarize_btn.setEnabled(False)
        self._summarize_btn.clicked.connect(self._on_summarize_clicked)
        t_layout.addWidget(self._summarize_btn)

        layout.addWidget(toolbar)

        # Messages scroll area
        self._chat_scroll = QScrollArea()
        self._chat_scroll.setWidgetResizable(True)
        self._chat_scroll.setStyleSheet("background: transparent; border: none;")

        self._chat_body = QWidget()
        self._chat_body.setStyleSheet(f"background-color: {Colors.BG_APP};")
        self._chat_layout = QVBoxLayout(self._chat_body)
        self._chat_layout.setContentsMargins(Spacing.LG, Spacing.LG, Spacing.LG, Spacing.LG)
        self._chat_layout.setSpacing(Spacing.MD)

        # Welcome placeholder
        self._welcome_widget = self._build_welcome_widget()
        self._chat_layout.addWidget(self._welcome_widget)
        self._chat_layout.addStretch()

        self._chat_scroll.setWidget(self._chat_body)
        layout.addWidget(self._chat_scroll, 1)

        # Composer bar
        composer_frame = QFrame()
        composer_frame.setStyleSheet(f"background-color: {Colors.BG_SURFACE}; border-top: 1px solid {Colors.BORDER};")
        c_layout = QHBoxLayout(composer_frame)
        c_layout.setContentsMargins(Spacing.LG, Spacing.MD, Spacing.LG, Spacing.MD)
        c_layout.setSpacing(Spacing.SM)

        self._input = QTextEdit()
        self._input.setPlaceholderText("Ask a question about the imported PDF documents… (Press Enter to send)")
        self._input.setFixedHeight(50)
        self._input.setStyleSheet(f"""
            QTextEdit {{
                background-color: {Colors.BG_SURFACE_ALT};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: {Radius.MD}px;
                padding: 10px 14px;
                font-size: {Fonts.SIZE_BASE}px;
                font-family: {Fonts.FAMILY_PRIMARY};
            }}
            QTextEdit:focus {{
                border-color: {Colors.ACCENT};
            }}
        """)
        self._input.installEventFilter(self)
        c_layout.addWidget(self._input, 1)

        self._send_btn = PrimaryButton("Ask")
        self._send_btn.setFixedSize(70, 50)
        self._send_btn.clicked.connect(self._on_send_clicked)
        c_layout.addWidget(self._send_btn)

        self._stop_btn = QPushButton("⏹ Stop")
        self._stop_btn.setFixedSize(80, 50)
        self._stop_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._stop_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.ERROR};
                color: #FFFFFF;
                border-radius: {Radius.MD}px;
                font-weight: 600;
                border: none;
            }}
            QPushButton:hover {{
                background-color: #D32F2F;
            }}
        """)
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        self._stop_btn.hide()
        c_layout.addWidget(self._stop_btn)

        layout.addWidget(composer_frame)

        return panel

    def _build_welcome_widget(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(Spacing.SM)

        icon = QLabel("📑")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet("font-size: 48px; background: transparent; border: none;")

        title = QLabel("Offline PDF Intelligence & RAG")
        title.setFont(Fonts.make(Fonts.SIZE_LG, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY};")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        desc = QLabel(
            "Import PDF documents on the left. PagePilot indexes text locally using "
            "nomic-embed-text and SQLite FTS5.\nAsk questions to get verified answers with "
            "exact page citations."
        )
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_SM}px; max-width: 500px;")

        layout.addWidget(icon)
        layout.addWidget(title)
        layout.addWidget(desc)
        return w

    # ── Document Library Management ───────────────────────────────────

    def _load_document_list(self):
        # Clear existing document cards
        while self._doc_list_layout.count() > 1:
            item = self._doc_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        docs = self._doc_manager.list_documents()

        if not docs:
            empty = QLabel("No documents imported.\nClick 'Import PDF' above.")
            empty.setStyleSheet(f"color: {Colors.TEXT_DISABLED}; font-size: {Fonts.SIZE_SM}px; padding: 16px;")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._doc_list_layout.insertWidget(0, empty)
            self._summarize_btn.setEnabled(False)
            return

        for doc in docs:
            card = self._make_doc_card(doc)
            self._doc_list_layout.insertWidget(self._doc_list_layout.count() - 1, card)

        self._update_scope_label()

    def _make_doc_card(self, doc: Document) -> QFrame:
        card = QFrame()
        is_selected = (self._selected_doc_id == doc.id)

        border_color = Colors.ACCENT if is_selected else Colors.BORDER
        bg_color = Colors.BG_SURFACE_ALT if is_selected else Colors.BG_ELEVATED

        card.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: {Radius.MD}px;
            }}
        """)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        # Title row
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(6)

        title_lbl = QLabel(doc.filename)
        title_lbl.setFont(Fonts.make(Fonts.SIZE_SM, QFont.Weight.DemiBold))
        title_lbl.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; border: none; background: transparent;")
        title_row.addWidget(title_lbl, 1)

        # Delete icon
        del_btn = IconButton("🗑", "Remove document", size=24)
        del_btn.clicked.connect(lambda chk=False, did=doc.id: self._on_delete_doc(did))
        title_row.addWidget(del_btn)
        layout.addLayout(title_row)

        # Meta & status row
        meta_row = QHBoxLayout()
        meta_row.setContentsMargins(0, 0, 0, 0)
        meta_row.setSpacing(6)

        info = f"{doc.page_count} pages • {doc.chunk_count} chunks"
        info_lbl = QLabel(info)
        info_lbl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_XS}px; border: none; background: transparent;")
        meta_row.addWidget(info_lbl, 1)

        badge_variant = "success" if doc.status == "ready" else ("warning" if doc.status == "indexing" else "error")
        badge_text = "Ready" if doc.status == "ready" else ("Indexing" if doc.status == "indexing" else "Error")
        if doc.error_message and "Scanned" in doc.error_message:
            badge_text = "Scanned"
            badge_variant = "warning"

        meta_row.addWidget(Badge(badge_text, badge_variant))
        layout.addLayout(meta_row)

        # Click handler to select doc
        card.mousePressEvent = lambda event, did=doc.id: self._on_doc_selected(did)
        card.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        return card

    def _select_all_docs(self):
        self._selected_doc_id = None
        self._all_docs_btn.setStyleSheet(self._filter_btn_style(active=True))
        self._load_document_list()

    def _on_doc_selected(self, doc_id: int):
        self._selected_doc_id = doc_id
        self._all_docs_btn.setStyleSheet(self._filter_btn_style(active=False))
        self._load_document_list()

    def _update_scope_label(self):
        if self._selected_doc_id is None:
            count = self._doc_manager.count_documents()
            chunks = self._doc_manager.count_chunks()
            self._scope_label.setText(f"Scope: All Documents ({count} PDFs, {chunks} chunks)")
            self._summarize_btn.setEnabled(chunks > 0)
        else:
            doc = self._doc_manager.get_document(self._selected_doc_id)
            if doc:
                self._scope_label.setText(f"Scope: {doc.filename} ({doc.page_count} pages, {doc.chunk_count} chunks)")
                self._summarize_btn.setEnabled(doc.chunk_count > 0)

    # ── Import & Ingestion ───────────────────────────────────────────

    def _on_import_clicked(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select PDF Documents",
            "",
            "PDF Files (*.pdf)",
        )
        if not paths:
            return

        self._start_ingestion(paths)

    def _start_ingestion(self, file_paths: list[str]):
        self._import_btn.setEnabled(False)
        self._progress_card.show()
        self._progress_bar.setValue(0)
        self._progress_label.setText("Starting indexing…")

        self._ingestion_worker = _IngestionWorker(
            service=self._ingestion_service,
            file_paths=file_paths,
            parent=self,
        )
        self._ingestion_worker.progress.connect(self._on_ingestion_progress)
        self._ingestion_worker.doc_completed.connect(self._on_ingestion_doc_completed)
        self._ingestion_worker.error.connect(self._on_ingestion_error)
        self._ingestion_worker.finished.connect(self._on_ingestion_finished)
        self._ingestion_worker.start()

    def _on_ingestion_progress(self, stage: str, current: int, total: int):
        if total > 0:
            pct = int((current / total) * 100)
            self._progress_bar.setValue(pct)
        stage_names = {
            "hashing": "Checking SHA-256…",
            "extracting": f"Extracting pages ({current}/{total})…",
            "embedding": f"Embedding chunks ({current}/{total})…",
            "indexing_vectors": "Storing vectors…",
            "indexing_fts": "Building keyword index…",
            "done": "Done!",
        }
        text = stage_names.get(stage, f"{stage} ({current}/{total})")
        self._progress_label.setText(text)

    def _on_ingestion_doc_completed(self, doc: Document):
        self._load_document_list()

    def _on_ingestion_error(self, err_msg: str):
        self._status_label.setText(f"Error: {err_msg[:40]}")
        self._status_label.setStyleSheet(f"color: {Colors.ERROR};")

    def _on_ingestion_finished(self):
        self._import_btn.setEnabled(True)
        self._progress_card.hide()
        self._load_document_list()

    def _on_delete_doc(self, doc_id: int):
        self._doc_manager.delete_document(doc_id)
        if self._selected_doc_id == doc_id:
            self._selected_doc_id = None
        self._load_document_list()

    # ── Q&A Execution ────────────────────────────────────────────────

    def _on_send_clicked(self):
        query = self._input.toPlainText().strip()
        if not query:
            return
        self._input.clear()
        self._ask(query)

    def _ask(self, question: str):
        if self._rag_worker and self._rag_worker.isRunning():
            return

        # Hide welcome widget if visible
        if self._welcome_widget.isVisible():
            self._welcome_widget.hide()

        # Add user question widget
        user_w = QAMessageWidget(role="user", text=question)
        self._chat_layout.insertWidget(self._chat_layout.count() - 1, user_w)

        # Add empty assistant widget for streaming
        asst_w = QAMessageWidget(role="assistant", text="Searching documents and generating answer…")
        asst_w.speak_requested.connect(self._on_speak)
        asst_w.stop_speech_requested.connect(self._on_stop_speech)
        self._chat_layout.insertWidget(self._chat_layout.count() - 1, asst_w)
        self._current_assistant_widget = asst_w

        # Scroll to bottom
        self._scroll_to_bottom()

        # Update UI buttons
        self._send_btn.hide()
        self._stop_btn.show()

        # Map filenames to paths for citation clicking
        doc_paths = {d.filename: d.file_path for d in self._doc_manager.list_documents()}

        target_doc_ids = [self._selected_doc_id] if self._selected_doc_id is not None else None

        self._rag_worker = RAGWorker(
            rag_service=self._rag_service,
            question=question,
            document_ids=target_doc_ids,
            is_summary=False,
            parent=self,
        )
        self._rag_worker.token_received.connect(self._on_token_received)
        self._rag_worker.citations_ready.connect(lambda cits: asst_w.set_citations(cits, doc_paths))
        self._rag_worker.finished.connect(self._on_rag_finished)
        self._rag_worker.error_occurred.connect(self._on_rag_error)
        self._rag_worker.cancelled.connect(self._on_rag_cancelled)
        self._rag_worker.start()

    def _on_summarize_clicked(self):
        if self._rag_worker and self._rag_worker.isRunning():
            return

        # Determine doc to summarize
        doc_id = self._selected_doc_id
        if doc_id is None:
            docs = self._doc_manager.list_documents()
            if not docs:
                return
            doc_id = docs[0].id

        doc = self._doc_manager.get_document(doc_id)
        if not doc:
            return

        if self._welcome_widget.isVisible():
            self._welcome_widget.hide()

        user_w = QAMessageWidget(role="user", text=f"Summarize document: {doc.filename}")
        self._chat_layout.insertWidget(self._chat_layout.count() - 1, user_w)

        asst_w = QAMessageWidget(role="assistant", text="Generating document summary…")
        asst_w.speak_requested.connect(self._on_speak)
        asst_w.stop_speech_requested.connect(self._on_stop_speech)
        self._chat_layout.insertWidget(self._chat_layout.count() - 1, asst_w)
        self._current_assistant_widget = asst_w

        self._scroll_to_bottom()
        self._send_btn.hide()
        self._stop_btn.show()

        doc_paths = {d.filename: d.file_path for d in self._doc_manager.list_documents()}

        self._rag_worker = RAGWorker(
            rag_service=self._rag_service,
            question="",
            document_ids=[doc_id],
            is_summary=True,
            summary_doc_id=doc_id,
            parent=self,
        )
        self._rag_worker.token_received.connect(self._on_token_received)
        self._rag_worker.citations_ready.connect(lambda cits: asst_w.set_citations(cits, doc_paths))
        self._rag_worker.finished.connect(self._on_rag_finished)
        self._rag_worker.error_occurred.connect(self._on_rag_error)
        self._rag_worker.cancelled.connect(self._on_rag_cancelled)
        self._rag_worker.start()

    def _on_token_received(self, token: str):
        if self._current_assistant_widget and self._rag_worker:
            self._current_assistant_widget.set_text(self._rag_worker._accumulated)
            self._scroll_to_bottom()

    def _on_rag_finished(self, full_text: str, citations: list):
        if self._current_assistant_widget:
            self._current_assistant_widget.set_text(full_text)
            doc_paths = {d.filename: d.file_path for d in self._doc_manager.list_documents()}
            self._current_assistant_widget.set_citations(citations, doc_paths)
        self._reset_generation_ui()

    def _on_rag_error(self, error_msg: str):
        if self._current_assistant_widget:
            self._current_assistant_widget.set_text(f"⚠️ {error_msg}")
        self._reset_generation_ui()

    def _on_rag_cancelled(self):
        if self._current_assistant_widget:
            self._current_assistant_widget.set_text(self._current_assistant_widget._raw_text + " *(Generation stopped)*")
        self._reset_generation_ui()

    def _on_stop_clicked(self):
        if self._rag_worker and self._rag_worker.isRunning():
            self._rag_worker.cancel()

    def _reset_generation_ui(self):
        self._stop_btn.hide()
        self._send_btn.show()
        self._scroll_to_bottom()

    def _scroll_to_bottom(self):
        QApplication.processEvents()
        scrollbar = self._chat_scroll.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    # ── Voice / TTS ──────────────────────────────────────────────────

    def _on_speak(self, text: str):
        sender = self.sender()
        if isinstance(sender, QAMessageWidget):
            self._active_speaking_widget = sender

        self._on_stop_speech()
        self._tts_thread = TtsThread(text=text, voice=PIPER_VOICE, parent=self)
        self._tts_thread.finished_playback.connect(self._on_tts_finished)
        self._tts_thread.error.connect(lambda _: self._on_tts_finished())
        self._tts_thread.start()

    def _on_stop_speech(self):
        if self._tts_thread and self._tts_thread.isRunning():
            self._tts_thread.stop()
            self._tts_thread.wait()
            self._tts_thread = None
        if self._active_speaking_widget:
            self._active_speaking_widget.set_speaking(False)

    def _on_tts_finished(self):
        if self._active_speaking_widget:
            self._active_speaking_widget.set_speaking(False)
            self._active_speaking_widget = None

    # ── Event Filter ─────────────────────────────────────────────────

    def eventFilter(self, obj, event):
        if obj is self._input and event.type() == event.Type.KeyPress:
            key_event = event
            if key_event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if not (key_event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
                    self._on_send_clicked()
                    return True
        return super().eventFilter(obj, event)
