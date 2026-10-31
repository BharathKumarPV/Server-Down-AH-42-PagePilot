"""
PagePilot – Chat screen (Stage 2).

ChatGPT-inspired layout:
  - Centered conversation area with spacious message layout
  - Streaming assistant responses with token-by-token rendering
  - Markdown rendering via QTextBrowser (safe: no script execution)
  - Copy / Regenerate message actions
  - Stop generation button
  - Multi-line composer with Enter-to-send / Shift+Enter-newline
  - Persistent SQLite history
  - Real Qwen2.5:3b inference through Ollama
"""

from __future__ import annotations

import os
import re
import textwrap
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QSize, QTimer, QThread
from PySide6.QtGui import QFont, QKeyEvent, QCursor, QClipboard
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QScrollArea, QSizePolicy,
    QTextEdit, QVBoxLayout, QWidget,
)

from pagepilot.ai.ollama_client import ChatMessage, OllamaClient, OllamaError, OllamaStatus
from pagepilot.ai.worker import InferenceWorker
from pagepilot.ai.rag_service import RAGService, RAGWorker
from pagepilot.config import (
    DOCS_DB_PATH, EMBED_MODEL, MODEL_NAME, NUM_CTX, OLLAMA_HOST, REQUEST_TIMEOUT,
    SYSTEM_PROMPT, MAX_HISTORY_MESSAGES,
    WHISPER_MODEL, WHISPER_LANGUAGE, PIPER_VOICE, PIPER_AUTO_SPEAK,
    ensure_chroma_dir,
)
from pagepilot.db.database import ConversationDB, DocumentDB, Message
from pagepilot.pdf.ingestion_service import IngestionService
from pagepilot.pdf.citation import open_pdf_document
from pagepilot.pdf.models import Citation, Document
from pagepilot.ocr.service import OCRService
from pagepilot.screens.ocr_review_dialog import OCRReviewDialog
from pagepilot.retrieval.embedding_client import EmbeddingClient
from pagepilot.retrieval.hybrid_retriever import HybridRetriever
from pagepilot.retrieval.keyword_search import KeywordSearch
from pagepilot.retrieval.vector_store import VectorStore
from pagepilot.theme import Colors, Fonts, Radius, Spacing
from pagepilot.ui.components import IconButton, PlannedBanner
from pagepilot.voice.recorder import RecorderThread, list_input_devices
from pagepilot.voice.whisper import TranscriptionThread, is_model_installed as is_whisper_installed
from pagepilot.voice.tts import TtsThread, is_voice_installed


# ──────────────────────────────────────────────────────────────────────────────
# Simple markdown → HTML converter (no external deps)
# ──────────────────────────────────────────────────────────────────────────────

def _md_to_html(text: str) -> str:
    """
    Convert a small subset of Markdown to HTML suitable for QTextBrowser.
    Handles: code fences, inline code, headings, bold, lists, paragraphs.
    Does NOT evaluate HTML or scripts from model output — text nodes are escaped.
    """
    import html as _html

    lines = text.split("\n")
    result: list[str] = []
    in_code = False
    code_buf: list[str] = []
    code_lang = ""

    def flush_code():
        nonlocal in_code, code_buf, code_lang
        body = _html.escape("\n".join(code_buf))
        lang_note = f"<span style='color:{Colors.TEXT_DISABLED};font-size:10px'>{code_lang}</span><br/>" if code_lang else ""
        result.append(
            f"<div style='background:{Colors.BG_ELEVATED};border-radius:6px;"
            f"padding:10px 14px;margin:6px 0;overflow-x:auto;position:relative;'>"
            f"{lang_note}"
            f"<pre style='margin:0;font-family:{Fonts.FAMILY_MONO};font-size:12px;"
            f"color:{Colors.TEXT_PRIMARY};white-space:pre;'>{body}</pre></div>"
        )
        in_code = False
        code_buf = []
        code_lang = ""

    def process_inline(s: str) -> str:
        # Inline code
        s = re.sub(
            r"`([^`]+)`",
            lambda m: (
                f"<code style='background:{Colors.BG_ELEVATED};"
                f"font-family:{Fonts.FAMILY_MONO};font-size:12px;"
                f"padding:1px 5px;border-radius:3px;color:{Colors.TEXT_PRIMARY};'>"
                f"{_html.escape(m.group(1))}</code>"
            ),
            s,
        )
        # Bold **text**
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        # Italic *text*
        s = re.sub(r"\*(.+?)\*", r"<i>\1</i>", s)
        return s

    i = 0
    while i < len(lines):
        line = lines[i]

        # Code fence toggle
        if line.startswith("```"):
            if in_code:
                flush_code()
            else:
                in_code = True
                code_lang = line[3:].strip()
            i += 1
            continue

        if in_code:
            code_buf.append(line)
            i += 1
            continue

        raw = _html.escape(line)

        # Headings
        if raw.startswith("### "):
            result.append(f"<h3 style='margin:8px 0 4px;color:{Colors.TEXT_PRIMARY}'>{process_inline(raw[4:])}</h3>")
        elif raw.startswith("## "):
            result.append(f"<h2 style='margin:10px 0 4px;color:{Colors.TEXT_PRIMARY}'>{process_inline(raw[3:])}</h2>")
        elif raw.startswith("# "):
            result.append(f"<h1 style='margin:12px 0 6px;color:{Colors.TEXT_PRIMARY}'>{process_inline(raw[2:])}</h1>")
        # Bullet lists
        elif raw.startswith("- ") or raw.startswith("* "):
            result.append(f"<li style='margin:2px 0;'>{process_inline(raw[2:])}</li>")
        # Numbered lists
        elif re.match(r"^\d+\. ", raw):
            content = re.sub(r"^\d+\. ", "", raw)
            result.append(f"<li style='margin:2px 0;'>{process_inline(content)}</li>")
        # Blank line
        elif raw.strip() == "":
            result.append("<br/>")
        # Normal paragraph line
        else:
            result.append(f"<p style='margin:3px 0;line-height:1.55;'>{process_inline(raw)}</p>")

        i += 1

    if in_code:
        flush_code()

    return (
        f"<html><body style='background:transparent;color:{Colors.TEXT_PRIMARY};"
        f"font-family:{Fonts.FAMILY_PRIMARY};font-size:{Fonts.SIZE_BASE}px;'>"
        + "".join(result)
        + "</body></html>"
    )


def _clean_markdown_for_speech(text: str) -> str:
    """Strip markdown formatting, code blocks, etc. for natural speech synthesis."""
    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", " [code block omitted] ", text)
    # Remove inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Remove headings
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    # Remove bold/italic
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    # Remove links [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove bullet/list markers
    text = re.sub(r"^[\*\-\+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)
    # Collapse multiple spaces/newlines
    text = re.sub(r"\n+", " ", text)
    return text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Message bubble widget
# ──────────────────────────────────────────────────────────────────────────────

class MessageWidget(QFrame):
    """
    A single chat message — user (right-aligned, tinted) or assistant (left, neutral).
    Assistant messages render Markdown via QTextBrowser.
    Provides Copy, Regenerate, and Read Aloud (Stage 3) actions.
    """

    copy_requested       = Signal(str)          # full message text
    regenerate_requested = Signal()             # ask parent to regenerate this turn
    speak_requested      = Signal(str)          # Stage 3: read aloud
    stop_speak_requested = Signal()             # Stage 3: stop speech

    def __init__(
        self,
        role: str,
        content: str = "",
        *,
        msg_id: int | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._role    = role
        self._content = content
        self._msg_id  = msg_id
        self._streaming = False
        self._citations_widget: QWidget | None = None

        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._flush_render)
        self._has_pending_render = False

        is_user = role == "user"
        self.setObjectName("MsgUser" if is_user else "MsgAssistant")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.setStyleSheet("QFrame { border: none; background: transparent; }")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 4, 0, 4)
        outer.setSpacing(0)

        # Row: avatar + bubble
        row = QWidget()
        row.setStyleSheet("background: transparent;")
        row_l = QHBoxLayout(row)
        row_l.setContentsMargins(0, 0, 0, 0)
        row_l.setSpacing(Spacing.MD)

        # Avatar label
        avatar = QLabel("👤" if is_user else "🧭")
        avatar.setFixedSize(32, 32)
        avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        avatar.setStyleSheet(
            f"font-size: 18px; background: {'#3D3875' if is_user else Colors.BG_ELEVATED};"
            f" border-radius: 16px; border: none;"
        )

        # Bubble frame
        bubble = QFrame()
        bubble.setObjectName("Bubble")
        bubble.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        bubble_style = (
            f"QFrame#Bubble {{ background-color: {'#2E2C52' if is_user else Colors.BG_SURFACE};"
            f" border-radius: {Radius.LG}px; border: 1px solid "
            f"{'#44407A' if is_user else Colors.BORDER}; }}"
        )
        bubble.setStyleSheet(bubble_style)

        b_layout = QVBoxLayout(bubble)
        b_layout.setContentsMargins(Spacing.LG, Spacing.MD, Spacing.LG, Spacing.MD)
        b_layout.setSpacing(6)
        self._bubble_layout = b_layout
        self._citations_widget: QWidget | None = None

        # Role header
        header_row = QWidget()
        header_row.setStyleSheet("background: transparent;")
        h_l = QHBoxLayout(header_row)
        h_l.setContentsMargins(0, 0, 0, 0)
        h_l.setSpacing(Spacing.SM)

        role_lbl = QLabel("You" if is_user else "PagePilot")
        role_lbl.setStyleSheet(
            f"font-size: {Fonts.SIZE_XS}px; font-weight: 700; "
            f"color: {'#9B95FF' if is_user else Colors.TEXT_SECONDARY}; "
            f"background: transparent; border: none;"
        )
        h_l.addWidget(role_lbl)
        h_l.addStretch()
        b_layout.addWidget(header_row)

        # Content display
        if is_user:
            self._text_lbl = QLabel()
            self._text_lbl.setWordWrap(True)
            self._text_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self._text_lbl.setStyleSheet(
                f"color: {Colors.TEXT_PRIMARY}; font-size: {Fonts.SIZE_BASE}px; "
                f"background: transparent; border: none; line-height: 1.55;"
            )
            self._text_lbl.setText(content)
            b_layout.addWidget(self._text_lbl)
        else:
            self._browser = QTextEdit()
            self._browser.setReadOnly(True)
            self._browser.setFrameShape(QFrame.Shape.NoFrame)
            self._browser.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
            self._browser.setStyleSheet(
                f"QTextEdit {{ background: transparent; border: none; "
                f"color: {Colors.TEXT_PRIMARY}; font-size: {Fonts.SIZE_BASE}px; "
                f"font-family: {Fonts.FAMILY_PRIMARY}; }}"
            )
            self._browser.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self._browser.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self._browser.document().setDocumentMargin(0)
            self._browser.document().contentsChanged.connect(self._adjust_browser_height)
            if content:
                self._browser.setHtml(_md_to_html(content))
                self._adjust_browser_height()
            b_layout.addWidget(self._browser)

        # Action bar (assistant only)
        if not is_user:
            self._action_bar = self._make_action_bar()
            b_layout.addWidget(self._action_bar)

        if is_user:
            row_l.addStretch()
            row_l.addWidget(bubble, 1)
            row_l.addWidget(avatar)
        else:
            row_l.addWidget(avatar)
            row_l.addWidget(bubble, 1)
            row_l.addStretch()

        outer.addWidget(row)

    def _make_action_bar(self) -> QWidget:
        bar = QWidget()
        bar.setStyleSheet("background: transparent;")
        h = QHBoxLayout(bar)
        h.setContentsMargins(0, 4, 0, 0)
        h.setSpacing(4)

        def _small_btn(label: str, tooltip: str) -> QPushButton:
            btn = QPushButton(label)
            btn.setToolTip(tooltip)
            btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            btn.setStyleSheet(
                f"QPushButton {{ background: transparent; color: {Colors.TEXT_DISABLED}; "
                f"font-size: {Fonts.SIZE_XS}px; border: none; padding: 2px 6px; border-radius: 4px; }}"
                f"QPushButton:hover {{ color: {Colors.TEXT_PRIMARY}; "
                f"background: {Colors.BG_ELEVATED}; }}"
            )
            return btn

        copy_btn  = _small_btn("⎘ Copy",       "Copy response to clipboard")
        regen_btn = _small_btn("↺ Regenerate", "Regenerate this response")
        speak_btn = _small_btn("🔊 Read Aloud", "Read this response aloud")

        copy_btn.clicked.connect(self._on_copy)
        regen_btn.clicked.connect(self.regenerate_requested)
        speak_btn.clicked.connect(self._on_speak_toggle)

        self._regen_btn = regen_btn
        self._speak_btn = speak_btn
        self._is_speaking = False

        h.addWidget(copy_btn)
        h.addWidget(regen_btn)
        h.addWidget(speak_btn)
        h.addStretch()
        return bar

    def _on_speak_toggle(self):
        if self._is_speaking:
            self.stop_speak_requested.emit()
        else:
            self.speak_requested.emit(self._content)

    def set_speaking(self, speaking: bool):
        self._is_speaking = speaking
        if hasattr(self, "_speak_btn"):
            if speaking:
                self._speak_btn.setText("⏹ Stop Speaking")
                self._speak_btn.setToolTip("Stop speech playback")
                self._speak_btn.setStyleSheet(
                    f"QPushButton {{ background: {Colors.BG_ELEVATED}; color: {Colors.ERROR}; "
                    f"font-size: {Fonts.SIZE_XS}px; border: none; padding: 2px 6px; border-radius: 4px; font-weight: 600; }}"
                    f"QPushButton:hover {{ background: #3d2525; color: #ff8888; }}"
                )
            else:
                self._speak_btn.setText("🔊 Read Aloud")
                self._speak_btn.setToolTip("Read this response aloud")
                self._speak_btn.setStyleSheet(
                    f"QPushButton {{ background: transparent; color: {Colors.TEXT_DISABLED}; "
                    f"font-size: {Fonts.SIZE_XS}px; border: none; padding: 2px 6px; border-radius: 4px; }}"
                    f"QPushButton:hover {{ color: {Colors.TEXT_PRIMARY}; "
                    f"background: {Colors.BG_ELEVATED}; }}"
                )

    def _adjust_browser_height(self):
        if hasattr(self, "_browser"):
            doc_height = int(self._browser.document().size().height()) + 4
            self._browser.setFixedHeight(max(doc_height, 20))

    def _flush_render(self):
        if self._has_pending_render and self._role != "user" and hasattr(self, "_browser"):
            self._has_pending_render = False
            self._browser.setHtml(_md_to_html(self._content))
            self._adjust_browser_height()

    # ── Public API ────────────────────────────────────────────────────

    def append_token(self, token: str):
        """Append a streaming token with throttled rendering for responsive UI."""
        self._content += token
        self._has_pending_render = True
        if not self._render_timer.isActive():
            self._render_timer.start(50)  # Smooth 50ms batching (~20 FPS)

    def set_content(self, text: str):
        self._render_timer.stop()
        self._has_pending_render = False
        self._content = text
        if self._role == "user":
            self._text_lbl.setText(text)
        else:
            self._browser.setHtml(_md_to_html(text))
            self._adjust_browser_height()

    def set_streaming(self, streaming: bool):
        self._streaming = streaming
        if not streaming:
            self._render_timer.stop()
            self._flush_render()
        if hasattr(self, "_regen_btn"):
            self._regen_btn.setEnabled(not streaming)
        if hasattr(self, "_speak_btn"):
            self._speak_btn.setEnabled(not streaming)

    def get_content(self) -> str:
        return self._content

    def set_error(self, msg: str):
        self._content = msg
        if self._role != "user":
            self._browser.setHtml(
                f"<p style='color:{Colors.ERROR};font-family:{Fonts.FAMILY_PRIMARY};"
                f"font-size:{Fonts.SIZE_BASE}px'>{msg}</p>"
            )
            self._adjust_browser_height()

    def set_citations(self, citations: list):
        """Display clickable source pills below assistant response."""
        if not citations or self._citations_widget is not None:
            return

        self._citations_widget = QWidget()
        self._citations_widget.setStyleSheet("background: transparent;")
        c_layout = QHBoxLayout(self._citations_widget)
        c_layout.setContentsMargins(0, 4, 0, 2)
        c_layout.setSpacing(6)
        c_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        lbl = QLabel("Sources:")
        lbl.setStyleSheet(f"color: {Colors.TEXT_DISABLED}; font-size: 11px; font-weight: 600; background: transparent; border: none;")
        c_layout.addWidget(lbl)

        for cit in citations:
            fname = getattr(cit, "filename", "Document")
            pnum = getattr(cit, "page_number", 1)
            pill = QPushButton(f"📄 {fname}, p. {pnum}")
            pill.setFixedHeight(24)
            pill.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            chunk_preview = getattr(cit, "chunk_text", "")[:120]
            pill.setToolTip(f"Open {fname} at page {pnum}\nPreview: {chunk_preview}…")
            pill.setStyleSheet(
                f"QPushButton {{ background: {Colors.BG_ELEVATED}; color: {Colors.TEXT_SECONDARY}; "
                f"border: 1px solid {Colors.BORDER}; border-radius: 12px; padding: 2px 8px; font-size: 11px; }}"
                f"QPushButton:hover {{ background: {Colors.ACCENT_MUTED}; color: {Colors.ACCENT}; border-color: {Colors.ACCENT}; }}"
            )
            pill.clicked.connect(lambda _, p=pnum: open_pdf_document("", page_number=p))
            c_layout.addWidget(pill)

        if hasattr(self, "_action_bar") and hasattr(self, "_bubble_layout"):
            idx = self._bubble_layout.indexOf(self._action_bar)
            if idx >= 0:
                self._bubble_layout.insertWidget(idx, self._citations_widget)
            else:
                self._bubble_layout.addWidget(self._citations_widget)
        elif hasattr(self, "_bubble_layout"):
            self._bubble_layout.addWidget(self._citations_widget)

    def _on_copy(self):
        QApplication.clipboard().setText(self._content)


# ──────────────────────────────────────────────────────────────────────────────
# Conversation scroll area
# ──────────────────────────────────────────────────────────────────────────────

class ConversationArea(QScrollArea):
    regenerate_requested = Signal(int)           # index of message pair to regenerate
    speak_requested      = Signal(object, str)   # (MessageWidget, text)
    stop_speak_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        self._container = QWidget()
        self._container.setStyleSheet("background: transparent;")
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(Spacing.XL, Spacing.LG, Spacing.XL, Spacing.LG)
        self._layout.setSpacing(2)
        self._layout.addStretch()
        self.setWidget(self._container)

        self._widgets: list[MessageWidget] = []

    def add_message(self, role: str, content: str = "", msg_id: int | None = None) -> MessageWidget:
        w = MessageWidget(role, content, msg_id=msg_id)
        if role == "assistant":
            idx = len(self._widgets)
            w.regenerate_requested.connect(lambda i=idx: self.regenerate_requested.emit(i))
            w.speak_requested.connect(lambda text, widget=w: self.speak_requested.emit(widget, text))
            w.stop_speak_requested.connect(self.stop_speak_requested)
        self._layout.insertWidget(self._layout.count() - 1, w)
        self._widgets.append(w)
        QTimer.singleShot(10, self._scroll_to_bottom)
        return w

    def remove_last_assistant(self):
        """Remove the last assistant MessageWidget (for regeneration)."""
        for i in range(len(self._widgets) - 1, -1, -1):
            if self._widgets[i]._role == "assistant":
                w = self._widgets.pop(i)
                self._layout.removeWidget(w)
                w.deleteLater()
                return

    def clear(self):
        while self._layout.count() > 1:
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._widgets.clear()

    def last_assistant(self) -> MessageWidget | None:
        for w in reversed(self._widgets):
            if w._role == "assistant":
                return w
        return None

    def _scroll_to_bottom(self, force: bool = False):
        sb = self.verticalScrollBar()
        # If force is True or user is already near bottom (<= 80px), follow stream
        if force or (sb.maximum() - sb.value() <= 80):
            sb.setValue(sb.maximum())

    @property
    def message_count(self) -> int:
        return len(self._widgets)


# ──────────────────────────────────────────────────────────────────────────────
# Composer (multi-line input + buttons)
# ──────────────────────────────────────────────────────────────────────────────

class _ComposerInput(QTextEdit):
    """
    Multi-line text editor for composing messages.
    Enter sends; Shift+Enter inserts a newline.
    """
    submit = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setPlaceholderText("Message PagePilot…  (Enter to send, Shift+Enter for new line)")
        self.setMaximumHeight(160)
        self.setMinimumHeight(48)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.document().contentsChanged.connect(self._auto_resize)
        self.setStyleSheet(
            f"QTextEdit {{ background: transparent; color: {Colors.TEXT_PRIMARY}; "
            f"font-size: {Fonts.SIZE_BASE}px; font-family: {Fonts.FAMILY_PRIMARY}; "
            f"border: none; padding: 8px 0; selection-background-color: {Colors.ACCENT_MUTED}; }}"
        )

    def _auto_resize(self):
        doc_h = int(self.document().size().height()) + 16
        self.setFixedHeight(min(max(doc_h, 48), 160))

    def keyPressEvent(self, event: QKeyEvent):
        if (event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
                and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            text = self.toPlainText().strip()
            if text:
                self.submit.emit(text)
                self.clear()
            return
        super().keyPressEvent(event)


class AttachmentChip(QFrame):
    """
    Visual attachment pill shown inside Composer.
    Supports PDF documents and note images.
    Displays: icon, filename, status (ingesting/ready/pending/failed), review action (if image), and remove button.
    """
    remove_clicked = Signal(str)  # file_path
    action_clicked = Signal(str)  # file_path (e.g. Review OCR)
    diagnostics_clicked = Signal(int)  # doc_id

    def __init__(
        self,
        file_path: str,
        doc_type: str = "pdf",
        status: str = "ready",
        status_text: str = "",
        doc_id: Optional[int] = None,
        page_count: int = 1,
        parent=None,
    ):
        super().__init__(parent)
        self.file_path = file_path
        self.doc_type = doc_type
        self.status = status
        self.doc_id = doc_id
        self.page_count = page_count

        self.setObjectName("AttachmentChip")
        self.setStyleSheet(
            f"QFrame#AttachmentChip {{ background: {Colors.BG_ELEVATED}; "
            f"border: 1px solid {Colors.BORDER}; border-radius: {Radius.MD}px; "
            f"padding: 2px 6px; }}"
        )
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.setSpacing(6)

        # Icon
        icon_str = "📄" if doc_type == "pdf" else "📷"
        self._icon_lbl = QLabel(icon_str)
        self._icon_lbl.setStyleSheet("font-size: 13px; background: transparent; border: none;")
        layout.addWidget(self._icon_lbl)

        # Truncated Filename
        fname = Path(file_path).name
        disp_name = fname if len(fname) <= 24 else fname[:12] + "…" + fname[-9:]
        self._name_lbl = QLabel(disp_name)
        self._name_lbl.setToolTip(f"{fname}\nPath: {file_path}")
        self._name_lbl.setStyleSheet(
            f"color: {Colors.TEXT_PRIMARY}; font-size: {Fonts.SIZE_XS}px; "
            f"font-weight: 600; background: transparent; border: none;"
        )
        layout.addWidget(self._name_lbl)

        # Status Label
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet(
            f"font-size: 10px; background: transparent; border: none;"
        )
        layout.addWidget(self._status_lbl)

        # Action Button (Review OCR for images)
        self._action_btn = QPushButton("🔍 Review")
        self._action_btn.setFixedHeight(20)
        self._action_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._action_btn.setStyleSheet(
            f"QPushButton {{ background: {Colors.ACCENT}; color: white; "
            f"font-size: 10px; font-weight: 600; border: none; border-radius: 4px; padding: 1px 6px; }}"
            f"QPushButton:hover {{ background: {Colors.ACCENT_HOVER}; }}"
        )
        self._action_btn.clicked.connect(lambda: self.action_clicked.emit(self.file_path))
        self._action_btn.hide()
        layout.addWidget(self._action_btn)

        # Diagnostics 'ℹ' button (for PDFs)
        self._diag_btn = QPushButton("ℹ")
        self._diag_btn.setFixedSize(18, 18)
        self._diag_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._diag_btn.setToolTip("View Document Diagnostics & Status")
        self._diag_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {Colors.TEXT_MUTED}; "
            f"font-size: 11px; font-weight: bold; border: 1px solid {Colors.BORDER}; border-radius: 9px; padding: 0px; }}"
            f"QPushButton:hover {{ color: {Colors.ACCENT}; border-color: {Colors.ACCENT}; background: {Colors.BG_SURFACE}; }}"
        )
        self._diag_btn.clicked.connect(self._on_diag_click)
        if doc_type != "pdf" or self.doc_id is None:
            self._diag_btn.hide()
        layout.addWidget(self._diag_btn)

        # Remove '✕' button
        self._remove_btn = QPushButton("✕")
        self._remove_btn.setFixedSize(16, 16)
        self._remove_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._remove_btn.setToolTip("Remove attachment")
        self._remove_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {Colors.TEXT_DISABLED}; "
            f"font-size: 10px; border: none; border-radius: 8px; }}"
            f"QPushButton:hover {{ color: {Colors.ERROR}; background: rgba(255, 100, 100, 0.2); }}"
        )
        self._remove_btn.clicked.connect(lambda: self.remove_clicked.emit(self.file_path))
        layout.addWidget(self._remove_btn)

        self.update_status(status, status_text, doc_id, page_count)

    def _on_diag_click(self):
        if self.doc_id is not None:
            self.diagnostics_clicked.emit(self.doc_id)

    def update_status(
        self,
        status: str,
        status_text: str = "",
        doc_id: Optional[int] = None,
        page_count: int = 1,
    ):
        self.status = status
        if doc_id is not None:
            self.doc_id = doc_id
        self.page_count = page_count

        if hasattr(self, "_diag_btn"):
            if self.doc_type == "pdf" and self.doc_id is not None:
                self._diag_btn.show()
            else:
                self._diag_btn.hide()

        if status == "ingesting":
            self._status_lbl.setText("⏳ Ingesting…")
            self._status_lbl.setStyleSheet(f"color: {Colors.WARNING}; font-size: 10px; background: transparent; border: none;")
            self._action_btn.hide()
        elif status == "ready":
            txt = status_text or (f"✓ {self.page_count}p" if self.doc_type == "pdf" else "✓ Ready")
            self._status_lbl.setText(txt)
            self._status_lbl.setStyleSheet(f"color: {Colors.SUCCESS}; font-size: 10px; background: transparent; border: none;")
            self._action_btn.hide()
        elif status == "pending_review":
            self._status_lbl.setText("Pending Review")
            self._status_lbl.setStyleSheet(f"color: {Colors.WARNING}; font-size: 10px; background: transparent; border: none;")
            self._action_btn.show()
        elif status == "error":
            self._status_lbl.setText("⚠️ Failed")
            self._status_lbl.setStyleSheet(f"color: {Colors.ERROR}; font-size: 10px; background: transparent; border: none;")
            self._action_btn.hide()
        else:
            self._status_lbl.setText(status_text)
            self._action_btn.hide()


class Composer(QFrame):
    """Rounded bottom composer bar with Stage 3 voice input controls and file attachment support."""

    send_requested               = Signal(str)
    stop_requested               = Signal()
    record_start_requested       = Signal()
    record_stop_requested        = Signal()
    record_cancel_requested      = Signal()
    attachment_requested         = Signal()
    attachment_remove_requested  = Signal(str)
    attachment_action_requested  = Signal(str)
    attachment_diagnostics_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Composer")
        self.setStyleSheet(
            f"QFrame#Composer {{ background: {Colors.BG_SURFACE}; "
            f"border: 1px solid {Colors.BORDER}; border-radius: {Radius.XL}px; }}"
        )
        self._is_recording = False
        self._is_processing_voice = False
        self._attachments: dict[str, dict] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(Spacing.LG, 6, Spacing.MD, 6)
        outer.setSpacing(Spacing.XS)

        # Attachments container (horizontal layout above input)
        self._attachments_container = QWidget()
        self._attachments_container.setObjectName("AttachmentsContainer")
        self._attachments_container.setStyleSheet("background: transparent;")
        self._attachments_layout = QHBoxLayout(self._attachments_container)
        self._attachments_layout.setContentsMargins(0, 0, 0, 4)
        self._attachments_layout.setSpacing(Spacing.SM)
        self._attachments_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._attachments_container.hide()
        outer.addWidget(self._attachments_container)

        self._input = _ComposerInput()
        self._input.submit.connect(self.send_requested)
        outer.addWidget(self._input)

        # Bottom toolbar
        toolbar = QWidget()
        toolbar.setStyleSheet("background: transparent;")
        tb_l = QHBoxLayout(toolbar)
        tb_l.setContentsMargins(0, 2, 0, 2)
        tb_l.setSpacing(Spacing.SM)

        # Attachment button (Paperclip)
        self._attach_btn = QPushButton("📎")
        self._attach_btn.setFixedSize(30, 30)
        self._attach_btn.setToolTip("Attach document (PDF) or note image (PNG, JPG, WebP)")
        self._attach_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._attach_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {Colors.TEXT_SECONDARY}; "
            f"font-size: 15px; border: none; border-radius: 15px; }}"
            f"QPushButton:hover {{ background: {Colors.BG_ELEVATED}; color: {Colors.TEXT_PRIMARY}; }}"
            f"QPushButton:disabled {{ color: {Colors.BG_ELEVATED}; }}"
        )
        self._attach_btn.clicked.connect(self.attachment_requested.emit)

        # Microphone button (Stage 3)
        self._mic_btn = QPushButton("🎙")
        self._mic_btn.setFixedSize(30, 30)
        self._mic_btn.setToolTip("Record voice input (Whisper)")
        self._mic_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._mic_btn.setStyleSheet(self._default_mic_style())
        self._mic_btn.clicked.connect(self._on_mic_clicked)

        # Cancel recording button (hidden until recording)
        self._cancel_rec_btn = QPushButton("✕ Cancel")
        self._cancel_rec_btn.setFixedHeight(26)
        self._cancel_rec_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._cancel_rec_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {Colors.TEXT_DISABLED}; "
            f"font-size: {Fonts.SIZE_XS}px; border: 1px solid {Colors.BORDER}; "
            f"border-radius: {Radius.SM}px; padding: 2px 8px; }}"
            f"QPushButton:hover {{ color: {Colors.ERROR}; border-color: {Colors.ERROR}; background: {Colors.BG_ELEVATED}; }}"
        )
        self._cancel_rec_btn.clicked.connect(self.record_cancel_requested)
        self._cancel_rec_btn.hide()

        # Status label during recording or processing
        self._voice_status_lbl = QLabel("")
        self._voice_status_lbl.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_XS}px; "
            f"background: transparent; border: none;"
        )
        self._voice_status_lbl.hide()

        tb_l.addWidget(self._attach_btn)
        tb_l.addWidget(self._mic_btn)
        tb_l.addWidget(self._cancel_rec_btn)
        tb_l.addWidget(self._voice_status_lbl)
        tb_l.addStretch()

        self._send_btn = self._send_button()
        self._stop_btn = self._stop_button()
        self._stop_btn.hide()

        tb_l.addWidget(self._stop_btn)
        tb_l.addWidget(self._send_btn)
        outer.addWidget(toolbar)

    def _default_mic_style(self) -> str:
        return (
            f"QPushButton {{ background: transparent; color: {Colors.TEXT_SECONDARY}; "
            f"font-size: 15px; border: none; border-radius: 15px; }}"
            f"QPushButton:hover {{ background: {Colors.BG_ELEVATED}; color: {Colors.TEXT_PRIMARY}; }}"
            f"QPushButton:disabled {{ color: {Colors.BG_ELEVATED}; }}"
        )

    def _on_mic_clicked(self):
        if self._is_recording:
            self.record_stop_requested.emit()
        elif self._is_processing_voice:
            return
        else:
            self.record_start_requested.emit()

    def set_recording_state(self, state: str, detail: str = ""):
        """state: 'idle' | 'recording' | 'processing' | 'transcribing' | 'error'"""
        st = state.lower()
        if st == "recording":
            self._is_recording = True
            self._is_processing_voice = False
            self._mic_btn.setEnabled(True)
            self._mic_btn.setText("⏹")
            self._mic_btn.setToolTip("Stop recording and transcribe")
            self._mic_btn.setStyleSheet(
                f"QPushButton {{ background: {Colors.ERROR}; color: white; "
                f"font-size: 14px; border: none; border-radius: 15px; }}"
                f"QPushButton:hover {{ background: #C04040; }}"
            )
            self._cancel_rec_btn.show()
            self._voice_status_lbl.setText("🔴 Recording speech… click ⏹ to finish")
            self._voice_status_lbl.show()
        elif st == "processing":
            self._is_recording = False
            self._is_processing_voice = True
            self._mic_btn.setEnabled(False)
            self._mic_btn.setText("⏳")
            self._mic_btn.setToolTip("Processing audio…")
            self._mic_btn.setStyleSheet(
                f"QPushButton {{ background: {Colors.BG_ELEVATED}; color: {Colors.TEXT_PRIMARY}; "
                f"font-size: 14px; border: none; border-radius: 15px; }}"
            )
            self._cancel_rec_btn.hide()
            self._voice_status_lbl.setText(detail or "Saving audio file…")
            self._voice_status_lbl.show()
        elif st == "transcribing":
            self._is_recording = False
            self._is_processing_voice = True
            self._mic_btn.setEnabled(False)
            self._mic_btn.setText("⏳")
            self._mic_btn.setToolTip("Transcribing audio with Whisper…")
            self._mic_btn.setStyleSheet(
                f"QPushButton {{ background: {Colors.BG_ELEVATED}; color: {Colors.TEXT_PRIMARY}; "
                f"font-size: 14px; border: none; border-radius: 15px; }}"
            )
            self._cancel_rec_btn.hide()
            self._voice_status_lbl.setText(detail or "Transcribing with Whisper…")
            self._voice_status_lbl.show()
        elif st == "error":
            self._is_recording = False
            self._is_processing_voice = False
            self._mic_btn.setEnabled(True)
            self._mic_btn.setText("🎙")
            self._mic_btn.setToolTip("Voice error — click to retry")
            self._mic_btn.setStyleSheet(self._default_mic_style())
            self._cancel_rec_btn.hide()
            self._voice_status_lbl.setText(f"⚠ {detail}" if detail else "⚠ Voice error")
            self._voice_status_lbl.show()
        else:  # idle
            self._is_recording = False
            self._is_processing_voice = False
            self._mic_btn.setEnabled(True)
            self._mic_btn.setText("🎙")
            self._mic_btn.setToolTip("Record voice input (Whisper)")
            self._mic_btn.setStyleSheet(self._default_mic_style())
            self._cancel_rec_btn.hide()
            self._voice_status_lbl.hide()

    def has_attachment(self, file_path: str) -> bool:
        norm = os.path.normcase(os.path.abspath(file_path))
        for p in self._attachments:
            if os.path.normcase(os.path.abspath(p)) == norm:
                return True
        return False

    def add_attachment(
        self,
        file_path: str,
        doc_type: str = "pdf",
        status: str = "ready",
        status_text: str = "",
        doc_id: Optional[int] = None,
        page_count: int = 1,
    ):
        if self.has_attachment(file_path):
            return
        chip = AttachmentChip(
            file_path=file_path,
            doc_type=doc_type,
            status=status,
            status_text=status_text,
            doc_id=doc_id,
            page_count=page_count,
            parent=self._attachments_container,
        )
        chip.remove_clicked.connect(self._on_chip_remove)
        chip.action_clicked.connect(self.attachment_action_requested.emit)
        chip.diagnostics_clicked.connect(self.attachment_diagnostics_requested.emit)
        self._attachments[file_path] = {
            "chip": chip,
            "doc_type": doc_type,
            "status": status,
            "doc_id": doc_id,
            "page_count": page_count,
        }
        self._attachments_layout.addWidget(chip)
        self._attachments_container.show()

    def update_attachment(
        self,
        file_path: str,
        status: str,
        status_text: str = "",
        doc_id: Optional[int] = None,
        page_count: int = 1,
    ):
        norm = os.path.normcase(os.path.abspath(file_path))
        for p, data in self._attachments.items():
            if os.path.normcase(os.path.abspath(p)) == norm:
                data["status"] = status
                if doc_id is not None:
                    data["doc_id"] = doc_id
                data["page_count"] = page_count
                data["chip"].update_status(status, status_text, doc_id, page_count)
                break

    def remove_attachment(self, file_path: str):
        norm = os.path.normcase(os.path.abspath(file_path))
        target_key = None
        for p in self._attachments:
            if os.path.normcase(os.path.abspath(p)) == norm:
                target_key = p
                break
        if target_key:
            data = self._attachments.pop(target_key)
            chip = data["chip"]
            self._attachments_layout.removeWidget(chip)
            chip.deleteLater()
            if not self._attachments:
                self._attachments_container.hide()

    def _on_chip_remove(self, file_path: str):
        self.remove_attachment(file_path)
        self.attachment_remove_requested.emit(file_path)

    def get_attachments(self) -> list[dict]:
        res = []
        for p, data in self._attachments.items():
            res.append({
                "file_path": p,
                "doc_type": data["doc_type"],
                "status": data["status"],
                "doc_id": data["doc_id"],
                "page_count": data["page_count"],
                "filename": Path(p).name,
            })
        return res

    def clear_attachments(self):
        for p, data in list(self._attachments.items()):
            chip = data["chip"]
            self._attachments_layout.removeWidget(chip)
            chip.deleteLater()
        self._attachments.clear()
        self._attachments_container.hide()

    def insert_text(self, text: str):
        """Append transcribed text into composer input for review and editing."""
        cur = self._input.toPlainText()
        if cur and not cur.endswith(" "):
            new_text = cur + " " + text
        elif cur:
            new_text = cur + text
        else:
            new_text = text
        self._input.setPlainText(new_text)
        cursor = self._input.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self._input.setTextCursor(cursor)
        self._input.setFocus()

    def _icon_btn(self, emoji: str, tooltip: str) -> QPushButton:
        btn = QPushButton(emoji)
        btn.setFixedSize(30, 30)
        btn.setToolTip(tooltip)
        btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        btn.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {Colors.TEXT_DISABLED}; "
            f"font-size: 15px; border: none; border-radius: 15px; }}"
            f"QPushButton:hover {{ background: {Colors.BG_ELEVATED}; color: {Colors.TEXT_SECONDARY}; }}"
            f"QPushButton:disabled {{ color: {Colors.BG_ELEVATED}; }}"
        )
        return btn

    def _send_button(self) -> QPushButton:
        btn = QPushButton("⬆")
        btn.setFixedSize(34, 34)
        btn.setToolTip("Send message (Enter)")
        btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        btn.setStyleSheet(
            f"QPushButton {{ background: {Colors.ACCENT}; color: white; "
            f"font-size: 16px; border: none; border-radius: 17px; }}"
            f"QPushButton:hover {{ background: {Colors.ACCENT_HOVER}; }}"
            f"QPushButton:pressed {{ background: {Colors.ACCENT_PRESSED}; }}"
            f"QPushButton:disabled {{ background: {Colors.BG_ELEVATED}; color: {Colors.TEXT_DISABLED}; }}"
        )
        btn.clicked.connect(self._on_send_click)
        return btn

    def _stop_button(self) -> QPushButton:
        btn = QPushButton("⏹")
        btn.setFixedSize(34, 34)
        btn.setToolTip("Stop generation")
        btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        btn.setStyleSheet(
            f"QPushButton {{ background: {Colors.ERROR}; color: white; "
            f"font-size: 14px; border: none; border-radius: 17px; }}"
            f"QPushButton:hover {{ background: #C04040; }}"
        )
        btn.clicked.connect(self.stop_requested)
        return btn

    def _on_send_click(self):
        text = self._input.toPlainText().strip()
        if text:
            self.send_requested.emit(text)
            self._input.clear()

    def set_generating(self, generating: bool):
        self._send_btn.setVisible(not generating)
        self._stop_btn.setVisible(generating)
        self._input.setReadOnly(generating)
        if hasattr(self, "_mic_btn"):
            self._mic_btn.setEnabled(not generating)
        if hasattr(self, "_attach_btn"):
            self._attach_btn.setEnabled(not generating)

    def clear(self):
        self._input.clear()

    def focus_input(self):
        self._input.setFocus()


# ──────────────────────────────────────────────────────────────────────────────
# Welcome / empty state
# ──────────────────────────────────────────────────────────────────────────────

class WelcomePanel(QWidget):
    """Shown when no conversation is active."""

    prompt_selected = Signal(str)

    _PROMPTS = [
        "What can you help me with?",
        "Explain how neural networks work",
        "Write a Python function to reverse a string",
        "Summarise the key ideas behind the Turing test",
    ]

    def __init__(self, model_status: OllamaStatus, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(Spacing.MD)

        logo = QLabel("🧭")
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo.setStyleSheet("font-size: 52px; background: transparent; border: none;")

        title = QLabel("What can I help with?")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(Fonts.make(Fonts.SIZE_2XL, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent; border: none;")

        layout.addStretch()
        layout.addWidget(logo)
        layout.addSpacing(Spacing.SM)
        layout.addWidget(title)
        layout.addSpacing(Spacing.LG)

        # Model status badge
        self._status_label = QLabel()
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setStyleSheet(
            f"font-size: {Fonts.SIZE_SM}px; background: transparent; border: none;"
        )
        self._update_status(model_status)
        layout.addWidget(self._status_label)

        layout.addSpacing(Spacing.LG)

        # Prompt suggestion chips
        chips_row = QWidget()
        chips_row.setStyleSheet("background: transparent;")
        chips_l = QHBoxLayout(chips_row)
        chips_l.setContentsMargins(0, 0, 0, 0)
        chips_l.setSpacing(Spacing.SM)
        chips_l.setAlignment(Qt.AlignmentFlag.AlignCenter)

        for prompt in self._PROMPTS:
            chip = self._make_chip(prompt)
            chips_l.addWidget(chip)
        layout.addWidget(chips_row)

        layout.addStretch()

    def _make_chip(self, text: str) -> QPushButton:
        btn = QPushButton(textwrap.shorten(text, width=32, placeholder="…"))
        btn.setToolTip(text)
        btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        btn.setStyleSheet(
            f"QPushButton {{ background: {Colors.BG_SURFACE}; color: {Colors.TEXT_SECONDARY}; "
            f"border: 1px solid {Colors.BORDER}; border-radius: {Radius.LG}px; "
            f"padding: 8px 14px; font-size: {Fonts.SIZE_SM}px; }}"
            f"QPushButton:hover {{ background: {Colors.BG_ELEVATED}; color: {Colors.TEXT_PRIMARY}; "
            f"border-color: {Colors.TEXT_DISABLED}; }}"
        )
        btn.clicked.connect(lambda _=False, p=text: self.prompt_selected.emit(p))
        return btn

    def _update_status(self, status: OllamaStatus):
        if status.ready:
            self._status_label.setText(f"✅  {status.model_name} · ready")
            self._status_label.setStyleSheet(
                f"color: {Colors.SUCCESS}; font-size: {Fonts.SIZE_SM}px; background: transparent; border: none;"
            )
        elif status.server_reachable:
            self._status_label.setText(f"⚠  {status.model_name} · model not installed")
            self._status_label.setStyleSheet(
                f"color: {Colors.WARNING}; font-size: {Fonts.SIZE_SM}px; background: transparent; border: none;"
            )
        else:
            self._status_label.setText("⚠  Ollama is not running")
            self._status_label.setStyleSheet(
                f"color: {Colors.WARNING}; font-size: {Fonts.SIZE_SM}px; background: transparent; border: none;"
            )

    def refresh_status(self, status: OllamaStatus):
        self._update_status(status)


# ──────────────────────────────────────────────────────────────────────────────
# Status check worker (non-blocking)
# ──────────────────────────────────────────────────────────────────────────────

from PySide6.QtCore import QThread, Signal as _Signal

class _StatusWorker(QThread):
    done = _Signal(object)   # OllamaStatus

    def __init__(self, client: OllamaClient, parent=None):
        super().__init__(parent)
        self._client = client

    def run(self):
        self.done.emit(self._client.check_status())


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar conversation list
# ──────────────────────────────────────────────────────────────────────────────

class ConvListItem(QFrame):
    selected   = Signal(int)   # conv_id
    delete_req = Signal(int)   # conv_id

    def __init__(self, conv_id: int, title: str, parent=None):
        super().__init__(parent)
        self.conv_id = conv_id
        self._active = False
        self.setObjectName("ConvItem")
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setFixedHeight(40)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._refresh_style()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(Spacing.SM, 0, Spacing.SM, 0)
        layout.setSpacing(4)

        self._lbl = QLabel(textwrap.shorten(title, width=22, placeholder="…"))
        self._lbl.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_SM}px; "
            f"background: transparent; border: none;"
        )
        self._lbl.setToolTip(title)

        del_btn = QPushButton("✕")
        del_btn.setFixedSize(18, 18)
        del_btn.setToolTip("Delete conversation")
        del_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        del_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {Colors.TEXT_DISABLED}; "
            f"font-size: 9px; border: none; border-radius: 9px; }}"
            f"QPushButton:hover {{ color: {Colors.ERROR}; background: {Colors.BG_ELEVATED}; }}"
        )
        del_btn.clicked.connect(lambda: self.delete_req.emit(self.conv_id))

        layout.addWidget(self._lbl, 1)
        layout.addWidget(del_btn)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.selected.emit(self.conv_id)
        super().mousePressEvent(event)

    def set_active(self, active: bool):
        self._active = active
        self._lbl.setStyleSheet(
            f"color: {Colors.TEXT_PRIMARY if active else Colors.TEXT_SECONDARY}; "
            f"font-size: {Fonts.SIZE_SM}px; "
            f"font-weight: {'600' if active else 'normal'}; "
            f"background: transparent; border: none;"
        )
        self._refresh_style()

    def _refresh_style(self):
        bg = Colors.NAV_ACTIVE_BG if self._active else "transparent"
        self.setStyleSheet(
            f"QFrame#ConvItem {{ background: {bg}; border-radius: {Radius.MD}px; border: none; }}"
            f"QFrame#ConvItem:hover {{ background: {Colors.BG_ELEVATED}; }}"
        )

    def update_title(self, title: str):
        self._lbl.setText(textwrap.shorten(title, width=22, placeholder="…"))
        self._lbl.setToolTip(title)


# ──────────────────────────────────────────────────────────────────────────────
# Main Chat Screen
# ──────────────────────────────────────────────────────────────────────────────

class _AttachmentIngestionWorker(QThread):
    """Background worker for PDF ingestion attached in chat composer."""
    progress  = Signal(str, int, int)
    completed = Signal(dict)
    failed    = Signal(str, str)

    def __init__(self, service: IngestionService, file_path: str, parent=None):
        super().__init__(parent)
        self.service = service
        self.file_path = file_path

    def run(self):
        try:
            doc = self.service.ingest_pdf(
                self.file_path,
                progress_callback=lambda stage, cur, tot: self.progress.emit(stage, cur, tot)
            )
            self.completed.emit({
                "file_path": self.file_path,
                "doc_id": doc.id,
                "page_count": doc.page_count,
                "filename": doc.filename,
            })
        except Exception as exc:
            self.failed.emit(self.file_path, str(exc))


class ChatScreen(QWidget):
    """
    Full Stage 2 chat screen.
    Left pane: new-chat button + conversation history.
    Right pane: conversation + composer.
    """

    # Signal forwarded to MainWindow to request navigation to History screen
    open_history = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ChatScreen")
        self.setStyleSheet(f"background-color: {Colors.BG_APP};")

        # Services
        self._client  = OllamaClient(
            host=OLLAMA_HOST, model=MODEL_NAME,
            num_ctx=NUM_CTX, timeout=REQUEST_TIMEOUT,
        )
        self._db      = ConversationDB()
        self._worker: InferenceWorker | None = None
        self._rag_worker: RAGWorker | None = None
        self._status_worker: _StatusWorker | None = None
        self._recorder: RecorderThread | None = None
        self._transcriber: TranscriptionThread | None = None
        self._tts_thread: TtsThread | None = None
        self._speaking_widget: MessageWidget | None = None
        self._current_conv_id: int | None = None
        self._current_stream_widget: MessageWidget | None = None
        self._pending_msg_id: int | None = None
        self._ollama_status = OllamaStatus()
        self._conv_items: dict[int, ConvListItem] = {}

        # Lazy document and RAG services
        self._doc_db: DocumentDB | None = None
        self._vector_store: VectorStore | None = None
        self._embedding_client: EmbeddingClient | None = None
        self._keyword_search: KeywordSearch | None = None
        self._hybrid_retriever: HybridRetriever | None = None
        self._ingestion_service: IngestionService | None = None
        self._ocr_service: OCRService | None = None
        self._rag_service: RAGService | None = None
        self._attachment_workers: list[_AttachmentIngestionWorker] = []
        self._is_recording_active = False

        # Layout: sidebar | main
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_chat_sidebar())
        root.addWidget(self._build_main_area(), 1)

        self._load_history_list()
        self._check_ollama_status()

    def _get_ingestion_service(self) -> IngestionService:
        if self._ingestion_service is None:
            ensure_chroma_dir()
            self._doc_db = DocumentDB(DOCS_DB_PATH)
            self._vector_store = VectorStore()
            self._embedding_client = EmbeddingClient(host=OLLAMA_HOST, model=EMBED_MODEL)
            self._ingestion_service = IngestionService(
                db=self._doc_db,
                vector_store=self._vector_store,
                embedding_client=self._embedding_client,
            )
        return self._ingestion_service

    def _get_ocr_service(self) -> OCRService:
        if self._ocr_service is None:
            self._get_ingestion_service()
            self._ocr_service = OCRService(
                doc_db=self._doc_db,
                vector_store=self._vector_store,
                embedding_client=self._embedding_client,
            )
        return self._ocr_service

    def _get_rag_service(self) -> RAGService:
        if self._rag_service is None:
            self._get_ingestion_service()
            self._keyword_search = KeywordSearch(db=self._doc_db)
            self._hybrid_retriever = HybridRetriever(
                vector_store=self._vector_store,
                keyword_search=self._keyword_search,
                embedding_client=self._embedding_client,
            )
            self._rag_service = RAGService(
                ollama_client=self._client,
                hybrid_retriever=self._hybrid_retriever,
                db=self._doc_db,
            )
        return self._rag_service

    # ──────────────────────────────────────────────────────────────
    # Build UI
    # ──────────────────────────────────────────────────────────────

    def _build_chat_sidebar(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("ChatSidebar")
        panel.setFixedWidth(240)
        panel.setStyleSheet(
            f"QWidget#ChatSidebar {{ background: {Colors.BG_SIDEBAR}; "
            f"border-right: 1px solid {Colors.BORDER}; }}"
        )

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(Spacing.SM, Spacing.MD, Spacing.SM, Spacing.MD)
        layout.setSpacing(Spacing.SM)

        # New chat button
        new_btn = QPushButton("＋  New Chat")
        new_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        new_btn.setFixedHeight(38)
        new_btn.setStyleSheet(
            f"QPushButton {{ background: {Colors.ACCENT_MUTED}; color: {Colors.TEXT_PRIMARY}; "
            f"border: 1px solid {Colors.ACCENT}; border-radius: {Radius.MD}px; "
            f"font-size: {Fonts.SIZE_SM}px; font-weight: 600; padding: 0 12px; }}"
            f"QPushButton:hover {{ background: {Colors.ACCENT}; }}"
        )
        new_btn.clicked.connect(self._on_new_chat)
        layout.addWidget(new_btn)

        # Search box
        self._search_input = QTextEdit()
        self._search_input.setPlaceholderText("Search conversations…")
        self._search_input.setFixedHeight(34)
        self._search_input.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._search_input.setStyleSheet(
            f"QTextEdit {{ background: {Colors.BG_SURFACE_ALT}; color: {Colors.TEXT_PRIMARY}; "
            f"border: 1px solid {Colors.BORDER}; border-radius: {Radius.MD}px; "
            f"padding: 4px 10px; font-size: {Fonts.SIZE_SM}px; }}"
            f"QTextEdit:focus {{ border-color: {Colors.BORDER_FOCUSED}; }}"
        )
        self._search_input.textChanged.connect(self._on_search_changed)
        layout.addWidget(self._search_input)

        # Section label
        sec = QLabel("RECENT")
        sec.setStyleSheet(
            f"color: {Colors.TEXT_DISABLED}; font-size: {Fonts.SIZE_XS}px; "
            f"font-weight: 700; letter-spacing: 1px; background: transparent; border: none; "
            f"padding: 4px 4px 2px;"
        )
        layout.addWidget(sec)

        # Conversation list scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        self._conv_list_container = QWidget()
        self._conv_list_container.setStyleSheet("background: transparent;")
        self._conv_list_layout = QVBoxLayout(self._conv_list_container)
        self._conv_list_layout.setContentsMargins(0, 0, 0, 0)
        self._conv_list_layout.setSpacing(2)
        self._conv_list_layout.addStretch()

        scroll.setWidget(self._conv_list_container)
        layout.addWidget(scroll, 1)

        return panel

    def _build_main_area(self) -> QWidget:
        main = QWidget()
        main.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(main)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Top bar
        layout.addWidget(self._build_top_bar())

        # Active Document Context Banner
        self._doc_context_banner = QFrame()
        self._doc_context_banner.setStyleSheet(
            f"background-color: {Colors.BG_SURFACE}; border-bottom: 1px solid {Colors.BORDER}; "
            f"padding: 4px 16px;"
        )
        banner_layout = QHBoxLayout(self._doc_context_banner)
        banner_layout.setContentsMargins(16, 4, 16, 4)
        banner_layout.setSpacing(8)
        self._doc_context_lbl = QLabel("")
        self._doc_context_lbl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px; font-weight: 500;")
        banner_layout.addWidget(self._doc_context_lbl)
        banner_layout.addStretch()
        self._doc_context_banner.hide()
        layout.addWidget(self._doc_context_banner)

        # Content: welcome or conversation
        self._welcome  = WelcomePanel(self._ollama_status)
        self._welcome.prompt_selected.connect(self._on_prompt_chip)

        self._convo = ConversationArea()
        self._convo.regenerate_requested.connect(self._on_regenerate)
        self._convo.speak_requested.connect(self._on_speak_requested)
        self._convo.stop_speak_requested.connect(self._stop_tts)
        self._convo.hide()

        content_holder = QWidget()
        content_holder.setStyleSheet("background: transparent;")
        ch_layout = QVBoxLayout(content_holder)
        ch_layout.setContentsMargins(0, 0, 0, 0)
        ch_layout.setSpacing(0)
        ch_layout.addWidget(self._welcome)
        ch_layout.addWidget(self._convo)
        layout.addWidget(content_holder, 1)

        # Composer wrapper (max width centred)
        composer_wrapper = QWidget()
        composer_wrapper.setStyleSheet(
            f"background: {Colors.BG_APP}; border-top: 1px solid {Colors.BORDER};"
        )
        cw_l = QVBoxLayout(composer_wrapper)
        cw_l.setContentsMargins(Spacing.XL, Spacing.MD, Spacing.XL, Spacing.LG)

        self._composer = Composer()
        self._composer.send_requested.connect(self._on_send)
        self._composer.stop_requested.connect(self._on_stop)
        self._composer.record_start_requested.connect(self._on_record_start)
        self._composer.record_stop_requested.connect(self._on_record_stop)
        self._composer.record_cancel_requested.connect(self._on_record_cancel)
        self._composer.attachment_requested.connect(self._on_attach_clicked)
        self._composer.attachment_remove_requested.connect(self._on_remove_attachment)
        self._composer.attachment_action_requested.connect(self._on_review_attachment)
        self._composer.attachment_diagnostics_requested.connect(self._show_document_diagnostics)
        cw_l.addWidget(self._composer)

        note = QLabel(
            "PagePilot uses a local Ollama model. No data leaves your device."
        )
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        note.setStyleSheet(
            f"color: {Colors.TEXT_DISABLED}; font-size: {Fonts.SIZE_XS}px; "
            f"background: transparent; border: none;"
        )
        cw_l.addWidget(note)

        layout.addWidget(composer_wrapper)
        return main

    def _build_top_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(50)
        bar.setObjectName("ChatTopBar")
        bar.setStyleSheet(
            f"QWidget#ChatTopBar {{ background: {Colors.BG_APP}; "
            f"border-bottom: 1px solid {Colors.BORDER}; }}"
        )
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(Spacing.LG, 0, Spacing.LG, 0)

        self._conv_title_lbl = QLabel("PagePilot")
        self._conv_title_lbl.setFont(Fonts.make(Fonts.SIZE_MD, QFont.Weight.DemiBold))
        self._conv_title_lbl.setStyleSheet(
            f"color: {Colors.TEXT_PRIMARY}; background: transparent; border: none;"
        )

        self._model_lbl = QLabel(f"{MODEL_NAME}")
        self._model_lbl.setStyleSheet(
            f"color: {Colors.TEXT_DISABLED}; font-size: {Fonts.SIZE_XS}px; "
            f"background: transparent; border: none;"
        )

        self._status_dot = QLabel("●")
        self._status_dot.setStyleSheet(
            f"color: {Colors.TEXT_DISABLED}; font-size: 10px; "
            f"background: transparent; border: none;"
        )

        layout.addWidget(self._conv_title_lbl)
        layout.addSpacing(Spacing.SM)
        layout.addWidget(self._status_dot)
        layout.addWidget(self._model_lbl)
        layout.addStretch()
        return bar

    # ──────────────────────────────────────────────────────────────
    # Ollama status
    # ──────────────────────────────────────────────────────────────

    def _check_ollama_status(self):
        if self._status_worker and self._status_worker.isRunning():
            return
        self._status_worker = _StatusWorker(self._client, parent=self)
        self._status_worker.done.connect(self._on_status_result)
        self._status_worker.start()

    def _on_status_result(self, status: OllamaStatus):
        self._ollama_status = status
        self._welcome.refresh_status(status)

        if status.ready:
            self._status_dot.setStyleSheet(
                f"color: {Colors.SUCCESS}; font-size: 10px; "
                f"background: transparent; border: none;"
            )
            self._model_lbl.setText(f"{status.model_name} · ready")
        elif status.server_reachable:
            self._status_dot.setStyleSheet(
                f"color: {Colors.WARNING}; font-size: 10px; "
                f"background: transparent; border: none;"
            )
            self._model_lbl.setText(f"{status.model_name} · not installed")
        else:
            self._status_dot.setStyleSheet(
                f"color: {Colors.ERROR}; font-size: 10px; "
                f"background: transparent; border: none;"
            )
            self._model_lbl.setText("Ollama not running")

    # ──────────────────────────────────────────────────────────────
    # Conversation list management
    # ──────────────────────────────────────────────────────────────

    def _load_history_list(self):
        """Load all conversations from DB into the sidebar list."""
        while self._conv_list_layout.count() > 1:
            item = self._conv_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._conv_items.clear()

        convs = self._db.list_conversations()
        for conv in convs:
            self._add_conv_item(conv.id, conv.title)

    def _add_conv_item(self, conv_id: int, title: str) -> ConvListItem:
        item = ConvListItem(conv_id, title)
        item.selected.connect(self._on_conv_selected)
        item.delete_req.connect(self._on_conv_delete)
        self._conv_items[conv_id] = item
        self._conv_list_layout.insertWidget(
            self._conv_list_layout.count() - 1, item
        )
        return item

    def _highlight_conv(self, conv_id: int | None):
        for cid, item in self._conv_items.items():
            item.set_active(cid == conv_id)

    # ──────────────────────────────────────────────────────────────
    # Slots — navigation
    # ──────────────────────────────────────────────────────────────

    def _on_new_chat(self):
        """Clear the view; actual DB record created on first message."""
        self._stop_tts()
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
        if self._rag_worker and self._rag_worker.isRunning():
            self._rag_worker.cancel()
        self._current_conv_id = None
        self._convo.clear()
        self._convo.hide()
        self._welcome.show()
        self._conv_title_lbl.setText("PagePilot")
        self._highlight_conv(None)
        self._composer.clear()
        self._composer.clear_attachments()
        self._update_context_banner()
        self._reset_generation_state()
        self._composer.focus_input()

    def _on_conv_selected(self, conv_id: int):
        if conv_id == self._current_conv_id:
            return
        if self._worker and self._worker.isRunning():
            return
        if self._rag_worker and self._rag_worker.isRunning():
            return

        self._stop_tts()
        self._current_conv_id = conv_id
        self._highlight_conv(conv_id)

        conv = self._db.get_conversation(conv_id)
        if conv:
            self._conv_title_lbl.setText(conv.title)

        messages = self._db.get_messages(conv_id)
        self._convo.clear()
        self._welcome.hide()
        self._convo.show()

        for msg in messages:
            if msg.role == "system":
                continue
            self._convo.add_message(msg.role, msg.content, msg_id=msg.id)

        # Restore attached documents for this conversation
        self._composer.clear_attachments()
        saved_doc_ids = self._db.get_conversation_documents(conv_id)
        if saved_doc_ids:
            doc_db = self._get_doc_db()
            for did in saved_doc_ids:
                d = doc_db.get_document(did)
                if d and Path(d.file_path).exists():
                    st_text = f"✓ {d.page_count}p" if d.status == "ready" else d.status.capitalize()
                    self._composer.add_attachment(
                        file_path=d.file_path,
                        doc_type="pdf",
                        status=d.status,
                        status_text=st_text,
                        doc_id=d.id,
                        page_count=d.page_count,
                    )
        self._update_context_banner()

        QTimer.singleShot(50, self._convo._scroll_to_bottom)

    def _update_context_banner(self):
        attachments = self._composer.get_attachments()
        pdf_attachments = [a for a in attachments if a.get("doc_type") == "pdf"]
        if pdf_attachments:
            items = []
            for a in pdf_attachments:
                p_cnt = a.get("page_count", 1)
                st = a.get("status", "ready")
                status_suffix = "" if st == "ready" else f" [{st}]"
                items.append(f"📄 {a['filename']} ({p_cnt}p{status_suffix})")
            self._doc_context_lbl.setText("Active Context: " + " · ".join(items))
            self._doc_context_banner.show()
        else:
            self._doc_context_banner.hide()

    def _show_document_diagnostics(self, doc_id: int):
        from pagepilot.screens.document_diagnostics_dialog import DocumentDiagnosticsDialog
        dlg = DocumentDiagnosticsDialog(
            doc_id=doc_id,
            doc_db=self._get_doc_db(),
            ingestion_service=self._get_ingestion_service(),
            parent=self,
        )
        dlg.reindexed.connect(self._on_document_reindexed)
        dlg.exec()

    def _on_document_reindexed(self, doc_id: int):
        doc = self._get_doc_db().get_document(doc_id)
        if doc:
            st_text = f"✓ {doc.page_count}p" if doc.status == "ready" else doc.status.capitalize()
            self._composer.update_attachment(
                file_path=doc.file_path,
                status=doc.status,
                status_text=st_text,
                doc_id=doc.id,
                page_count=doc.page_count,
            )
            self._update_context_banner()

    def _on_conv_delete(self, conv_id: int):
        reply = QMessageBox.question(
            self,
            "Delete Conversation",
            "Delete this conversation? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._db.delete_conversation(conv_id)
        if conv_id in self._conv_items:
            item = self._conv_items.pop(conv_id)
            self._conv_list_layout.removeWidget(item)
            item.deleteLater()
        if self._current_conv_id == conv_id:
            self._on_new_chat()

    def _on_search_changed(self):
        query = self._search_input.toPlainText().strip().lower()
        for conv_id, item in self._conv_items.items():
            lbl_text = item._lbl.toolTip().lower()
            item.setVisible(not query or query in lbl_text)

    def _on_prompt_chip(self, text: str):
        """Populate the composer with the chip text (do not auto-send)."""
        self._composer._input.setPlainText(text)
        self._composer._input.setFocus()

    # ──────────────────────────────────────────────────────────────
    # Slots — attachment handling (PDF & Notes OCR)
    # ──────────────────────────────────────────────────────────────

    def _on_attach_clicked(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Attach Document or Note Image",
            "",
            "Supported Files (*.pdf *.png *.jpg *.jpeg *.webp);;PDF Documents (*.pdf);;Images (*.png *.jpg *.jpeg *.webp);;All Files (*.*)",
        )
        if not file_paths:
            return

        for fpath in file_paths:
            path_obj = Path(fpath)
            if not path_obj.exists() or not path_obj.is_file():
                self._show_attachment_error(f"File not found: {fpath}")
                continue

            try:
                size_bytes = path_obj.stat().st_size
            except Exception as exc:
                self._show_attachment_error(f"Cannot access file '{path_obj.name}': {exc}")
                continue

            if size_bytes == 0:
                self._show_attachment_error(f"Cannot attach empty file: {path_obj.name}")
                continue

            ext = path_obj.suffix.lower()
            if ext not in {".pdf", ".png", ".jpg", ".jpeg", ".webp"}:
                self._show_attachment_error(
                    f"Unsupported file format '{ext}'. Supported formats: PDF, PNG, JPG, JPEG, WebP."
                )
                continue

            size_mb = size_bytes / (1024 * 1024)
            if ext == ".pdf" and size_mb > 100:
                self._show_attachment_error(f"PDF exceeds 100 MB limit: {path_obj.name} ({size_mb:.1f} MB)")
                continue
            elif ext != ".pdf" and size_mb > 50:
                self._show_attachment_error(f"Image exceeds 50 MB limit: {path_obj.name} ({size_mb:.1f} MB)")
                continue

            if self._composer.has_attachment(fpath):
                self._show_attachment_error(f"File '{path_obj.name}' is already attached.")
                continue

            if ext == ".pdf":
                self._composer.add_attachment(
                    file_path=fpath,
                    doc_type="pdf",
                    status="ingesting",
                    status_text="Ingesting PDF…",
                )
                worker = _AttachmentIngestionWorker(self._get_ingestion_service(), fpath, parent=self)
                worker.completed.connect(self._on_pdf_ingestion_completed)
                worker.failed.connect(self._on_pdf_ingestion_failed)
                self._attachment_workers.append(worker)
                worker.start()
            else:
                self._composer.add_attachment(
                    file_path=fpath,
                    doc_type="image",
                    status="pending_review",
                    status_text="Importing…",
                )
                try:
                    ocr_svc = self._get_ocr_service()
                    doc = ocr_svc.import_image_note(fpath)
                    self._composer.update_attachment(
                        file_path=fpath,
                        status="pending_review",
                        status_text="Pending Review",
                        doc_id=doc.id,
                        page_count=1,
                    )
                except Exception as exc:
                    self._composer.update_attachment(
                        file_path=fpath,
                        status="error",
                        status_text="Import failed",
                    )
                    self._show_attachment_error(f"Failed to import note image '{path_obj.name}': {exc}")

    def _on_pdf_ingestion_completed(self, data: dict):
        fpath = data["file_path"]
        doc_id = data["doc_id"]
        page_count = data["page_count"]
        self._composer.update_attachment(
            file_path=fpath,
            status="ready",
            status_text=f"✓ Ready ({page_count}p)",
            doc_id=doc_id,
            page_count=page_count,
        )
        self._update_context_banner()
        if self._current_conv_id is not None:
            cur_att = self._composer.get_attachments()
            doc_ids = [a["doc_id"] for a in cur_att if a.get("doc_id")]
            self._db.set_conversation_documents(self._current_conv_id, doc_ids)

    def _on_pdf_ingestion_failed(self, fpath: str, err: str):
        self._composer.update_attachment(
            file_path=fpath,
            status="error",
            status_text="Ingestion failed",
        )
        self._update_context_banner()
        self._show_attachment_error(f"Failed to ingest PDF '{Path(fpath).name}': {err}")

    def _on_remove_attachment(self, fpath: str):
        self._update_context_banner()
        if self._current_conv_id is not None:
            cur_att = self._composer.get_attachments()
            doc_ids = [a["doc_id"] for a in cur_att if a.get("doc_id")]
            self._db.set_conversation_documents(self._current_conv_id, doc_ids)

    def _on_review_attachment(self, fpath: str):
        attachments = self._composer.get_attachments()
        target = next((a for a in attachments if os.path.normcase(os.path.abspath(a["file_path"])) == os.path.normcase(os.path.abspath(fpath))), None)
        if not target or not target.get("doc_id"):
            self._show_attachment_error("Image metadata not found for review.")
            return

        dialog = OCRReviewDialog(
            image_path=fpath,
            document_id=target["doc_id"],
            page_number=1,
            ocr_service=self._get_ocr_service(),
            parent=self,
        )
        if dialog.exec():
            # Promoted and approved
            self._composer.update_attachment(
                file_path=fpath,
                status="ready",
                status_text="✓ Approved & Indexed",
                doc_id=target["doc_id"],
            )
            self._update_context_banner()
            if self._current_conv_id is not None:
                cur_att = self._composer.get_attachments()
                doc_ids = [a["doc_id"] for a in cur_att if a.get("doc_id")]
                self._db.set_conversation_documents(self._current_conv_id, doc_ids)

    def _show_attachment_error(self, message: str):
        QMessageBox.warning(self, "PagePilot — Attachment", message)

    # ──────────────────────────────────────────────────────────────
    # Slots — message sending & RAG / Ollama inference
    # ──────────────────────────────────────────────────────────────

    def _on_send(self, text: str):
        if not text.strip():
            return
        if self._worker and self._worker.isRunning():
            return
        if self._rag_worker and self._rag_worker.isRunning():
            return

        attachments = self._composer.get_attachments()

        # Check if any attachment is still ingesting
        ingesting = [a for a in attachments if a.get("status") == "ingesting"]
        if ingesting:
            self._show_attachment_error(f"Please wait for '{ingesting[0]['filename']}' to finish ingesting before sending.")
            return

        # Check if any image note is unapproved (anti-hallucination guard)
        pending_notes = [a for a in attachments if a.get("doc_type") == "image" and a.get("status") == "pending_review"]
        if pending_notes:
            self._show_attachment_error(
                f"Note image '{pending_notes[0]['filename']}' is pending OCR review.\n"
                f"Click '🔍 Review' on the attachment chip to verify and approve it before using it in chat."
            )
            return

        self._stop_tts()

        # Create DB conversation on first message
        if self._current_conv_id is None:
            title = text[:60].strip()
            conv  = self._db.create_conversation(title)
            self._current_conv_id = conv.id
            self._conv_title_lbl.setText(title)
            item = self._add_conv_item(conv.id, title)
            self._highlight_conv(conv.id)
            self._welcome.hide()
            self._convo.show()

        # Save + display user message
        user_msg = self._db.add_message(self._current_conv_id, "user", text)
        self._convo.add_message("user", text, msg_id=user_msg.id)

        # Save pending assistant message (will update on completion)
        asst_msg = self._db.add_message(
            self._current_conv_id, "assistant", "", status="pending"
        )
        self._pending_msg_id = asst_msg.id

        # Determine if we have attached documents ready for RAG grounding
        ready_doc_ids = [a["doc_id"] for a in attachments if a.get("doc_id") and a.get("status") in ("ready", "approved")]

        if ready_doc_ids:
            self._db.set_conversation_documents(self._current_conv_id, ready_doc_ids)
            self._start_rag_inference(text, ready_doc_ids)
        else:
            self._start_inference(text)

    def _start_rag_inference(self, text: str, doc_ids: list[int]):
        """Run grounded RAG inference on attached document(s)."""
        rag_svc = self._get_rag_service()
        stream_widget = self._convo.add_message("assistant", "")
        stream_widget.set_streaming(True)
        self._current_stream_widget = stream_widget

        self._rag_worker = RAGWorker(
            rag_service=rag_svc,
            question=text,
            document_ids=doc_ids,
            is_verified_mode=True,
            parent=self,
        )
        self._rag_worker.token_received.connect(self._on_token)
        self._rag_worker.citations_ready.connect(self._on_rag_citations)
        self._rag_worker.finished.connect(self._on_rag_generation_done)
        self._rag_worker.error_occurred.connect(self._on_generation_error)
        self._rag_worker.cancelled.connect(self._on_generation_cancelled)
        self._rag_worker.start()

        self._composer.set_generating(True)

    def _on_rag_citations(self, citations: list):
        if self._current_stream_widget and citations:
            self._current_stream_widget.set_citations(citations)

    def _on_rag_generation_done(self, full_text: str, citations: list):
        if self._current_stream_widget:
            self._current_stream_widget.set_streaming(False)
            if citations:
                self._current_stream_widget.set_citations(citations)
            if PIPER_AUTO_SPEAK:
                self._on_speak_requested(self._current_stream_widget, full_text)
        if self._pending_msg_id is not None:
            self._db.update_message(self._pending_msg_id, full_text, "complete")
        self._reset_generation_state()
        QTimer.singleShot(20, lambda: self._convo._scroll_to_bottom(force=True))

    def _start_inference(self, _trigger_text: str):
        """Build context and launch the standard Ollama inference worker."""
        if not self._ollama_status.ready:
            self._check_ollama_status()
            self._show_error(
                "Ollama is not ready.\n\n"
                + (self._ollama_status.error or "Check that Ollama is running and the model is installed.")
            )
            return

        # Build message list (system + recent history)
        history = self._db.get_messages(self._current_conv_id)
        context: list[ChatMessage] = [ChatMessage("system", SYSTEM_PROMPT)]
        relevant = [m for m in history if not (m.role == "assistant" and m.status == "pending")]
        for msg in relevant[-MAX_HISTORY_MESSAGES:]:
            context.append(ChatMessage(msg.role, msg.content))

        # Create streaming widget
        stream_widget = self._convo.add_message("assistant", "")
        stream_widget.set_streaming(True)
        self._current_stream_widget = stream_widget

        # Launch worker
        self._worker = InferenceWorker(self._client, context, parent=self)
        self._worker.token_received.connect(self._on_token)
        self._worker.finished.connect(self._on_generation_done)
        self._worker.error_occurred.connect(self._on_generation_error)
        self._worker.cancelled.connect(self._on_generation_cancelled)
        self._worker.start()

        self._composer.set_generating(True)

    def _on_token(self, token: str):
        if self._current_stream_widget:
            self._current_stream_widget.append_token(token)
            self._convo._scroll_to_bottom(force=False)

    def _on_generation_done(self, full_text: str):
        if self._current_stream_widget:
            self._current_stream_widget.set_streaming(False)
            if PIPER_AUTO_SPEAK:
                self._on_speak_requested(self._current_stream_widget, full_text)
        if self._pending_msg_id is not None:
            self._db.update_message(self._pending_msg_id, full_text, "complete")
        self._reset_generation_state()
        QTimer.singleShot(20, lambda: self._convo._scroll_to_bottom(force=True))

    def _on_generation_error(self, error_msg: str):
        if self._current_stream_widget:
            self._current_stream_widget.set_error(f"Error: {error_msg}")
            self._current_stream_widget.set_streaming(False)
        if self._pending_msg_id is not None:
            self._db.update_message(self._pending_msg_id, f"[Error] {error_msg}", "error")
        self._reset_generation_state()
        self._check_ollama_status()

    def _on_generation_cancelled(self):
        partial = ""
        if self._current_stream_widget:
            partial = self._current_stream_widget.get_content()
            self._current_stream_widget.set_streaming(False)
        if self._pending_msg_id is not None and partial:
            self._db.update_message(self._pending_msg_id, f"[Cancelled] {partial}", "cancelled")
        elif self._pending_msg_id is not None:
            self._db.update_message(self._pending_msg_id, "[Generation cancelled]", "cancelled")
        self._reset_generation_state()

    def _reset_generation_state(self):
        self._current_stream_widget = None
        self._pending_msg_id = None
        self._composer.set_generating(False)
        self._worker = None
        self._rag_worker = None

    def _on_stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
        if self._rag_worker and self._rag_worker.isRunning():
            self._rag_worker.cancel()

    def _on_regenerate(self, _widget_index: int):
        if (self._worker and self._worker.isRunning()) or (self._rag_worker and self._rag_worker.isRunning()):
            return
        if self._current_conv_id is None:
            return

        self._db.delete_last_assistant_message(self._current_conv_id)
        self._convo.remove_last_assistant()

        messages = self._db.get_messages(self._current_conv_id)
        last_user = next(
            (m for m in reversed(messages) if m.role == "user"), None
        )
        if not last_user:
            return

        asst_msg = self._db.add_message(
            self._current_conv_id, "assistant", "", status="pending"
        )
        self._pending_msg_id = asst_msg.id

        attachments = self._composer.get_attachments()
        ready_doc_ids = [a["doc_id"] for a in attachments if a.get("doc_id") and a.get("status") in ("ready", "approved")]
        if ready_doc_ids:
            self._start_rag_inference(last_user.content, ready_doc_ids)
        else:
            self._start_inference(last_user.content)

    def _show_error(self, message: str):
        w = self._convo.add_message("assistant", "")
        w.set_error(message)

    # ──────────────────────────────────────────────────────────────
    # Slots — voice recording & transcription (Stage 3)
    # ──────────────────────────────────────────────────────────────

    def _on_record_start(self):
        self._stop_tts()
        if self._is_recording_active:
            return

        if not is_whisper_installed(WHISPER_MODEL):
            self._composer.set_recording_state(
                "error", f"Whisper model 'ggml-{WHISPER_MODEL}' not found. Download in Settings."
            )
            return

        # Pre-flight check for microphone devices
        devices = list_input_devices()
        if not devices:
            self._composer.set_recording_state(
                "error", "No microphone detected. Check Windows sound settings."
            )
            return

        # Terminate any previously running recorder
        if self._recorder and self._recorder.isRunning():
            self._recorder.stop()
            self._recorder.wait(300)
        self._recorder = None

        self._is_recording_active = True
        self._recorder = RecorderThread(parent=self)
        self._recorder.state_changed.connect(self._on_recorder_state)
        self._recorder.finished_path.connect(self._on_recorder_path)
        self._recorder.error_occurred.connect(self._on_recorder_error)
        self._recorder.start()
        self._composer.set_recording_state("recording")

    def _on_record_stop(self):
        if self._recorder and self._recorder.isRunning():
            self._composer.set_recording_state("processing", "Saving audio…")
            self._recorder.stop()

    def _on_record_cancel(self):
        self._is_recording_active = False
        if self._recorder and self._recorder.isRunning():
            self._recorder.cancel()
        self._composer.set_recording_state("idle")

    def _on_recorder_state(self, state: str):
        if state == "cancelled":
            self._is_recording_active = False
            self._composer.set_recording_state("idle")
        elif state == "recording":
            self._composer.set_recording_state("recording")
        elif state == "stopping":
            self._composer.set_recording_state("processing", "Saving audio…")

    def _on_recorder_error(self, err: str):
        self._is_recording_active = False
        self._composer.set_recording_state("error", err)

    def _on_recorder_path(self, wav_path: str):
        self._is_recording_active = False
        self._composer.set_recording_state("transcribing", "Transcribing with Whisper…")

        if self._transcriber and self._transcriber.isRunning():
            self._transcriber.wait(300)
        self._transcriber = None

        self._transcriber = TranscriptionThread(
            wav_path=wav_path,
            model_name=WHISPER_MODEL,
            language=WHISPER_LANGUAGE,
            parent=self,
        )
        self._transcriber.progress.connect(
            lambda msg: self._composer.set_recording_state("transcribing", msg)
        )
        self._transcriber.transcription_done.connect(
            lambda text, path=wav_path: self._on_transcription_done(text, path)
        )
        self._transcriber.error_occurred.connect(
            lambda err, path=wav_path: self._on_transcription_error(err, path)
        )
        self._transcriber.start()

    def _on_transcription_done(self, text: str, wav_path: str):
        try:
            if os.path.exists(wav_path):
                os.unlink(wav_path)
        except Exception:
            pass
        self._composer.set_recording_state("idle")
        if text and text.strip():
            self._composer.insert_text(text.strip())

    def _on_transcription_error(self, err: str, wav_path: str):
        try:
            if os.path.exists(wav_path):
                os.unlink(wav_path)
        except Exception:
            pass
        self._composer.set_recording_state("error", err)

    # ──────────────────────────────────────────────────────────────
    # Slots — text-to-speech (Stage 3)
    # ──────────────────────────────────────────────────────────────

    def _on_speak_requested(self, widget: MessageWidget, text: str):
        self._stop_tts()
        if not is_voice_installed(PIPER_VOICE):
            self._show_error(
                f"Piper voice '{PIPER_VOICE}' is not installed.\n"
                f"Open Settings → Voice to install voice models."
            )
            return

        clean = _clean_markdown_for_speech(text)
        if not clean:
            return

        self._speaking_widget = widget
        self._tts_thread = TtsThread(clean, voice_name=PIPER_VOICE, parent=self)
        self._tts_thread.state_changed.connect(self._on_tts_state)
        self._tts_thread.error_occurred.connect(self._on_tts_error)
        self._tts_thread.start()

    def _stop_tts(self):
        if self._tts_thread and self._tts_thread.isRunning():
            self._tts_thread.stop()
            self._tts_thread.wait(300)
        if self._speaking_widget:
            self._speaking_widget.set_speaking(False)
            self._speaking_widget = None
        self._tts_thread = None

    def _on_tts_state(self, state: str):
        if state == "playing" and self._speaking_widget:
            self._speaking_widget.set_speaking(True)
        elif state in ("done", "stopped", "error"):
            if self._speaking_widget:
                self._speaking_widget.set_speaking(False)
                self._speaking_widget = None
            self._tts_thread = None

    def _on_tts_error(self, err: str):
        self._stop_tts()
        self._show_error(f"Speech playback error: {err}")

    def closeEvent(self, event):
        """Safely cancel background recording and threads on screen close."""
        self._stop_tts()
        if self._recorder and self._recorder.isRunning():
            self._recorder.cancel()
            self._recorder.wait(300)
        if self._transcriber and self._transcriber.isRunning():
            self._transcriber.wait(300)
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(300)
        if self._rag_worker and self._rag_worker.isRunning():
            self._rag_worker.cancel()
            self._rag_worker.wait(300)
        super().closeEvent(event)

    # ──────────────────────────────────────────────────────────────
    # Public API (called by MainWindow / Sidebar)
    # ──────────────────────────────────────────────────────────────

    def refresh_status(self):
        """Re-check Ollama. Called when the screen becomes visible."""
        self._check_ollama_status()
