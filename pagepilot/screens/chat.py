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

import re
import textwrap
from datetime import datetime

from PySide6.QtCore import Qt, Signal, QSize, QTimer
from PySide6.QtGui import QFont, QKeyEvent, QCursor, QClipboard
from PySide6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QSizePolicy,
    QTextEdit, QVBoxLayout, QWidget,
)

from pagepilot.ai.ollama_client import ChatMessage, OllamaClient, OllamaError, OllamaStatus
from pagepilot.ai.worker import InferenceWorker
from pagepilot.config import (
    MODEL_NAME, NUM_CTX, OLLAMA_HOST, REQUEST_TIMEOUT,
    SYSTEM_PROMPT, MAX_HISTORY_MESSAGES,
)
from pagepilot.db.database import ConversationDB, Message
from pagepilot.theme import Colors, Fonts, Radius, Spacing
from pagepilot.ui.components import IconButton, PlannedBanner


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


# ──────────────────────────────────────────────────────────────────────────────
# Message bubble widget
# ──────────────────────────────────────────────────────────────────────────────

class MessageWidget(QFrame):
    """
    A single chat message — user (right-aligned, tinted) or assistant (left, neutral).
    Assistant messages render Markdown via QTextBrowser.
    Provides Copy and Regenerate actions.
    """

    copy_requested       = Signal(str)          # full message text
    regenerate_requested = Signal()             # ask parent to regenerate this turn

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

        copy_btn.clicked.connect(self._on_copy)
        regen_btn.clicked.connect(self.regenerate_requested)

        self._regen_btn = regen_btn
        h.addWidget(copy_btn)
        h.addWidget(regen_btn)
        h.addStretch()
        return bar

    def _adjust_browser_height(self):
        doc_height = int(self._browser.document().size().height()) + 4
        self._browser.setFixedHeight(max(doc_height, 20))

    # ── Public API ────────────────────────────────────────────────────

    def append_token(self, token: str):
        """Append a streaming token (assistant only)."""
        self._content += token
        self._browser.setHtml(_md_to_html(self._content))
        self._adjust_browser_height()

    def set_content(self, text: str):
        self._content = text
        if self._role == "user":
            self._text_lbl.setText(text)
        else:
            self._browser.setHtml(_md_to_html(text))
            self._adjust_browser_height()

    def set_streaming(self, streaming: bool):
        self._streaming = streaming
        if hasattr(self, "_regen_btn"):
            self._regen_btn.setEnabled(not streaming)

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

    def _on_copy(self):
        QApplication.clipboard().setText(self._content)


# ──────────────────────────────────────────────────────────────────────────────
# Conversation scroll area
# ──────────────────────────────────────────────────────────────────────────────

class ConversationArea(QScrollArea):
    regenerate_requested = Signal(int)   # index of message pair to regenerate

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

    def _scroll_to_bottom(self):
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

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


class Composer(QFrame):
    """Rounded bottom composer bar."""

    send_requested = Signal(str)
    stop_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Composer")
        self.setStyleSheet(
            f"QFrame#Composer {{ background: {Colors.BG_SURFACE}; "
            f"border: 1px solid {Colors.BORDER}; border-radius: {Radius.XL}px; }}"
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(Spacing.LG, 4, Spacing.MD, 4)
        outer.setSpacing(0)

        self._input = _ComposerInput()
        self._input.submit.connect(self.send_requested)
        outer.addWidget(self._input)

        # Bottom toolbar
        toolbar = QWidget()
        toolbar.setStyleSheet("background: transparent;")
        tb_l = QHBoxLayout(toolbar)
        tb_l.setContentsMargins(0, 2, 0, 2)
        tb_l.setSpacing(Spacing.SM)

        # Planned attachment btn
        attach_btn = self._icon_btn("📎", "Attachments — planned in a future stage")
        attach_btn.setEnabled(False)

        # Planned mic btn
        mic_btn = self._icon_btn("🎙", "Voice input — planned in Stage 3")
        mic_btn.setEnabled(False)

        tb_l.addWidget(attach_btn)
        tb_l.addWidget(mic_btn)
        tb_l.addStretch()

        self._send_btn = self._send_button()
        self._stop_btn = self._stop_button()
        self._stop_btn.hide()

        tb_l.addWidget(self._stop_btn)
        tb_l.addWidget(self._send_btn)
        outer.addWidget(toolbar)

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
        self._status_worker: _StatusWorker | None = None
        self._current_conv_id: int | None = None
        self._current_stream_widget: MessageWidget | None = None
        self._pending_msg_id: int | None = None
        self._ollama_status = OllamaStatus()
        self._conv_items: dict[int, ConvListItem] = {}

        # Layout: sidebar | main
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_chat_sidebar())
        root.addWidget(self._build_main_area(), 1)

        self._load_history_list()
        self._check_ollama_status()

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

        # Content: welcome or conversation
        self._welcome  = WelcomePanel(self._ollama_status)
        self._welcome.prompt_selected.connect(self._on_prompt_chip)

        self._convo = ConversationArea()
        self._convo.regenerate_requested.connect(self._on_regenerate)
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
        # Clear existing
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
        # Insert before stretch
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
        self._current_conv_id = None
        self._convo.clear()
        self._convo.hide()
        self._welcome.show()
        self._conv_title_lbl.setText("PagePilot")
        self._highlight_conv(None)
        self._composer.focus_input()

    def _on_conv_selected(self, conv_id: int):
        if conv_id == self._current_conv_id:
            return
        if self._worker and self._worker.isRunning():
            return   # don't switch mid-generation

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
        QTimer.singleShot(50, self._convo._scroll_to_bottom)

    def _on_conv_delete(self, conv_id: int):
        from PySide6.QtWidgets import QMessageBox
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
    # Slots — message sending
    # ──────────────────────────────────────────────────────────────

    def _on_send(self, text: str):
        if not text.strip():
            return
        if self._worker and self._worker.isRunning():
            return

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

        self._start_inference(text)

    def _start_inference(self, _trigger_text: str):
        """Build context and launch the inference worker."""
        if not self._ollama_status.ready:
            # Re-check first
            self._check_ollama_status()
            self._show_error(
                "Ollama is not ready.\n\n"
                + (self._ollama_status.error or "Check that Ollama is running and the model is installed.")
            )
            return

        # Build message list (system + recent history)
        history = self._db.get_messages(self._current_conv_id)
        context: list[ChatMessage] = [ChatMessage("system", SYSTEM_PROMPT)]
        # Keep last MAX_HISTORY_MESSAGES (excluding pending assistant)
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
            # Auto-scroll
            QTimer.singleShot(5, self._convo._scroll_to_bottom)

    def _on_generation_done(self, full_text: str):
        if self._current_stream_widget:
            self._current_stream_widget.set_streaming(False)
        if self._pending_msg_id is not None:
            self._db.update_message(self._pending_msg_id, full_text, "complete")
        self._reset_generation_state()

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

    def _on_stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()

    def _on_regenerate(self, _widget_index: int):
        if self._worker and self._worker.isRunning():
            return
        if self._current_conv_id is None:
            return

        # Remove last assistant message from DB and UI
        self._db.delete_last_assistant_message(self._current_conv_id)
        self._convo.remove_last_assistant()

        # Get last user message content
        messages = self._db.get_messages(self._current_conv_id)
        last_user = next(
            (m for m in reversed(messages) if m.role == "user"), None
        )
        if not last_user:
            return

        # Save new pending assistant slot
        asst_msg = self._db.add_message(
            self._current_conv_id, "assistant", "", status="pending"
        )
        self._pending_msg_id = asst_msg.id
        self._start_inference(last_user.content)

    def _show_error(self, message: str):
        w = self._convo.add_message("assistant", "")
        w.set_error(message)

    # ──────────────────────────────────────────────────────────────
    # Public API (called by MainWindow / Sidebar)
    # ──────────────────────────────────────────────────────────────

    def refresh_status(self):
        """Re-check Ollama. Called when the screen becomes visible."""
        self._check_ollama_status()
