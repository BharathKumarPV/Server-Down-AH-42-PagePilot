"""
PagePilot – Chat screen.

Displays a welcome state, conversation bubbles, a message input bar
with Send and (placeholder) Microphone buttons.

No AI back-end is wired in Stage 1.
"""

from __future__ import annotations

import datetime

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QFont, QKeyEvent
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from pagepilot.theme import Colors, Fonts, Radius, Spacing
from pagepilot.ui.components import (
    IconButton, PlannedBanner, PrimaryButton, TextInput,
)


# ──────────────────────────────────────────────────────────────────────────────
# Individual message bubble
# ──────────────────────────────────────────────────────────────────────────────
class MessageBubble(QFrame):
    """A single chat message rendered as a styled bubble."""

    def __init__(self, text: str, role: str = "user", parent=None):
        """
        role: 'user' | 'assistant'
        """
        super().__init__(parent)
        self.setObjectName("MessageBubble")

        is_user = (role == "user")
        bg = Colors.BUBBLE_USER if is_user else Colors.BUBBLE_ASSISTANT
        border = Colors.ACCENT_MUTED if is_user else Colors.BORDER
        align = Qt.AlignmentFlag.AlignRight if is_user else Qt.AlignmentFlag.AlignLeft

        self.setStyleSheet(f"""
            QFrame#MessageBubble {{
                background-color: {bg};
                border: 1px solid {border};
                border-radius: {Radius.LG}px;
            }}
        """)
        self.setMaximumWidth(680)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        inner = QVBoxLayout(self)
        inner.setContentsMargins(Spacing.MD, Spacing.SM, Spacing.MD, Spacing.SM)
        inner.setSpacing(4)

        # Role label
        role_label = QLabel("You" if is_user else "PagePilot")
        role_label.setStyleSheet(
            f"color: {'#9B95FF' if is_user else Colors.TEXT_DISABLED};"
            f" font-size: {Fonts.SIZE_XS}px; font-weight: 700;"
            f" background: transparent; border: none;"
        )
        inner.addWidget(role_label)

        # Message text
        msg = QLabel(text)
        msg.setWordWrap(True)
        msg.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        msg.setStyleSheet(
            f"color: {Colors.TEXT_PRIMARY}; font-size: {Fonts.SIZE_BASE}px;"
            f" background: transparent; border: none; line-height: 1.5;"
        )
        inner.addWidget(msg)

        # Timestamp
        ts = QLabel(datetime.datetime.now().strftime("%H:%M"))
        ts.setAlignment(Qt.AlignmentFlag.AlignRight)
        ts.setStyleSheet(
            f"color: {Colors.TEXT_DISABLED}; font-size: {Fonts.SIZE_XS}px;"
            f" background: transparent; border: none;"
        )
        inner.addWidget(ts)


# ──────────────────────────────────────────────────────────────────────────────
# Conversation scroll area
# ──────────────────────────────────────────────────────────────────────────────
class ConversationArea(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet(f"background: transparent; border: none;")

        self._container = QWidget()
        self._container.setStyleSheet("background: transparent;")
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(Spacing.XL, Spacing.LG, Spacing.XL, Spacing.LG)
        self._layout.setSpacing(Spacing.MD)
        self._layout.addStretch()

        self.setWidget(self._container)
        self._message_count = 0

    def add_message(self, text: str, role: str = "user"):
        bubble = MessageBubble(text, role)
        align = (
            Qt.AlignmentFlag.AlignRight
            if role == "user"
            else Qt.AlignmentFlag.AlignLeft
        )
        # Wrap in a row widget to push bubble to the correct side
        row = QWidget()
        row.setStyleSheet("background: transparent;")
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        if role == "user":
            row_layout.addStretch()
            row_layout.addWidget(bubble)
        else:
            row_layout.addWidget(bubble)
            row_layout.addStretch()

        # Insert before the trailing stretch
        self._layout.insertWidget(self._layout.count() - 1, row)
        self._message_count += 1

        # Scroll to bottom
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
        return bubble

    def clear_messages(self):
        # Remove all items except the trailing stretch
        while self._layout.count() > 1:
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._message_count = 0

    @property
    def has_messages(self) -> bool:
        return self._message_count > 0


# ──────────────────────────────────────────────────────────────────────────────
# Message input bar
# ──────────────────────────────────────────────────────────────────────────────
class _EnterTrapInput(TextInput):
    """TextInput that emits a signal when Enter is pressed (Shift+Enter = newline)."""
    submit = Signal(str)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Return and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            text = self.text().strip()
            if text:
                self.submit.emit(text)
                self.clear()
        else:
            super().keyPressEvent(event)


class InputBar(QWidget):
    """Bottom input row: mic + text field + send button."""

    send_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("InputBar")
        self.setStyleSheet(f"""
            QWidget#InputBar {{
                background-color: {Colors.BG_SURFACE};
                border-top: 1px solid {Colors.BORDER};
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(Spacing.LG, Spacing.MD, Spacing.LG, Spacing.MD)
        layout.setSpacing(Spacing.SM)

        # Microphone (placeholder)
        self._mic_btn = IconButton("🎙", "Voice input — coming in a future stage", size=40)
        self._mic_btn.setEnabled(False)
        self._mic_btn.setStyleSheet(self._mic_btn.styleSheet() + f"""
            QPushButton:disabled {{
                color: {Colors.TEXT_DISABLED};
                background: transparent;
                border: 1px dashed {Colors.BORDER};
                border-radius: 20px;
            }}
        """)

        # Text field
        self._input = _EnterTrapInput("Ask PagePilot something… (Enter to send)")
        self._input.submit.connect(self._on_send)

        # Send button
        self._send_btn = PrimaryButton("Send ➤")
        self._send_btn.setFixedWidth(90)
        self._send_btn.clicked.connect(self._on_send_clicked)

        layout.addWidget(self._mic_btn)
        layout.addWidget(self._input, 1)
        layout.addWidget(self._send_btn)

    def _on_send_clicked(self):
        text = self._input.text().strip()
        if text:
            self.send_requested.emit(text)
            self._input.clear()

    def _on_send(self, text: str):
        self.send_requested.emit(text)


# ──────────────────────────────────────────────────────────────────────────────
# Welcome / empty state
# ──────────────────────────────────────────────────────────────────────────────
class WelcomePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(Spacing.MD)

        logo = QLabel("🧭")
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo.setStyleSheet("font-size: 56px; background: transparent; border: none;")

        title = QLabel("PagePilot")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(Fonts.make(Fonts.SIZE_3XL, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent; border: none;")

        subtitle = QLabel("Your offline-first AI assistant")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setFont(Fonts.make(Fonts.SIZE_MD))
        subtitle.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent; border: none;")

        hint = QLabel("Type a message below to start a conversation.")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet(f"color: {Colors.TEXT_DISABLED}; font-size: {Fonts.SIZE_SM}px; background: transparent; border: none;")

        stage_note = PlannedBanner("AI inference is not connected in Stage 1. Messages will not receive AI responses.")

        layout.addStretch()
        layout.addWidget(logo)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(Spacing.SM)
        layout.addWidget(hint)
        layout.addSpacing(Spacing.LG)
        layout.addWidget(stage_note)
        layout.addStretch()


# ──────────────────────────────────────────────────────────────────────────────
# Chat Screen (top-level widget for the Chat page)
# ──────────────────────────────────────────────────────────────────────────────
class ChatScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ChatScreen")
        self.setStyleSheet(f"background-color: {Colors.BG_APP};")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Top header bar ────────────────────────────────────────────
        header = self._build_header()
        root.addWidget(header)

        # ── Stacked content: welcome or conversation ──────────────────
        self._welcome = WelcomePanel()
        self._convo = ConversationArea()
        self._convo.hide()

        # Container that holds both (only one visible at a time)
        content_container = QWidget()
        content_container.setStyleSheet("background: transparent;")
        self._content_layout = QVBoxLayout(content_container)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(0)
        self._content_layout.addWidget(self._welcome)
        self._content_layout.addWidget(self._convo)

        root.addWidget(content_container, 1)

        # ── Input bar ─────────────────────────────────────────────────
        self._input_bar = InputBar()
        self._input_bar.send_requested.connect(self._on_send)
        root.addWidget(self._input_bar)

    # ── header ───────────────────────────────────────────────────────
    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setFixedHeight(56)
        header.setObjectName("ChatHeader")
        header.setStyleSheet(f"""
            QWidget#ChatHeader {{
                background-color: {Colors.BG_SURFACE};
                border-bottom: 1px solid {Colors.BORDER};
            }}
        """)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(Spacing.LG, 0, Spacing.LG, 0)

        title = QLabel("💬  New Conversation")
        title.setFont(Fonts.make(Fonts.SIZE_MD, QFont.Weight.DemiBold))
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent; border: none;")

        self._new_chat_btn = IconButton("＋", "Start new conversation", size=36)
        self._new_chat_btn.clicked.connect(self._on_new_chat)

        layout.addWidget(title)
        layout.addStretch()
        layout.addWidget(self._new_chat_btn)
        return header

    # ── slots ─────────────────────────────────────────────────────────
    def _on_send(self, text: str):
        # Show conversation area on first message
        if not self._convo.has_messages:
            self._welcome.hide()
            self._convo.show()

        self._convo.add_message(text, role="user")

        # Stage 1: no AI response — show a clear placeholder reply
        placeholder = (
            "⏳  AI inference is not yet connected.\n"
            "Your message has been received. Responses will be available in a future stage."
        )
        self._convo.add_message(placeholder, role="assistant")

    def _on_new_chat(self):
        self._convo.clear_messages()
        self._convo.hide()
        self._welcome.show()

