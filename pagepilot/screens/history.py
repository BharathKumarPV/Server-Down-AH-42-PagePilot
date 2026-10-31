"""
PagePilot – Chat History screen (Stage 2).

Lists all persisted conversations with title and timestamp.
Clicking a row navigates to the Chat screen and loads that conversation.
"""

from __future__ import annotations

import textwrap

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QCursor
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QScrollArea,
    QSizePolicy, QVBoxLayout, QWidget,
)

from pagepilot.db.database import ConversationDB, Conversation
from pagepilot.theme import Colors, Fonts, Radius, Spacing
from pagepilot.ui.components import EmptyState


class HistoryItem(QFrame):
    """A single row in the history list."""

    open_requested = Signal(int)   # conv_id

    def __init__(self, conv: Conversation, parent=None):
        super().__init__(parent)
        self.conv_id = conv.id
        self.setObjectName("HistoryItem")
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setFixedHeight(58)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setStyleSheet(f"""
            QFrame#HistoryItem {{
                background: {Colors.BG_SURFACE};
                border: 1px solid {Colors.BORDER};
                border-radius: {Radius.MD}px;
            }}
            QFrame#HistoryItem:hover {{
                background: {Colors.BG_SURFACE_ALT};
                border-color: {Colors.TEXT_DISABLED};
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(Spacing.MD, 0, Spacing.MD, 0)
        layout.setSpacing(Spacing.MD)

        icon = QLabel("💬")
        icon.setStyleSheet("font-size: 18px; background: transparent; border: none;")
        icon.setFixedWidth(28)

        info = QWidget()
        info.setStyleSheet("background: transparent;")
        info_l = QVBoxLayout(info)
        info_l.setContentsMargins(0, 0, 0, 0)
        info_l.setSpacing(2)

        title_lbl = QLabel(textwrap.shorten(conv.title, width=60, placeholder="…"))
        title_lbl.setFont(Fonts.make(Fonts.SIZE_BASE, QFont.Weight.Medium))
        title_lbl.setStyleSheet(
            f"color: {Colors.TEXT_PRIMARY}; background: transparent; border: none;"
        )

        # Format timestamp nicely
        ts = conv.updated_at[:10] if conv.updated_at else ""
        date_lbl = QLabel(ts)
        date_lbl.setStyleSheet(
            f"color: {Colors.TEXT_DISABLED}; font-size: {Fonts.SIZE_XS}px; "
            f"background: transparent; border: none;"
        )

        info_l.addWidget(title_lbl)
        info_l.addWidget(date_lbl)

        arrow = QLabel("›")
        arrow.setStyleSheet(
            f"color: {Colors.TEXT_DISABLED}; font-size: 18px; "
            f"background: transparent; border: none;"
        )

        layout.addWidget(icon)
        layout.addWidget(info, 1)
        layout.addWidget(arrow)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.open_requested.emit(self.conv_id)
        super().mousePressEvent(event)


class HistoryScreen(QWidget):
    """Shows all saved conversations; emits open_conversation to navigate."""

    open_conversation = Signal(int)   # conv_id → MainWindow routes to Chat screen

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("HistoryScreen")
        self.setStyleSheet(f"background-color: {Colors.BG_APP};")
        self._db = ConversationDB()

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_header())

        # Scrollable body
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background: transparent; border: none;")

        self._body = QWidget()
        self._body.setStyleSheet(f"background-color: {Colors.BG_APP};")
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(Spacing.XL, Spacing.XL, Spacing.XL, Spacing.XL)
        self._body_layout.setSpacing(Spacing.SM)
        self._body_layout.addStretch()

        scroll.setWidget(self._body)
        root.addWidget(scroll, 1)

        self._load()

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setFixedHeight(56)
        header.setObjectName("HistoryHeader")
        header.setStyleSheet(f"""
            QWidget#HistoryHeader {{
                background-color: {Colors.BG_SURFACE};
                border-bottom: 1px solid {Colors.BORDER};
            }}
        """)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(Spacing.LG, 0, Spacing.LG, 0)
        title = QLabel("🕘  Chat History")
        title.setFont(Fonts.make(Fonts.SIZE_MD, QFont.Weight.DemiBold))
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent; border: none;")
        layout.addWidget(title)
        return header

    def _load(self):
        # Remove all except trailing stretch
        while self._body_layout.count() > 1:
            item = self._body_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        convs = self._db.list_conversations()
        if not convs:
            empty = EmptyState(
                "🕘",
                "No conversations yet",
                "Start a new chat — your history will appear here.",
            )
            self._body_layout.insertWidget(0, empty)
        else:
            for conv in convs:
                row = HistoryItem(conv)
                row.open_requested.connect(self.open_conversation)
                self._body_layout.insertWidget(self._body_layout.count() - 1, row)

    def refresh(self):
        """Reload from DB. Called when this screen becomes visible."""
        self._load()
