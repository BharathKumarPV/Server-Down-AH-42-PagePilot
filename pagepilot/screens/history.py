"""
PagePilot – Chat History screen.

Displays past conversation sessions. No persistence is wired in Stage 1.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QScrollArea,
    QSizePolicy, QVBoxLayout, QWidget,
)

from pagepilot.theme import Colors, Fonts, Radius, Spacing
from pagepilot.ui.components import EmptyState, PlannedBanner


class HistoryScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("HistoryScreen")
        self.setStyleSheet(f"background-color: {Colors.BG_APP};")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header
        header = QWidget()
        header.setFixedHeight(56)
        header.setObjectName("HistoryHeader")
        header.setStyleSheet(f"""
            QWidget#HistoryHeader {{
                background-color: {Colors.BG_SURFACE};
                border-bottom: 1px solid {Colors.BORDER};
            }}
        """)
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(Spacing.LG, 0, Spacing.LG, 0)
        title = QLabel("🕘  Chat History")
        title.setFont(Fonts.make(Fonts.SIZE_MD, QFont.Weight.DemiBold))
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent; border: none;")
        h_layout.addWidget(title)
        root.addWidget(header)

        # Body
        body = QWidget()
        body.setStyleSheet(f"background-color: {Colors.BG_APP};")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(Spacing.XL, Spacing.XL, Spacing.XL, Spacing.XL)
        body_layout.setSpacing(Spacing.LG)

        banner = PlannedBanner(
            "Chat history persistence is not yet implemented. "
            "Conversations are stored in memory only and will be lost when the application is closed."
        )
        body_layout.addWidget(banner)

        empty = EmptyState(
            "🕘",
            "No conversation history",
            "Your past chats will appear here once chat persistence is implemented.",
        )
        body_layout.addWidget(empty, 1)

        root.addWidget(body, 1)

