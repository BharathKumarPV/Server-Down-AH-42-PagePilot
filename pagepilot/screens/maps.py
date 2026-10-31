"""
PagePilot – Offline Maps screen.

All mapping and routing features are placeholders in Stage 1.
No mapping library or network call is made.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QScrollArea,
    QSizePolicy, QVBoxLayout, QWidget,
)

from pagepilot.theme import Colors, Fonts, Radius, Spacing
from pagepilot.ui.components import (
    Card, PlannedBanner, PrimaryButton, TextInput,
)


class MapPlaceholder(QFrame):
    """Visual stand-in for the map tile area."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("MapPlaceholder")
        self.setMinimumHeight(320)
        self.setStyleSheet(f"""
            QFrame#MapPlaceholder {{
                background-color: {Colors.BG_SURFACE};
                border: 2px dashed {Colors.BORDER};
                border-radius: {Radius.LG}px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(Spacing.SM)

        icon = QLabel("🗺")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet(f"font-size: 52px; background: transparent; border: none; color: {Colors.TEXT_DISABLED};")

        title = QLabel("Offline Map")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(Fonts.make(Fonts.SIZE_LG, QFont.Weight.DemiBold))
        title.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent; border: none;")

        note = QLabel(
            "Offline map tiles will be rendered here.\n"
            "Map integration is planned for a future stage."
        )
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {Colors.TEXT_DISABLED}; font-size: {Fonts.SIZE_SM}px; background: transparent; border: none;")

        layout.addWidget(icon)
        layout.addWidget(title)
        layout.addWidget(note)


class MapsScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("MapsScreen")
        self.setStyleSheet(f"background-color: {Colors.BG_APP};")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())

        # Main body: two-column on wide screens
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background: transparent; border: none;")

        body = QWidget()
        body.setStyleSheet(f"background-color: {Colors.BG_APP};")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(Spacing.XL, Spacing.XL, Spacing.XL, Spacing.XL)
        body_layout.setSpacing(Spacing.LG)

        body_layout.addWidget(PlannedBanner(
            "Offline maps and routing are not yet implemented. "
            "This screen shows the planned UI layout for Stage 1 review."
        ))

        # Map tile area
        body_layout.addWidget(MapPlaceholder())

        # Controls row: search + route
        ctrl_row = QWidget()
        ctrl_row.setStyleSheet("background: transparent;")
        ctrl_layout = QHBoxLayout(ctrl_row)
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        ctrl_layout.setSpacing(Spacing.LG)

        ctrl_layout.addWidget(self._build_search_card(), 1)
        ctrl_layout.addWidget(self._build_route_card(), 1)
        body_layout.addWidget(ctrl_row)

        body_layout.addStretch()
        scroll.setWidget(body)
        root.addWidget(scroll, 1)

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setFixedHeight(56)
        header.setObjectName("MapsHeader")
        header.setStyleSheet(f"""
            QWidget#MapsHeader {{
                background-color: {Colors.BG_SURFACE};
                border-bottom: 1px solid {Colors.BORDER};
            }}
        """)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(Spacing.LG, 0, Spacing.LG, 0)
        title = QLabel("🗺  Offline Maps")
        title.setFont(Fonts.make(Fonts.SIZE_MD, QFont.Weight.DemiBold))
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent; border: none;")
        layout.addWidget(title)
        return header

    def _build_search_card(self) -> Card:
        card = Card(title="Location Search")
        layout = card.body_layout()

        self._search_input = TextInput("Search for a location…")
        search_btn = PrimaryButton("🔍  Search  (coming soon)")
        search_btn.setEnabled(False)

        layout.addWidget(self._search_input)
        layout.addWidget(search_btn)
        return card

    def _build_route_card(self) -> Card:
        card = Card(title="Plan a Route")
        layout = card.body_layout()

        origin_label = QLabel("Origin:")
        origin_label.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_SM}px;"
            f" font-weight: 600; background: transparent; border: none;"
        )
        self._origin_input = TextInput("Enter starting location…")

        dest_label = QLabel("Destination:")
        dest_label.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_SM}px;"
            f" font-weight: 600; background: transparent; border: none;"
        )
        self._dest_input = TextInput("Enter destination…")

        route_btn = PrimaryButton("🧭  Calculate Route  (coming soon)")
        route_btn.setEnabled(False)

        layout.addWidget(origin_label)
        layout.addWidget(self._origin_input)
        layout.addWidget(dest_label)
        layout.addWidget(self._dest_input)
        layout.addWidget(route_btn)
        return card

