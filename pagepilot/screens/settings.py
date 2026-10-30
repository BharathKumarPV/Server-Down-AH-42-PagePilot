"""
PagePilot – Settings screen.

Displays application info, model status, offline mode, and storage info.
All data is static in Stage 1.
"""

from __future__ import annotations

import platform
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QScrollArea,
    QVBoxLayout, QWidget,
)

from pagepilot.config import APP_NAME, APP_VERSION, APP_DESCRIPTION, APP_AUTHOR
from pagepilot.theme import Colors, Fonts, Spacing
from pagepilot.ui.components import Badge, Card, Divider, InfoRow, SectionLabel


class SettingsScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("SettingsScreen")
        self.setStyleSheet(f"background-color: {Colors.BG_APP};")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background: transparent; border: none;")

        body = QWidget()
        body.setStyleSheet(f"background-color: {Colors.BG_APP};")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(Spacing.XL, Spacing.XL, Spacing.XL, Spacing.XL)
        body_layout.setSpacing(Spacing.LG)

        body_layout.addWidget(self._build_app_card())
        body_layout.addWidget(self._build_status_card())
        body_layout.addWidget(self._build_storage_card())
        body_layout.addWidget(self._build_system_card())
        body_layout.addStretch()

        scroll.setWidget(body)
        root.addWidget(scroll, 1)

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setFixedHeight(56)
        header.setObjectName("SettingsHeader")
        header.setStyleSheet(f"""
            QWidget#SettingsHeader {{
                background-color: {Colors.BG_SURFACE};
                border-bottom: 1px solid {Colors.BORDER};
            }}
        """)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(Spacing.LG, 0, Spacing.LG, 0)
        title = QLabel("⚙  Settings")
        title.setFont(Fonts.make(Fonts.SIZE_MD, QFont.Weight.DemiBold))
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent; border: none;")
        layout.addWidget(title)
        return header

    def _build_app_card(self) -> Card:
        card = Card(title="Application")
        layout = card.body_layout()

        # Branding row
        brand_row = QWidget()
        brand_row.setStyleSheet("background: transparent;")
        br_layout = QHBoxLayout(brand_row)
        br_layout.setContentsMargins(0, 0, 0, 0)
        br_layout.setSpacing(Spacing.MD)

        logo = QLabel("🧭")
        logo.setStyleSheet("font-size: 32px; background: transparent; border: none;")

        info_col = QWidget()
        info_col.setStyleSheet("background: transparent;")
        info_layout = QVBoxLayout(info_col)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(2)

        name_lbl = QLabel(APP_NAME)
        name_lbl.setFont(Fonts.make(Fonts.SIZE_XL, QFont.Weight.Bold))
        name_lbl.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent; border: none;")

        desc_lbl = QLabel(APP_DESCRIPTION)
        desc_lbl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_SM}px; background: transparent; border: none;")

        info_layout.addWidget(name_lbl)
        info_layout.addWidget(desc_lbl)

        br_layout.addWidget(logo)
        br_layout.addWidget(info_col, 1)
        br_layout.addWidget(Badge(f"v{APP_VERSION}", "accent"))

        layout.addWidget(brand_row)
        layout.addWidget(Divider())

        layout.addWidget(InfoRow("Author", APP_AUTHOR))
        layout.addWidget(InfoRow("Version", APP_VERSION))
        layout.addWidget(InfoRow("Stage", "Stage 1 — Windows Frontend (UI only)"))
        return card

    def _build_status_card(self) -> Card:
        card = Card(title="Status")
        layout = card.body_layout()

        def status_row(label: str, badge_text: str, variant: str, detail: str = "") -> QWidget:
            w = QWidget()
            w.setStyleSheet("background: transparent;")
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(Spacing.MD)
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_SM}px; background: transparent; border: none; min-width: 160px;")
            badge = Badge(badge_text, variant)
            h.addWidget(lbl)
            h.addWidget(badge)
            if detail:
                d = QLabel(detail)
                d.setStyleSheet(f"color: {Colors.TEXT_DISABLED}; font-size: {Fonts.SIZE_XS}px; background: transparent; border: none;")
                h.addWidget(d)
            h.addStretch()
            return w

        layout.addWidget(status_row(
            "AI Chat",
            "Not Implemented",
            "warning",
            "Planned — Stage 2"
        ))
        layout.addWidget(status_row(
            "Offline Mode",
            "No online services configured",
            "default",
            "Stage 1"
        ))
        layout.addWidget(status_row(
            "Voice Recognition",
            "Not Implemented",
            "default",
            "Planned — Stage 3"
        ))
        layout.addWidget(status_row(
            "PDF Processing",
            "Not Implemented",
            "default",
            "Planned — Stage 4"
        ))
        layout.addWidget(status_row(
            "DOCX Generation",
            "Not Implemented",
            "default",
            "Planned — Stage 5"
        ))
        layout.addWidget(status_row(
            "Android Application",
            "Not Implemented",
            "default",
            "Planned — Stage 6"
        ))
        layout.addWidget(status_row(
            "Offline Maps",
            "Not Implemented",
            "default",
            "Planned — Stage 7"
        ))
        return card

    def _build_storage_card(self) -> Card:
        card = Card(title="Storage")
        layout = card.body_layout()
        layout.addWidget(InfoRow("User Data", "Not stored — Stage 1 holds no data on disk"))
        layout.addWidget(InfoRow("Models", "No models loaded"))
        layout.addWidget(InfoRow("Cache", "None"))
        layout.addWidget(InfoRow("Map Tiles", "None"))
        return card

    def _build_system_card(self) -> Card:
        card = Card(title="System")
        layout = card.body_layout()
        layout.addWidget(InfoRow("Operating System", platform.system() + " " + platform.release()))
        layout.addWidget(InfoRow("Python", sys.version.split(" ")[0]))
        try:
            import PySide6
            pyside_ver = PySide6.__version__
        except Exception:
            pyside_ver = "unknown"
        layout.addWidget(InfoRow("PySide6", pyside_ver))
        layout.addWidget(InfoRow("Architecture", platform.machine()))
        return card

