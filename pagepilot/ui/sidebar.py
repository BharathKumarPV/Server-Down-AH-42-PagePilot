"""
PagePilot – Sidebar navigation widget.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QFont, QCursor
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QSpacerItem, QVBoxLayout, QWidget,
)

from pagepilot.config import NAV_ITEMS, SIDEBAR_EXPANDED_WIDTH, APP_NAME
from pagepilot.theme import Colors, Fonts, Radius, Spacing


class NavItem(QPushButton):
    """Single sidebar navigation entry."""

    def __init__(self, nav_id: str, label: str, emoji: str, parent=None):
        super().__init__(parent)
        self.nav_id = nav_id
        self._label = label
        self._emoji = emoji
        self._active = False

        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setCheckable(False)
        self.setFixedHeight(44)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setToolTip(label)

        self._build_content()
        self._refresh_style()

    def _build_content(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(Spacing.MD, 0, Spacing.MD, 0)
        layout.setSpacing(Spacing.MD)

        self._icon_lbl = QLabel(self._emoji)
        self._icon_lbl.setStyleSheet("background: transparent; border: none; font-size: 16px;")
        self._icon_lbl.setFixedWidth(22)
        self._icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._text_lbl = QLabel(self._label)
        self._text_lbl.setStyleSheet(f"background: transparent; border: none; font-size: {Fonts.SIZE_BASE}px; font-family: {Fonts.FAMILY_PRIMARY};")

        self._active_bar = QFrame()
        self._active_bar.setFixedWidth(3)
        self._active_bar.setStyleSheet(f"background: transparent; border: none; border-radius: 2px;")

        layout.addWidget(self._active_bar)
        layout.addWidget(self._icon_lbl)
        layout.addWidget(self._text_lbl, 1)

    def set_active(self, active: bool):
        self._active = active
        self._refresh_style()

    def _refresh_style(self):
        if self._active:
            bg = Colors.NAV_ACTIVE_BG
            fg = Colors.TEXT_PRIMARY
            bar_bg = Colors.NAV_ACTIVE_INDICATOR
        else:
            bg = "transparent"
            fg = Colors.TEXT_SECONDARY
            bar_bg = "transparent"

        self._active_bar.setStyleSheet(f"background: {bar_bg}; border: none; border-radius: 2px;")
        self._text_lbl.setStyleSheet(
            f"background: transparent; border: none; color: {fg}; "
            f"font-size: {Fonts.SIZE_BASE}px; font-family: {Fonts.FAMILY_PRIMARY};"
            + (" font-weight: 600;" if self._active else "")
        )
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg};
                border: none;
                border-radius: {Radius.MD}px;
                text-align: left;
            }}
            QPushButton:hover {{
                background-color: {Colors.BG_ELEVATED if not self._active else Colors.NAV_ACTIVE_BG};
            }}
        """)


class Sidebar(QWidget):
    """Left navigation panel."""

    nav_changed = Signal(str)   # emits nav_id on selection

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Sidebar")
        self.setFixedWidth(SIDEBAR_EXPANDED_WIDTH)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(f"""
            QWidget#Sidebar {{
                background-color: {Colors.BG_SIDEBAR};
                border-right: 1px solid {Colors.BORDER};
            }}
        """)

        self._items: dict[str, NavItem] = {}
        self._current_id: str = ""

        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(Spacing.SM, 0, Spacing.SM, Spacing.LG)
        root.setSpacing(0)

        # ── Logo / brand header ──────────────────────────────────────
        header = QWidget()
        header.setFixedHeight(64)
        header.setStyleSheet("background: transparent;")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(Spacing.MD, 0, Spacing.MD, 0)
        h_layout.setSpacing(Spacing.SM)

        logo = QLabel("🧭")
        logo.setStyleSheet("background: transparent; border: none; font-size: 22px;")

        brand = QLabel(APP_NAME)
        brand.setFont(Fonts.make(Fonts.SIZE_LG, QFont.Weight.Bold))
        brand.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent; border: none;")

        h_layout.addWidget(logo)
        h_layout.addWidget(brand)
        h_layout.addStretch()

        root.addWidget(header)

        # ── Divider ──────────────────────────────────────────────────
        div = QFrame()
        div.setFixedHeight(1)
        div.setStyleSheet(f"background: {Colors.BORDER}; border: none;")
        root.addWidget(div)
        root.addSpacing(Spacing.SM)

        # ── Nav items ────────────────────────────────────────────────
        for nav_id, label, emoji in NAV_ITEMS:
            item = NavItem(nav_id, label, emoji)
            item.clicked.connect(lambda checked=False, nid=nav_id: self._on_nav_clicked(nid))
            self._items[nav_id] = item
            root.addWidget(item)
            root.addSpacing(2)

        root.addStretch()

        # ── Bottom version label ──────────────────────────────────────
        ver = QLabel("Stage 1 — UI only")
        ver.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ver.setStyleSheet(f"color: {Colors.TEXT_DISABLED}; font-size: {Fonts.SIZE_XS}px; background: transparent; border: none;")
        root.addWidget(ver)

    def _on_nav_clicked(self, nav_id: str):
        self.select(nav_id)
        self.nav_changed.emit(nav_id)

    def select(self, nav_id: str):
        if self._current_id == nav_id:
            return
        if self._current_id and self._current_id in self._items:
            self._items[self._current_id].set_active(False)
        self._current_id = nav_id
        if nav_id in self._items:
            self._items[nav_id].set_active(True)

