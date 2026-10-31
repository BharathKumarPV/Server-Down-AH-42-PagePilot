"""
PagePilot – Reusable UI components.

All widgets here are theme-aware and import from pagepilot.theme only.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QFont, QCursor
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QSizePolicy, QTextEdit,
    QVBoxLayout, QWidget, QComboBox, QScrollArea,
)

from pagepilot.theme import Colors, Fonts, Radius, Spacing


# ──────────────────────────────────────────────────────────────────────────────
# Base card / surface
# ──────────────────────────────────────────────────────────────────────────────
class Card(QFrame):
    """Rounded surface with optional title label."""

    def __init__(self, parent=None, *, title: str = "", padding: int = Spacing.LG):
        super().__init__(parent)
        self.setObjectName("Card")
        self.setStyleSheet(f"""
            QFrame#Card {{
                background-color: {Colors.BG_SURFACE};
                border: 1px solid {Colors.BORDER};
                border-radius: {Radius.LG}px;
            }}
        """)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(padding, padding, padding, padding)
        self._layout.setSpacing(Spacing.MD)

        if title:
            lbl = QLabel(title)
            lbl.setFont(Fonts.make(Fonts.SIZE_MD, QFont.Weight.DemiBold))
            lbl.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent; border: none;")
            self._layout.addWidget(lbl)

    def body_layout(self) -> QVBoxLayout:
        return self._layout


# ──────────────────────────────────────────────────────────────────────────────
# Divider
# ──────────────────────────────────────────────────────────────────────────────
class Divider(QFrame):
    def __init__(self, parent=None, *, orientation=Qt.Orientation.Horizontal):
        super().__init__(parent)
        if orientation == Qt.Orientation.Horizontal:
            self.setFrameShape(QFrame.Shape.HLine)
            self.setFixedHeight(1)
        else:
            self.setFrameShape(QFrame.Shape.VLine)
            self.setFixedWidth(1)
        self.setStyleSheet(f"background-color: {Colors.BORDER}; border: none;")


# ──────────────────────────────────────────────────────────────────────────────
# Primary button
# ──────────────────────────────────────────────────────────────────────────────
class PrimaryButton(QPushButton):
    def __init__(self, text: str, parent=None, *, icon_text: str = ""):
        super().__init__(text, parent)
        self._apply_style()
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setMinimumHeight(38)

    def _apply_style(self):
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.ACCENT};
                color: {Colors.TEXT_ON_ACCENT};
                border-radius: {Radius.MD}px;
                padding: 8px 20px;
                font-size: {Fonts.SIZE_BASE}px;
                font-weight: 600;
                font-family: {Fonts.FAMILY_PRIMARY};
                border: none;
            }}
            QPushButton:hover {{
                background-color: {Colors.ACCENT_HOVER};
            }}
            QPushButton:pressed {{
                background-color: {Colors.ACCENT_PRESSED};
            }}
            QPushButton:disabled {{
                background-color: {Colors.BG_ELEVATED};
                color: {Colors.TEXT_DISABLED};
            }}
        """)


# ──────────────────────────────────────────────────────────────────────────────
# Secondary / ghost button
# ──────────────────────────────────────────────────────────────────────────────
class SecondaryButton(QPushButton):
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setMinimumHeight(38)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: {Radius.MD}px;
                padding: 8px 20px;
                font-size: {Fonts.SIZE_BASE}px;
                font-family: {Fonts.FAMILY_PRIMARY};
            }}
            QPushButton:hover {{
                background-color: {Colors.BG_ELEVATED};
                border-color: {Colors.TEXT_DISABLED};
            }}
            QPushButton:pressed {{
                background-color: {Colors.BG_SURFACE_ALT};
            }}
            QPushButton:disabled {{
                color: {Colors.TEXT_DISABLED};
                border-color: {Colors.BORDER};
            }}
        """)


# ──────────────────────────────────────────────────────────────────────────────
# Icon / round button (for toolbar-style actions)
# ──────────────────────────────────────────────────────────────────────────────
class IconButton(QPushButton):
    def __init__(self, emoji: str, tooltip: str = "", parent=None, *, size: int = 36):
        super().__init__(emoji, parent)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setFixedSize(size, size)
        if tooltip:
            self.setToolTip(tooltip)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {Colors.TEXT_SECONDARY};
                border-radius: {size // 2}px;
                font-size: {Fonts.SIZE_LG}px;
                border: none;
            }}
            QPushButton:hover {{
                background-color: {Colors.BG_ELEVATED};
                color: {Colors.TEXT_PRIMARY};
            }}
            QPushButton:pressed {{
                background-color: {Colors.BG_SURFACE_ALT};
            }}
        """)


# ──────────────────────────────────────────────────────────────────────────────
# Text input (single line)
# ──────────────────────────────────────────────────────────────────────────────
class TextInput(QLineEdit):
    def __init__(self, placeholder: str = "", parent=None):
        super().__init__(parent)
        self.setPlaceholderText(placeholder)
        self.setMinimumHeight(40)
        self.setStyleSheet(f"""
            QLineEdit {{
                background-color: {Colors.BG_SURFACE_ALT};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: {Radius.MD}px;
                padding: 8px 14px;
                font-size: {Fonts.SIZE_BASE}px;
                font-family: {Fonts.FAMILY_PRIMARY};
                selection-background-color: {Colors.ACCENT_MUTED};
            }}
            QLineEdit:focus {{
                border-color: {Colors.BORDER_FOCUSED};
                background-color: {Colors.BG_SURFACE};
            }}
            QLineEdit:disabled {{
                color: {Colors.TEXT_DISABLED};
            }}
        """)


# ──────────────────────────────────────────────────────────────────────────────
# Multiline text area
# ──────────────────────────────────────────────────────────────────────────────
class TextArea(QTextEdit):
    def __init__(self, placeholder: str = "", parent=None, *, read_only: bool = False):
        super().__init__(parent)
        self.setPlaceholderText(placeholder)
        self.setReadOnly(read_only)
        self.setStyleSheet(f"""
            QTextEdit {{
                background-color: {Colors.BG_SURFACE_ALT};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: {Radius.MD}px;
                padding: 10px 14px;
                font-size: {Fonts.SIZE_BASE}px;
                font-family: {Fonts.FAMILY_PRIMARY};
                selection-background-color: {Colors.ACCENT_MUTED};
            }}
            QTextEdit:focus {{
                border-color: {Colors.BORDER_FOCUSED};
                background-color: {Colors.BG_SURFACE};
            }}
            QTextEdit:disabled {{
                color: {Colors.TEXT_DISABLED};
            }}
        """)


# ──────────────────────────────────────────────────────────────────────────────
# Dropdown (combobox)
# ──────────────────────────────────────────────────────────────────────────────
class DropDown(QComboBox):
    def __init__(self, items: list[str], parent=None):
        super().__init__(parent)
        self.addItems(items)
        self.setMinimumHeight(38)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setStyleSheet(f"""
            QComboBox {{
                background-color: {Colors.BG_SURFACE_ALT};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: {Radius.MD}px;
                padding: 8px 14px;
                font-size: {Fonts.SIZE_BASE}px;
                font-family: {Fonts.FAMILY_PRIMARY};
            }}
            QComboBox:hover {{
                border-color: {Colors.TEXT_DISABLED};
            }}
            QComboBox:focus {{
                border-color: {Colors.BORDER_FOCUSED};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 28px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid {Colors.TEXT_SECONDARY};
                margin-right: 8px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {Colors.BG_ELEVATED};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: {Radius.SM}px;
                selection-background-color: {Colors.ACCENT_MUTED};
                outline: none;
                padding: 4px;
            }}
        """)


# ──────────────────────────────────────────────────────────────────────────────
# Badge / pill label
# ──────────────────────────────────────────────────────────────────────────────
class Badge(QLabel):
    """Small coloured pill for status indicators."""

    COLORS = {
        "default": (Colors.BG_ELEVATED, Colors.TEXT_SECONDARY),
        "success": ("#1E3D2F", Colors.SUCCESS),
        "warning": ("#3D2E00", Colors.WARNING),
        "error":   ("#3D1515", Colors.ERROR),
        "info":    ("#162B42", Colors.INFO),
        "accent":  (Colors.ACCENT_MUTED, Colors.ACCENT),
    }

    def __init__(self, text: str, variant: str = "default", parent=None):
        super().__init__(text, parent)
        bg, fg = self.COLORS.get(variant, self.COLORS["default"])
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {bg};
                color: {fg};
                border-radius: 10px;
                padding: 2px 10px;
                font-size: {Fonts.SIZE_XS}px;
                font-weight: 600;
                font-family: {Fonts.FAMILY_PRIMARY};
                border: none;
            }}
        """)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)


# ──────────────────────────────────────────────────────────────────────────────
# Section header label
# ──────────────────────────────────────────────────────────────────────────────
class SectionLabel(QLabel):
    def __init__(self, text: str, parent=None):
        super().__init__(text.upper(), parent)
        self.setStyleSheet(f"""
            QLabel {{
                color: {Colors.TEXT_DISABLED};
                font-size: {Fonts.SIZE_XS}px;
                font-weight: 700;
                letter-spacing: 1px;
                font-family: {Fonts.FAMILY_PRIMARY};
                background: transparent;
                border: none;
            }}
        """)


# ──────────────────────────────────────────────────────────────────────────────
# Info row (label : value)
# ──────────────────────────────────────────────────────────────────────────────
class InfoRow(QWidget):
    def __init__(self, label: str, value: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Spacing.SM)

        lbl = QLabel(label)
        lbl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_SM}px; background: transparent; border: none;")
        lbl.setFixedWidth(160)

        val = QLabel(value)
        val.setObjectName("InfoValue")
        val.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: {Fonts.SIZE_SM}px; background: transparent; border: none;")
        val.setWordWrap(True)

        layout.addWidget(lbl)
        layout.addWidget(val, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Placeholder / empty state widget
# ──────────────────────────────────────────────────────────────────────────────
class EmptyState(QWidget):
    """Centred icon + title + subtitle placeholder."""

    def __init__(self, emoji: str, title: str, subtitle: str = "", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(Spacing.SM)

        icon = QLabel(emoji)
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet(f"font-size: 48px; background: transparent; border: none; color: {Colors.TEXT_DISABLED};")

        ttl = QLabel(title)
        ttl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ttl.setFont(Fonts.make(Fonts.SIZE_LG, QFont.Weight.DemiBold))
        ttl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent; border: none;")

        layout.addWidget(icon)
        layout.addWidget(ttl)

        if subtitle:
            sub = QLabel(subtitle)
            sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
            sub.setWordWrap(True)
            sub.setStyleSheet(f"color: {Colors.TEXT_DISABLED}; font-size: {Fonts.SIZE_SM}px; background: transparent; border: none;")
            layout.addWidget(sub)


# ──────────────────────────────────────────────────────────────────────────────
# "Not implemented" banner
# ──────────────────────────────────────────────────────────────────────────────
class PlannedBanner(QFrame):
    """Subtle banner indicating a feature is planned but not yet active."""

    def __init__(self, message: str = "This feature is planned and not yet implemented.", parent=None):
        super().__init__(parent)
        self.setObjectName("PlannedBanner")
        self.setStyleSheet(f"""
            QFrame#PlannedBanner {{
                background-color: #1C1E0F;
                border: 1px solid #3A3B00;
                border-radius: {Radius.MD}px;
            }}
        """)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(Spacing.MD, Spacing.SM, Spacing.MD, Spacing.SM)
        layout.setSpacing(Spacing.SM)

        icon = QLabel("🔧")
        icon.setStyleSheet("background: transparent; border: none; font-size: 14px;")

        lbl = QLabel(message)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f"color: {Colors.WARNING}; font-size: {Fonts.SIZE_SM}px; background: transparent; border: none;")

        layout.addWidget(icon)
        layout.addWidget(lbl, 1)

