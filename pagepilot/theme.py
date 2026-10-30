"""
PagePilot – Centralised design-system / theme.

All colours, fonts, spacing values and stylesheet snippets are defined here.
Components and screens import from this module – no colour literals elsewhere.
"""

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication


# ──────────────────────────────────────────────
# Palette
# ──────────────────────────────────────────────
class Colors:
    # Backgrounds
    BG_APP          = "#0F1117"   # outermost shell
    BG_SIDEBAR      = "#16181F"   # sidebar panel
    BG_SURFACE      = "#1E2028"   # cards, panels
    BG_SURFACE_ALT  = "#252830"   # alternate surface (hover, input)
    BG_ELEVATED     = "#2C2F3A"   # raised elements

    # Accent
    ACCENT          = "#6C63FF"   # primary brand purple
    ACCENT_HOVER    = "#7B74FF"
    ACCENT_PRESSED  = "#5A52E0"
    ACCENT_MUTED    = "#3D3875"   # subtle tint

    # Status / semantic
    SUCCESS         = "#4CAF7D"
    WARNING         = "#F0A500"
    ERROR           = "#E05252"
    INFO            = "#4A9EFF"

    # Text
    TEXT_PRIMARY    = "#F0F2F8"
    TEXT_SECONDARY  = "#9198AE"
    TEXT_DISABLED   = "#555B6E"
    TEXT_ON_ACCENT  = "#FFFFFF"

    # Borders / dividers
    BORDER          = "#2A2D38"
    BORDER_FOCUSED  = "#6C63FF"

    # Chat bubbles
    BUBBLE_USER     = "#3D3875"
    BUBBLE_ASSISTANT= "#1E2028"

    # Sidebar active
    NAV_ACTIVE_BG   = "#252238"
    NAV_ACTIVE_INDICATOR = "#6C63FF"


# ──────────────────────────────────────────────
# Typography
# ──────────────────────────────────────────────
class Fonts:
    FAMILY_PRIMARY  = "Segoe UI"
    FAMILY_MONO     = "Consolas"

    SIZE_XS   = 10
    SIZE_SM   = 12
    SIZE_BASE = 13
    SIZE_MD   = 14
    SIZE_LG   = 16
    SIZE_XL   = 20
    SIZE_2XL  = 24
    SIZE_3XL  = 30

    @staticmethod
    def make(size: int, weight=QFont.Weight.Normal, family: str = "Segoe UI") -> QFont:
        f = QFont(family, size)
        f.setWeight(weight)
        return f


# ──────────────────────────────────────────────
# Spacing / radii
# ──────────────────────────────────────────────
class Spacing:
    XS  = 4
    SM  = 8
    MD  = 12
    LG  = 16
    XL  = 24
    XXL = 32

class Radius:
    SM  = 4
    MD  = 8
    LG  = 12
    XL  = 16
    PILL= 999


# ──────────────────────────────────────────────
# Global application stylesheet
# ──────────────────────────────────────────────
GLOBAL_STYLESHEET = f"""
/* ── reset ── */
* {{
    outline: none;
    border: none;
    margin: 0;
    padding: 0;
}}

QMainWindow, QWidget#AppRoot {{
    background-color: {Colors.BG_APP};
}}

/* ── scroll bars ── */
QScrollBar:vertical {{
    background: {Colors.BG_SURFACE};
    width: 6px;
    border-radius: 3px;
}}
QScrollBar::handle:vertical {{
    background: {Colors.BG_ELEVATED};
    border-radius: 3px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {Colors.TEXT_DISABLED};
}}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: {Colors.BG_SURFACE};
    height: 6px;
    border-radius: 3px;
}}
QScrollBar::handle:horizontal {{
    background: {Colors.BG_ELEVATED};
    border-radius: 3px;
    min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {Colors.TEXT_DISABLED};
}}
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* ── tooltips ── */
QToolTip {{
    background-color: {Colors.BG_ELEVATED};
    color: {Colors.TEXT_PRIMARY};
    border: 1px solid {Colors.BORDER};
    border-radius: {Radius.SM}px;
    padding: 4px 8px;
    font-size: {Fonts.SIZE_SM}px;
}}
"""


def apply_theme(app: QApplication) -> None:
    """Apply the global dark theme to the QApplication."""
    app.setStyle("Fusion")
    app.setStyleSheet(GLOBAL_STYLESHEET)

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window,          QColor(Colors.BG_APP))
    palette.setColor(QPalette.ColorRole.WindowText,      QColor(Colors.TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Base,            QColor(Colors.BG_SURFACE))
    palette.setColor(QPalette.ColorRole.AlternateBase,   QColor(Colors.BG_SURFACE_ALT))
    palette.setColor(QPalette.ColorRole.Text,            QColor(Colors.TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Button,          QColor(Colors.BG_SURFACE))
    palette.setColor(QPalette.ColorRole.ButtonText,      QColor(Colors.TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Highlight,       QColor(Colors.ACCENT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(Colors.TEXT_ON_ACCENT))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(Colors.TEXT_DISABLED))
    palette.setColor(QPalette.ColorRole.ToolTipBase,     QColor(Colors.BG_ELEVATED))
    palette.setColor(QPalette.ColorRole.ToolTipText,     QColor(Colors.TEXT_PRIMARY))
    app.setPalette(palette)

