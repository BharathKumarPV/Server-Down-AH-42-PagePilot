"""
PagePilot – Main window.

Assembles the sidebar and the stacked screen area into a single QMainWindow.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QMainWindow, QStackedWidget, QWidget,
)

from pagepilot.config import (
    NAV_CHAT, NAV_DOCX, NAV_HISTORY, NAV_MAPS, NAV_PDF, NAV_SETTINGS,
    WINDOW_DEFAULT_HEIGHT, WINDOW_DEFAULT_WIDTH,
    WINDOW_MIN_HEIGHT, WINDOW_MIN_WIDTH,
    APP_NAME,
)
from pagepilot.screens.chat import ChatScreen
from pagepilot.screens.docx_creator import DocxScreen
from pagepilot.screens.history import HistoryScreen
from pagepilot.screens.maps import MapsScreen
from pagepilot.screens.pdf_assistant import PdfScreen
from pagepilot.screens.settings import SettingsScreen
from pagepilot.ui.sidebar import Sidebar


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.resize(WINDOW_DEFAULT_WIDTH, WINDOW_DEFAULT_HEIGHT)

        # ── Root layout ──────────────────────────────────────────────
        root_widget = QWidget()
        root_widget.setObjectName("AppRoot")
        self.setCentralWidget(root_widget)

        root_layout = QHBoxLayout(root_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── Sidebar ──────────────────────────────────────────────────
        self._sidebar = Sidebar()
        self._sidebar.nav_changed.connect(self._switch_screen)

        # ── Stacked screens ──────────────────────────────────────────
        self._stack = QStackedWidget()
        self._stack.setObjectName("ScreenStack")

        self._screens: dict[str, QWidget] = {
            NAV_CHAT:     ChatScreen(),
            NAV_HISTORY:  HistoryScreen(),
            NAV_PDF:      PdfScreen(),
            NAV_DOCX:     DocxScreen(),
            NAV_MAPS:     MapsScreen(),
            NAV_SETTINGS: SettingsScreen(),
        }
        for screen in self._screens.values():
            self._stack.addWidget(screen)

        root_layout.addWidget(self._sidebar)
        root_layout.addWidget(self._stack, 1)

        # Start on the Chat screen
        self._switch_screen(NAV_CHAT)
        self._sidebar.select(NAV_CHAT)

    def _switch_screen(self, nav_id: str):
        if nav_id in self._screens:
            self._stack.setCurrentWidget(self._screens[nav_id])

