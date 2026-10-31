"""
PagePilot – Main window (Stage 2).

Assembles the global sidebar + stacked screen area.
Routes cross-screen signals:
  - History screen → Chat screen (open a saved conversation)
  - Screen visibility changes (trigger data refresh)
"""

from __future__ import annotations

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

        # ── Screens ──────────────────────────────────────────────────
        self._stack = QStackedWidget()
        self._stack.setObjectName("ScreenStack")

        self._chat_screen     = ChatScreen()
        self._history_screen  = HistoryScreen()
        self._pdf_screen      = PdfScreen()
        self._docx_screen     = DocxScreen()
        self._maps_screen     = MapsScreen()
        self._settings_screen = SettingsScreen()

        self._screens: dict[str, QWidget] = {
            NAV_CHAT:     self._chat_screen,
            NAV_HISTORY:  self._history_screen,
            NAV_PDF:      self._pdf_screen,
            NAV_DOCX:     self._docx_screen,
            NAV_MAPS:     self._maps_screen,
            NAV_SETTINGS: self._settings_screen,
        }
        for screen in self._screens.values():
            self._stack.addWidget(screen)

        root_layout.addWidget(self._sidebar)
        root_layout.addWidget(self._stack, 1)

        # ── Cross-screen wiring ───────────────────────────────────────
        # History item → navigate to Chat and load conversation
        self._history_screen.open_conversation.connect(self._open_conversation_from_history)

        # Start on Chat
        self._switch_screen(NAV_CHAT)
        self._sidebar.select(NAV_CHAT)

    def _switch_screen(self, nav_id: str):
        if nav_id not in self._screens:
            return
        self._stack.setCurrentWidget(self._screens[nav_id])

        # Trigger refresh on screens that display live data
        if nav_id == NAV_HISTORY:
            self._history_screen.refresh()
        elif nav_id == NAV_SETTINGS:
            self._settings_screen.refresh()
        elif nav_id == NAV_CHAT:
            self._chat_screen.refresh_status()
        elif nav_id == NAV_DOCX:
            self._docx_screen.refresh_documents()

    def _open_conversation_from_history(self, conv_id: int):
        """Switch to the Chat screen and load the selected conversation."""
        self._switch_screen(NAV_CHAT)
        self._sidebar.select(NAV_CHAT)
        self._chat_screen._on_conv_selected(conv_id)
