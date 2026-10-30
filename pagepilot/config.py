"""
PagePilot – Application configuration constants.
All tuneable values live here so screens and components stay free of magic numbers.
"""

APP_NAME = "PagePilot"
APP_VERSION = "0.1.0-stage1"
APP_DESCRIPTION = "Offline-first AI assistant"
APP_AUTHOR = "PagePilot Team"

# Window geometry
WINDOW_MIN_WIDTH = 900
WINDOW_MIN_HEIGHT = 600
WINDOW_DEFAULT_WIDTH = 1200
WINDOW_DEFAULT_HEIGHT = 780

# Sidebar
SIDEBAR_COLLAPSED_WIDTH = 64
SIDEBAR_EXPANDED_WIDTH = 220

# Chat
CHAT_MAX_MESSAGE_WIDTH = 680
CHAT_BUBBLE_RADIUS = 12

# Navigation item IDs  (used as page keys in the stacked widget)
NAV_CHAT = "chat"
NAV_HISTORY = "history"
NAV_PDF = "pdf"
NAV_DOCX = "docx"
NAV_MAPS = "maps"
NAV_SETTINGS = "settings"

NAV_ITEMS = [
    (NAV_CHAT,     "New Chat",      "💬"),
    (NAV_HISTORY,  "Chat History",  "🕘"),
    (NAV_PDF,      "PDF Assistant", "📄"),
    (NAV_DOCX,     "DOCX Creator",  "📝"),
    (NAV_MAPS,     "Offline Maps",  "🗺"),
    (NAV_SETTINGS, "Settings",      "⚙"),
]

