"""
PagePilot – Application configuration constants.
All tuneable values live here so screens and components stay free of magic numbers.
"""

import os
from pathlib import Path

APP_NAME = "PagePilot"
APP_VERSION = "0.3.0-stage3"
APP_DESCRIPTION = "Offline-first AI assistant"
APP_AUTHOR = "PagePilot Team"

# Window geometry
WINDOW_MIN_WIDTH = 900
WINDOW_MIN_HEIGHT = 600
WINDOW_DEFAULT_WIDTH = 1260
WINDOW_DEFAULT_HEIGHT = 800

# Sidebar
SIDEBAR_COLLAPSED_WIDTH = 64
SIDEBAR_EXPANDED_WIDTH = 260

# Chat display
CHAT_MAX_MESSAGE_WIDTH = 720
CHAT_BUBBLE_RADIUS = 12

# ──────────────────────────────────────────────
# Ollama / AI configuration
# ──────────────────────────────────────────────
OLLAMA_HOST     = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL_NAME      = os.environ.get("PAGEPILOT_MODEL", "qwen2.5:3b")
NUM_CTX         = int(os.environ.get("PAGEPILOT_NUM_CTX", "4096"))
REQUEST_TIMEOUT = int(os.environ.get("PAGEPILOT_TIMEOUT", "120"))   # seconds
MAX_HISTORY_MESSAGES = 20   # max messages sent as context (keeps RAM+tokens bounded)

# System prompt sent at the start of every conversation
SYSTEM_PROMPT = (
    "You are PagePilot, a helpful, concise and honest offline AI assistant. "
    "Answer the user's questions clearly. When you don't know something, say so. "
    "Do not hallucinate facts. Keep responses appropriately detailed."
)

# ──────────────────────────────────────────────
# Local storage
# ──────────────────────────────────────────────
_APP_DATA_DIR = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "PagePilot"
DB_PATH = _APP_DATA_DIR / "conversations.db"

def ensure_data_dir() -> Path:
    """Create the application data directory if it does not exist."""
    _APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return _APP_DATA_DIR

# ──────────────────────────────────────────────
# Voice / Stage 3 configuration
# ──────────────────────────────────────────────

# Whisper speech recognition
WHISPER_MODEL     = os.environ.get("PAGEPILOT_WHISPER_MODEL", "base")
WHISPER_LANGUAGE  = os.environ.get("PAGEPILOT_WHISPER_LANG", "auto")
WHISPER_THREADS   = int(os.environ.get("PAGEPILOT_WHISPER_THREADS", "4"))

_WHISPER_MODEL_DIR = _APP_DATA_DIR / "models" / "whisper"
WHISPER_MODEL_DIR  = _WHISPER_MODEL_DIR

def ensure_whisper_dir() -> Path:
    _WHISPER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return _WHISPER_MODEL_DIR

# Piper TTS
PIPER_VOICE        = os.environ.get("PAGEPILOT_PIPER_VOICE", "en_US-lessac-medium")
PIPER_AUTO_SPEAK   = os.environ.get("PAGEPILOT_AUTO_SPEAK", "0") == "1"

_PIPER_VOICE_DIR   = _APP_DATA_DIR / "models" / "piper"
PIPER_VOICE_DIR    = _PIPER_VOICE_DIR

def ensure_piper_dir() -> Path:
    _PIPER_VOICE_DIR.mkdir(parents=True, exist_ok=True)
    return _PIPER_VOICE_DIR

# ──────────────────────────────────────────────
# Navigation item IDs  (used as page keys in stacked widget)
# ──────────────────────────────────────────────
NAV_CHAT     = "chat"
NAV_HISTORY  = "history"
NAV_PDF      = "pdf"
NAV_DOCX     = "docx"
NAV_MAPS     = "maps"
NAV_SETTINGS = "settings"

NAV_ITEMS = [
    (NAV_CHAT,     "New Chat",      "💬"),
    (NAV_HISTORY,  "Chat History",  "🕘"),
    (NAV_PDF,      "PDF Assistant", "📄"),
    (NAV_DOCX,     "DOCX Creator",  "📝"),
    (NAV_MAPS,     "Offline Maps",  "🗺"),
    (NAV_SETTINGS, "Settings",      "⚙"),
]
