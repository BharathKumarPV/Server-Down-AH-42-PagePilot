"""
PagePilot – Settings screen (Stage 2).

Shows live Ollama/model status, database location, and configuration.
Status is fetched in a background thread — never faked.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QSizePolicy, QVBoxLayout, QWidget,
)

from pagepilot.ai.ollama_client import OllamaClient, OllamaStatus
from pagepilot.config import (
    APP_AUTHOR, APP_DESCRIPTION, APP_NAME, APP_VERSION,
    DB_PATH, DOCS_DB_PATH, CHROMA_DIR, EMBED_MODEL,
    MODEL_NAME, NUM_CTX, OLLAMA_HOST, REQUEST_TIMEOUT,
    WHISPER_MODEL_DIR, PIPER_VOICE_DIR, DOCX_OUTPUT_DIR,
)
from pagepilot.theme import Colors, Fonts, Radius, Spacing
from pagepilot.ui.components import Badge, Card, Divider, InfoRow, SectionLabel
from pagepilot.voice.settings_widget import VoiceSettingsCard


# ──────────────────────────────────────────────────────────────────────────────
# Background status worker
# ──────────────────────────────────────────────────────────────────────────────

class _StatusWorker(QThread):
    done = Signal(object)   # OllamaStatus

    def __init__(self, client: OllamaClient, parent=None):
        super().__init__(parent)
        self._client = client

    def run(self):
        self.done.emit(self._client.check_status())


# ──────────────────────────────────────────────────────────────────────────────
# Settings screen
# ──────────────────────────────────────────────────────────────────────────────

class SettingsScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("SettingsScreen")
        self.setStyleSheet(f"background-color: {Colors.BG_APP};")

        self._client = OllamaClient(host=OLLAMA_HOST, model=MODEL_NAME)
        self._status_worker: _StatusWorker | None = None

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
        body_layout.addWidget(self._build_ai_card())
        body_layout.addWidget(VoiceSettingsCard())
        body_layout.addWidget(self._build_pdf_card())
        body_layout.addWidget(self._build_roadmap_card())
        body_layout.addWidget(self._build_storage_card())
        body_layout.addWidget(self._build_system_card())
        body_layout.addStretch()

        scroll.setWidget(body)
        root.addWidget(scroll, 1)

        # Kick off initial status check
        self._refresh_status()

    # ── Header ───────────────────────────────────────────────────────

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

    # ── Application card ─────────────────────────────────────────────

    def _build_app_card(self) -> Card:
        card = Card(title="Application")
        layout = card.body_layout()

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
        desc_lbl.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_SM}px; "
            f"background: transparent; border: none;"
        )

        info_layout.addWidget(name_lbl)
        info_layout.addWidget(desc_lbl)

        br_layout.addWidget(logo)
        br_layout.addWidget(info_col, 1)
        br_layout.addWidget(Badge(f"v{APP_VERSION}", "accent"))
        layout.addWidget(brand_row)
        layout.addWidget(Divider())

        layout.addWidget(InfoRow("Author",  APP_AUTHOR))
        layout.addWidget(InfoRow("Version", APP_VERSION))
        layout.addWidget(InfoRow("Stage",   "Stage 5 — Professional DOCX Generation, PDF Intelligence, Voice & AI Chat"))
        return card

    # ── AI / Ollama card ─────────────────────────────────────────────

    def _build_ai_card(self) -> Card:
        card = Card(title="AI Model & Ollama")
        layout = card.body_layout()

        # Status section (dynamic)
        status_section = QWidget()
        status_section.setStyleSheet("background: transparent;")
        self._status_layout = QVBoxLayout(status_section)
        self._status_layout.setContentsMargins(0, 0, 0, 0)
        self._status_layout.setSpacing(Spacing.SM)

        self._server_row  = self._make_status_row("Ollama Server", "Checking…", "default")
        self._model_row   = self._make_status_row("Model", "Checking…",          "default")
        self._status_layout.addWidget(self._server_row["widget"])
        self._status_layout.addWidget(self._model_row["widget"])

        layout.addWidget(status_section)
        layout.addWidget(Divider())

        # Configuration (static)
        layout.addWidget(InfoRow("Ollama Host",      OLLAMA_HOST))
        layout.addWidget(InfoRow("Model",            MODEL_NAME))
        layout.addWidget(InfoRow("Context Window",   f"{NUM_CTX:,} tokens"))
        layout.addWidget(InfoRow("Request Timeout",  f"{REQUEST_TIMEOUT}s"))

        layout.addWidget(Divider())

        # Action row
        btn_row = QWidget()
        btn_row.setStyleSheet("background: transparent;")
        btn_l = QHBoxLayout(btn_row)
        btn_l.setContentsMargins(0, 0, 0, 0)
        btn_l.setSpacing(Spacing.SM)

        self._retry_btn = QPushButton("↺  Retry Connection")
        self._retry_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._retry_btn.setFixedHeight(34)
        self._retry_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {Colors.TEXT_PRIMARY}; "
            f"border: 1px solid {Colors.BORDER}; border-radius: {Radius.MD}px; "
            f"font-size: {Fonts.SIZE_SM}px; padding: 0 16px; }}"
            f"QPushButton:hover {{ background: {Colors.BG_ELEVATED}; }}"
            f"QPushButton:disabled {{ color: {Colors.TEXT_DISABLED}; }}"
        )
        self._retry_btn.clicked.connect(self._refresh_status)
        btn_l.addWidget(self._retry_btn)
        btn_l.addStretch()
        layout.addWidget(btn_row)

        # Install instructions
        self._install_lbl = QLabel()
        self._install_lbl.setWordWrap(True)
        self._install_lbl.setStyleSheet(
            f"color: {Colors.TEXT_DISABLED}; font-size: {Fonts.SIZE_XS}px; "
            f"background: transparent; border: none;"
        )
        self._install_lbl.hide()
        layout.addWidget(self._install_lbl)

        return card

    def _make_status_row(self, label: str, badge_text: str, variant: str) -> dict:
        w = QWidget()
        w.setStyleSheet("background: transparent;")
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(Spacing.MD)

        lbl = QLabel(label)
        lbl.setFixedWidth(160)
        lbl.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_SM}px; "
            f"background: transparent; border: none;"
        )

        badge = Badge(badge_text, variant)
        h.addWidget(lbl)
        h.addWidget(badge)
        h.addStretch()
        return {"widget": w, "badge": badge}

    def _update_status_badge(self, row: dict, text: str, variant: str):
        old_badge = row["badge"]
        new_badge = Badge(text, variant)
        # Replace in layout
        layout = row["widget"].layout()
        idx = layout.indexOf(old_badge)
        if idx >= 0:
            layout.removeWidget(old_badge)
            old_badge.deleteLater()
            layout.insertWidget(idx, new_badge)
        row["badge"] = new_badge

    # ── PDF Intelligence card (Stage 4) ───────────────────────────────

    def _build_pdf_card(self) -> Card:
        card = Card(title="PDF Intelligence & RAG")
        layout = card.body_layout()

        layout.addWidget(InfoRow("Embed Model",       EMBED_MODEL))
        layout.addWidget(InfoRow("Vector Dimension",  "768 float32"))
        layout.addWidget(InfoRow("Retrieval Engine",  "Hybrid RRF (Cosine Vector + SQLite FTS5)"))
        layout.addWidget(InfoRow("Vector Database",   "ChromaDB (Local, Telemetry Disabled)"))
        layout.addWidget(InfoRow("Text Extractor",    "PyMuPDF (Page-level provenance)"))
        layout.addWidget(InfoRow("Citation Fidelity", "Verified 1-based page metadata"))
        return card

    # ── Roadmap card ─────────────────────────────────────────────────

    def _build_roadmap_card(self) -> Card:
        card = Card(title="Feature Roadmap")
        layout = card.body_layout()

        def feature_row(label: str, badge_text: str, variant: str, detail: str = "") -> QWidget:
            w = QWidget()
            w.setStyleSheet("background: transparent;")
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(Spacing.MD)
            lbl = QLabel(label)
            lbl.setFixedWidth(180)
            lbl.setStyleSheet(
                f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_SM}px; "
                f"background: transparent; border: none;"
            )
            badge = Badge(badge_text, variant)
            h.addWidget(lbl)
            h.addWidget(badge)
            if detail:
                d = QLabel(detail)
                d.setStyleSheet(
                    f"color: {Colors.TEXT_DISABLED}; font-size: {Fonts.SIZE_XS}px; "
                    f"background: transparent; border: none;"
                )
                h.addWidget(d)
            h.addStretch()
            return w

        layout.addWidget(feature_row("AI Chat",              "Active",          "success", "Stage 2 — qwen2.5:3b via Ollama"))
        layout.addWidget(feature_row("Offline Mode",         "No cloud APIs",   "info",    "All inference is local"))
        layout.addWidget(feature_row("Voice Recognition",    "Active",          "success", "Stage 3 — Whisper.cpp + Piper TTS"))
        layout.addWidget(feature_row("PDF Processing",       "Active",          "success", "Stage 4 — PyMuPDF + ChromaDB + nomic-embed-text"))
        layout.addWidget(feature_row("DOCX Generation",      "Active",          "success", "Stage 5 — python-docx + 5 templates + citations"))
        layout.addWidget(feature_row("Multi-Doc Reasoning",  "Active",          "success", "Stage 6A — Faithful to Every Sentence (Cross-Doc & Claim Verifier)"))
        layout.addWidget(feature_row("Handwriting & Math",   "Not Implemented", "default", "Planned — Stage 6B"))
        layout.addWidget(feature_row("Android Application",  "Not Implemented", "default", "Planned — Future"))
        layout.addWidget(feature_row("Offline Maps",         "Not Implemented", "default", "Planned — Future"))
        return card

    # ── Storage card ─────────────────────────────────────────────────

    def _build_storage_card(self) -> Card:
        card = Card(title="Storage")
        layout = card.body_layout()

        db_path = str(DB_PATH)
        db_exists = DB_PATH.exists()
        db_size = f"{DB_PATH.stat().st_size / 1024:.1f} KB" if db_exists else "not yet created"

        docs_db_exists = DOCS_DB_PATH.exists()
        docs_db_size = f"{DOCS_DB_PATH.stat().st_size / 1024:.1f} KB" if docs_db_exists else "not yet created"

        layout.addWidget(InfoRow("Chat Database",     db_path))
        layout.addWidget(InfoRow("Chat DB Size",      db_size))
        layout.addWidget(InfoRow("Documents DB",      str(DOCS_DB_PATH)))
        layout.addWidget(InfoRow("Documents DB Size", docs_db_size))
        layout.addWidget(InfoRow("Vector Store",      str(CHROMA_DIR)))
        layout.addWidget(InfoRow("DOCX Output",       str(DOCX_OUTPUT_DIR)))
        layout.addWidget(InfoRow("Whisper Models",    str(WHISPER_MODEL_DIR)))
        layout.addWidget(InfoRow("Piper Voices",      str(PIPER_VOICE_DIR)))
        layout.addWidget(InfoRow("Ollama Models",     "Managed by Ollama runtime"))
        layout.addWidget(InfoRow("Map Tiles",         "None — Stage 7"))
        return card

    # ── System card ──────────────────────────────────────────────────

    def _build_system_card(self) -> Card:
        card = Card(title="System")
        layout = card.body_layout()
        layout.addWidget(InfoRow("Operating System", platform.system() + " " + platform.release()))
        layout.addWidget(InfoRow("Python",           sys.version.split(" ")[0]))
        try:
            import PySide6
            pyside_ver = PySide6.__version__
        except Exception:
            pyside_ver = "unknown"
        layout.addWidget(InfoRow("PySide6",      pyside_ver))
        layout.addWidget(InfoRow("Architecture", platform.machine()))
        return card

    # ── Status refresh ───────────────────────────────────────────────

    def _refresh_status(self):
        if self._status_worker and self._status_worker.isRunning():
            return
        self._retry_btn.setEnabled(False)
        self._retry_btn.setText("Checking…")
        self._update_status_badge(self._server_row, "Checking…", "default")
        self._update_status_badge(self._model_row,  "Checking…", "default")
        self._install_lbl.hide()

        self._status_worker = _StatusWorker(self._client, parent=self)
        self._status_worker.done.connect(self._on_status_done)
        self._status_worker.start()

    def _on_status_done(self, status: OllamaStatus):
        self._retry_btn.setEnabled(True)
        self._retry_btn.setText("↺  Retry Connection")

        # Server row
        if status.server_reachable:
            self._update_status_badge(self._server_row, "Running", "success")
        else:
            self._update_status_badge(self._server_row, "Not Running", "error")

        # Model row
        if status.model_available:
            self._update_status_badge(self._model_row, "Installed", "success")
        elif status.server_reachable:
            self._update_status_badge(self._model_row, "Not Installed", "warning")
        else:
            self._update_status_badge(self._model_row, "Unknown", "default")

        # Show install instructions when needed
        if status.error:
            self._install_lbl.setText(f"ℹ  {status.error}")
            self._install_lbl.show()
        else:
            self._install_lbl.hide()

    # ── Public API ───────────────────────────────────────────────────

    def refresh(self):
        """Called by MainWindow when this screen becomes visible."""
        self._refresh_status()
