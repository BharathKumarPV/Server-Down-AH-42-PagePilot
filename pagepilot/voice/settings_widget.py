"""
pagepilot.voice.settings_widget — Stage 3: Voice configuration UI card.

Shown inside the Settings screen.  Lets the user:
  • See whether Whisper and Piper models are installed.
  • Download the Whisper model.
  • Open the voice model directory.
  • Choose the active Piper voice.
  • Toggle auto-speak.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QProgressBar, QPushButton,
    QVBoxLayout, QWidget, QComboBox, QCheckBox,
)

from pagepilot.config import (
    WHISPER_MODEL, PIPER_VOICE, PIPER_AUTO_SPEAK,
    PIPER_VOICE_DIR, WHISPER_MODEL_DIR,
    ensure_whisper_dir, ensure_piper_dir,
)
from pagepilot.theme import Colors, Fonts, Radius, Spacing
from pagepilot.ui.components import Badge, Card, Divider, InfoRow, SectionLabel
from pagepilot.voice.whisper import (
    is_model_installed, model_download_url, model_path, WHISPER_MODEL_DIR as WMD,
)
from pagepilot.voice.tts import (
    is_voice_installed, list_installed_voices,
    PIPER_VOICE_DIR as PVD, DEFAULT_VOICE, voice_download_instructions,
)


# ── Download worker ─────────────────────────────────────────────────────────────

class _DownloadWorker(QThread):
    """Downloads a Whisper model from Hugging Face into the model directory."""

    progress  = Signal(int)    # 0–100
    done      = Signal()
    failed    = Signal(str)

    def __init__(self, url: str, dest: Path, parent=None):
        super().__init__(parent)
        self._url  = url
        self._dest = dest

    def run(self):
        try:
            import urllib.request
            ensure_whisper_dir()
            tmp = self._dest.with_suffix(".tmp")

            def _reporthook(block_num, block_size, total_size):
                if total_size > 0:
                    pct = min(100, int(block_num * block_size * 100 / total_size))
                    self.progress.emit(pct)

            urllib.request.urlretrieve(self._url, tmp, reporthook=_reporthook)
            tmp.rename(self._dest)
            self.done.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


# ── Voice settings card ─────────────────────────────────────────────────────────

class VoiceSettingsCard(QWidget):
    """
    A self-contained settings card for Stage 3 voice configuration.
    Drop this directly into a QVBoxLayout.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self._dl_worker: _DownloadWorker | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Spacing.MD)

        layout.addWidget(self._build_whisper_card())
        layout.addWidget(self._build_piper_card())

    # ── Whisper card ──────────────────────────────────────────────────────────

    def _build_whisper_card(self) -> Card:
        card = Card(title="Speech Recognition — Whisper")
        layout = card.body_layout()

        # Status row
        installed = is_model_installed(WHISPER_MODEL)
        status_row = QWidget()
        status_row.setStyleSheet("background: transparent;")
        sr_l = QHBoxLayout(status_row)
        sr_l.setContentsMargins(0, 0, 0, 0)
        sr_l.setSpacing(Spacing.MD)

        sr_lbl = QLabel("Model")
        sr_lbl.setFixedWidth(160)
        sr_lbl.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_SM}px; "
            f"background: transparent; border: none;"
        )
        self._whisper_badge = Badge(
            "Installed" if installed else "Not Installed",
            "success" if installed else "warning",
        )
        sr_l.addWidget(sr_lbl)
        sr_l.addWidget(self._whisper_badge)
        sr_l.addStretch()
        layout.addWidget(status_row)

        layout.addWidget(InfoRow("Model", f"ggml-{WHISPER_MODEL} (multilingual)"))
        layout.addWidget(InfoRow("Engine", "whisper.cpp via pywhispercpp"))
        layout.addWidget(InfoRow("License", "MIT (whisper.cpp, pywhispercpp, OpenAI weights)"))
        layout.addWidget(InfoRow("Storage", str(WMD)))

        layout.addWidget(Divider())

        # Download progress bar (hidden until download starts)
        self._dl_progress = QProgressBar()
        self._dl_progress.setRange(0, 100)
        self._dl_progress.setValue(0)
        self._dl_progress.setFixedHeight(6)
        self._dl_progress.setStyleSheet(
            f"QProgressBar {{ background: {Colors.BG_ELEVATED}; border-radius: 3px; border: none; }}"
            f"QProgressBar::chunk {{ background: {Colors.ACCENT}; border-radius: 3px; }}"
        )
        self._dl_progress.hide()
        layout.addWidget(self._dl_progress)

        self._dl_status_lbl = QLabel("")
        self._dl_status_lbl.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_XS}px; "
            f"background: transparent; border: none;"
        )
        self._dl_status_lbl.hide()
        layout.addWidget(self._dl_status_lbl)

        # Button row
        btn_row = QWidget()
        btn_row.setStyleSheet("background: transparent;")
        br_l = QHBoxLayout(btn_row)
        br_l.setContentsMargins(0, 0, 0, 0)
        br_l.setSpacing(Spacing.SM)

        self._dl_btn = QPushButton("⬇  Download ggml-base model (~148 MB)")
        self._dl_btn.setFixedHeight(34)
        self._dl_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._dl_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {Colors.TEXT_PRIMARY}; "
            f"border: 1px solid {Colors.BORDER}; border-radius: {Radius.MD}px; "
            f"font-size: {Fonts.SIZE_SM}px; padding: 0 16px; }}"
            f"QPushButton:hover {{ background: {Colors.BG_ELEVATED}; }}"
            f"QPushButton:disabled {{ color: {Colors.TEXT_DISABLED}; }}"
        )
        self._dl_btn.clicked.connect(self._on_download_whisper)
        if installed:
            self._dl_btn.setEnabled(False)
            self._dl_btn.setText("✓  Model already installed")

        open_dir_btn = QPushButton("📁  Open folder")
        open_dir_btn.setFixedHeight(34)
        open_dir_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        open_dir_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {Colors.TEXT_SECONDARY}; "
            f"border: 1px solid {Colors.BORDER}; border-radius: {Radius.MD}px; "
            f"font-size: {Fonts.SIZE_SM}px; padding: 0 16px; }}"
            f"QPushButton:hover {{ background: {Colors.BG_ELEVATED}; }}"
        )
        open_dir_btn.clicked.connect(self._on_open_whisper_dir)

        br_l.addWidget(self._dl_btn)
        br_l.addWidget(open_dir_btn)
        br_l.addStretch()
        layout.addWidget(btn_row)

        note = QLabel(
            "ℹ  The model is downloaded from Hugging Face once and stored locally. "
            "No data leaves your device during transcription."
        )
        note.setWordWrap(True)
        note.setStyleSheet(
            f"color: {Colors.TEXT_DISABLED}; font-size: {Fonts.SIZE_XS}px; "
            f"background: transparent; border: none;"
        )
        layout.addWidget(note)

        return card

    # ── Piper card ────────────────────────────────────────────────────────────

    def _build_piper_card(self) -> Card:
        card = Card(title="Text-to-Speech — Piper")
        layout = card.body_layout()

        voices = list_installed_voices()
        installed = is_voice_installed(PIPER_VOICE)

        # Status
        status_row = QWidget()
        status_row.setStyleSheet("background: transparent;")
        sr_l = QHBoxLayout(status_row)
        sr_l.setContentsMargins(0, 0, 0, 0)
        sr_l.setSpacing(Spacing.MD)
        sr_lbl = QLabel("Active voice")
        sr_lbl.setFixedWidth(160)
        sr_lbl.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_SM}px; "
            f"background: transparent; border: none;"
        )
        self._piper_badge = Badge(
            "Installed" if installed else "Not Installed",
            "success" if installed else "warning",
        )
        sr_l.addWidget(sr_lbl)
        sr_l.addWidget(self._piper_badge)
        sr_l.addStretch()
        layout.addWidget(status_row)

        layout.addWidget(InfoRow("Engine", "Piper 1.8.0 (OHF-Voice)"))
        layout.addWidget(InfoRow("License", "GPL-3.0-or-later — see THIRD_PARTY_NOTICES.md"))
        layout.addWidget(InfoRow("Voice storage", str(PVD)))

        layout.addWidget(Divider())

        # Voice selector
        voice_sel_row = QWidget()
        voice_sel_row.setStyleSheet("background: transparent;")
        vs_l = QHBoxLayout(voice_sel_row)
        vs_l.setContentsMargins(0, 0, 0, 0)
        vs_l.setSpacing(Spacing.MD)
        vs_lbl = QLabel("Voice")
        vs_lbl.setFixedWidth(160)
        vs_lbl.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_SM}px; "
            f"background: transparent; border: none;"
        )
        self._voice_combo = QComboBox()
        self._voice_combo.setFixedHeight(32)
        self._voice_combo.setStyleSheet(
            f"QComboBox {{ background: {Colors.BG_ELEVATED}; color: {Colors.TEXT_PRIMARY}; "
            f"border: 1px solid {Colors.BORDER}; border-radius: {Radius.SM}px; "
            f"font-size: {Fonts.SIZE_SM}px; padding: 0 8px; }}"
            f"QComboBox::drop-down {{ border: none; }}"
            f"QComboBox QAbstractItemView {{ background: {Colors.BG_SURFACE}; "
            f"color: {Colors.TEXT_PRIMARY}; selection-background-color: {Colors.ACCENT_MUTED}; }}"
        )
        if voices:
            for v in voices:
                self._voice_combo.addItem(v)
            if PIPER_VOICE in voices:
                self._voice_combo.setCurrentText(PIPER_VOICE)
        else:
            self._voice_combo.addItem("No voices installed")
            self._voice_combo.setEnabled(False)

        vs_l.addWidget(vs_lbl)
        vs_l.addWidget(self._voice_combo)
        vs_l.addStretch()
        layout.addWidget(voice_sel_row)

        layout.addWidget(Divider())

        # Open voice dir button
        btn_row = QWidget()
        btn_row.setStyleSheet("background: transparent;")
        br_l = QHBoxLayout(btn_row)
        br_l.setContentsMargins(0, 0, 0, 0)
        br_l.setSpacing(Spacing.SM)

        open_dir_btn = QPushButton("📁  Open voice folder")
        open_dir_btn.setFixedHeight(34)
        open_dir_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        open_dir_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {Colors.TEXT_SECONDARY}; "
            f"border: 1px solid {Colors.BORDER}; border-radius: {Radius.MD}px; "
            f"font-size: {Fonts.SIZE_SM}px; padding: 0 16px; }}"
            f"QPushButton:hover {{ background: {Colors.BG_ELEVATED}; }}"
        )
        open_dir_btn.clicked.connect(self._on_open_piper_dir)

        huggingface_btn = QPushButton("🌐  Browse voice catalogue")
        huggingface_btn.setFixedHeight(34)
        huggingface_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        huggingface_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {Colors.TEXT_SECONDARY}; "
            f"border: 1px solid {Colors.BORDER}; border-radius: {Radius.MD}px; "
            f"font-size: {Fonts.SIZE_SM}px; padding: 0 16px; }}"
            f"QPushButton:hover {{ background: {Colors.BG_ELEVATED}; }}"
        )
        huggingface_btn.clicked.connect(self._on_browse_voices)

        br_l.addWidget(open_dir_btn)
        br_l.addWidget(huggingface_btn)
        br_l.addStretch()
        layout.addWidget(btn_row)

        note = QLabel(
            "ℹ  Download .onnx and .onnx.json files for your chosen voice and place them "
            "in the voice folder above.  Each voice model has its own licence — "
            "review it before redistribution.  piper-tts itself is GPL-3.0."
        )
        note.setWordWrap(True)
        note.setStyleSheet(
            f"color: {Colors.TEXT_DISABLED}; font-size: {Fonts.SIZE_XS}px; "
            f"background: transparent; border: none;"
        )
        layout.addWidget(note)

        return card

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _on_download_whisper(self):
        from pagepilot.voice.whisper import model_download_url, model_path
        ensure_whisper_dir()
        dest = model_path(WHISPER_MODEL)
        url  = model_download_url(WHISPER_MODEL)

        self._dl_btn.setEnabled(False)
        self._dl_btn.setText("Downloading…")
        self._dl_progress.setValue(0)
        self._dl_progress.show()
        self._dl_status_lbl.setText(f"Downloading from {url}")
        self._dl_status_lbl.show()

        self._dl_worker = _DownloadWorker(url, dest, parent=self)
        self._dl_worker.progress.connect(self._dl_progress.setValue)
        self._dl_worker.done.connect(self._on_download_done)
        self._dl_worker.failed.connect(self._on_download_failed)
        self._dl_worker.start()

    def _on_download_done(self):
        self._dl_progress.setValue(100)
        self._dl_progress.hide()
        self._dl_status_lbl.hide()
        self._dl_btn.setText("✓  Model installed")
        self._dl_btn.setEnabled(False)
        # Refresh badge
        old = self._whisper_badge
        new = Badge("Installed", "success")
        old.parent().layout().replaceWidget(old, new)
        old.deleteLater()
        self._whisper_badge = new

    def _on_download_failed(self, err: str):
        self._dl_progress.hide()
        self._dl_status_lbl.setText(f"Download failed: {err}")
        self._dl_status_lbl.show()
        self._dl_btn.setEnabled(True)
        self._dl_btn.setText("⬇  Retry download")

    def _on_open_whisper_dir(self):
        ensure_whisper_dir()
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(WMD)))

    def _on_open_piper_dir(self):
        ensure_piper_dir()
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(PVD)))

    def _on_browse_voices(self):
        QDesktopServices.openUrl(
            QUrl("https://huggingface.co/rhasspy/piper-voices/tree/main")
        )

    def selected_voice(self) -> str:
        return self._voice_combo.currentText()

