"""
pagepilot.voice.whisper — Stage 3: Offline transcription via pywhispercpp.

Runs Whisper (base multilingual model) in a background QThread.
Never calls any cloud service.

Model
-----
Default:  ggml-base (multilingual, ~148 MB)
English:  ggml-base.en (~145 MB)

Models are stored in the application data directory
(%APPDATA%\\PagePilot\\models\\whisper\\) — outside the Git repository.

Licensing
---------
whisper.cpp  — MIT License  (https://github.com/ggml-org/whisper.cpp)
pywhispercpp — MIT License  (https://github.com/abdeladim-s/pywhispercpp)
Whisper weights — MIT License (OpenAI)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Signal

# ── Model storage ─────────────────────────────────────────────────────────────

_APP_DATA = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
WHISPER_MODEL_DIR = _APP_DATA / "PagePilot" / "models" / "whisper"


def model_path(model_name: str = "base") -> Path:
    """Return the expected local path for a given Whisper model name."""
    return WHISPER_MODEL_DIR / f"ggml-{model_name}.bin"


def is_model_installed(model_name: str = "base") -> bool:
    return model_path(model_name).is_file()


def model_download_url(model_name: str = "base") -> str:
    return (
        f"https://huggingface.co/ggerganov/whisper.cpp"
        f"/resolve/main/ggml-{model_name}.bin"
    )


# ── Transcription worker ───────────────────────────────────────────────────────

class TranscriptionError(Exception):
    """Raised for known transcription problems."""


class TranscriptionThread(QThread):
    """
    Transcribes a WAV file in the background using pywhispercpp.

    Signals
    -------
    transcription_done(str)  — full transcribed text
    error_occurred(str)      — human-readable error
    progress(str)            — short status message for the UI
    """

    transcription_done = Signal(str)
    error_occurred     = Signal(str)
    progress           = Signal(str)

    def __init__(
        self,
        wav_path: str,
        model_name: str = "base",
        language: str = "auto",
        n_threads: int = 4,
        parent=None,
    ):
        super().__init__(parent)
        self._wav_path   = wav_path
        self._model_name = model_name
        self._language   = language
        self._n_threads  = n_threads

    def run(self):
        # ── Pre-flight checks ────────────────────────────────────────────────
        try:
            from pywhispercpp.model import Model
        except ImportError:
            self.error_occurred.emit(
                "pywhispercpp is not installed.  "
                "Run: pip install pywhispercpp"
            )
            return

        mpath = model_path(self._model_name)
        if not mpath.is_file():
            self.error_occurred.emit(
                f"Whisper model '{self._model_name}' not found.\n"
                f"Expected at: {mpath}\n\n"
                f"Download it from the Settings → Voice screen."
            )
            return

        if not Path(self._wav_path).is_file():
            self.error_occurred.emit(f"Audio file not found: {self._wav_path}")
            return

        # ── Transcription ────────────────────────────────────────────────────
        self.progress.emit("Loading model…")
        try:
            model = Model(
                str(mpath),
                n_threads=self._n_threads,
                print_progress=False,
                print_realtime=False,
            )
        except Exception as exc:
            self.error_occurred.emit(f"Failed to load Whisper model: {exc}")
            return

        self.progress.emit("Transcribing…")
        try:
            kwargs: dict = {}
            if self._language and self._language != "auto":
                kwargs["language"] = self._language

            segments = model.transcribe(self._wav_path, **kwargs)
            clean_segments = []
            for seg in segments:
                t = seg.text.strip()
                t_lower = t.lower()
                # Filter out Whisper silence and hallucination artifacts
                if t and not any(tag in t_lower for tag in [
                    "[blank_audio]", "[no speech]", "[silence]", "(silence)",
                    "[music]", "(music)", "[applause]", "(applause)"
                ]):
                    clean_segments.append(t)
            text = " ".join(clean_segments).strip()

            if not text:
                self.error_occurred.emit(
                    "No speech detected.  "
                    "Try speaking closer to the microphone."
                )
                return

            self.transcription_done.emit(text)

        except Exception as exc:
            self.error_occurred.emit(f"Transcription failed: {exc}")

