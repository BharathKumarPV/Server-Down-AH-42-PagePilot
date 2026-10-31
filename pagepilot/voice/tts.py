"""
pagepilot.voice.tts  —  Stage 3: Offline text-to-speech via piper-tts.

Synthesises speech from text using a locally stored Piper voice model
(.onnx + .onnx.json) and plays it back through the system's default audio
output.  No cloud service is involved at any point.

Licensing notice
----------------
piper-tts 1.8.0 is distributed under the GNU General Public License v3.0
or later (GPL-3.0-or-later).  Source: http://github.com/OHF-voice/piper1-gpl

If you distribute PagePilot as a compiled binary that links piper-tts, the
GPL-3.0 copyleft obligations apply to the combined work.  Personal, internal,
and open-source use is unrestricted.

Individual voice models are distributed separately.  Check the licence for
each downloaded voice on the Piper voice repository before redistribution.

Voice model storage
-------------------
Voice files are stored in:
  %APPDATA%\\PagePilot\\models\\piper\\
outside the Git repository.

Expected layout:
  %APPDATA%\\PagePilot\\models\\piper\\en_US-lessac-medium.onnx
  %APPDATA%\\PagePilot\\models\\piper\\en_US-lessac-medium.onnx.json

Official voice catalogue:
  https://huggingface.co/rhasspy/piper-voices/tree/main
"""

from __future__ import annotations

import io
import os
import tempfile
import wave
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Signal

# ── Voice model storage ────────────────────────────────────────────────────────

_APP_DATA = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
PIPER_VOICE_DIR = _APP_DATA / "PagePilot" / "models" / "piper"

# Default bundled voice name (user may change in Settings)
DEFAULT_VOICE = "en_US-lessac-medium"


def voice_path(voice_name: str) -> Path:
    """Return the expected .onnx path for a given voice."""
    return PIPER_VOICE_DIR / f"{voice_name}.onnx"


def is_voice_installed(voice_name: str = DEFAULT_VOICE) -> bool:
    """Return True if both .onnx and .onnx.json exist for this voice."""
    p = voice_path(voice_name)
    return p.is_file() and p.with_suffix(".onnx.json").is_file()


def voice_download_instructions(voice_name: str = DEFAULT_VOICE) -> str:
    """Return human-readable download instructions for the given voice."""
    return (
        f"Voice model '{voice_name}' is not installed.\n\n"
        f"Download it from:\n"
        f"  https://huggingface.co/rhasspy/piper-voices/tree/main\n\n"
        f"Place both files in:\n"
        f"  {PIPER_VOICE_DIR}\\\n"
        f"  {voice_name}.onnx\n"
        f"  {voice_name}.onnx.json\n\n"
        f"Note: piper-tts engine is GPL-3.0.  Voice models are licensed individually.  "
        f"Check each model's licence before redistribution."
    )


def list_installed_voices() -> list[str]:
    """Return names of all installed Piper voices (those with both files)."""
    if not PIPER_VOICE_DIR.is_dir():
        return []
    names = []
    for f in PIPER_VOICE_DIR.glob("*.onnx"):
        if f.with_suffix(".onnx.json").is_file():
            names.append(f.stem)
    return sorted(names)


# ── TTS worker ─────────────────────────────────────────────────────────────────

class TtsThread(QThread):
    """
    Synthesise text to speech in a background thread, then play it.

    Signals
    -------
    state_changed(str)  — 'synthesising' | 'playing' | 'done' | 'stopped' | 'error'
    error_occurred(str) — human-readable error message
    """

    state_changed  = Signal(str)
    error_occurred = Signal(str)

    def __init__(
        self,
        text: str,
        voice_name: str = DEFAULT_VOICE,
        parent=None,
    ):
        super().__init__(parent)
        self._text       = text
        self._voice_name = voice_name
        self._cancel     = False

    def stop(self):
        """Signal the thread to stop playback as soon as possible."""
        self._cancel = True

    # ── QThread entry ──────────────────────────────────────────────────────────

    def run(self):
        # ── Import check ────────────────────────────────────────────────────
        try:
            from piper import PiperVoice
        except ImportError:
            self.error_occurred.emit(
                "piper-tts is not installed.  Run: pip install piper-tts"
            )
            self.state_changed.emit("error")
            return

        try:
            import sounddevice as sd
        except ImportError:
            self.error_occurred.emit(
                "sounddevice is not installed.  Run: pip install sounddevice"
            )
            self.state_changed.emit("error")
            return

        # ── Voice model check ───────────────────────────────────────────────
        onnx_path = voice_path(self._voice_name)
        if not onnx_path.is_file():
            self.error_occurred.emit(voice_download_instructions(self._voice_name))
            self.state_changed.emit("error")
            return

        json_path = onnx_path.with_suffix(".onnx.json")
        if not json_path.is_file():
            self.error_occurred.emit(
                f"Voice config file missing: {json_path}\n"
                "Re-download the voice model."
            )
            self.state_changed.emit("error")
            return

        # ── Synthesis ────────────────────────────────────────────────────────
        self.state_changed.emit("synthesising")
        try:
            voice = PiperVoice.load(str(onnx_path))
        except Exception as exc:
            self.error_occurred.emit(f"Failed to load voice model: {exc}")
            self.state_changed.emit("error")
            return

        if self._cancel:
            self.state_changed.emit("stopped")
            return

        # Synthesise to an in-memory WAV buffer
        buf = io.BytesIO()
        try:
            with wave.open(buf, "wb") as wf:
                voice.synthesize(self._text, wf)
        except Exception as exc:
            self.error_occurred.emit(f"Speech synthesis failed: {exc}")
            self.state_changed.emit("error")
            return

        if self._cancel:
            self.state_changed.emit("stopped")
            return

        # ── Playback via sounddevice ─────────────────────────────────────────
        buf.seek(0)
        try:
            import numpy as np

            self.state_changed.emit("playing")
            with wave.open(buf, "rb") as wf:
                sample_rate = wf.getframerate()
                n_channels  = wf.getnchannels()
                n_frames    = wf.getnframes()
                raw         = wf.readframes(n_frames)

            audio = np.frombuffer(raw, dtype=np.int16).reshape(-1, n_channels)

            # Play in chunks so we can honour cancel requests
            chunk_size = sample_rate // 10   # 100 ms chunks
            with sd.OutputStream(
                samplerate=sample_rate,
                channels=n_channels,
                dtype="int16",
            ) as stream:
                offset = 0
                while offset < len(audio):
                    if self._cancel:
                        break
                    end   = min(offset + chunk_size, len(audio))
                    stream.write(audio[offset:end])
                    offset = end

        except Exception as exc:
            self.error_occurred.emit(f"Audio playback failed: {exc}")
            self.state_changed.emit("error")
            return

        if self._cancel:
            self.state_changed.emit("stopped")
        else:
            self.state_changed.emit("done")

