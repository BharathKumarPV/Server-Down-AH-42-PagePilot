"""
pagepilot.voice.recorder — Stage 3: Microphone recording via sounddevice.

Records audio from the default (or selected) input device into a temporary
WAV file at 16 kHz mono — the format expected by Whisper.

States
------
    IDLE       → waiting for user to press the mic button
    RECORDING  → audio is streaming from the mic
    STOPPING   → stop() called; flushing buffer to disk
    DONE       → WAV written; `path` property is valid
    ERROR      → something went wrong; `error` property is set
    CANCELLED  → user cancelled before saving

All heavy work runs in a QThread.  UI signals are emitted on the Qt main thread.
"""

from __future__ import annotations

import os
import queue
import tempfile
import wave
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QThread, Signal

# Whisper expects 16 kHz mono 16-bit PCM
SAMPLE_RATE = 16_000
CHANNELS    = 1
DTYPE       = "int16"
CHUNK       = 1_024   # frames per sounddevice callback


class RecorderError(Exception):
    """Raised for known recorder problems with user-readable messages."""


class RecorderThread(QThread):
    """
    Records microphone audio in a background thread.

    Signals
    -------
    state_changed(str)   — 'recording' | 'stopping' | 'done' | 'error' | 'cancelled'
    level_updated(float) — RMS amplitude [0.0 – 1.0] for a VU meter (~ 30 fps)
    finished_path(str)   — absolute path of the saved WAV (emitted before 'done')
    error_occurred(str)  — human-readable error message
    """

    state_changed  = Signal(str)
    level_updated  = Signal(float)
    finished_path  = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancel_flag = False
        self._stop_flag   = False
        self._q: queue.Queue[Optional[np.ndarray]] = queue.Queue()
        self._tmp_path: str | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def stop(self):
        """Stop recording and save the WAV."""
        self._stop_flag = True

    def cancel(self):
        """Cancel recording; discard audio."""
        self._cancel_flag = True
        self._stop_flag   = True
        if self._tmp_path and os.path.exists(self._tmp_path):
            try:
                os.unlink(self._tmp_path)
            except Exception:
                pass
            self._tmp_path = None

    @property
    def path(self) -> str | None:
        return self._tmp_path

    # ── QThread entry point ───────────────────────────────────────────────────

    def run(self):
        try:
            import sounddevice as sd
        except ImportError:
            msg = (
                "sounddevice is not installed.  "
                "Run: pip install sounddevice"
            )
            self.error_occurred.emit(msg)
            self.state_changed.emit("error")
            return

        # Check for at least one input device
        try:
            devices = sd.query_devices()
            has_input = any(d["max_input_channels"] > 0 for d in devices)
            if not has_input:
                raise RecorderError("No microphone or audio input device found.")
        except RecorderError as exc:
            self.error_occurred.emit(str(exc))
            self.state_changed.emit("error")
            return
        except Exception as exc:
            self.error_occurred.emit(f"Audio device query failed: {exc}")
            self.state_changed.emit("error")
            return

        self.state_changed.emit("recording")
        frames: list[np.ndarray] = []
        _level_counter = 0

        def _callback(indata, frame_count, time_info, status):
            nonlocal _level_counter
            chunk = indata.copy()
            self._q.put(chunk)
            # Emit level every ~10 callbacks ≈ 30 fps at 16 kHz/1024
            _level_counter += 1
            if _level_counter % 10 == 0:
                rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
                level = min(1.0, rms / 8000.0)
                self.level_updated.emit(level)

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=CHUNK,
                callback=_callback,
            ):
                while not self._stop_flag:
                    try:
                        chunk = self._q.get(timeout=0.05)
                        frames.append(chunk)
                    except queue.Empty:
                        pass

            # Drain any remaining chunks
            while not self._q.empty():
                frames.append(self._q.get_nowait())

        except Exception as exc:
            err_str = str(exc)
            if "permission" in err_str.lower() or "access" in err_str.lower():
                msg = f"Microphone access denied: {exc}. Please check Windows Privacy Settings."
            elif "device unavailable" in err_str.lower() or "busy" in err_str.lower():
                msg = f"Microphone is busy or already in use: {exc}."
            else:
                msg = f"Recording failed: {exc}"
            self.error_occurred.emit(msg)
            self.state_changed.emit("error")
            return

        if self._cancel_flag:
            self.state_changed.emit("cancelled")
            return

        if not frames:
            self.error_occurred.emit("No audio was captured.")
            self.state_changed.emit("error")
            return

        # Write WAV
        self.state_changed.emit("stopping")
        try:
            tmp = tempfile.NamedTemporaryFile(
                suffix=".wav", prefix="pagepilot_rec_", delete=False
            )
            tmp.close()
            audio = np.concatenate(frames, axis=0)
            with wave.open(tmp.name, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)           # 16-bit → 2 bytes
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio.tobytes())
            self._tmp_path = tmp.name
            self.finished_path.emit(tmp.name)
            self.state_changed.emit("done")
        except Exception as exc:
            if "tmp" in locals() and os.path.exists(tmp.name):
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass
            self.error_occurred.emit(f"Failed to save recording: {exc}")
            self.state_changed.emit("error")


def list_input_devices() -> list[dict]:
    """Return a list of available audio input devices (name + index)."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        return [
            {"index": i, "name": d["name"]}
            for i, d in enumerate(devices)
            if d["max_input_channels"] > 0
        ]
    except Exception:
        return []

