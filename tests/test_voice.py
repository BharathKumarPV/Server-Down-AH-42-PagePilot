"""
tests/test_voice.py — Unit and integration tests for Stage 3 Voice capabilities:
  - Markdown text sanitizer for speech
  - Voice recording thread and device query
  - Whisper transcription worker and model detection
  - Piper text-to-speech worker and voice detection
  - Voice settings UI card
  - Composer mic button state transitions
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ──────────────────────────────────────────────────────────────────────────────
# Speech Text Cleaner Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestSpeechTextCleaner(unittest.TestCase):

    def test_clean_markdown_strips_code_blocks(self):
        from pagepilot.screens.chat import _clean_markdown_for_speech
        raw = "Here is Python code:\n```python\ndef add(a, b):\n    return a + b\n```\nDone."
        cleaned = _clean_markdown_for_speech(raw)
        self.assertNotIn("def add", cleaned)
        self.assertIn("Here is Python code:", cleaned)
        self.assertIn("Done.", cleaned)

    def test_clean_markdown_strips_inline_code_and_formatting(self):
        from pagepilot.screens.chat import _clean_markdown_for_speech
        raw = "Use the `os.path.join` method with **bold** and *italic* text."
        cleaned = _clean_markdown_for_speech(raw)
        self.assertEqual(cleaned, "Use the os.path.join method with bold and italic text.")

    def test_clean_markdown_strips_headings_and_lists(self):
        from pagepilot.screens.chat import _clean_markdown_for_speech
        raw = "### Steps\n- First step\n- Second step\n1. Numbered one"
        cleaned = _clean_markdown_for_speech(raw)
        self.assertNotIn("###", cleaned)
        self.assertIn("Steps First step Second step Numbered one", cleaned)

    def test_clean_markdown_strips_links(self):
        from pagepilot.screens.chat import _clean_markdown_for_speech
        raw = "Visit [PagePilot Website](https://example.com) for details."
        cleaned = _clean_markdown_for_speech(raw)
        self.assertEqual(cleaned, "Visit PagePilot Website for details.")


# ──────────────────────────────────────────────────────────────────────────────
# Audio Recorder Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestAudioRecorder(unittest.TestCase):

    def test_list_input_devices_returns_list(self):
        from pagepilot.voice.recorder import list_input_devices
        devices = list_input_devices()
        self.assertIsInstance(devices, list)

    def test_recorder_thread_initial_state(self):
        from pagepilot.voice.recorder import RecorderThread
        recorder = RecorderThread()
        self.assertFalse(recorder._stop_flag)
        self.assertFalse(recorder._cancel_flag)
        self.assertIsNone(recorder.path)

    def test_recorder_thread_cancel(self):
        from pagepilot.voice.recorder import RecorderThread
        recorder = RecorderThread()
        recorder.cancel()
        self.assertTrue(recorder._cancel_flag)
        self.assertTrue(recorder._stop_flag)

    def test_recorder_thread_cancel_cleans_existing_temp_file(self):
        from pagepilot.voice.recorder import RecorderThread
        with tempfile.NamedTemporaryFile(suffix=".wav", prefix="pagepilot_test_", delete=False) as tf:
            tf.write(b"dummy")
            temp_path = tf.name
        self.assertTrue(os.path.exists(temp_path))
        recorder = RecorderThread()
        recorder._tmp_path = temp_path
        recorder.cancel()
        self.assertFalse(os.path.exists(temp_path))
        self.assertIsNone(recorder._tmp_path)

    def test_recorder_thread_stop(self):
        from pagepilot.voice.recorder import RecorderThread
        recorder = RecorderThread()
        recorder.stop()
        self.assertTrue(recorder._stop_flag)
        self.assertFalse(recorder._cancel_flag)


# ──────────────────────────────────────────────────────────────────────────────
# Whisper Worker & Model Detection Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestWhisperIntegration(unittest.TestCase):

    def test_whisper_model_path_resolution(self):
        from pagepilot.voice.whisper import model_path, model_download_url
        p = model_path("base")
        self.assertTrue(str(p).endswith("ggml-base.bin"))
        url = model_download_url("base")
        self.assertIn("ggml-base.bin", url)

    def test_transcription_thread_fails_when_model_missing(self):
        from pagepilot.voice.whisper import TranscriptionThread
        worker = TranscriptionThread("dummy.wav", model_name="nonexistent_model_12345")
        errors = []
        worker.error_occurred.connect(errors.append)
        worker.run()
        self.assertTrue(len(errors) > 0)
        self.assertIn("not found", errors[0])

    def test_transcription_thread_fails_when_audio_file_missing(self):
        from pagepilot.voice.whisper import TranscriptionThread
        with patch("pagepilot.voice.whisper.is_model_installed", return_value=True), \
             patch("pagepilot.voice.whisper.model_path", return_value=Path(__file__)):
            worker = TranscriptionThread("non_existent_audio_path_xyz.wav", model_name="base")
            errors = []
            worker.error_occurred.connect(errors.append)
            worker.run()
            self.assertTrue(len(errors) > 0)
            self.assertIn("not found", errors[0])


# ──────────────────────────────────────────────────────────────────────────────
# Piper TTS Worker & Voice Detection Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestPiperTtsIntegration(unittest.TestCase):

    def test_voice_path_resolution(self):
        from pagepilot.voice.tts import voice_path, voice_download_instructions
        vp = voice_path("en_US-lessac-medium")
        self.assertTrue(str(vp).endswith("en_US-lessac-medium.onnx"))
        inst = voice_download_instructions("en_US-lessac-medium")
        self.assertIn("en_US-lessac-medium", inst)
        self.assertIn("GPL-3.0", inst)

    def test_tts_thread_fails_when_voice_missing(self):
        from pagepilot.voice.tts import TtsThread
        worker = TtsThread("Hello world", voice_name="nonexistent_voice_xyz")
        states = []
        errors = []
        worker.state_changed.connect(states.append)
        worker.error_occurred.connect(errors.append)
        worker.run()
        self.assertIn("error", states)
        self.assertTrue(len(errors) > 0)

    def test_tts_thread_stop(self):
        from pagepilot.voice.tts import TtsThread
        worker = TtsThread("Hello", voice_name="en_US-lessac-medium")
        self.assertFalse(worker._cancel)
        worker.stop()
        self.assertTrue(worker._cancel)


# ──────────────────────────────────────────────────────────────────────────────
# UI & Composer Voice State Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestVoiceUIIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls._app = QApplication.instance() or QApplication(sys.argv)

    def test_composer_recording_state_transitions(self):
        from pagepilot.screens.chat import Composer
        comp = Composer()
        self.assertFalse(comp._is_recording)
        self.assertFalse(comp._is_processing_voice)

        # Transition to recording
        comp.set_recording_state("recording")
        self.assertTrue(comp._is_recording)
        self.assertFalse(comp._cancel_rec_btn.isHidden())
        self.assertEqual(comp._mic_btn.text(), "⏹")

        # Transition to processing
        comp.set_recording_state("processing")
        self.assertFalse(comp._is_recording)
        self.assertTrue(comp._is_processing_voice)
        self.assertEqual(comp._mic_btn.text(), "⏳")

        # Transition to idle
        comp.set_recording_state("idle")
        self.assertFalse(comp._is_recording)
        self.assertFalse(comp._is_processing_voice)
        self.assertEqual(comp._mic_btn.text(), "🎙")

    def test_composer_insert_text_does_not_autosend(self):
        from pagepilot.screens.chat import Composer
        comp = Composer()
        sent = []
        comp.send_requested.connect(sent.append)

        comp.insert_text("Transcribed speech from whisper")
        self.assertEqual(comp._input.toPlainText(), "Transcribed speech from whisper")
        # Ensure it was not sent automatically!
        self.assertEqual(len(sent), 0)

    def test_message_widget_read_aloud_button(self):
        from pagepilot.screens.chat import MessageWidget
        msg = MessageWidget("assistant", "Here is a helpful answer.")
        self.assertIsNotNone(msg._speak_btn)
        self.assertEqual(msg._speak_btn.text(), "🔊 Read Aloud")

        # Toggle speaking state
        msg.set_speaking(True)
        self.assertEqual(msg._speak_btn.text(), "⏹ Stop Speaking")
        msg.set_speaking(False)
        self.assertEqual(msg._speak_btn.text(), "🔊 Read Aloud")

    def test_voice_settings_card_instantiates(self):
        from pagepilot.voice.settings_widget import VoiceSettingsCard
        card = VoiceSettingsCard()
        self.assertIsNotNone(card)


if __name__ == "__main__":
    unittest.main(verbosity=2)

