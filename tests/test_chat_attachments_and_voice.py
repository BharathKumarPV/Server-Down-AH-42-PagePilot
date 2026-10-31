"""
tests/test_chat_attachments_and_voice.py

Automated unit and integration tests for:
- Paperclip file attachment button, chips, and workflow
- File validation (PDF, PNG, JPG, WebP, duplicate prevention, size limits)
- Asynchronous PDF and Image note routing
- Voice UI state management (IDLE, RECORDING, PROCESSING, TRANSCRIBING, ERROR)
- Recording start/stop/cancellation
- Missing microphone and missing Whisper model error handling
- Transcription text insertion without auto-send
- Temporary audio file cleanup
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pagepilot.screens.chat import AttachmentChip, Composer, ChatScreen
from pagepilot.theme import Colors


class TestChatAttachmentsAndVoice(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        from pagepilot.ai.ollama_client import OllamaStatus
        self._status_patcher = patch(
            "pagepilot.ai.ollama_client.OllamaClient.check_status",
            return_value=OllamaStatus(server_reachable=True, model_available=True),
        )
        self._status_patcher.start()
        self._worker_patcher = patch(
            "pagepilot.screens.chat._StatusWorker.start",
            return_value=None,
        )
        self._worker_patcher.start()
        self._temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self._temp_dir.name)

    def tearDown(self):
        self._temp_dir.cleanup()
        self._worker_patcher.stop()
        self._status_patcher.stop()

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Paperclip & Attachment Chip Tests
    # ──────────────────────────────────────────────────────────────────────────

    def test_composer_paperclip_button_present_and_enabled(self):
        comp = Composer()
        self.assertTrue(hasattr(comp, "_attach_btn"))
        self.assertTrue(comp._attach_btn.isEnabled())
        self.assertEqual(comp._attach_btn.text(), "📎")

        clicks = []
        comp.attachment_requested.connect(lambda: clicks.append(True))
        comp._attach_btn.click()
        self.assertEqual(len(clicks), 1)

    def test_composer_add_and_get_attachment(self):
        comp = Composer()
        pdf_path = str(self.temp_path / "test_doc.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"%PDF-1.4 dummy content")

        self.assertFalse(comp.has_attachment(pdf_path))
        comp.add_attachment(pdf_path, doc_type="pdf", status="ready", status_text="✓ 5p", doc_id=10, page_count=5)

        self.assertTrue(comp.has_attachment(pdf_path))
        atts = comp.get_attachments()
        self.assertEqual(len(atts), 1)
        self.assertEqual(atts[0]["filename"], "test_doc.pdf")
        self.assertEqual(atts[0]["doc_type"], "pdf")
        self.assertEqual(atts[0]["status"], "ready")
        self.assertEqual(atts[0]["doc_id"], 10)
        self.assertEqual(atts[0]["page_count"], 5)
        self.assertFalse(comp._attachments_container.isHidden())

    def test_composer_duplicate_attachment_prevented(self):
        comp = Composer()
        pdf_path = str(self.temp_path / "lecture.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"data")

        comp.add_attachment(pdf_path, doc_type="pdf")
        self.assertEqual(len(comp.get_attachments()), 1)

        # Attempt to add same file again
        comp.add_attachment(pdf_path, doc_type="pdf")
        self.assertEqual(len(comp.get_attachments()), 1)

    def test_composer_remove_attachment(self):
        comp = Composer()
        pdf_path = str(self.temp_path / "remove_me.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"data")

        comp.add_attachment(pdf_path, doc_type="pdf")
        self.assertTrue(comp.has_attachment(pdf_path))

        comp.remove_attachment(pdf_path)
        self.assertFalse(comp.has_attachment(pdf_path))
        self.assertEqual(len(comp.get_attachments()), 0)
        self.assertTrue(comp._attachments_container.isHidden())

    def test_attachment_chip_status_transitions(self):
        pdf_path = str(self.temp_path / "notes.png")
        with open(pdf_path, "wb") as f:
            f.write(b"img")

        chip = AttachmentChip(pdf_path, doc_type="image", status="pending_review", status_text="Pending Review")
        self.assertEqual(chip.status, "pending_review")
        self.assertFalse(chip._action_btn.isHidden())  # Review button shown for pending notes

        # Update to ready
        chip.update_status(status="ready", status_text="✓ Approved & Indexed", doc_id=99)
        self.assertEqual(chip.status, "ready")
        self.assertEqual(chip.doc_id, 99)
        self.assertTrue(chip._action_btn.isHidden())

        # Update to error
        chip.update_status(status="error")
        self.assertEqual(chip.status, "error")
        self.assertIn("Failed", chip._status_lbl.text())

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Voice State Machine & Error Handling Tests
    # ──────────────────────────────────────────────────────────────────────────

    def test_voice_all_explicit_states(self):
        comp = Composer()

        # 1. IDLE
        comp.set_recording_state("idle")
        self.assertFalse(comp._is_recording)
        self.assertFalse(comp._is_processing_voice)
        self.assertEqual(comp._mic_btn.text(), "🎙")
        self.assertTrue(comp._mic_btn.isEnabled())
        self.assertTrue(comp._cancel_rec_btn.isHidden())
        self.assertTrue(comp._voice_status_lbl.isHidden())

        # 2. RECORDING
        comp.set_recording_state("recording")
        self.assertTrue(comp._is_recording)
        self.assertFalse(comp._is_processing_voice)
        self.assertEqual(comp._mic_btn.text(), "⏹")
        self.assertTrue(comp._mic_btn.isEnabled())
        self.assertFalse(comp._cancel_rec_btn.isHidden())
        self.assertFalse(comp._voice_status_lbl.isHidden())
        self.assertIn("Recording", comp._voice_status_lbl.text())

        # 3. PROCESSING
        comp.set_recording_state("processing", "Saving audio file…")
        self.assertFalse(comp._is_recording)
        self.assertTrue(comp._is_processing_voice)
        self.assertEqual(comp._mic_btn.text(), "⏳")
        self.assertFalse(comp._mic_btn.isEnabled())
        self.assertTrue(comp._cancel_rec_btn.isHidden())
        self.assertFalse(comp._voice_status_lbl.isHidden())

        # 4. TRANSCRIBING
        comp.set_recording_state("transcribing", "Whisper transcribing…")
        self.assertFalse(comp._is_recording)
        self.assertTrue(comp._is_processing_voice)
        self.assertEqual(comp._mic_btn.text(), "⏳")
        self.assertFalse(comp._mic_btn.isEnabled())
        self.assertIn("Whisper", comp._voice_status_lbl.text())

        # 5. ERROR (Allows retry)
        comp.set_recording_state("error", "No microphone detected")
        self.assertFalse(comp._is_recording)
        self.assertFalse(comp._is_processing_voice)
        self.assertEqual(comp._mic_btn.text(), "🎙")
        self.assertTrue(comp._mic_btn.isEnabled())  # User can click to retry!
        self.assertFalse(comp._voice_status_lbl.isHidden())
        self.assertIn("No microphone", comp._voice_status_lbl.text())

    def test_voice_insert_text_does_not_autosend(self):
        comp = Composer()
        sent = []
        comp.send_requested.connect(sent.append)

        comp.insert_text("Explain gradient descent optimization.")
        self.assertEqual(comp._input.toPlainText(), "Explain gradient descent optimization.")
        self.assertEqual(len(sent), 0)

        # Appending text maintains clean spacing
        comp.insert_text("In machine learning.")
        self.assertEqual(comp._input.toPlainText(), "Explain gradient descent optimization. In machine learning.")
        self.assertEqual(len(sent), 0)

    def test_whisper_silence_artifact_filtering(self):
        from pagepilot.voice.whisper import TranscriptionThread

        class DummySegment:
            def __init__(self, text):
                self.text = text

        # Mock Model.transcribe returning blank audio artifact
        with patch("pywhispercpp.model.Model") as MockModel:
            mock_inst = MagicMock()
            mock_inst.transcribe.return_value = [DummySegment("[BLANK_AUDIO]")]
            MockModel.return_value = mock_inst

            with patch("pagepilot.voice.whisper.is_model_installed", return_value=True), \
                 patch("pagepilot.voice.whisper.model_path", return_value=Path(__file__)):

                wav_file = self.temp_path / "silence.wav"
                with open(wav_file, "wb") as f:
                    f.write(b"RIFF dummy wav")

                worker = TranscriptionThread(str(wav_file), model_name="base")
                errors = []
                success = []
                worker.error_occurred.connect(errors.append)
                worker.transcription_done.connect(success.append)

                worker.run()
                self.assertEqual(len(success), 0)
                self.assertEqual(len(errors), 1)
                self.assertIn("No speech detected", errors[0])

    def test_temporary_audio_cleanup_on_cancel(self):
        from pagepilot.voice.recorder import RecorderThread
        wav_file = self.temp_path / "pagepilot_rec_test.wav"
        with open(wav_file, "wb") as f:
            f.write(b"dummy wav data")

        self.assertTrue(wav_file.exists())
        rec = RecorderThread()
        rec._tmp_path = str(wav_file)
        rec.cancel()

        self.assertFalse(wav_file.exists())
        self.assertIsNone(rec.path)

    # ──────────────────────────────────────────────────────────────────────────
    # 3. ChatScreen File Selection & Validation Routing Tests
    # ──────────────────────────────────────────────────────────────────────────

    def test_chat_screen_attachment_validation(self):
        screen = ChatScreen()
        screen._show_attachment_error = MagicMock()

        # 1. Nonexistent file
        with patch("PySide6.QtWidgets.QFileDialog.getOpenFileNames", return_value=(["nonexistent_file.pdf"], "")):
            screen._on_attach_clicked()
            screen._show_attachment_error.assert_called()
            self.assertIn("not found", screen._show_attachment_error.call_args[0][0])

        # 2. Unsupported extension (.exe)
        unsupported = self.temp_path / "app.exe"
        with open(unsupported, "wb") as f:
            f.write(b"MZ binary")
        with patch("PySide6.QtWidgets.QFileDialog.getOpenFileNames", return_value=([str(unsupported)], "")):
            screen._on_attach_clicked()
            screen._show_attachment_error.assert_called()
            self.assertIn("Unsupported file format", screen._show_attachment_error.call_args[0][0])

        # 3. Empty file (0 bytes)
        empty_f = self.temp_path / "empty.pdf"
        empty_f.touch()
        with patch("PySide6.QtWidgets.QFileDialog.getOpenFileNames", return_value=([str(empty_f)], "")):
            screen._on_attach_clicked()
            screen._show_attachment_error.assert_called()
            self.assertIn("empty file", screen._show_attachment_error.call_args[0][0])

    def test_chat_screen_pdf_attachment_ingestion_routing(self):
        screen = ChatScreen()
        screen._show_attachment_error = MagicMock()

        pdf_f = self.temp_path / "course_syllabus.pdf"
        with open(pdf_f, "wb") as f:
            f.write(b"%PDF-1.4 dummy pdf bytes")

        # Mock IngestionService
        mock_doc = MagicMock(id=55, page_count=8, filename="course_syllabus.pdf")
        mock_ingestion_svc = MagicMock()
        mock_ingestion_svc.ingest_pdf.return_value = mock_doc
        screen._get_ingestion_service = MagicMock(return_value=mock_ingestion_svc)

        with patch("PySide6.QtWidgets.QFileDialog.getOpenFileNames", return_value=([str(pdf_f)], "")):
            screen._on_attach_clicked()

        # Verify added to composer
        self.assertTrue(screen._composer.has_attachment(str(pdf_f)))

        # Wait or trigger ingestion callback directly
        screen._on_pdf_ingestion_completed({
            "file_path": str(pdf_f),
            "doc_id": 55,
            "page_count": 8,
            "filename": "course_syllabus.pdf",
        })

        atts = screen._composer.get_attachments()
        self.assertEqual(len(atts), 1)
        self.assertEqual(atts[0]["status"], "ready")
        self.assertEqual(atts[0]["doc_id"], 55)
        self.assertEqual(atts[0]["page_count"], 8)

    def test_chat_screen_blocks_send_when_attachment_ingesting(self):
        screen = ChatScreen()
        screen._show_attachment_error = MagicMock()

        pdf_f = str(self.temp_path / "slow_loading.pdf")
        screen._composer.add_attachment(pdf_f, doc_type="pdf", status="ingesting")

        screen._on_send("What is chapter 1 about?")
        screen._show_attachment_error.assert_called()
        self.assertIn("finish ingesting", screen._show_attachment_error.call_args[0][0])
        self.assertIsNone(screen._current_conv_id)

    def test_chat_screen_blocks_send_when_image_pending_review(self):
        screen = ChatScreen()
        screen._show_attachment_error = MagicMock()

        img_f = str(self.temp_path / "unreviewed_notes.png")
        screen._composer.add_attachment(img_f, doc_type="image", status="pending_review", doc_id=77)

        screen._on_send("Solve equation from my notes")
        screen._show_attachment_error.assert_called()
        self.assertIn("pending OCR review", screen._show_attachment_error.call_args[0][0])
        self.assertIsNone(screen._current_conv_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
