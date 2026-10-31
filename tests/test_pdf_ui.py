"""
Tests for PDF Assistant UI Screen (Stage 4).
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from pagepilot.pdf.models import Citation, Document
from pagepilot.screens.pdf_assistant import CitationPill, PdfScreen, QAMessageWidget


class TestPdfAssistantUI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication(sys.argv)

    def test_screen_initialization(self):
        screen = PdfScreen()
        assert screen is not None
        assert screen._import_btn is not None
        assert screen._summarize_btn is not None
        assert screen._input is not None
        assert screen._send_btn is not None
        assert screen._stop_btn is not None

    def test_citation_pill_widget(self):
        cit = Citation(1, "manual.pdf", 7, "Sample snippet text from page 7", 0.8)
        pill = CitationPill(cit, file_path="/fake/path.pdf")
        assert "manual.pdf, p. 7" in pill.text()
        assert "Sample snippet" in pill.toolTip()

    def test_qa_message_widget_assistant_and_user(self):
        # User message
        u_msg = QAMessageWidget(role="user", text="How to start the engine?")
        assert u_msg.role == "user"

        # Assistant message with citations
        cit = Citation(2, "guide.pdf", 12, "Engine start steps...", 0.9)
        a_msg = QAMessageWidget(role="assistant", text="Follow step 1 to 3 [guide.pdf, p. 12].", citations=[cit])
        assert a_msg.role == "assistant"
        assert a_msg._browser is not None
        assert not a_msg._citations_box.isHidden()
