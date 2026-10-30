"""
tests/test_chat_integration.py

Integration tests for the chat flow:
  - Conversation creation and DB persistence
  - Message context building
  - Regeneration logic
  - Markdown renderer (no Qt required)

UI-level tests use offscreen Qt rendering.

Run with:
    python -m pytest tests/test_chat_integration.py -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Qt must be initialised before importing any QWidget
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ──────────────────────────────────────────────────────────────────────────────
# Markdown renderer tests (pure Python, no Qt)
# ──────────────────────────────────────────────────────────────────────────────

class TestMarkdownRenderer(unittest.TestCase):

    def _render(self, text: str) -> str:
        from pagepilot.screens.chat import _md_to_html
        return _md_to_html(text)

    def test_plain_text_wrapped_in_paragraph(self):
        html = self._render("Hello world")
        self.assertIn("Hello world", html)

    def test_bold_renders(self):
        html = self._render("**bold text**")
        self.assertIn("<b>bold text</b>", html)

    def test_heading_renders(self):
        html = self._render("## Section")
        self.assertIn("<h2", html)
        self.assertIn("Section", html)

    def test_code_fence_renders(self):
        html = self._render("```python\nprint('hi')\n```")
        self.assertIn("<pre", html)
        self.assertIn("print(&#x27;hi&#x27;)", html)   # html-escaped

    def test_inline_code_renders(self):
        html = self._render("Use `os.path.join`")
        self.assertIn("<code", html)
        self.assertIn("os.path.join", html)

    def test_bullet_list_renders(self):
        html = self._render("- item one\n- item two")
        self.assertIn("<li", html)
        self.assertIn("item one", html)

    def test_no_script_tags_from_model_output(self):
        """Model output must not inject executable HTML."""
        html = self._render("<script>alert('xss')</script>")
        self.assertNotIn("<script>", html)

    def test_empty_string(self):
        html = self._render("")
        self.assertIsInstance(html, str)


# ──────────────────────────────────────────────────────────────────────────────
# Database-backed conversation flow tests
# ──────────────────────────────────────────────────────────────────────────────

class TestConversationFlow(unittest.TestCase):

    def setUp(self):
        from pagepilot.db.database import ConversationDB
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.db = ConversationDB(db_path=Path(self._tmp.name))

    def tearDown(self):
        os.unlink(self._tmp.name)

    def test_new_conversation_and_first_message(self):
        conv = self.db.create_conversation("What is AI?")
        msg  = self.db.add_message(conv.id, "user", "What is AI?")
        msgs = self.db.get_messages(conv.id)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].content, "What is AI?")
        self.assertEqual(msgs[0].role,    "user")

    def test_assistant_response_saved(self):
        conv = self.db.create_conversation("Q&A")
        self.db.add_message(conv.id, "user",      "What is 2+2?")
        msg  = self.db.add_message(conv.id, "assistant", "", status="pending")
        self.db.update_message(msg.id, "2+2 equals 4.", "complete")
        msgs = self.db.get_messages(conv.id)
        self.assertEqual(msgs[1].content, "2+2 equals 4.")
        self.assertEqual(msgs[1].status,  "complete")

    def test_follow_up_messages_ordered(self):
        conv = self.db.create_conversation("Multi-turn")
        self.db.add_message(conv.id, "user",      "Hello")
        self.db.add_message(conv.id, "assistant", "Hi!")
        self.db.add_message(conv.id, "user",      "How are you?")
        self.db.add_message(conv.id, "assistant", "I'm fine.")
        msgs = self.db.get_messages(conv.id)
        self.assertEqual(len(msgs), 4)
        self.assertEqual(msgs[0].role, "user")
        self.assertEqual(msgs[1].role, "assistant")
        self.assertEqual(msgs[2].role, "user")
        self.assertEqual(msgs[3].role, "assistant")

    def test_regeneration_removes_last_assistant(self):
        conv = self.db.create_conversation("Regen test")
        self.db.add_message(conv.id, "user",      "Tell me a joke")
        self.db.add_message(conv.id, "assistant", "Why did the chicken...")
        self.db.delete_last_assistant_message(conv.id)
        msgs = self.db.get_messages(conv.id)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].role, "user")

    def test_cancelled_response_stored_with_status(self):
        conv = self.db.create_conversation("Cancelled")
        self.db.add_message(conv.id, "user", "Count to 1000")
        msg  = self.db.add_message(conv.id, "assistant", "", status="pending")
        self.db.update_message(msg.id, "[Cancelled] 1, 2, 3", "cancelled")
        msgs = self.db.get_messages(conv.id)
        self.assertEqual(msgs[1].status, "cancelled")

    def test_error_response_stored_with_status(self):
        conv = self.db.create_conversation("Error test")
        self.db.add_message(conv.id, "user", "Hello")
        msg  = self.db.add_message(conv.id, "assistant", "", status="pending")
        self.db.update_message(msg.id, "[Error] Connection lost", "error")
        msgs = self.db.get_messages(conv.id)
        self.assertEqual(msgs[1].status, "error")

    def test_multiple_conversations_independent(self):
        c1 = self.db.create_conversation("Conv 1")
        c2 = self.db.create_conversation("Conv 2")
        self.db.add_message(c1.id, "user", "Message in conv 1")
        self.db.add_message(c2.id, "user", "Message in conv 2")
        msgs1 = self.db.get_messages(c1.id)
        msgs2 = self.db.get_messages(c2.id)
        self.assertEqual(len(msgs1), 1)
        self.assertEqual(len(msgs2), 1)
        self.assertEqual(msgs1[0].content, "Message in conv 1")
        self.assertEqual(msgs2[0].content, "Message in conv 2")

    def test_title_truncation_pattern(self):
        """Titles are set to first 60 chars of user message in the app."""
        long_msg = "A" * 100
        title    = long_msg[:60].strip()
        conv     = self.db.create_conversation(title)
        self.assertEqual(len(conv.title), 60)

    def test_switch_conversation_loads_correct_messages(self):
        c1 = self.db.create_conversation("First")
        c2 = self.db.create_conversation("Second")
        self.db.add_message(c1.id, "user", "Q from first")
        self.db.add_message(c2.id, "user", "Q from second")
        # Simulate switching to c1
        msgs = self.db.get_messages(c1.id)
        self.assertEqual(msgs[0].content, "Q from first")
        # Simulate switching to c2
        msgs = self.db.get_messages(c2.id)
        self.assertEqual(msgs[0].content, "Q from second")

    def test_delete_conversation_confirmed(self):
        conv = self.db.create_conversation("To delete")
        self.db.add_message(conv.id, "user", "Hello")
        self.db.delete_conversation(conv.id)
        self.assertIsNone(self.db.get_conversation(conv.id))
        self.assertEqual(self.db.get_messages(conv.id), [])


# ──────────────────────────────────────────────────────────────────────────────
# UI smoke tests (offscreen Qt)
# ──────────────────────────────────────────────────────────────────────────────

class TestChatScreenUI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls._app = QApplication.instance() or QApplication(sys.argv)
        from pagepilot.theme import apply_theme
        apply_theme(cls._app)

    def setUp(self):
        from PySide6.QtWidgets import QApplication
        from pagepilot.ai.ollama_client import OllamaStatus

        # Patch OllamaClient.check_status so UI tests never hit the network.
        self._status_patcher = patch(
            "pagepilot.ai.ollama_client.OllamaClient.check_status",
            return_value=OllamaStatus(server_reachable=True, model_available=True),
        )
        self._status_patcher.start()

        # Suppress the background _StatusWorker QThread so it never blocks
        # the event loop or outlives the test's patch context.
        self._worker_patcher = patch(
            "pagepilot.screens.chat._StatusWorker.start",
            return_value=None,
        )
        self._worker_patcher.start()

        self._screens = []   # keep references; closed in tearDown

    def tearDown(self):
        from PySide6.QtWidgets import QApplication
        for screen in self._screens:
            try:
                screen.close()
                screen.deleteLater()
            except RuntimeError:
                pass
        self._screens.clear()
        QApplication.processEvents()
        self._worker_patcher.stop()
        self._status_patcher.stop()

    def _make_screen(self):
        """
        Create a ChatScreen that is shown inside a top-level window.

        Qt propagates visibility from parent → child: a widget's isVisible()
        returns False unless every ancestor up to the top-level window has
        been shown.  Showing the container here makes isVisible() / isHidden()
        behave identically to what happens at runtime, so all assertions remain
        meaningful without being weakened.
        """
        from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout
        from pagepilot.screens.chat import ChatScreen

        container = QWidget()
        container.setWindowTitle("PagePilot – test")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        screen = ChatScreen(parent=container)
        layout.addWidget(screen)

        container.resize(900, 600)
        container.show()
        QApplication.processEvents()

        # Keep both alive for the test's duration
        self._screens.append(container)
        return screen

    def test_chat_screen_instantiates(self):
        screen = self._make_screen()
        self.assertIsNotNone(screen)

    def test_welcome_shown_initially(self):
        screen = self._make_screen()
        self.assertTrue(screen._welcome.isVisible())
        self.assertFalse(screen._convo.isVisible())

    def test_convo_hidden_initially(self):
        screen = self._make_screen()
        self.assertFalse(screen._convo.isVisible())

    def test_new_chat_resets_to_welcome(self):
        screen = self._make_screen()
        # Simulate a conversation being active
        screen._current_conv_id = 42
        screen._convo.show()
        screen._welcome.hide()
        screen._on_new_chat()
        self.assertTrue(screen._welcome.isVisible())
        self.assertFalse(screen._convo.isVisible())
        self.assertIsNone(screen._current_conv_id)

    def test_composer_send_blocked_when_empty(self):
        screen = self._make_screen()
        sent = []
        screen._on_send("")
        # No conversation should have been created
        self.assertIsNone(screen._current_conv_id)

    def test_sidebar_navigation_all_screens(self):
        """All 6 nav items render without error."""
        from pagepilot.ui.main_window import MainWindow
        win = MainWindow()
        from pagepilot.config import NAV_CHAT, NAV_HISTORY, NAV_PDF, NAV_DOCX, NAV_MAPS, NAV_SETTINGS
        for nav_id in [NAV_HISTORY, NAV_PDF, NAV_DOCX, NAV_MAPS, NAV_SETTINGS, NAV_CHAT]:
            win._switch_screen(nav_id)
            win._sidebar.select(nav_id)

    def test_window_resize(self):
        from pagepilot.ui.main_window import MainWindow
        win = MainWindow()
        win.resize(1400, 900)
        win.resize(900, 600)
        self.assertGreaterEqual(win.width(), 900)

    def test_stop_button_hidden_when_not_generating(self):
        screen = self._make_screen()
        self.assertFalse(screen._composer._stop_btn.isVisible())
        self.assertTrue(screen._composer._send_btn.isVisible())

    def test_generating_state_shows_stop_button(self):
        screen = self._make_screen()
        screen._composer.set_generating(True)
        self.assertTrue(screen._composer._stop_btn.isVisible())
        self.assertFalse(screen._composer._send_btn.isVisible())
        screen._composer.set_generating(False)
        self.assertFalse(screen._composer._stop_btn.isVisible())


if __name__ == "__main__":
    unittest.main(verbosity=2)

