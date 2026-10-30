"""
tests/test_database.py

Unit tests for SQLite conversation storage.

Run with:
    python -m pytest tests/test_database.py -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pagepilot.db.database import ConversationDB, Conversation, Message


class TestConversationDB(unittest.TestCase):
    """Uses a temp file DB so tests are fully isolated."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.db = ConversationDB(db_path=Path(self._tmp.name))

    def tearDown(self):
        os.unlink(self._tmp.name)

    # ── Conversation CRUD ─────────────────────────────────────────────

    def test_create_conversation_returns_id(self):
        conv = self.db.create_conversation("Test chat")
        self.assertIsInstance(conv.id, int)
        self.assertGreater(conv.id, 0)
        self.assertEqual(conv.title, "Test chat")

    def test_list_conversations_empty(self):
        convs = self.db.list_conversations()
        self.assertEqual(convs, [])

    def test_list_conversations_ordered_by_updated_at_desc(self):
        c1 = self.db.create_conversation("First")
        c2 = self.db.create_conversation("Second")
        convs = self.db.list_conversations()
        # Most recently updated first
        self.assertEqual(convs[0].id, c2.id)
        self.assertEqual(convs[1].id, c1.id)

    def test_get_conversation(self):
        conv = self.db.create_conversation("My conversation")
        fetched = self.db.get_conversation(conv.id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.title, "My conversation")

    def test_get_conversation_not_found(self):
        result = self.db.get_conversation(99999)
        self.assertIsNone(result)

    def test_update_title(self):
        conv = self.db.create_conversation("Old title")
        self.db.update_title(conv.id, "New title")
        fetched = self.db.get_conversation(conv.id)
        self.assertEqual(fetched.title, "New title")

    def test_delete_conversation(self):
        conv = self.db.create_conversation("To delete")
        self.db.delete_conversation(conv.id)
        self.assertIsNone(self.db.get_conversation(conv.id))

    def test_delete_conversation_cascades_messages(self):
        conv = self.db.create_conversation("With messages")
        self.db.add_message(conv.id, "user", "Hello")
        self.db.delete_conversation(conv.id)
        msgs = self.db.get_messages(conv.id)
        self.assertEqual(msgs, [])

    # ── Message CRUD ──────────────────────────────────────────────────

    def test_add_and_get_messages(self):
        conv = self.db.create_conversation("Chat")
        self.db.add_message(conv.id, "user",      "Hello!")
        self.db.add_message(conv.id, "assistant", "Hi there!")
        msgs = self.db.get_messages(conv.id)
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0].role,    "user")
        self.assertEqual(msgs[0].content, "Hello!")
        self.assertEqual(msgs[1].role,    "assistant")

    def test_message_default_status_is_complete(self):
        conv = self.db.create_conversation("Chat")
        msg  = self.db.add_message(conv.id, "user", "Test")
        self.assertEqual(msg.status, "complete")

    def test_add_message_with_pending_status(self):
        conv = self.db.create_conversation("Chat")
        msg  = self.db.add_message(conv.id, "assistant", "", status="pending")
        self.assertEqual(msg.status, "pending")

    def test_update_message(self):
        conv = self.db.create_conversation("Chat")
        msg  = self.db.add_message(conv.id, "assistant", "", status="pending")
        self.db.update_message(msg.id, "Final answer.", "complete")
        msgs = self.db.get_messages(conv.id)
        self.assertEqual(msgs[0].content, "Final answer.")
        self.assertEqual(msgs[0].status,  "complete")

    def test_update_message_error_status(self):
        conv = self.db.create_conversation("Chat")
        msg  = self.db.add_message(conv.id, "assistant", "", status="pending")
        self.db.update_message(msg.id, "[Error] timeout", "error")
        msgs = self.db.get_messages(conv.id)
        self.assertEqual(msgs[0].status, "error")

    def test_delete_last_assistant_message(self):
        conv = self.db.create_conversation("Chat")
        self.db.add_message(conv.id, "user",      "Q1")
        self.db.add_message(conv.id, "assistant", "A1")
        self.db.add_message(conv.id, "user",      "Q2")
        self.db.add_message(conv.id, "assistant", "A2")
        removed = self.db.delete_last_assistant_message(conv.id)
        self.assertTrue(removed)
        msgs = self.db.get_messages(conv.id)
        # A2 should be gone; A1 should still be there
        contents = [m.content for m in msgs]
        self.assertNotIn("A2", contents)
        self.assertIn("A1", contents)

    def test_delete_last_assistant_no_messages(self):
        conv = self.db.create_conversation("Empty")
        removed = self.db.delete_last_assistant_message(conv.id)
        self.assertFalse(removed)

    # ── Persistence ───────────────────────────────────────────────────

    def test_messages_persist_across_db_instances(self):
        db_path = Path(self._tmp.name)
        # Write with first instance
        db1 = ConversationDB(db_path=db_path)
        conv = db1.create_conversation("Persistent")
        db1.add_message(conv.id, "user", "Remember this!")

        # Read with second instance (simulates restart)
        db2 = ConversationDB(db_path=db_path)
        msgs = db2.get_messages(conv.id)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].content, "Remember this!")

    def test_conversation_title_persists(self):
        db_path = Path(self._tmp.name)
        db1 = ConversationDB(db_path=db_path)
        conv = db1.create_conversation("Saved title")
        db1.update_title(conv.id, "Updated title")

        db2 = ConversationDB(db_path=db_path)
        fetched = db2.get_conversation(conv.id)
        self.assertEqual(fetched.title, "Updated title")

    # ── Touch / timestamps ────────────────────────────────────────────

    def test_add_message_bumps_conversation_updated_at(self):
        conv = self.db.create_conversation("Chat")
        old_ts = self.db.get_conversation(conv.id).updated_at
        import time; time.sleep(0.01)  # ensure clock advances
        self.db.add_message(conv.id, "user", "Hello")
        new_ts = self.db.get_conversation(conv.id).updated_at
        self.assertGreaterEqual(new_ts, old_ts)

    # ── DB path ───────────────────────────────────────────────────────

    def test_get_db_path(self):
        path = self.db.get_db_path()
        self.assertTrue(path.endswith(".db"))


if __name__ == "__main__":
    unittest.main(verbosity=2)

