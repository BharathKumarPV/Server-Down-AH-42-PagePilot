"""
PagePilot – SQLite conversation database.

Schema
------
conversations(id, title, created_at, updated_at)
messages(id, conversation_id, role, content, created_at, status)

Status values: 'complete' | 'cancelled' | 'error' | 'pending'

All public methods acquire a connection per call and close it cleanly.
The DB is stored in %APPDATA%/PagePilot/conversations.db.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

from pagepilot.config import DB_PATH, ensure_data_dir


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Conversation:
    id: int
    title: str
    created_at: str
    updated_at: str


@dataclass
class Message:
    id: int
    conversation_id: int
    role: str
    content: str
    created_at: str
    status: str = "complete"


# ──────────────────────────────────────────────────────────────────────────────
# Database manager
# ──────────────────────────────────────────────────────────────────────────────

class ConversationDB:
    """Manages all SQLite access for PagePilot chat history."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        ensure_data_dir()
        self._init_schema()

    # ── Internal helpers ─────────────────────────────────────────────

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        con = sqlite3.connect(str(self.db_path))
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _init_schema(self):
        with self._conn() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    title      TEXT    NOT NULL DEFAULT 'New conversation',
                    created_at TEXT    NOT NULL,
                    updated_at TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role            TEXT    NOT NULL CHECK(role IN ('system','user','assistant')),
                    content         TEXT    NOT NULL,
                    created_at      TEXT    NOT NULL,
                    status          TEXT    NOT NULL DEFAULT 'complete'
                                            CHECK(status IN ('complete','cancelled','error','pending'))
                );

                CREATE INDEX IF NOT EXISTS idx_messages_conv
                    ON messages(conversation_id, created_at);
            """)

    # ── Conversations ────────────────────────────────────────────────

    def create_conversation(self, title: str = "New conversation") -> Conversation:
        now = self._now()
        with self._conn() as con:
            cur = con.execute(
                "INSERT INTO conversations(title, created_at, updated_at) VALUES(?,?,?)",
                (title, now, now),
            )
            return Conversation(id=cur.lastrowid, title=title,
                                created_at=now, updated_at=now)

    def list_conversations(self) -> list[Conversation]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT id, title, created_at, updated_at "
                "FROM conversations ORDER BY updated_at DESC"
            ).fetchall()
            return [Conversation(**dict(r)) for r in rows]

    def get_conversation(self, conv_id: int) -> Conversation | None:
        with self._conn() as con:
            row = con.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id=?",
                (conv_id,),
            ).fetchone()
            return Conversation(**dict(row)) if row else None

    def update_title(self, conv_id: int, title: str):
        with self._conn() as con:
            con.execute(
                "UPDATE conversations SET title=?, updated_at=? WHERE id=?",
                (title, self._now(), conv_id),
            )

    def touch_conversation(self, conv_id: int):
        with self._conn() as con:
            con.execute(
                "UPDATE conversations SET updated_at=? WHERE id=?",
                (self._now(), conv_id),
            )

    def delete_conversation(self, conv_id: int):
        with self._conn() as con:
            con.execute("DELETE FROM conversations WHERE id=?", (conv_id,))

    # ── Messages ─────────────────────────────────────────────────────

    def add_message(
        self,
        conv_id: int,
        role: str,
        content: str,
        status: str = "complete",
    ) -> Message:
        now = self._now()
        with self._conn() as con:
            cur = con.execute(
                "INSERT INTO messages(conversation_id, role, content, created_at, status) "
                "VALUES(?,?,?,?,?)",
                (conv_id, role, content, now, status),
            )
            # Also bump conversation updated_at
            con.execute(
                "UPDATE conversations SET updated_at=? WHERE id=?",
                (now, conv_id),
            )
            return Message(
                id=cur.lastrowid,
                conversation_id=conv_id,
                role=role,
                content=content,
                created_at=now,
                status=status,
            )

    def update_message(self, msg_id: int, content: str, status: str = "complete"):
        with self._conn() as con:
            con.execute(
                "UPDATE messages SET content=?, status=? WHERE id=?",
                (content, status, msg_id),
            )

    def get_messages(self, conv_id: int) -> list[Message]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT id, conversation_id, role, content, created_at, status "
                "FROM messages WHERE conversation_id=? ORDER BY created_at ASC",
                (conv_id,),
            ).fetchall()
            return [Message(**dict(r)) for r in rows]

    def delete_last_assistant_message(self, conv_id: int) -> bool:
        """Remove the most recent assistant message. Used for regeneration."""
        with self._conn() as con:
            row = con.execute(
                "SELECT id FROM messages WHERE conversation_id=? AND role='assistant' "
                "ORDER BY created_at DESC LIMIT 1",
                (conv_id,),
            ).fetchone()
            if row:
                con.execute("DELETE FROM messages WHERE id=?", (row["id"],))
                return True
            return False

    def get_db_path(self) -> str:
        return str(self.db_path)

