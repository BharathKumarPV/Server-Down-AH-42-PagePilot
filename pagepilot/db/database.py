"""
PagePilot – SQLite databases for conversations and documents.

ConversationDB:
  conversations(id, title, created_at, updated_at)
  messages(id, conversation_id, role, content, created_at, status)

DocumentDB (Stage 4):
  documents(id, filename, file_path, sha256_hash, page_count, import_time, status, embed_model, chunk_count, error_message)
  chunks(chunk_id, document_id, filename, page_number, chunk_index, text, token_count, section_heading)
  chunks_fts (FTS5 virtual table for keyword retrieval)
"""

from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, List, Optional

from pagepilot.config import DB_PATH, DOCS_DB_PATH, ensure_data_dir
from pagepilot.pdf.models import Chunk, Document


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
# ConversationDB manager
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


# ──────────────────────────────────────────────────────────────────────────────
# DocumentDB manager (Stage 4)
# ──────────────────────────────────────────────────────────────────────────────

class DocumentDB:
    """Manages SQLite storage for PDF metadata, chunks, and FTS5 search."""

    def __init__(self, db_path: Path = DOCS_DB_PATH):
        self.db_path = Path(db_path)
        ensure_data_dir()
        self._fts_available = False
        self._init_schema()

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
                CREATE TABLE IF NOT EXISTS documents (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename      TEXT    NOT NULL,
                    file_path     TEXT    NOT NULL,
                    sha256_hash   TEXT    UNIQUE NOT NULL,
                    page_count    INTEGER NOT NULL,
                    import_time   TEXT    NOT NULL,
                    status        TEXT    NOT NULL DEFAULT 'pending',
                    embed_model   TEXT    NOT NULL,
                    chunk_count   INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT    DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id        TEXT PRIMARY KEY,
                    document_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    filename        TEXT NOT NULL,
                    page_number     INTEGER NOT NULL,
                    chunk_index     INTEGER NOT NULL,
                    text            TEXT NOT NULL,
                    token_count     INTEGER NOT NULL,
                    section_heading TEXT DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_doc
                    ON chunks(document_id);
            """)

            # Initialize FTS5 table if supported
            try:
                con.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                        chunk_id UNINDEXED,
                        document_id UNINDEXED,
                        filename,
                        page_number UNINDEXED,
                        text
                    );
                """)
                self._fts_available = True
            except sqlite3.OperationalError:
                self._fts_available = False

    @property
    def fts_available(self) -> bool:
        return self._fts_available

    # ── Document operations ──────────────────────────────────────────

    def create_document(
        self,
        filename: str,
        file_path: str,
        sha256_hash: str,
        page_count: int,
        embed_model: str,
        status: str = "pending",
    ) -> Document:
        now = self._now()
        with self._conn() as con:
            cur = con.execute(
                """
                INSERT INTO documents (
                    filename, file_path, sha256_hash, page_count,
                    import_time, status, embed_model, chunk_count, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, '')
                """,
                (filename, file_path, sha256_hash, page_count, now, status, embed_model),
            )
            return Document(
                id=cur.lastrowid,
                filename=filename,
                file_path=file_path,
                sha256_hash=sha256_hash,
                page_count=page_count,
                import_time=now,
                status=status,
                embed_model=embed_model,
                chunk_count=0,
                error_message="",
            )

    def get_document_by_hash(self, sha256_hash: str) -> Document | None:
        with self._conn() as con:
            row = con.execute(
                "SELECT * FROM documents WHERE sha256_hash = ?", (sha256_hash,)
            ).fetchone()
            return Document(**dict(row)) if row else None

    def get_document(self, doc_id: int) -> Document | None:
        with self._conn() as con:
            row = con.execute(
                "SELECT * FROM documents WHERE id = ?", (doc_id,)
            ).fetchone()
            return Document(**dict(row)) if row else None

    def list_documents(self) -> list[Document]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM documents ORDER BY import_time DESC"
            ).fetchall()
            return [Document(**dict(r)) for r in rows]

    def update_document_status(
        self,
        doc_id: int,
        status: str,
        chunk_count: int = 0,
        error_message: str = "",
    ):
        with self._conn() as con:
            con.execute(
                """
                UPDATE documents
                SET status = ?, chunk_count = ?, error_message = ?
                WHERE id = ?
                """,
                (status, chunk_count, error_message, doc_id),
            )

    def delete_document(self, doc_id: int):
        with self._conn() as con:
            if self._fts_available:
                try:
                    con.execute("DELETE FROM chunks_fts WHERE document_id = ?", (doc_id,))
                except Exception:
                    pass
            con.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
            con.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    def count_documents(self) -> int:
        with self._conn() as con:
            row = con.execute("SELECT COUNT(*) as cnt FROM documents").fetchone()
            return row["cnt"] if row else 0

    def count_chunks(self) -> int:
        with self._conn() as con:
            row = con.execute("SELECT COUNT(*) as cnt FROM chunks").fetchone()
            return row["cnt"] if row else 0

    # ── Chunk operations ─────────────────────────────────────────────

    def insert_chunks(self, chunks: list[Chunk]):
        if not chunks:
            return
        with self._conn() as con:
            chunk_records = [
                (
                    c.chunk_id,
                    c.document_id,
                    c.filename,
                    c.page_number,
                    c.chunk_index,
                    c.text,
                    c.token_count,
                    c.section_heading,
                )
                for c in chunks
            ]
            con.executemany(
                """
                INSERT OR REPLACE INTO chunks (
                    chunk_id, document_id, filename, page_number,
                    chunk_index, text, token_count, section_heading
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                chunk_records,
            )

            if self._fts_available:
                fts_records = [
                    (c.chunk_id, c.document_id, c.filename, c.page_number, c.text)
                    for c in chunks
                ]
                try:
                    con.executemany(
                        """
                        INSERT INTO chunks_fts (
                            chunk_id, document_id, filename, page_number, text
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        fts_records,
                    )
                except Exception:
                    pass

    def get_chunks_for_document(self, doc_id: int) -> list[Chunk]:
        with self._conn() as con:
            rows = con.execute(
                """
                SELECT chunk_id, document_id, filename, page_number,
                       chunk_index, text, token_count, section_heading
                FROM chunks WHERE document_id = ?
                ORDER BY page_number ASC, chunk_index ASC
                """,
                (doc_id,),
            ).fetchall()
            return [Chunk(**dict(r)) for r in rows]

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        with self._conn() as con:
            row = con.execute(
                """
                SELECT chunk_id, document_id, filename, page_number,
                       chunk_index, text, token_count, section_heading
                FROM chunks WHERE chunk_id = ?
                """,
                (chunk_id,),
            ).fetchone()
            return Chunk(**dict(row)) if row else None

    # ── Keyword search (FTS5 with LIKE fallback) ─────────────────────

    def search_keyword(
        self,
        query: str,
        document_ids: list[int] | None = None,
        top_k: int = 5,
    ) -> list[Chunk]:
        """
        Search chunks using SQLite FTS5 matching, falling back to LIKE if FTS5 fails.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        doc_filter = ""
        params: list[object] = []

        if document_ids:
            placeholders = ",".join("?" for _ in document_ids)
            doc_filter = f" AND document_id IN ({placeholders})"

        if self._fts_available:
            try:
                # Sanitize query for FTS5: extract alphanumeric tokens
                words = re.findall(r"[A-Za-z0-9_]+", cleaned_query)
                if words:
                    # Match all terms
                    match_expr = " AND ".join(f'"{w}"*' for w in words)
                    sql = f"""
                        SELECT c.chunk_id, c.document_id, c.filename, c.page_number,
                               c.chunk_index, c.text, c.token_count, c.section_heading
                        FROM chunks c
                        JOIN chunks_fts f ON c.chunk_id = f.chunk_id
                        WHERE chunks_fts MATCH ? {doc_filter}
                        LIMIT ?
                    """
                    sql_params = [match_expr]
                    if document_ids:
                        sql_params.extend(document_ids)
                    sql_params.append(top_k)

                    with self._conn() as con:
                        rows = con.execute(sql, sql_params).fetchall()
                        if rows:
                            return [Chunk(**dict(r)) for r in rows]
            except sqlite3.OperationalError:
                # Fall through to LIKE fallback
                pass

        # LIKE fallback
        like_term = f"%{cleaned_query}%"
        sql = f"""
            SELECT chunk_id, document_id, filename, page_number,
                   chunk_index, text, token_count, section_heading
            FROM chunks
            WHERE text LIKE ? {doc_filter}
            ORDER BY page_number ASC, chunk_index ASC
            LIMIT ?
        """
        sql_params = [like_term]
        if document_ids:
            sql_params.extend(document_ids)
        sql_params.append(top_k)

        with self._conn() as con:
            rows = con.execute(sql, sql_params).fetchall()
            return [Chunk(**dict(r)) for r in rows]
