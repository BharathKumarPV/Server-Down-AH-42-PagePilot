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

                CREATE TABLE IF NOT EXISTS conversation_documents (
                    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    document_id     INTEGER NOT NULL,
                    PRIMARY KEY (conversation_id, document_id)
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

    def set_conversation_documents(self, conv_id: int, doc_ids: list[int]):
        """Persist selected document IDs associated with this conversation."""
        with self._conn() as con:
            con.execute("DELETE FROM conversation_documents WHERE conversation_id = ?", (conv_id,))
            con.executemany(
                "INSERT OR IGNORE INTO conversation_documents (conversation_id, document_id) VALUES (?, ?)",
                [(conv_id, did) for did in doc_ids],
            )

    def get_conversation_documents(self, conv_id: int) -> list[int]:
        """Retrieve persisted document IDs for this conversation."""
        with self._conn() as con:
            rows = con.execute(
                "SELECT document_id FROM conversation_documents WHERE conversation_id = ? ORDER BY document_id ASC",
                (conv_id,),
            ).fetchall()
            return [int(r["document_id"]) for r in rows]

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
                    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename              TEXT    NOT NULL,
                    file_path             TEXT    NOT NULL,
                    sha256_hash           TEXT    UNIQUE NOT NULL,
                    page_count            INTEGER NOT NULL,
                    import_time           TEXT    NOT NULL,
                    status                TEXT    NOT NULL DEFAULT 'pending',
                    embed_model           TEXT    NOT NULL,
                    chunk_count           INTEGER NOT NULL DEFAULT 0,
                    error_message         TEXT    DEFAULT '',
                    extracted_pages_count INTEGER NOT NULL DEFAULT 0,
                    scanned_pages_count   INTEGER NOT NULL DEFAULT 0,
                    char_count            INTEGER NOT NULL DEFAULT 0,
                    embedded_chunks_count INTEGER NOT NULL DEFAULT 0,
                    indexed_pages_count   INTEGER NOT NULL DEFAULT 0
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

                CREATE TABLE IF NOT EXISTS ocr_pages (
                    document_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    page_number     INTEGER NOT NULL,
                    image_path      TEXT NOT NULL,
                    status          TEXT NOT NULL DEFAULT 'pending_review',
                    created_at      TEXT NOT NULL,
                    updated_at      TEXT NOT NULL,
                    error_message   TEXT DEFAULT '',
                    PRIMARY KEY (document_id, page_number)
                );

                CREATE TABLE IF NOT EXISTS ocr_regions (
                    region_id           TEXT PRIMARY KEY,
                    document_id         INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    page_number         INTEGER NOT NULL,
                    region_type         TEXT NOT NULL DEFAULT 'text',
                    box_x0              REAL NOT NULL,
                    box_y0              REAL NOT NULL,
                    box_x1              REAL NOT NULL,
                    box_y1              REAL NOT NULL,
                    recognized_text     TEXT NOT NULL,
                    corrected_text      TEXT,
                    latex_expression    TEXT,
                    confidence_score    REAL DEFAULT 0.0,
                    recognition_engine  TEXT DEFAULT '',
                    model_version       TEXT DEFAULT '',
                    approval_status     TEXT NOT NULL DEFAULT 'pending_review',
                    image_crop_path     TEXT DEFAULT '',
                    created_at          TEXT NOT NULL,
                    updated_at          TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_ocr_regions_doc_page
                    ON ocr_regions(document_id, page_number);
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

            # Add diagnostics columns if upgrading from earlier schema
            for col, default_val in [
                ("extracted_pages_count", "0"),
                ("scanned_pages_count", "0"),
                ("char_count", "0"),
                ("embedded_chunks_count", "0"),
                ("indexed_pages_count", "0"),
            ]:
                try:
                    con.execute(f"ALTER TABLE documents ADD COLUMN {col} INTEGER NOT NULL DEFAULT {default_val}")
                except sqlite3.OperationalError:
                    pass

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
        extracted_pages_count: int = 0,
        scanned_pages_count: int = 0,
        char_count: int = 0,
        embedded_chunks_count: int = 0,
        indexed_pages_count: int = 0,
    ) -> Document:
        now = self._now()
        with self._conn() as con:
            cur = con.execute(
                """
                INSERT INTO documents (
                    filename, file_path, sha256_hash, page_count,
                    import_time, status, embed_model, chunk_count, error_message,
                    extracted_pages_count, scanned_pages_count, char_count,
                    embedded_chunks_count, indexed_pages_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, '', ?, ?, ?, ?, ?)
                """,
                (
                    filename, file_path, sha256_hash, page_count, now, status, embed_model,
                    extracted_pages_count, scanned_pages_count, char_count,
                    embedded_chunks_count, indexed_pages_count,
                ),
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
                extracted_pages_count=extracted_pages_count,
                scanned_pages_count=scanned_pages_count,
                char_count=char_count,
                embedded_chunks_count=embedded_chunks_count,
                indexed_pages_count=indexed_pages_count,
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
        extracted_pages_count: Optional[int] = None,
        scanned_pages_count: Optional[int] = None,
        char_count: Optional[int] = None,
        embedded_chunks_count: Optional[int] = None,
        indexed_pages_count: Optional[int] = None,
    ):
        with self._conn() as con:
            updates = ["status = ?", "chunk_count = ?", "error_message = ?"]
            params: list[object] = [status, chunk_count, error_message]
            if extracted_pages_count is not None:
                updates.append("extracted_pages_count = ?")
                params.append(extracted_pages_count)
            if scanned_pages_count is not None:
                updates.append("scanned_pages_count = ?")
                params.append(scanned_pages_count)
            if char_count is not None:
                updates.append("char_count = ?")
                params.append(char_count)
            if embedded_chunks_count is not None:
                updates.append("embedded_chunks_count = ?")
                params.append(embedded_chunks_count)
            if indexed_pages_count is not None:
                updates.append("indexed_pages_count = ?")
                params.append(indexed_pages_count)
            params.append(doc_id)
            con.execute(
                f"UPDATE documents SET {', '.join(updates)} WHERE id = ?",
                params,
            )

    def get_document_diagnostics(self, doc_id: int) -> dict | None:
        """Fetch accurate, non-fabricated metadata for the document diagnostics panel."""
        doc = self.get_document(doc_id)
        if not doc:
            return None
        return {
            "id": doc.id,
            "filename": doc.filename,
            "file_path": doc.file_path,
            "sha256_hash": doc.sha256_hash,
            "total_pages": doc.page_count,
            "extracted_text_pages": doc.extracted_pages_count,
            "scanned_pages": doc.scanned_pages_count,
            "total_characters": doc.char_count,
            "generated_chunks": doc.chunk_count,
            "embedded_chunks": doc.embedded_chunks_count,
            "indexed_pages": doc.indexed_pages_count,
            "status": doc.status,
            "error_message": doc.error_message,
            "import_time": doc.import_time,
        }

    def delete_document(self, doc_id: int):
        with self._conn() as con:
            if self._fts_available:
                try:
                    con.execute("DELETE FROM chunks_fts WHERE document_id = ?", (doc_id,))
                except Exception:
                    pass
            con.execute("DELETE FROM ocr_regions WHERE document_id = ?", (doc_id,))
            con.execute("DELETE FROM ocr_pages WHERE document_id = ?", (doc_id,))
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

        doc_filter_fts = ""
        doc_filter_like = ""
        params: list[object] = []

        if document_ids:
            placeholders = ",".join("?" for _ in document_ids)
            doc_filter_fts = f" AND c.document_id IN ({placeholders})"
            doc_filter_like = f" AND document_id IN ({placeholders})"

        if self._fts_available:
            try:
                # Sanitize query for FTS5: extract alphanumeric tokens
                words = re.findall(r"[A-Za-z0-9_]+", cleaned_query)
                stopwords = {
                    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with",
                    "is", "are", "was", "were", "of", "by", "as", "it", "its", "what",
                    "this", "that", "these", "those", "from", "pdf", "uploaded", "document",
                    "explain", "summarize", "detail", "about", "please", "show", "tell",
                    "in", "the", "uploaded", "pdfs"
                }
                sig_words = [w for w in words if w.lower() not in stopwords]
                target_words = sig_words if sig_words else words
                if target_words:
                    # Match significant terms using OR for recall, ranked by bm25
                    match_expr = " OR ".join(f'"{w}"*' for w in target_words)
                    sql = f"""
                        SELECT c.chunk_id, c.document_id, c.filename, c.page_number,
                               c.chunk_index, c.text, c.token_count, c.section_heading
                        FROM chunks c
                        JOIN chunks_fts f ON c.chunk_id = f.chunk_id
                        WHERE chunks_fts MATCH ? {doc_filter_fts}
                        ORDER BY bm25(chunks_fts) ASC
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

        # LIKE fallback across significant terms
        target_terms = [w for w in re.findall(r"[A-Za-z0-9_]+", cleaned_query) if len(w) > 2]
        if not target_terms:
            target_terms = [cleaned_query]

        like_conditions = " OR ".join("text LIKE ?" for _ in target_terms[:3])
        sql = f"""
            SELECT chunk_id, document_id, filename, page_number,
                   chunk_index, text, token_count, section_heading
            FROM chunks
            WHERE ({like_conditions}) {doc_filter_like}
            ORDER BY page_number ASC, chunk_index ASC
            LIMIT ?
        """
        sql_params = [f"%{t}%" for t in target_terms[:3]]
        if document_ids:
            sql_params.extend(document_ids)
        sql_params.append(top_k)

        with self._conn() as con:
            rows = con.execute(sql, sql_params).fetchall()
            return [Chunk(**dict(r)) for r in rows]

    # ── OCR Operations (Stage 6B) ───────────────────────────────────

    def save_ocr_page(
        self,
        document_id: int,
        page_number: int,
        image_path: str,
        status: str = "pending_review",
        error_message: str = "",
    ):
        now = self._now()
        with self._conn() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO ocr_pages (
                    document_id, page_number, image_path, status, created_at, updated_at, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (document_id, page_number, image_path, status, now, now, error_message),
            )

    def get_ocr_page(self, document_id: int, page_number: int) -> dict | None:
        with self._conn() as con:
            row = con.execute(
                """
                SELECT document_id, page_number, image_path, status, created_at, updated_at, error_message
                FROM ocr_pages
                WHERE document_id = ? AND page_number = ?
                """,
                (document_id, page_number),
            ).fetchone()
            return dict(row) if row else None

    def list_ocr_pages(self, document_id: int) -> list[dict]:
        with self._conn() as con:
            rows = con.execute(
                """
                SELECT document_id, page_number, image_path, status, created_at, updated_at, error_message
                FROM ocr_pages
                WHERE document_id = ?
                ORDER BY page_number ASC
                """,
                (document_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def save_ocr_regions(self, regions: list[dict]):
        if not regions:
            return
        now = self._now()
        records = [
            (
                r["region_id"],
                r["document_id"],
                r["page_number"],
                r.get("region_type", "text"),
                r["box_x0"],
                r["box_y0"],
                r["box_x1"],
                r["box_y1"],
                r.get("recognized_text", ""),
                r.get("corrected_text"),
                r.get("latex_expression"),
                r.get("confidence_score", 0.0),
                r.get("recognition_engine", "paddleocr_ppocrv5"),
                r.get("model_version", "5.0.0"),
                r.get("approval_status", "pending_review"),
                r.get("image_crop_path", ""),
                now,
                now,
            )
            for r in regions
        ]
        with self._conn() as con:
            con.executemany(
                """
                INSERT OR REPLACE INTO ocr_regions (
                    region_id, document_id, page_number, region_type,
                    box_x0, box_y0, box_x1, box_y1,
                    recognized_text, corrected_text, latex_expression,
                    confidence_score, recognition_engine, model_version,
                    approval_status, image_crop_path, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )

    def get_ocr_regions(self, document_id: int, page_number: int | None = None) -> list[dict]:
        with self._conn() as con:
            if page_number is not None:
                rows = con.execute(
                    """
                    SELECT region_id, document_id, page_number, region_type,
                           box_x0, box_y0, box_x1, box_y1,
                           recognized_text, corrected_text, latex_expression,
                           confidence_score, recognition_engine, model_version,
                           approval_status, image_crop_path, created_at, updated_at
                    FROM ocr_regions
                    WHERE document_id = ? AND page_number = ?
                    ORDER BY box_y0 ASC, box_x0 ASC
                    """,
                    (document_id, page_number),
                ).fetchall()
            else:
                rows = con.execute(
                    """
                    SELECT region_id, document_id, page_number, region_type,
                           box_x0, box_y0, box_x1, box_y1,
                           recognized_text, corrected_text, latex_expression,
                           confidence_score, recognition_engine, model_version,
                           approval_status, image_crop_path, created_at, updated_at
                    FROM ocr_regions
                    WHERE document_id = ?
                    ORDER BY page_number ASC, box_y0 ASC, box_x0 ASC
                    """,
                    (document_id,),
                ).fetchall()
            return [dict(r) for r in rows]

    def update_ocr_region(
        self,
        region_id: str,
        approval_status: str | None = None,
        corrected_text: str | None = None,
        latex_expression: str | None = None,
        region_type: str | None = None,
    ):
        updates = []
        params = []
        if approval_status is not None:
            updates.append("approval_status = ?")
            params.append(approval_status)
        if corrected_text is not None:
            updates.append("corrected_text = ?")
            params.append(corrected_text)
        if latex_expression is not None:
            updates.append("latex_expression = ?")
            params.append(latex_expression)
        if region_type is not None:
            updates.append("region_type = ?")
            params.append(region_type)

        if not updates:
            return

        updates.append("updated_at = ?")
        params.append(self._now())
        params.append(region_id)

        sql = f"UPDATE ocr_regions SET {', '.join(updates)} WHERE region_id = ?"
        with self._conn() as con:
            con.execute(sql, params)

    def approve_all_ocr_regions(self, document_id: int, page_number: int | None = None):
        now = self._now()
        with self._conn() as con:
            if page_number is not None:
                con.execute(
                    """
                    UPDATE ocr_regions
                    SET approval_status = 'approved', updated_at = ?
                    WHERE document_id = ? AND page_number = ? AND approval_status != 'rejected'
                    """,
                    (now, document_id, page_number),
                )
                con.execute(
                    """
                    UPDATE ocr_pages
                    SET status = 'approved', updated_at = ?
                    WHERE document_id = ? AND page_number = ?
                    """,
                    (now, document_id, page_number),
                )
            else:
                con.execute(
                    """
                    UPDATE ocr_regions
                    SET approval_status = 'approved', updated_at = ?
                    WHERE document_id = ? AND approval_status != 'rejected'
                    """,
                    (now, document_id),
                )
                con.execute(
                    """
                    UPDATE ocr_pages
                    SET status = 'approved', updated_at = ?
                    WHERE document_id = ?
                    """,
                    (now, document_id),
                )

    def get_approved_ocr_regions(self, document_id: int, page_number: int | None = None) -> list[dict]:
        with self._conn() as con:
            if page_number is not None:
                rows = con.execute(
                    """
                    SELECT region_id, document_id, page_number, region_type,
                           box_x0, box_y0, box_x1, box_y1,
                           recognized_text, corrected_text, latex_expression,
                           confidence_score, recognition_engine, model_version,
                           approval_status, image_crop_path, created_at, updated_at
                    FROM ocr_regions
                    WHERE document_id = ? AND page_number = ? AND approval_status = 'approved'
                    ORDER BY box_y0 ASC, box_x0 ASC
                    """,
                    (document_id, page_number),
                ).fetchall()
            else:
                rows = con.execute(
                    """
                    SELECT region_id, document_id, page_number, region_type,
                           box_x0, box_y0, box_x1, box_y1,
                           recognized_text, corrected_text, latex_expression,
                           confidence_score, recognition_engine, model_version,
                           approval_status, image_crop_path, created_at, updated_at
                    FROM ocr_regions
                    WHERE document_id = ? AND approval_status = 'approved'
                    ORDER BY page_number ASC, box_y0 ASC, box_x0 ASC
                    """,
                    (document_id,),
                ).fetchall()
            return [dict(r) for r in rows]
