"""
PagePilot – Stage 4 data models for PDF intelligence.

All public APIs use 1-based page numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Document:
    """Represents a PDF document imported into PagePilot."""
    id: int
    filename: str
    file_path: str
    sha256_hash: str
    page_count: int
    import_time: str
    status: str          # 'pending' | 'indexing' | 'ready' | 'error'
    embed_model: str
    chunk_count: int = 0
    error_message: str = ""


@dataclass
class PageText:
    """Extracted text from a single PDF page (1-based page number)."""
    document_id: int
    page_number: int          # 1-based
    text: str
    char_count: int
    is_empty: bool
    is_scanned: bool          # True when text is empty (likely image-only page)


@dataclass
class Chunk:
    """A text chunk derived from a PageText, used for embedding and retrieval."""
    chunk_id: str             # Deterministic SHA-256 based ID
    document_id: int
    filename: str
    page_number: int          # 1-based
    chunk_index: int          # 0-based index within the page
    text: str
    token_count: int
    section_heading: str = ""


@dataclass
class Citation:
    """A source reference to a specific chunk in a document."""
    document_id: int
    filename: str
    page_number: int          # 1-based
    chunk_text: str
    relevance_score: float = 0.0

    def format_short(self) -> str:
        return f"[{self.filename}, p. {self.page_number}]"

    def format_long(self) -> str:
        preview = self.chunk_text[:120].replace("\n", " ")
        if len(self.chunk_text) > 120:
            preview += "…"
        return f"[{self.filename}, p. {self.page_number}]: \"{preview}\""


@dataclass
class RetrievalResult:
    """Result of a hybrid retrieval operation."""
    chunks: list[Chunk] = field(default_factory=list)
    method: str = "hybrid"   # 'semantic' | 'keyword' | 'hybrid'
