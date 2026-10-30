"""
PagePilot – Keyword search service using SQLite FTS5 / LIKE (Stage 4).

Wraps DocumentDB keyword search with query sanitization and error handling.
"""

from __future__ import annotations

import re
from typing import List, Optional

from pagepilot.db.database import DocumentDB
from pagepilot.pdf.models import Chunk


class KeywordSearch:
    """
    Provides exact-term and keyword retrieval across indexed document chunks.
    """

    def __init__(self, db: DocumentDB):
        self.db = db

    def search(
        self,
        query: str,
        document_ids: list[int] | None = None,
        top_k: int = 5,
    ) -> list[Chunk]:
        """
        Execute keyword search with ranking and filtering.
        """
        cleaned = query.strip()
        if not cleaned:
            return []

        # Remove control characters
        cleaned = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", cleaned)
        if not cleaned:
            return []

        return self.db.search_keyword(
            query=cleaned,
            document_ids=document_ids,
            top_k=top_k,
        )

