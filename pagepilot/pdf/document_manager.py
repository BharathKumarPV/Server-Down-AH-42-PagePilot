"""
PagePilot – Document Manager (Stage 4).

High-level interface for querying and managing imported documents.
"""

from __future__ import annotations

from typing import List, Optional

from pagepilot.db.database import DocumentDB
from pagepilot.pdf.models import Chunk, Document
from pagepilot.retrieval.vector_store import VectorStore


class DocumentManager:
    """
    Coordinates document metadata lookups, listings, and deletions.
    """

    def __init__(self, db: DocumentDB, vector_store: VectorStore):
        self.db = db
        self.vector_store = vector_store

    def list_documents(self) -> list[Document]:
        """Return all imported documents, ordered newest first."""
        return self.db.list_documents()

    def get_document(self, doc_id: int) -> Document | None:
        """Lookup a document by its database ID."""
        return self.db.get_document(doc_id)

    def delete_document(self, doc_id: int):
        """Remove a document from SQLite and ChromaDB."""
        self.vector_store.delete_document(doc_id)
        self.db.delete_document(doc_id)

    def get_chunks_for_document(self, doc_id: int) -> list[Chunk]:
        """Fetch all chunks belonging to a document."""
        return self.db.get_chunks_for_document(doc_id)

    def count_documents(self) -> int:
        """Total number of documents."""
        return self.db.count_documents()

    def count_chunks(self) -> int:
        """Total number of indexed chunks."""
        return self.db.count_chunks()

