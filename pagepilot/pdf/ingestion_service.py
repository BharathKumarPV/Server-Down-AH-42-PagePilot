"""
PagePilot – PDF Ingestion Pipeline (Stage 4).

Coordinates:
- SHA-256 deduplication
- Page extraction (generator-based, RAM bounded)
- Page-level chunking
- Dense vector generation (batched via Ollama)
- ChromaDB vector storage
- SQLite FTS5 index updates
- Automatic rollback/cleanup on error
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

from pagepilot.config import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS, EMBED_MODEL
from pagepilot.db.database import DocumentDB
from pagepilot.pdf.chunker import chunk_page
from pagepilot.pdf.extractor import ExtractorError, compute_sha256, extract_pages, get_page_count
from pagepilot.pdf.models import Chunk, Document, PageText
from pagepilot.retrieval.embedding_client import EmbeddingClient, EmbeddingError
from pagepilot.retrieval.vector_store import VectorStore


class IngestionError(Exception):
    """Raised when document ingestion fails."""


class IngestionService:
    """
    End-to-end service for ingesting PDF files into PagePilot.
    """

    def __init__(
        self,
        db: DocumentDB,
        vector_store: VectorStore,
        embedding_client: EmbeddingClient,
    ):
        self.db = db
        self.vector_store = vector_store
        self.embedding_client = embedding_client

    def ingest_pdf(
        self,
        file_path: str,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> Document:
        """
        Ingest and index a PDF file into the local hybrid retrieval engine.

        Parameters
        ----------
        file_path:
            Absolute path to the PDF on disk.
        progress_callback:
            Optional callback(stage_name: str, step: int, total: int).

        Returns
        -------
        Document record after successful indexing.
        """
        path = Path(file_path)
        if not path.is_file():
            raise IngestionError(f"PDF file does not exist: {file_path}")

        filename = path.name

        def report(stage: str, cur: int, tot: int):
            if progress_callback:
                progress_callback(stage, cur, tot)

        report("hashing", 0, 1)
        sha256 = compute_sha256(str(path))
        report("hashing", 1, 1)

        # 1. Deduplication check
        existing = self.db.get_document_by_hash(sha256)
        if existing and existing.status == "ready":
            report("done", 1, 1)
            return existing
        elif existing:
            # Clean previous partial or failed document record
            self.delete_document(existing.id)

        # 2. Page count & initial DB record
        try:
            page_count = get_page_count(str(path))
        except ExtractorError as exc:
            raise IngestionError(f"Failed to inspect PDF: {exc}") from exc

        doc = self.db.create_document(
            filename=filename,
            file_path=str(path.resolve()),
            sha256_hash=sha256,
            page_count=page_count,
            embed_model=self.embedding_client.model,
            status="indexing",
        )

        all_chunks: list[Chunk] = []

        try:
            # 3. Extraction & Chunking page-by-page
            report("extracting", 0, page_count)
            scanned_pages = 0

            for page_text in extract_pages(str(path), doc.id):
                if page_text.is_empty:
                    scanned_pages += 1
                page_chunks = chunk_page(
                    page=page_text,
                    filename=filename,
                    target_tokens=CHUNK_SIZE_TOKENS,
                    overlap_tokens=CHUNK_OVERLAP_TOKENS,
                )
                all_chunks.extend(page_chunks)
                report("extracting", page_text.page_number, page_count)

            # Check if document has no extractable text (e.g. image-only scan)
            if not all_chunks:
                warn_msg = "PDF contains only scanned images or empty pages. No extractable text found."
                self.db.update_document_status(
                    doc.id,
                    status="ready",
                    chunk_count=0,
                    error_message=warn_msg,
                )
                doc.status = "ready"
                doc.chunk_count = 0
                doc.error_message = warn_msg
                report("done", 1, 1)
                return doc

            # 4. Dense Vector Generation
            total_chunks = len(all_chunks)
            report("embedding", 0, total_chunks)

            def on_embed_progress(completed: int, total: int):
                report("embedding", completed, total)

            chunk_texts = [c.text for c in all_chunks]
            embeddings = self.embedding_client.embed_documents(
                chunk_texts,
                progress_callback=on_embed_progress,
            )

            # 5. ChromaDB vector storage
            report("indexing_vectors", 0, total_chunks)
            self.vector_store.add_chunks(all_chunks, embeddings)
            report("indexing_vectors", total_chunks, total_chunks)

            # 6. SQLite FTS5 keyword indexing
            report("indexing_fts", 0, total_chunks)
            self.db.insert_chunks(all_chunks)
            report("indexing_fts", total_chunks, total_chunks)

            # 7. Mark ready
            self.db.update_document_status(
                doc.id,
                status="ready",
                chunk_count=total_chunks,
                error_message="",
            )
            doc.status = "ready"
            doc.chunk_count = total_chunks
            report("done", 1, 1)
            return doc

        except Exception as exc:
            # Rollback on failure
            self.db.update_document_status(
                doc.id,
                status="error",
                chunk_count=0,
                error_message=str(exc),
            )
            try:
                self.vector_store.delete_document(doc.id)
            except Exception:
                pass
            raise IngestionError(f"Failed to ingest document '{filename}': {exc}") from exc

    def delete_document(self, doc_id: int):
        """Remove document and its vectors from both vector store and SQLite."""
        self.vector_store.delete_document(doc_id)
        self.db.delete_document(doc_id)

