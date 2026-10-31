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
from pagepilot.pdf.models import (
    DOC_STATUS_EMBEDDING,
    DOC_STATUS_EXTRACTED,
    DOC_STATUS_EXTRACTING,
    DOC_STATUS_FAILED,
    DOC_STATUS_IMPORTED,
    DOC_STATUS_INDEXING,
    DOC_STATUS_PARTIALLY_INDEXED,
    DOC_STATUS_READY,
    DOC_STATUS_SCANNED_OCR_REQUIRED,
    Chunk,
    Document,
    PageText,
)
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
        force: bool = False,
        existing_doc_id: Optional[int] = None,
    ) -> Document:
        """
        Ingest and index a PDF file into the local hybrid retrieval engine.

        Parameters
        ----------
        file_path:
            Absolute path to the PDF on disk.
        progress_callback:
            Optional callback(stage_name: str, step: int, total: int).
        force:
            If True, re-indexes even if already indexed in database.
        existing_doc_id:
            If provided, re-indexes into this specific document ID without creating a new record.

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

        # 2. Page count inspection
        try:
            page_count = get_page_count(str(path))
        except ExtractorError as exc:
            raise IngestionError(f"Failed to inspect PDF: {exc}") from exc

        if existing_doc_id is not None:
            doc = self.db.get_document(existing_doc_id)
            if not doc:
                raise IngestionError(f"Document {existing_doc_id} not found for re-indexing.")
            # Wipe previous chunks and vectors for this document
            self.vector_store.delete_document(existing_doc_id)
            with self.db._conn() as con:
                if self.db.fts_available:
                    try:
                        con.execute("DELETE FROM chunks_fts WHERE document_id = ?", (existing_doc_id,))
                    except Exception:
                        pass
                con.execute("DELETE FROM chunks WHERE document_id = ?", (existing_doc_id,))
                con.execute("DELETE FROM ocr_regions WHERE document_id = ?", (existing_doc_id,))
                con.execute("DELETE FROM ocr_pages WHERE document_id = ?", (existing_doc_id,))
            self.db.update_document_status(existing_doc_id, status=DOC_STATUS_IMPORTED)
        else:
            # 1. Deduplication check
            existing = self.db.get_document_by_hash(sha256)
            if not force and existing and existing.status in (DOC_STATUS_READY, DOC_STATUS_PARTIALLY_INDEXED):
                report("done", 1, 1)
                return existing
            elif existing:
                self.delete_document(existing.id)

            doc = self.db.create_document(
                filename=filename,
                file_path=str(path.resolve()),
                sha256_hash=sha256,
                page_count=page_count,
                embed_model=self.embedding_client.model,
                status=DOC_STATUS_IMPORTED,
            )

        all_chunks: list[Chunk] = []

        try:
            # 3. Extraction & Chunking page-by-page
            report("extracting", 0, page_count)
            self.db.update_document_status(doc.id, status=DOC_STATUS_EXTRACTING)

            extracted_pages_count = 0
            scanned_pages_count = 0
            total_char_count = 0

            for page_text in extract_pages(str(path), doc.id):
                total_char_count += page_text.char_count
                if not page_text.is_empty:
                    extracted_pages_count += 1
                if page_text.is_scanned:
                    scanned_pages_count += 1

                page_chunks = chunk_page(
                    page=page_text,
                    filename=filename,
                    target_tokens=CHUNK_SIZE_TOKENS,
                    overlap_tokens=CHUNK_OVERLAP_TOKENS,
                )
                all_chunks.extend(page_chunks)
                report("extracting", page_text.page_number, page_count)

            # Check if document has no extractable text
            if not all_chunks:
                if scanned_pages_count > 0:
                    status = DOC_STATUS_SCANNED_OCR_REQUIRED
                    warn_msg = f"PDF contains {scanned_pages_count} scanned pages. Real OCR is required to extract text."
                else:
                    status = DOC_STATUS_FAILED
                    warn_msg = "PDF contains empty pages with no extractable text."

                self.db.update_document_status(
                    doc.id,
                    status=status,
                    chunk_count=0,
                    error_message=warn_msg,
                    extracted_pages_count=extracted_pages_count,
                    scanned_pages_count=scanned_pages_count,
                    char_count=total_char_count,
                    embedded_chunks_count=0,
                    indexed_pages_count=0,
                )
                doc.status = status
                doc.chunk_count = 0
                doc.error_message = warn_msg
                doc.extracted_pages_count = extracted_pages_count
                doc.scanned_pages_count = scanned_pages_count
                doc.char_count = total_char_count
                doc.embedded_chunks_count = 0
                doc.indexed_pages_count = 0
                report("done", 1, 1)
                return doc

            # 4. Dense Vector Generation
            total_chunks = len(all_chunks)
            report("embedding", 0, total_chunks)
            self.db.update_document_status(doc.id, status=DOC_STATUS_EMBEDDING)

            def on_embed_progress(completed: int, total: int):
                report("embedding", completed, total)

            chunk_texts = [c.text for c in all_chunks]
            embeddings = self.embedding_client.embed_documents(
                chunk_texts,
                progress_callback=on_embed_progress,
            )

            # 5. ChromaDB vector storage & SQLite FTS5 keyword indexing
            report("indexing_vectors", 0, total_chunks)
            self.db.update_document_status(doc.id, status=DOC_STATUS_INDEXING)
            self.vector_store.add_chunks(all_chunks, embeddings)
            report("indexing_vectors", total_chunks, total_chunks)

            report("indexing_fts", 0, total_chunks)
            self.db.insert_chunks(all_chunks)
            report("indexing_fts", total_chunks, total_chunks)

            # 6. Accurate Final Status
            indexed_pages = len({c.page_number for c in all_chunks})
            if scanned_pages_count > 0:
                final_status = DOC_STATUS_PARTIALLY_INDEXED
                final_error = f"{scanned_pages_count} of {page_count} pages are scanned images without selectable text."
            else:
                final_status = DOC_STATUS_READY
                final_error = ""

            self.db.update_document_status(
                doc.id,
                status=final_status,
                chunk_count=total_chunks,
                error_message=final_error,
                extracted_pages_count=extracted_pages_count,
                scanned_pages_count=scanned_pages_count,
                char_count=total_char_count,
                embedded_chunks_count=len(embeddings),
                indexed_pages_count=indexed_pages,
            )
            doc.status = final_status
            doc.chunk_count = total_chunks
            doc.error_message = final_error
            doc.extracted_pages_count = extracted_pages_count
            doc.scanned_pages_count = scanned_pages_count
            doc.char_count = total_char_count
            doc.embedded_chunks_count = len(embeddings)
            doc.indexed_pages_count = indexed_pages

            report("done", 1, 1)
            return doc

        except Exception as exc:
            # Rollback on failure
            self.db.update_document_status(
                doc.id,
                status=DOC_STATUS_FAILED,
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

    def reindex_document(
        self,
        doc_id: int,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> Document:
        """
        Safely re-index an existing document without creating duplicate chunks or vectors.
        """
        doc = self.db.get_document(doc_id)
        if not doc:
            raise IngestionError(f"Document {doc_id} not found for re-indexing.")

        file_path = doc.file_path
        if not Path(file_path).is_file():
            raise IngestionError(f"Document source file no longer exists at: {file_path}")

        # Run extraction & indexing again into existing_doc_id with force=True
        return self.ingest_pdf(
            file_path,
            progress_callback=progress_callback,
            force=True,
            existing_doc_id=doc_id,
        )

