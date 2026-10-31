"""
PagePilot – Stage 6B OCR Hybrid RAG Indexer.

Converts approved OCR text and mathematical formulas into page-aware chunks
and indexes them into SQLite FTS5 and ChromaDB.

CRITICAL ANTI-HALLUCINATION ENFORCEMENT:
Unapproved (pending review or rejected) OCR regions are strictly excluded
from the retrieval index.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from pagepilot.db.database import DocumentDB
from pagepilot.ocr.models import ApprovalStatus, BoundingBox, OCRRegion, PageOCRResult, RegionType
from pagepilot.pdf.chunker import chunk_page
from pagepilot.pdf.models import Chunk, Document, PageText
from pagepilot.retrieval.embedding_client import EmbeddingClient
from pagepilot.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class IndexingError(Exception):
    """Raised when indexing approved OCR content fails."""


class OCRIndexer:
    """Coordinates chunking, full-text indexing, and vector embeddings for approved OCR content."""

    def __init__(
        self,
        db: DocumentDB,
        vector_store: VectorStore,
        embedding_client: Optional[EmbeddingClient] = None,
    ):
        self.db = db
        self.vector_store = vector_store
        self.embedding_client = embedding_client

    def index_approved_page(
        self,
        document_id: int,
        page_number: int,
    ) -> List[Chunk]:
        """
        Extract approved text and LaTeX from a specific page, chunk it,
        and index into SQLite FTS5 and ChromaDB.
        """
        doc = self.db.get_document(document_id)
        if not doc:
            raise IndexingError(f"Document ID {document_id} not found in database.")

        raw_regions = self.db.get_approved_ocr_regions(document_id, page_number)
        if not raw_regions:
            logger.info("No approved OCR regions found for document %d page %d; skipping indexing.", document_id, page_number)
            return []

        regions = [OCRRegion.from_dict(r) for r in raw_regions]
        page_result = PageOCRResult(
            document_id=document_id,
            page_number=page_number,
            image_path="",
            regions=regions,
            page_status=ApprovalStatus.APPROVED,
        )

        full_approved_text = page_result.approved_text()
        if not full_approved_text.strip():
            return []

        # Construct PageText for standard chunking
        page_text = PageText(
            document_id=document_id,
            page_number=page_number,
            text=full_approved_text,
            char_count=len(full_approved_text),
            is_empty=False,
            is_scanned=False,
        )

        chunks = chunk_page(page_text, doc.filename)
        if not chunks:
            return []

        # Tag chunk IDs with OCR prefix for clear provenance identification
        for idx, c in enumerate(chunks):
            c.chunk_id = f"ocr_{document_id}_p{page_number}_c{idx}_{doc.sha256_hash[:8]}"

        # 1. Insert into SQLite and FTS5
        self.db.insert_chunks(chunks)

        # 2. Add embeddings to ChromaDB if service is available
        if self.embedding_client and self.embedding_client.is_available():
            try:
                texts = [c.text for c in chunks]
                embeddings = self.embedding_client.embed_documents(texts)
                self.vector_store.add_chunks(chunks, embeddings)
            except Exception as exc:
                logger.warning("Vector indexing failed for OCR document %d: %s. FTS5 remains available.", document_id, exc)

        # Update document status and chunk count
        all_chunks = self.db.get_chunks_for_document(document_id)
        self.db.update_document_status(
            document_id,
            status="ready",
            chunk_count=len(all_chunks),
        )

        return chunks

    def index_approved_document(self, document_id: int) -> int:
        """
        Process and index all approved pages belonging to an imported image note or scanned PDF.
        Returns the total number of chunks indexed.
        """
        doc = self.db.get_document(document_id)
        if not doc:
            raise IndexingError(f"Document ID {document_id} not found.")

        # Find all unique page numbers in ocr_regions
        raw_regions = self.db.get_approved_ocr_regions(document_id)
        if not raw_regions:
            return 0

        page_numbers = sorted({int(r["page_number"]) for r in raw_regions})
        total_chunks = 0
        for p in page_numbers:
            chunks = self.index_approved_page(document_id, p)
            total_chunks += len(chunks)

        return total_chunks
