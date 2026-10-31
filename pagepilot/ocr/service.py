"""
PagePilot – Stage 6B OCR Service Orchestrator.

High-level interface coordinating:
- Image import and validation (PNG, JPG, WebP)
- Scanned PDF page rendering and OCR dispatch
- Worker process lifecycle management
- Review state persistence in SQLite
- Human-approved indexing into Hybrid RAG (SQLite FTS5 + ChromaDB)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional

from pagepilot.config import EMBED_MODEL
from pagepilot.db.database import DocumentDB
from pagepilot.ocr.image_loader import (
    compute_sha256,
    is_image_file,
    is_pdf_file,
    load_and_validate_image,
    render_pdf_page_to_image,
)
from pagepilot.ocr.indexer import OCRIndexer
from pagepilot.ocr.model_manager import OCRModelManager
from pagepilot.ocr.models import (
    ApprovalStatus,
    BoundingBox,
    OCRRegion,
    PageOCRResult,
    RegionType,
)
from pagepilot.ocr.worker_client import OCRWorkerClient
from pagepilot.pdf.models import Document
from pagepilot.retrieval.embedding_client import EmbeddingClient
from pagepilot.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class OCRService:
    """Coordinates local OCR extraction, review updates, and trusted RAG indexing."""

    def __init__(
        self,
        db: DocumentDB,
        vector_store: VectorStore,
        embedding_client: Optional[EmbeddingClient] = None,
        worker_client: Optional[OCRWorkerClient] = None,
        model_manager: Optional[OCRModelManager] = None,
        simulation_mode: Optional[bool] = None,
        mock_engine: Optional[Any] = None,
    ):
        self.db = db
        self.vector_store = vector_store
        self.embedding_client = embedding_client
        self.worker = worker_client or OCRWorkerClient(
            simulation_mode=simulation_mode,
            mock_engine=mock_engine,
        )
        self.models = model_manager or OCRModelManager()
        self.indexer = OCRIndexer(db, vector_store, embedding_client)

    # ── Input Handling ────────────────────────────────────────────────────────

    def import_image_note(self, file_path: str) -> Document:
        """
        Import a photographed student note (PNG, JPG, JPEG, WebP).
        Performs SHA-256 deduplication and inserts document record in SQLite.
        """
        meta = load_and_validate_image(file_path)
        sha256 = meta["sha256"]

        # Check existing document with identical content
        existing = self.db.get_document_by_hash(sha256)
        if existing:
            return existing

        doc = self.db.create_document(
            filename=meta["filename"],
            file_path=meta["file_path"],
            sha256_hash=sha256,
            page_count=1,
            embed_model=EMBED_MODEL,
            status="ocr_pending_review",
        )
        return doc

    # ── Processing ────────────────────────────────────────────────────────────

    def process_document_ocr(
        self,
        document_id: int,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> List[PageOCRResult]:
        """
        Execute layout detection, text OCR, and formula recognition across document pages.
        Persists results with PENDING_REVIEW status awaiting user verification.
        """
        doc = self.db.get_document(document_id)
        if not doc:
            raise ValueError(f"Document ID {document_id} not found.")

        file_path = doc.file_path
        results: List[PageOCRResult] = []

        if is_pdf_file(file_path):
            try:
                import fitz
                pdf_doc = fitz.open(file_path)
                total_pages = len(pdf_doc)
                pdf_doc.close()
            except Exception as exc:
                raise ValueError(f"Failed to open PDF {file_path}: {exc}") from exc

            for p_num in range(1, total_pages + 1):
                if progress_callback:
                    progress_callback(f"Rendering scanned page {p_num}/{total_pages}", p_num, total_pages)

                img_path = render_pdf_page_to_image(file_path, p_num)

                if progress_callback:
                    progress_callback(f"Running OCR on page {p_num}/{total_pages}", p_num, total_pages)

                regions = self.worker.process_page(
                    image_path=img_path,
                    document_id=document_id,
                    page_number=p_num,
                    filename=doc.filename,
                    file_hash=doc.sha256_hash,
                )

                # Persist in SQLite
                self.db.save_ocr_page(document_id, p_num, img_path, status=ApprovalStatus.PENDING_REVIEW.value)
                self.db.save_ocr_regions([r.to_dict() for r in regions])

                page_res = PageOCRResult(
                    document_id=document_id,
                    page_number=p_num,
                    image_path=img_path,
                    regions=regions,
                    page_status=ApprovalStatus.PENDING_REVIEW,
                )
                results.append(page_res)

        else:
            # Single image note
            if progress_callback:
                progress_callback("Running OCR on photographed note", 1, 1)

            regions = self.worker.process_page(
                image_path=file_path,
                document_id=document_id,
                page_number=1,
                filename=doc.filename,
                file_hash=doc.sha256_hash,
            )

            self.db.save_ocr_page(document_id, 1, file_path, status=ApprovalStatus.PENDING_REVIEW.value)
            self.db.save_ocr_regions([r.to_dict() for r in regions])

            page_res = PageOCRResult(
                document_id=document_id,
                page_number=1,
                image_path=file_path,
                regions=regions,
                page_status=ApprovalStatus.PENDING_REVIEW,
            )
            results.append(page_res)

        return results

    # ── Review & Verification ─────────────────────────────────────────────────

    def get_page_ocr_result(self, document_id: int, page_number: int) -> Optional[PageOCRResult]:
        """Fetch OCR page metadata and all region entries from SQLite."""
        page_dict = self.db.get_ocr_page(document_id, page_number)
        if not page_dict:
            return None

        raw_regions = self.db.get_ocr_regions(document_id, page_number)
        regions = [OCRRegion.from_dict(r) for r in raw_regions]

        return PageOCRResult(
            document_id=document_id,
            page_number=page_number,
            image_path=page_dict.get("image_path", ""),
            regions=regions,
            page_status=ApprovalStatus(page_dict.get("status", "pending_review")),
            error_message=page_dict.get("error_message", ""),
        )

    def update_region(
        self,
        region_id: str,
        approval_status: Optional[str] = None,
        corrected_text: Optional[str] = None,
        latex_expression: Optional[str] = None,
        region_type: Optional[str] = None,
    ):
        """Update user review corrections or status on an individual region."""
        self.db.update_ocr_region(
            region_id=region_id,
            approval_status=approval_status,
            corrected_text=corrected_text,
            latex_expression=latex_expression,
            region_type=region_type,
        )

    def approve_page(self, document_id: int, page_number: int):
        """Approve all un-rejected regions on a page."""
        self.db.approve_all_ocr_regions(document_id, page_number)

    def approve_document(self, document_id: int):
        """Approve all un-rejected regions across an entire document."""
        self.db.approve_all_ocr_regions(document_id)

    # ── Indexing into Trusted RAG ─────────────────────────────────────────────

    def index_approved_content(
        self,
        document_id: int,
        page_number: Optional[int] = None,
    ) -> int:
        """
        Index verified content into SQLite FTS5 and ChromaDB.
        Excludes unapproved and rejected regions.
        """
        if page_number is not None:
            chunks = self.indexer.index_approved_page(document_id, page_number)
            return len(chunks)
        return self.indexer.index_approved_document(document_id)

