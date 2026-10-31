"""
Tests for Stage 6B OCR Indexing: Validating that only approved regions are indexed into SQLite FTS5 and ChromaDB.
"""

from pagepilot.db.database import DocumentDB
from pagepilot.ocr.indexer import OCRIndexer
from pagepilot.ocr.models import ApprovalStatus, BoundingBox, OCRRegion, RegionType
from pagepilot.ocr.service import OCRService
from pagepilot.retrieval.keyword_search import KeywordSearch
from pagepilot.retrieval.vector_store import VectorStore


def test_indexing_excludes_unapproved_and_indexes_approved(tmp_path):
    db = DocumentDB(db_path=tmp_path / "test_docs.db")
    vstore = VectorStore(persist_dir=tmp_path / "vstore")
    kw_search = KeywordSearch(db)
    service = OCRService(db=db, vector_store=vstore)

    img_file = tmp_path / "calc_notes.png"
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05"
        b"\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    img_file.write_bytes(png_bytes)

    doc = service.import_image_note(str(img_file))
    service.process_document_ocr(doc.id)

    # 1. Initially, all regions are PENDING_REVIEW.
    # Attempting to index should produce 0 chunks (anti-hallucination guarantee).
    chunks_indexed = service.index_approved_content(doc.id)
    assert chunks_indexed == 0
    assert db.count_chunks() == 0

    # 2. Approve page
    service.approve_page(doc.id, 1)

    # Now index approved content
    chunks_indexed = service.index_approved_content(doc.id)
    assert chunks_indexed > 0
    assert db.count_chunks() == chunks_indexed

    # Check searchability in SQLite FTS5
    # The simulation fallback for 'calc' produces quadratic formula text
    results = kw_search.search("quadratic", document_ids=[doc.id])
    assert len(results) > 0
    assert "quadratic" in results[0].text.lower()
    assert results[0].page_number == 1
    assert results[0].chunk_id.startswith(f"ocr_{doc.id}_p1")


def test_rejected_region_strictly_omitted(tmp_path):
    db = DocumentDB(db_path=tmp_path / "test_docs.db")
    vstore = VectorStore(persist_dir=tmp_path / "vstore")
    kw_search = KeywordSearch(db)
    service = OCRService(db=db, vector_store=vstore)

    doc = db.create_document("note.png", str(tmp_path / "note.png"), "hash123", 1, "test-embed")

    # Save 2 regions: one approved, one rejected
    reg1 = {
        "region_id": "R1",
        "document_id": doc.id,
        "page_number": 1,
        "box_x0": 10.0, "box_y0": 10.0, "box_x1": 200.0, "box_y1": 50.0,
        "recognized_text": "Photosynthesis produces glucose and oxygen from sunlight.",
        "approval_status": "approved",
    }
    reg2 = {
        "region_id": "R2",
        "document_id": doc.id,
        "page_number": 1,
        "box_x0": 10.0, "box_y0": 60.0, "box_x1": 200.0, "box_y1": 100.0,
        "recognized_text": "Hallucinated erroneous statement about gravity.",
        "approval_status": "rejected",
    }
    db.save_ocr_regions([reg1, reg2])

    service.index_approved_content(doc.id)

    # Search for approved term
    res_approved = kw_search.search("photosynthesis", document_ids=[doc.id])
    assert len(res_approved) > 0

    # Search for rejected term
    res_rejected = kw_search.search("erroneous", document_ids=[doc.id])
    assert len(res_rejected) == 0

