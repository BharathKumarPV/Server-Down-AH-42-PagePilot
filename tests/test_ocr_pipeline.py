"""
Tests for Stage 6B OCR Pipeline: Document import, OCR extraction, and human review updates.
"""

from pagepilot.db.database import DocumentDB
from pagepilot.ocr.models import ApprovalStatus, BoundingBox, OCRRegion, RegionType
from pagepilot.ocr.service import OCRService
from pagepilot.retrieval.vector_store import VectorStore


def test_ocr_import_and_deduplication(tmp_path):
    db = DocumentDB(db_path=tmp_path / "test_docs.db")
    vstore = VectorStore(persist_dir=tmp_path / "vstore")
    service = OCRService(db=db, vector_store=vstore)

    img_file = tmp_path / "lecture_note.png"
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05"
        b"\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    img_file.write_bytes(png_bytes)

    # First import
    doc1 = service.import_image_note(str(img_file))
    assert doc1.filename == "lecture_note.png"
    assert doc1.status == "ocr_pending_review"

    # Second import (same content) should deduplicate and return same document
    doc2 = service.import_image_note(str(img_file))
    assert doc2.id == doc1.id
    assert db.count_documents() == 1


def test_ocr_processing_and_human_review_updates(tmp_path):
    db = DocumentDB(db_path=tmp_path / "test_docs.db")
    vstore = VectorStore(persist_dir=tmp_path / "vstore")
    service = OCRService(db=db, vector_store=vstore)

    img_file = tmp_path / "printed_quadratic.png"
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05"
        b"\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    img_file.write_bytes(png_bytes)

    doc = service.import_image_note(str(img_file))
    results = service.process_document_ocr(doc.id)
    assert len(results) == 1
    page1 = results[0]
    assert len(page1.regions) >= 2

    # Verify initial status in DB is pending review
    page_db = service.get_page_ocr_result(doc.id, 1)
    assert page_db.page_status == ApprovalStatus.PENDING_REVIEW

    # User modifies region 1 text
    r1 = page_db.regions[0]
    service.update_region(r1.region_id, corrected_text="User verified header text", approval_status=ApprovalStatus.APPROVED.value)

    # User rejects region 2
    r2 = page_db.regions[1]
    service.update_region(r2.region_id, approval_status=ApprovalStatus.REJECTED.value)

    updated = service.get_page_ocr_result(doc.id, 1)
    reg_map = {r.region_id: r for r in updated.regions}
    assert reg_map[r1.region_id].effective_text == "User verified header text"
    assert reg_map[r1.region_id].approval_status == ApprovalStatus.APPROVED
    assert reg_map[r2.region_id].approval_status == ApprovalStatus.REJECTED

