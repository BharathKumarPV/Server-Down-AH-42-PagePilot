"""
Tests for PDF text extraction and metadata processing (Stage 4).
"""

import tempfile
from pathlib import Path
import pytest
import pymupdf as fitz

from pagepilot.pdf.extractor import (
    ExtractorError,
    compute_sha256,
    extract_pages,
    get_page_count,
)


@pytest.fixture
def sample_pdf(tmp_path) -> Path:
    """Create a real 3-page PDF file with known content for testing."""
    pdf_path = tmp_path / "test_doc.pdf"
    doc = fitz.open()

    # Page 1: normal text
    p1 = doc.new_page()
    p1.insert_text((50, 50), "First Page: PagePilot Offline AI Architecture")

    # Page 2: empty page (simulating scanned / image-only)
    doc.new_page()

    # Page 3: multi-line content
    p3 = doc.new_page()
    p3.insert_text((50, 50), "Third Page: Retrieval Augmented Generation details.")

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_sha256_computation(sample_pdf):
    sha = compute_sha256(str(sample_pdf))
    assert isinstance(sha, str)
    assert len(sha) == 64
    # Deterministic
    assert compute_sha256(str(sample_pdf)) == sha


def test_get_page_count(sample_pdf):
    count = get_page_count(str(sample_pdf))
    assert count == 3


def test_extract_pages_page_numbers_are_one_based(sample_pdf):
    pages = list(extract_pages(str(sample_pdf), document_id=42))
    assert len(pages) == 3

    page_numbers = [p.page_number for p in pages]
    assert page_numbers == [1, 2, 3]


def test_extract_pages_content_and_empty_detection(sample_pdf):
    pages = list(extract_pages(str(sample_pdf), document_id=42))

    # Page 1
    assert "PagePilot Offline AI" in pages[0].text
    assert not pages[0].is_empty
    assert not pages[0].is_scanned
    assert pages[0].document_id == 42

    # Page 2 (empty)
    assert pages[1].is_empty
    assert pages[1].is_scanned
    assert pages[1].char_count == 0

    # Page 3
    assert "Retrieval Augmented Generation" in pages[2].text
    assert not pages[2].is_empty


def test_extract_pages_missing_file():
    with pytest.raises(ExtractorError, match="File not found"):
        list(extract_pages("non_existent_file.pdf", document_id=1))


def test_extract_pages_corrupt_file(tmp_path):
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_text("Not a real PDF file header")
    with pytest.raises(ExtractorError):
        list(extract_pages(str(corrupt), document_id=1))

