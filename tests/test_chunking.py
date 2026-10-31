"""
Tests for text chunking and page-level provenance (Stage 4).
"""

from pagepilot.pdf.chunker import (
    chunk_document,
    chunk_page,
    estimate_tokens,
    make_chunk_id,
)
from pagepilot.pdf.models import PageText


def test_token_estimation():
    text = "Short sentence."
    tokens = estimate_tokens(text)
    assert tokens >= 1
    # ~4 chars per token
    assert estimate_tokens("A" * 400) == 100


def test_deterministic_chunk_id():
    id1 = make_chunk_id(1, 2, 0, "sample text snippet")
    id2 = make_chunk_id(1, 2, 0, "sample text snippet")
    id3 = make_chunk_id(1, 2, 1, "different chunk index")
    assert id1 == id2
    assert id1 != id3
    assert len(id1) == 24


def test_empty_page_yields_no_chunks():
    empty_page = PageText(
        document_id=1,
        page_number=1,
        text="",
        char_count=0,
        is_empty=True,
        is_scanned=True,
    )
    chunks = chunk_page(empty_page, filename="test.pdf")
    assert chunks == []


def test_short_page_single_chunk():
    page = PageText(
        document_id=5,
        page_number=3,
        text="Introduction\n\nThis is a brief page that easily fits within the target token size.",
        char_count=80,
        is_empty=False,
        is_scanned=False,
    )
    chunks = chunk_page(page, filename="report.pdf", target_tokens=500)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.document_id == 5
    assert c.page_number == 3
    assert c.chunk_index == 0
    assert c.filename == "report.pdf"
    assert "brief page" in c.text
    assert c.section_heading == "Introduction"


def test_long_page_splits_with_preserved_page_number():
    long_text = "\n\n".join([f"Paragraph {i}: " + ("data " * 100) for i in range(10)])
    page = PageText(
        document_id=10,
        page_number=7,
        text=long_text,
        char_count=len(long_text),
        is_empty=False,
        is_scanned=False,
    )
    chunks = chunk_page(page, filename="large.pdf", target_tokens=100, overlap_tokens=10)
    assert len(chunks) > 1

    # Every chunk MUST have page_number == 7
    for idx, c in enumerate(chunks):
        assert c.page_number == 7
        assert c.document_id == 10
        assert c.chunk_index == idx
        assert c.filename == "large.pdf"


def test_chunk_document_never_crosses_pages():
    p1 = PageText(1, 1, "Page one content.", 16, False, False)
    p2 = PageText(1, 2, "Page two content.", 16, False, False)
    p3 = PageText(1, 3, "", 0, True, True)  # empty scanned page

    chunks = chunk_document([p1, p2, p3], filename="multi.pdf")
    assert len(chunks) == 2

    assert chunks[0].page_number == 1
    assert chunks[1].page_number == 2

