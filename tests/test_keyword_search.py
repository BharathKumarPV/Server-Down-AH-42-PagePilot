"""
Tests for SQLite FTS5 and keyword search (Stage 4).
"""

from pagepilot.db.database import DocumentDB
from pagepilot.pdf.models import Chunk
from pagepilot.retrieval.keyword_search import KeywordSearch


def test_keyword_search_basic(tmp_path):
    db = DocumentDB(tmp_path / "docs.db")
    doc1 = db.create_document("specs.pdf", "/path/specs.pdf", "hash1", 5, "nomic-embed-text")
    doc2 = db.create_document("notes.pdf", "/path/notes.pdf", "hash2", 2, "nomic-embed-text")

    c1 = Chunk("c1", doc1.id, "specs.pdf", 1, 0, "The GPU architecture requires 8GB of VRAM.", 10)
    c2 = Chunk("c2", doc1.id, "specs.pdf", 2, 0, "Local offline speech synthesis runs via Piper.", 10)
    c3 = Chunk("c3", doc2.id, "notes.pdf", 1, 0, "Meeting minutes regarding GPU budgeting.", 10)

    db.insert_chunks([c1, c2, c3])

    ks = KeywordSearch(db)

    # Search for "Piper"
    res1 = ks.search("Piper")
    assert len(res1) == 1
    assert res1[0].chunk_id == "c2"
    assert res1[0].page_number == 2

    # Search for "GPU"
    res2 = ks.search("GPU")
    assert len(res2) == 2
    chunk_ids = {c.chunk_id for c in res2}
    assert chunk_ids == {"c1", "c3"}


def test_keyword_search_document_filter(tmp_path):
    db = DocumentDB(tmp_path / "docs.db")
    doc1 = db.create_document("doc1.pdf", "/p1", "h1", 1, "model")
    doc2 = db.create_document("doc2.pdf", "/p2", "h2", 1, "model")

    c1 = Chunk("c1", doc1.id, "doc1.pdf", 1, 0, "Quantum computing algorithms", 5)
    c2 = Chunk("c2", doc2.id, "doc2.pdf", 1, 0, "Quantum error correction methods", 5)
    db.insert_chunks([c1, c2])

    ks = KeywordSearch(db)

    # Filter to only doc2
    res = ks.search("Quantum", document_ids=[doc2.id])
    assert len(res) == 1
    assert res[0].chunk_id == "c2"
    assert res[0].document_id == doc2.id


def test_keyword_search_sanitization_and_empty(tmp_path):
    db = DocumentDB(tmp_path / "docs.db")
    ks = KeywordSearch(db)

    # Empty query
    assert ks.search("") == []
    assert ks.search("   ") == []

    # Special punctuation shouldn't crash
    assert ks.search("***---???") == []

