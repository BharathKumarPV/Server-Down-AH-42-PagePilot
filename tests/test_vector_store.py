"""
Tests for ChromaDB local vector store (Stage 4).
"""

from pagepilot.pdf.models import Chunk
from pagepilot.retrieval.vector_store import VectorStore


def test_vector_store_add_and_query(tmp_path):
    store = VectorStore(persist_dir=tmp_path / "chroma")

    c1 = Chunk("c1", 1, "doc1.pdf", 1, 0, "Machine learning with local inference", 10, "Intro")
    c2 = Chunk("c2", 1, "doc1.pdf", 2, 0, "Database indexing using FTS5 and SQLite", 10, "DB")
    c3 = Chunk("c3", 2, "doc2.pdf", 1, 0, "Cooking recipes and culinary history", 10, "Food")

    # Distinct 768-dim embeddings
    emb1 = [1.0] * 384 + [0.0] * 384
    emb2 = [0.0] * 384 + [1.0] * 384
    emb3 = [0.5] * 768

    store.add_chunks([c1, c2, c3], [emb1, emb2, emb3])
    assert store.count_vectors() == 3

    # Query with vector close to emb1
    query_emb = [0.9] * 384 + [0.0] * 384
    results = store.query(query_emb, top_k=2)

    assert len(results) == 2
    assert results[0].chunk_id == "c1"
    assert results[0].page_number == 1
    assert results[0].filename == "doc1.pdf"
    assert results[0].section_heading == "Intro"


def test_vector_store_filter_by_document_id(tmp_path):
    store = VectorStore(persist_dir=tmp_path / "chroma")

    c1 = Chunk("c1", 10, "doc10.pdf", 1, 0, "Text from doc 10", 5)
    c2 = Chunk("c2", 20, "doc20.pdf", 1, 0, "Text from doc 20", 5)

    emb = [0.1] * 768
    store.add_chunks([c1, c2], [emb, emb])

    # Filter to only doc 20
    results = store.query(emb, top_k=5, document_ids=[20])
    assert len(results) == 1
    assert results[0].document_id == 20
    assert results[0].chunk_id == "c2"


def test_vector_store_delete_document(tmp_path):
    store = VectorStore(persist_dir=tmp_path / "chroma")

    c1 = Chunk("c1", 1, "doc1.pdf", 1, 0, "Content 1", 5)
    c2 = Chunk("c2", 1, "doc1.pdf", 2, 0, "Content 2", 5)
    c3 = Chunk("c3", 2, "doc2.pdf", 1, 0, "Content 3", 5)

    emb = [0.2] * 768
    store.add_chunks([c1, c2, c3], [emb, emb, emb])
    assert store.count_vectors() == 3

    # Delete doc 1
    store.delete_document(1)
    assert store.count_vectors() == 1

    # Remaining chunk is doc 2
    res = store.query(emb, top_k=5)
    assert len(res) == 1
    assert res[0].document_id == 2


def test_vector_store_empty_query(tmp_path):
    store = VectorStore(persist_dir=tmp_path / "chroma")
    res = store.query([0.1] * 768, top_k=5)
    assert res == []


def test_vector_store_reset(tmp_path):
    store = VectorStore(persist_dir=tmp_path / "chroma")
    c1 = Chunk("c1", 1, "doc1.pdf", 1, 0, "Content", 5)
    store.add_chunks([c1], [[0.1] * 768])
    assert store.count_vectors() == 1

    store.reset()
    assert store.count_vectors() == 0

