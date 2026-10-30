"""
Tests for IngestionService and DocumentManager (Stage 4).
"""

from unittest.mock import MagicMock
import pymupdf as fitz
import pytest

from pagepilot.db.database import DocumentDB
from pagepilot.pdf.document_manager import DocumentManager
from pagepilot.pdf.ingestion_service import IngestionService
from pagepilot.retrieval.embedding_client import EmbeddingClient
from pagepilot.retrieval.vector_store import VectorStore


@pytest.fixture
def test_pdf(tmp_path):
    path = tmp_path / "research.pdf"
    doc = fitz.open()

    p1 = doc.new_page()
    p1.insert_text((50, 50), "Neural Networks in Robotics\n\nRobotics applications benefit from local inference.")

    p2 = doc.new_page()
    p2.insert_text((50, 50), "Edge Computing\n\nEdge devices require low power consumption and high efficiency.")

    doc.save(str(path))
    doc.close()
    return path


def test_ingestion_service_end_to_end(tmp_path, test_pdf):
    db = DocumentDB(tmp_path / "docs.db")
    vector_store = VectorStore(persist_dir=tmp_path / "chroma")

    embedding_client = MagicMock(spec=EmbeddingClient)
    embedding_client.model = "nomic-embed-text"
    dummy_embed = lambda texts, progress_callback=None: [[0.1] * 768 for _ in texts]
    embedding_client.embed.side_effect = dummy_embed
    embedding_client.embed_documents.side_effect = dummy_embed

    service = IngestionService(db, vector_store, embedding_client)

    progress_events = []

    def on_prog(stage, cur, tot):
        progress_events.append(stage)

    doc = service.ingest_pdf(str(test_pdf), progress_callback=on_prog)

    assert doc.id is not None
    assert doc.status == "ready"
    assert doc.page_count == 2
    assert doc.chunk_count == 2
    assert doc.filename == "research.pdf"

    # Verify progress events
    assert "hashing" in progress_events
    assert "extracting" in progress_events
    assert "embedding" in progress_events
    assert "done" in progress_events

    # Verify vectors in VectorStore
    assert vector_store.count_vectors() == 2

    # Verify chunks in DB
    chunks = db.get_chunks_for_document(doc.id)
    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert chunks[1].page_number == 2

    # Verify deduplication: re-ingesting same file
    doc2 = service.ingest_pdf(str(test_pdf))
    assert doc2.id == doc.id
    # No extra vectors added
    assert vector_store.count_vectors() == 2


def test_document_manager_listing_and_deletion(tmp_path, test_pdf):
    db = DocumentDB(tmp_path / "docs.db")
    vector_store = VectorStore(persist_dir=tmp_path / "chroma")

    embedding_client = MagicMock(spec=EmbeddingClient)
    embedding_client.model = "nomic-embed-text"
    dummy_embed = lambda texts, progress_callback=None: [[0.1] * 768 for _ in texts]
    embedding_client.embed.side_effect = dummy_embed
    embedding_client.embed_documents.side_effect = dummy_embed

    service = IngestionService(db, vector_store, embedding_client)
    manager = DocumentManager(db, vector_store)

    doc = service.ingest_pdf(str(test_pdf))
    assert manager.count_documents() == 1
    assert manager.count_chunks() == 2

    docs = manager.list_documents()
    assert len(docs) == 1
    assert docs[0].id == doc.id

    # Delete
    manager.delete_document(doc.id)
    assert manager.count_documents() == 0
    assert manager.count_chunks() == 0
    assert vector_store.count_vectors() == 0

