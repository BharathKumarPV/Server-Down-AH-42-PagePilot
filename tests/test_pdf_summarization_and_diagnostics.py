"""
PagePilot – Critical Master Fix Verification Suite.

Tests:
1. Intent classification & query cleaning (Document summary, code explanation, page ranges).
2. Diagnostic metadata tracking (Total pages, text pages, scanned pages, chunks, vectors).
3. Safe duplicate-free document re-indexing.
4. Hierarchical summarization across sections and multi-document synthesis.
5. Code explanation extraction and prompt structuring.
6. Conversation document scope persistence across sessions.
7. Reduced motion setting and UI performance configuration.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pymupdf as fitz
import pytest

from pagepilot import config
from pagepilot.ai.ollama_client import ChatMessage, OllamaClient
from pagepilot.ai.prompts import (
    build_code_explanation_prompt,
    build_hierarchical_summary_prompt,
    build_multi_doc_summary_prompt,
)
from pagepilot.ai.query_planner import (
    INTENT_CODE_EXPLANATION,
    INTENT_DOCUMENT_QUESTION,
    INTENT_DOCUMENT_SUMMARY,
    INTENT_GENERAL_CHAT,
    INTENT_MULTI_DOCUMENT_COMPARISON,
    INTENT_TARGETED_LOOKUP,
    QueryPlanner,
)
from pagepilot.ai.rag_service import RAGService
from pagepilot.db.database import ConversationDB, DocumentDB
from pagepilot.pdf.models import (
    Chunk,
    DOC_STATUS_READY,
    DOC_STATUS_SCANNED_OCR_REQUIRED,
    Document,
)
from pagepilot.pdf.extractor import extract_pages
from pagepilot.pdf.ingestion_service import IngestionService
from pagepilot.retrieval.hybrid_retriever import HybridRetriever, RetrievalResult
from pagepilot.retrieval.keyword_search import KeywordSearch
from pagepilot.retrieval.vector_store import VectorStore


# ──────────────────────────────────────────────────────────────────────────────
# 1. Intent Classification & Query Cleaning
# ──────────────────────────────────────────────────────────────────────────────

def test_intent_classification_document_summary():
    planner = QueryPlanner()
    assert planner.classify_intent("Explain what's in the PDF and summarize it.") == INTENT_DOCUMENT_SUMMARY
    assert planner.classify_intent("Can you provide an overview of the document?") == INTENT_DOCUMENT_SUMMARY
    assert planner.classify_intent("Give me the tl;dr of this file.") == INTENT_DOCUMENT_SUMMARY
    assert planner.classify_intent("What does this PDF contain?") == INTENT_DOCUMENT_SUMMARY
    assert planner.classify_intent("Summarize pages 20 to 30") == INTENT_DOCUMENT_SUMMARY


def test_intent_classification_page_range_extraction():
    planner = QueryPlanner()
    span1 = planner.extract_page_range("Summarize pages 20 to 30")
    assert span1 == (20, 30)

    span2 = planner.extract_page_range("Explain page 5-12 in detail")
    assert span2 == (5, 12)

    span3 = planner.extract_page_range("Summarize pages 40 through 35")
    assert span3 == (35, 40)

    assert planner.extract_page_range("Explain the introduction") is None


def test_intent_classification_code_explanation():
    planner = QueryPlanner()
    assert planner.classify_intent("Explain code in detail in the uploaded PDF.") == INTENT_CODE_EXPLANATION
    assert planner.classify_intent("Walk through the Python implementation of quicksort.") == INTENT_CODE_EXPLANATION
    assert planner.classify_intent("How is the train_step function implemented?") == INTENT_CODE_EXPLANATION
    assert planner.classify_intent("Describe the script in the document.") == INTENT_CODE_EXPLANATION


def test_intent_classification_comparison_and_general():
    planner = QueryPlanner()
    assert planner.classify_intent("Compare the two uploaded documents.") == INTENT_MULTI_DOCUMENT_COMPARISON
    assert planner.classify_intent("What is the difference between doc A and doc B?") == INTENT_MULTI_DOCUMENT_COMPARISON
    assert planner.classify_intent("What is machine learning?", has_documents=False) == INTENT_GENERAL_CHAT


def test_clean_retrieval_query():
    planner = QueryPlanner()
    q1 = "Explain code in detail in the uploaded PDF."
    cleaned1 = planner.clean_retrieval_query(q1)
    assert "uploaded" not in cleaned1.lower()
    assert "pdf" not in cleaned1.lower()
    assert "code" in cleaned1.lower()

    q2 = "Summarize the key findings in this document?"
    cleaned2 = planner.clean_retrieval_query(q2)
    assert "document" not in cleaned2.lower()
    assert "key findings" in cleaned2.lower()


# ──────────────────────────────────────────────────────────────────────────────
# 2. Database Diagnostics Metadata Tracking
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def temp_dbs(tmp_path):
    doc_db_path = tmp_path / "test_docs.db"
    conv_db_path = tmp_path / "test_conv.db"
    return DocumentDB(doc_db_path), ConversationDB(conv_db_path)


def test_document_diagnostics_retrieval(temp_dbs):
    doc_db, _ = temp_dbs
    doc = doc_db.create_document(
        filename="Pandas_Series.pdf",
        file_path="/path/to/Pandas_Series.pdf",
        sha256_hash="abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
        page_count=48,
        status="ready",
        embed_model="nomic-embed-text",
        extracted_pages_count=48,
        scanned_pages_count=0,
        char_count=125400,
        embedded_chunks_count=96,
        indexed_pages_count=48,
    )

    diag = doc_db.get_document_diagnostics(doc.id)
    assert diag is not None
    assert diag["id"] == doc.id
    assert diag["filename"] == "Pandas_Series.pdf"
    assert diag["total_pages"] == 48
    assert diag["extracted_text_pages"] == 48
    assert diag["scanned_pages"] == 0
    assert diag["total_characters"] == 125400
    assert diag["embedded_chunks"] == 96
    assert diag["indexed_pages"] == 48
    assert diag["status"] == "ready"


def test_conversation_document_scope_persistence(temp_dbs):
    _, conv_db = temp_dbs
    conv = conv_db.create_conversation("RAG Investigation")

    # Initial scope is empty
    assert conv_db.get_conversation_documents(conv.id) == []

    # Persist document association
    conv_db.set_conversation_documents(conv.id, [101, 102])
    assert conv_db.get_conversation_documents(conv.id) == [101, 102]

    # Update scope
    conv_db.set_conversation_documents(conv.id, [101])
    assert conv_db.get_conversation_documents(conv.id) == [101]

    # Clear scope
    conv_db.set_conversation_documents(conv.id, [])
    assert conv_db.get_conversation_documents(conv.id) == []


# ──────────────────────────────────────────────────────────────────────────────
# 3. Safe Duplicate-Free Re-indexing
# ──────────────────────────────────────────────────────────────────────────────

def test_safe_duplicate_free_reindexing(tmp_path):
    pdf_path = tmp_path / "sample_doc.pdf"
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((50, 50), "Section 1: Data Structures in Python. Lists, tuples, and dictionaries.")
    p2 = doc.new_page()
    p2.insert_text((50, 50), "Section 2: Object-Oriented Programming. Classes and inheritance.")
    doc.save(str(pdf_path))
    doc.close()

    db_path = tmp_path / "test_reindex.db"
    doc_db = DocumentDB(db_path)

    # Mock vector store
    mock_vs = MagicMock(spec=VectorStore)
    mock_vs.add_chunks.return_value = None
    mock_vs.delete_document.return_value = None

    # Mock embedding client
    mock_embed = MagicMock()
    mock_embed.model = "nomic-embed-text"
    mock_embed.embed_batch.return_value = [[0.1] * 768, [0.2] * 768]

    service = IngestionService(
        db=doc_db,
        vector_store=mock_vs,
        embedding_client=mock_embed,
    )

    # Initial Ingestion
    ingested_doc = service.ingest_pdf(str(pdf_path))
    assert ingested_doc.page_count == 2
    assert ingested_doc.status == DOC_STATUS_READY

    initial_chunks = doc_db.get_chunks_for_document(ingested_doc.id)
    initial_chunk_count = len(initial_chunks)
    assert initial_chunk_count > 0

    # Re-index Document
    reindexed_doc = service.reindex_document(ingested_doc.id)
    assert reindexed_doc.id == ingested_doc.id
    assert mock_vs.delete_document.called

    after_chunks = doc_db.get_chunks_for_document(reindexed_doc.id)
    # Ensure chunk count did not double!
    assert len(after_chunks) == initial_chunk_count


# ──────────────────────────────────────────────────────────────────────────────
# 4. Hierarchical Document Summarization & Code Workflows
# ──────────────────────────────────────────────────────────────────────────────

def test_hierarchical_summary_prompt_construction():
    sections = [
        {"pages_label": "pp. 1–10", "text": "Overview of Series and DataFrame indexing."},
        {"pages_label": "pp. 11–20", "text": "Vectorized string operations and regular expressions."},
    ]
    prompt = build_hierarchical_summary_prompt("Pandas_Manual.pdf", sections)
    assert "Pandas_Manual.pdf" in prompt
    assert "pp. 1" in prompt
    assert "pp. 11" in prompt
    assert "Produce a complete, structured summary" in prompt


def test_multi_doc_summary_prompt_construction():
    doc_summaries = [
        {"filename": "Report_A.pdf", "summary": "Quarterly sales increased by 15%."},
        {"filename": "Report_B.pdf", "summary": "Customer acquisition cost decreased by 5%."},
    ]
    prompt = build_multi_doc_summary_prompt(doc_summaries)
    assert "Report_A.pdf" in prompt
    assert "Report_B.pdf" in prompt
    assert "Quarterly sales increased" in prompt
    assert "cross-document" in prompt.lower()


def test_code_explanation_prompt_construction():
    chunk = Chunk(
        chunk_id="chunk-1",
        document_id=1,
        filename="fib.py",
        page_number=4,
        chunk_index=0,
        text="def compute_fibonacci(n):\n    if n <= 1:\n        return n\n    return compute_fibonacci(n-1) + compute_fibonacci(n-2)",
        token_count=35,
    )
    prompt = build_code_explanation_prompt("Explain the recursive algorithm in the code", [chunk])
    assert "compute_fibonacci" in prompt
    assert "Page 4" in prompt
    assert "Preserve the code syntax" in prompt
    assert "explain its purpose" in prompt


def test_rag_service_hierarchical_summarization_flow(temp_dbs):
    doc_db, _ = temp_dbs
    doc = doc_db.create_document(
        filename="Python_Guide.pdf",
        file_path="/path/to/Python_Guide.pdf",
        sha256_hash="1111222233334444",
        page_count=25,
        status="ready",
        embed_model="nomic-embed-text",
    )

    chunks = [
        Chunk(
            chunk_id=f"c-{i}",
            document_id=doc.id,
            filename=doc.filename,
            page_number=i,
            chunk_index=0,
            text=f"Content for page {i} discussing core features.",
            token_count=20,
        )
        for i in range(1, 26)
    ]
    doc_db.insert_chunks(chunks)

    mock_ollama = MagicMock(spec=OllamaClient)
    mock_ollama.chat_stream.return_value = iter(["This ", "document ", "covers ", "Python ", "core ", "features."])

    mock_retriever = MagicMock(spec=HybridRetriever)
    service = RAGService(
        ollama_client=mock_ollama,
        hybrid_retriever=mock_retriever,
        db=doc_db,
    )

    stream, citations = service.summarize_document_hierarchical(doc.id, page_range=(1, 10))
    tokens = list(stream)
    assert "".join(tokens) == "This document covers Python core features."
    assert len(citations) > 0
    assert all(c.page_number <= 10 for c in citations)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Accessibility & Reduced Motion Configuration
# ──────────────────────────────────────────────────────────────────────────────

def test_reduced_motion_configuration():
    assert hasattr(config, "REDUCED_MOTION")
    orig_val = config.REDUCED_MOTION

    config.REDUCED_MOTION = True
    assert config.REDUCED_MOTION is True

    config.REDUCED_MOTION = False
    assert config.REDUCED_MOTION is False

    config.REDUCED_MOTION = orig_val
