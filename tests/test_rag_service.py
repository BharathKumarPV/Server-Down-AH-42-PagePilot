"""
Tests for RAG service and background worker (Stage 4).
"""

from unittest.mock import MagicMock
import sys
from PySide6.QtWidgets import QApplication

from pagepilot.ai.rag_service import RAGService, RAGWorker
from pagepilot.pdf.models import Chunk, Document, RetrievalResult


_app = QApplication.instance() or QApplication(sys.argv)


def test_rag_service_insufficient_evidence():
    ollama_client = MagicMock()
    hybrid_retriever = MagicMock()
    db = MagicMock()

    # Empty retrieval
    hybrid_retriever.retrieve.return_value = RetrievalResult(chunks=[], method="none")

    service = RAGService(ollama_client, hybrid_retriever, db)
    stream, citations = service.answer_question("Where is the secret base?")

    assert citations == []
    answer = "".join(list(stream))
    assert "could not find sufficient information" in answer
    # Ollama chat_stream was NOT called since evidence was absent
    ollama_client.chat_stream.assert_not_called()


def test_rag_service_evidence_grounded():
    ollama_client = MagicMock()
    hybrid_retriever = MagicMock()
    db = MagicMock()

    c = Chunk("c1", 10, "handbook.pdf", 5, 0, "Server port is 8080.", 8)
    hybrid_retriever.retrieve.return_value = RetrievalResult(chunks=[c], method="hybrid")

    # Mock streaming tokens
    ollama_client.chat_stream.return_value = iter(["The ", "port ", "is ", "8080 ", "[handbook.pdf, p. 5]."])

    service = RAGService(ollama_client, hybrid_retriever, db)
    stream, citations = service.answer_question("What is the server port?")

    assert len(citations) == 1
    assert citations[0].filename == "handbook.pdf"
    assert citations[0].page_number == 5

    tokens = list(stream)
    assert "".join(tokens) == "The port is 8080 [handbook.pdf, p. 5]."
    ollama_client.chat_stream.assert_called_once()


def test_rag_service_summarize_document():
    ollama_client = MagicMock()
    hybrid_retriever = MagicMock()
    db = MagicMock()

    doc = Document(1, "brief.pdf", "/p", "h", 2, "time", "ready", "model", 2)
    db.get_document.return_value = doc

    c1 = Chunk("c1", 1, "brief.pdf", 1, 0, "Chapter 1 summary content.", 10)
    c2 = Chunk("c2", 1, "brief.pdf", 2, 0, "Chapter 2 conclusion content.", 10)
    db.get_chunks_for_document.return_value = [c1, c2]

    ollama_client.chat_stream.return_value = iter(["Summary ", "of ", "brief.pdf"])

    service = RAGService(ollama_client, hybrid_retriever, db)
    stream, citations = service.summarize_document(1)

    assert len(citations) == 2
    summary = "".join(list(stream))
    assert summary == "Summary of brief.pdf"


def test_rag_worker_signals():
    """Test RAGWorker emitting tokens and citations via Qt signals."""
    service = MagicMock()
    c = Chunk("c1", 1, "doc.pdf", 1, 0, "Some text", 5)
    citations = [MagicMock()]

    service.answer_question.return_value = (iter(["token1 ", "token2"]), citations)

    worker = RAGWorker(service, "question", parent=None)

    received_tokens = []
    received_citations = []
    finished_args = []

    worker.token_received.connect(lambda t: received_tokens.append(t))
    worker.citations_ready.connect(lambda c: received_citations.append(c))
    worker.finished.connect(lambda text, cits: finished_args.append((text, cits)))

    worker.start()
    worker.wait(5000)
    _app.processEvents()

    assert received_tokens == ["token1 ", "token2"]
    assert len(received_citations) == 1
    assert len(finished_args) == 1
    assert finished_args[0][0] == "token1 token2"
