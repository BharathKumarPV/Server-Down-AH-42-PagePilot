"""
PagePilot – RAG Service and Background Worker (Stage 4).

Coordinates:
- Hybrid chunk retrieval
- Prompt construction with anti-injection boundaries
- Streaming Qwen2.5:3b generation via Ollama
- Deterministic citation generation from chunk metadata
- PySide6 QThread worker for responsive UI
"""

from __future__ import annotations

from typing import Callable, Generator, List, Optional, Tuple

from PySide6.QtCore import QThread, Signal

from pagepilot.ai.ollama_client import ChatMessage, OllamaClient, OllamaError
from pagepilot.ai.prompts import (
    RAG_SYSTEM_PROMPT,
    SUMMARIZE_SYSTEM_PROMPT,
    build_rag_user_prompt,
    build_summarize_doc_prompt,
)
from pagepilot.config import RETRIEVAL_TOP_K
from pagepilot.db.database import DocumentDB
from pagepilot.pdf.citation import build_citations
from pagepilot.pdf.models import Chunk, Citation
from pagepilot.retrieval.hybrid_retriever import HybridRetriever


class RAGService:
    """
    Coordinates retrieval, prompt construction, and generation for document Q&A and summarization.
    """

    def __init__(
        self,
        ollama_client: OllamaClient,
        hybrid_retriever: HybridRetriever,
        db: DocumentDB,
    ):
        self.client = ollama_client
        self.retriever = hybrid_retriever
        self.db = db

    def answer_question(
        self,
        question: str,
        document_ids: list[int] | None = None,
        cancel_flag: list[bool] | None = None,
    ) -> Tuple[Generator[str, None, None], list[Citation]]:
        """
        Retrieve relevant passages and stream an evidence-grounded answer.

        Returns
        -------
        tuple(generator_yielding_tokens, list_of_citations)
        """
        retrieval_res = self.retriever.retrieve(
            query=question,
            document_ids=document_ids,
            top_k=RETRIEVAL_TOP_K,
        )

        chunks = retrieval_res.chunks
        if not chunks:
            citations: list[Citation] = []

            def empty_stream():
                yield (
                    "Based on the provided documents, I could not find sufficient "
                    "information to answer this question."
                )

            return empty_stream(), citations

        citations = build_citations(chunks)
        user_prompt = build_rag_user_prompt(question, chunks)
        messages = [
            ChatMessage(role="system", content=RAG_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ]

        stream = self.client.chat_stream(messages, cancel_flag=cancel_flag)
        return stream, citations

    def summarize_document(
        self,
        document_id: int,
        cancel_flag: list[bool] | None = None,
    ) -> Tuple[Generator[str, None, None], list[Citation]]:
        """
        Generate a structured summary of an indexed document.
        """
        doc = self.db.get_document(document_id)
        if not doc:
            raise ValueError(f"Document {document_id} not found.")

        chunks = self.db.get_chunks_for_document(document_id)
        if not chunks:
            def no_text_stream():
                yield "This document does not contain extractable text to summarize."

            return no_text_stream(), []

        # Select representative chunks across the document to stay within context
        # (e.g. first chunks, middle chunks, concluding chunks up to 8 chunks)
        if len(chunks) > 8:
            step = max(1, len(chunks) // 8)
            selected_chunks = [chunks[i] for i in range(0, len(chunks), step)][:8]
        else:
            selected_chunks = chunks

        citations = build_citations(selected_chunks)
        user_prompt = build_summarize_doc_prompt(doc.filename, selected_chunks)
        messages = [
            ChatMessage(role="system", content=SUMMARIZE_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ]

        stream = self.client.chat_stream(messages, cancel_flag=cancel_flag)
        return stream, citations


# ──────────────────────────────────────────────────────────────────────────────
# Background Worker for PySide6
# ──────────────────────────────────────────────────────────────────────────────

class RAGWorker(QThread):
    """
    QThread worker for running RAG retrieval and streaming inference in background.
    """

    token_received   = Signal(str)
    citations_ready  = Signal(list)     # list[Citation]
    finished         = Signal(str, list) # (full_text, list[Citation])
    error_occurred   = Signal(str)
    cancelled        = Signal()

    def __init__(
        self,
        rag_service: RAGService,
        question: str,
        document_ids: list[int] | None = None,
        is_summary: bool = False,
        summary_doc_id: int | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._service = rag_service
        self._question = question
        self._doc_ids = document_ids
        self._is_summary = is_summary
        self._summary_doc_id = summary_doc_id
        self._cancel = [False]
        self._accumulated = ""
        self._citations: list[Citation] = []

    def cancel(self):
        """Thread-safe cancellation request."""
        self._cancel[0] = True

    def run(self):
        try:
            if self._is_summary and self._summary_doc_id is not None:
                stream, citations = self._service.summarize_document(
                    self._summary_doc_id,
                    cancel_flag=self._cancel,
                )
            else:
                stream, citations = self._service.answer_question(
                    self._question,
                    document_ids=self._doc_ids,
                    cancel_flag=self._cancel,
                )

            self._citations = citations
            self.citations_ready.emit(citations)

            for token in stream:
                if self._cancel[0]:
                    self.cancelled.emit()
                    return
                self._accumulated += token
                self.token_received.emit(token)

            if self._cancel[0]:
                self.cancelled.emit()
            else:
                self.finished.emit(self._accumulated, self._citations)

        except OllamaError as exc:
            self.error_occurred.emit(str(exc))
        except Exception as exc:
            self.error_occurred.emit(f"RAG Error: {exc}")

