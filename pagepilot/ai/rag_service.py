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
from pagepilot.ai.claim_models import (
    Claim,
    Contradiction,
    Evidence,
    InformationRequirement,
    VerificationStatus,
    VerifiedAnswer,
)
from pagepilot.ai.claim_verifier import ClaimVerifier
from pagepilot.ai.query_planner import QueryPlanner
from pagepilot.ai.prompts import (
    RAG_SYSTEM_PROMPT,
    STRUCTURED_CLAIM_SYSTEM_PROMPT,
    SUMMARIZE_SYSTEM_PROMPT,
    build_rag_user_prompt,
    build_structured_claim_prompt,
    build_summarize_doc_prompt,
    parse_structured_claims_json,
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

    def answer_question_verified(
        self,
        question: str,
        document_ids: list[int] | None = None,
        cancel_flag: list[bool] | None = None,
    ) -> VerifiedAnswer:
        """
        Stage 6A Multi-Document Reasoning & Claim-Level Verification Pipeline.
        1. Query planning and decomposition into bounded requirements.
        2. Cross-document hybrid retrieval balancing multi-source evidence.
        3. Contradiction detection across documents.
        4. Structured atomic factual claim generation.
        5. Conservative claim verification against cited evidence IDs.
        6. Sentence-level citation rendering with explicit abstention for missing facts.
        """
        planner = QueryPlanner(self.client)
        requirements = planner.plan(question)

        evidence_list, updated_reqs = self.retriever.retrieve_cross_document(
            requirements=requirements,
            document_ids=document_ids,
            top_k_per_req=4,
            max_total_evidence=12,
            balance_diversity=True,
        )

        evidence_map: dict[str, Evidence] = {e.evidence_id: e for e in evidence_list}

        # Track requirements with insufficient evidence
        unanswered: list[str] = [
            req.description or req.subquery
            for req in updated_reqs
            if not req.has_sufficient_evidence
        ]

        verifier = ClaimVerifier()
        contradictions = verifier.detect_cross_document_contradictions(evidence_list)

        if not evidence_list:
            return VerifiedAnswer(
                question=question,
                claims=[],
                evidence_map={},
                contradictions=[],
                unanswered_requirements=[question] if not unanswered else unanswered,
                contributing_documents=[],
                raw_response="",
            )

        user_prompt = build_structured_claim_prompt(
            question=question,
            evidence_list=evidence_list,
            requirements=updated_reqs,
        )
        messages = [
            ChatMessage(role="system", content=STRUCTURED_CLAIM_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ]

        # Call model to obtain structured JSON claims
        raw_output_tokens: list[str] = []
        for token in self.client.chat_stream(messages, cancel_flag=cancel_flag):
            if cancel_flag and cancel_flag[0]:
                break
            raw_output_tokens.append(token)
        raw_output = "".join(raw_output_tokens)

        # Parse structured claims
        parsed_claims, model_unanswered = parse_structured_claims_json(
            raw_output,
            available_evidence_ids=set(evidence_map.keys()),
        )

        for u in model_unanswered:
            if u not in unanswered:
                unanswered.append(u)

        # Verify claims against evidence
        verified_claims = verifier.verify_all(parsed_claims, evidence_map)

        # Determine contributing documents from supported claims
        contributing_docs: set[str] = set()
        for c in verified_claims:
            if c.verification_status == VerificationStatus.SUPPORTED:
                for eid in c.evidence_ids:
                    ev = evidence_map.get(eid)
                    if ev:
                        contributing_docs.add(ev.filename)

        return VerifiedAnswer(
            question=question,
            claims=verified_claims,
            evidence_map=evidence_map,
            contradictions=contradictions,
            unanswered_requirements=unanswered,
            contributing_documents=sorted(list(contributing_docs)),
            raw_response=raw_output,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Background Worker for PySide6
# ──────────────────────────────────────────────────────────────────────────────

class RAGWorker(QThread):
    """
    QThread worker for running RAG retrieval and streaming inference in background.
    Supports both standard streaming and Stage 6A verified multi-document mode.
    """

    token_received        = Signal(str)
    citations_ready       = Signal(list)     # list[Citation]
    finished              = Signal(str, list) # (full_text, list[Citation])
    verified_answer_ready = Signal(object)   # VerifiedAnswer
    error_occurred        = Signal(str)
    cancelled             = Signal()

    def __init__(
        self,
        rag_service: RAGService,
        question: str,
        document_ids: list[int] | None = None,
        is_summary: bool = False,
        summary_doc_id: int | None = None,
        is_verified_mode: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self._service = rag_service
        self._question = question
        self._doc_ids = document_ids
        self._is_summary = is_summary
        self._summary_doc_id = summary_doc_id
        self._is_verified_mode = is_verified_mode
        self._cancel = [False]
        self._accumulated = ""
        self._citations: list[Citation] = []

    def cancel(self):
        """Thread-safe cancellation request."""
        self._cancel[0] = True

    def run(self):
        try:
            if self._is_verified_mode and not self._is_summary:
                # Stage 6A: Verified Multi-Document Answer Generation
                verified_ans = self._service.answer_question_verified(
                    self._question,
                    document_ids=self._doc_ids,
                    cancel_flag=self._cancel,
                )

                if self._cancel[0]:
                    self.cancelled.emit()
                    return

                # Convert evidence_map to Citation objects for citation pills
                citations_list: list[Citation] = []
                seen_pairs: set[tuple[int, int]] = set()
                for ev in verified_ans.evidence_map.values():
                    pair = (ev.document_id, ev.page_number)
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        citations_list.append(
                            Citation(
                                document_id=ev.document_id,
                                filename=ev.filename,
                                page_number=ev.page_number,
                                chunk_text=ev.original_text,
                                relevance_score=1.0 / ev.retrieval_rank,
                            )
                        )

                self._citations = citations_list
                self.citations_ready.emit(citations_list)

                rendered_text = verified_ans.render_sentence_level_text()
                self._accumulated = rendered_text
                self.token_received.emit(rendered_text)

                self.verified_answer_ready.emit(verified_ans)
                self.finished.emit(rendered_text, citations_list)
                return

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

