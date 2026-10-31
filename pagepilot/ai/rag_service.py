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

from typing import Callable, Generator, List, Optional, Tuple, TYPE_CHECKING

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
from pagepilot.ai.query_planner import (
    INTENT_CODE_EXPLANATION,
    INTENT_DOCUMENT_QUESTION,
    INTENT_DOCUMENT_SUMMARY,
    INTENT_GENERAL_CHAT,
    INTENT_MULTI_DOCUMENT_COMPARISON,
    INTENT_TARGETED_LOOKUP,
    QueryPlanner,
)
from pagepilot.ai.prompts import (
    CODE_EXPLANATION_SYSTEM_PROMPT,
    RAG_SYSTEM_PROMPT,
    STRUCTURED_CLAIM_SYSTEM_PROMPT,
    SUMMARIZE_HIERARCHICAL_SYSTEM_PROMPT,
    SUMMARIZE_SYSTEM_PROMPT,
    build_code_explanation_prompt,
    build_hierarchical_summary_prompt,
    build_multi_doc_summary_prompt,
    build_rag_user_prompt,
    build_structured_claim_prompt,
    build_summarize_doc_prompt,
    parse_structured_claims_json,
)
from pagepilot.config import RETRIEVAL_TOP_K
from pagepilot.db.database import DocumentDB
from pagepilot.pdf.citation import build_citations
from pagepilot.pdf.models import Chunk, Citation
if TYPE_CHECKING:
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

    def summarize_document_hierarchical(
        self,
        document_id: int,
        page_range: Optional[tuple[int, int]] = None,
        cancel_flag: list[bool] | None = None,
    ) -> Tuple[Generator[str, None, None], list[Citation]]:
        """
        Produce a hierarchical, evidence-grounded summary of a single PDF document.
        Handles large documents (e.g. 48 pages) section-by-section without context overflow.
        """
        doc = self.db.get_document(document_id)
        if not doc:
            raise ValueError(f"Document {document_id} not found.")

        all_chunks = self.db.get_chunks_for_document(document_id)
        if not all_chunks:
            def no_text():
                if doc.status == "scanned_ocr_required" or doc.scanned_pages_count > 0:
                    yield (
                        f"Document **{doc.filename}** contains {doc.scanned_pages_count} scanned pages.\n\n"
                        "Real OCR processing and review are required before this document can be summarized."
                    )
                else:
                    yield f"Document **{doc.filename}** contains no extractable text to summarize."
            return no_text(), []

        # Filter by page range if requested (e.g. "pages 20 to 30")
        if page_range:
            start_p, end_p = page_range
            chunks = [c for c in all_chunks if start_p <= c.page_number <= end_p]
            if not chunks:
                def range_empty():
                    yield (
                        f"No extractable text found in pages {start_p} to {end_p} of **{doc.filename}** "
                        f"(document has {doc.page_count} pages total)."
                    )
                return range_empty(), []
        else:
            chunks = all_chunks

        # Group chunks into sections across page spans
        pages_in_chunks = sorted(list({c.page_number for c in chunks}))
        sections: list[dict] = []
        section_size = 6

        for i in range(0, len(pages_in_chunks), section_size):
            p_slice = pages_in_chunks[i : i + section_size]
            s_chunks = [c for c in chunks if c.page_number in p_slice]
            p_label = f"pp. {p_slice[0]}–{p_slice[-1]}" if len(p_slice) > 1 else f"p. {p_slice[0]}"

            sec_texts = []
            for c in s_chunks:
                heading = f"[{c.section_heading}] " if c.section_heading else ""
                clean_t = c.text.strip()
                if len(clean_t) > 300:
                    clean_t = clean_t[:300] + "…"
                sec_texts.append(f"{heading}{clean_t}")

            section_content = "\n".join(sec_texts[:4])
            sections.append({
                "pages_label": p_label,
                "text": section_content,
                "chunks": s_chunks,
            })

        citations = build_citations(chunks)
        user_prompt = build_hierarchical_summary_prompt(doc.filename, sections)
        messages = [
            ChatMessage(role="system", content=SUMMARIZE_HIERARCHICAL_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ]

        stream = self.client.chat_stream(messages, cancel_flag=cancel_flag)
        return stream, citations

    def summarize_multiple_documents(
        self,
        document_ids: list[int],
        cancel_flag: list[bool] | None = None,
    ) -> Tuple[Generator[str, None, None], list[Citation]]:
        """
        Summarize multiple selected documents with individual breakdowns and a cross-document overview.
        """
        doc_summaries: list[dict] = []
        all_citations: list[Citation] = []

        for did in document_ids:
            doc = self.db.get_document(did)
            if not doc:
                continue
            chunks = self.db.get_chunks_for_document(did)
            if not chunks:
                doc_summaries.append({
                    "filename": doc.filename,
                    "summary": f"[No extractable text; status: {doc.status}]",
                })
                continue

            all_citations.extend(build_citations(chunks))

            if len(chunks) > 8:
                step = max(1, len(chunks) // 8)
                sampled = [chunks[i] for i in range(0, len(chunks), step)][:8]
            else:
                sampled = chunks

            doc_text = "\n".join(f"[p. {c.page_number}] {c.text[:250]}" for c in sampled)
            doc_summaries.append({
                "filename": doc.filename,
                "summary": doc_text,
            })

        user_prompt = build_multi_doc_summary_prompt(doc_summaries)
        messages = [
            ChatMessage(role="system", content=SUMMARIZE_HIERARCHICAL_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ]

        seen = set()
        dedup_cits: list[Citation] = []
        for c in all_citations:
            key = (c.document_id, c.page_number)
            if key not in seen:
                seen.add(key)
                dedup_cits.append(c)

        stream = self.client.chat_stream(messages, cancel_flag=cancel_flag)
        return stream, dedup_cits

    def explain_code(
        self,
        question: str,
        document_ids: list[int] | None = None,
        cancel_flag: list[bool] | None = None,
    ) -> Tuple[Generator[str, None, None], list[Citation]]:
        """
        Extract and explain code blocks and implementation passages from documents.
        """
        planner = QueryPlanner(self.client)
        cleaned_query = planner.clean_retrieval_query(question)

        # Retrieve relevant passages containing code patterns or programming terms
        retrieval_res = self.retriever.retrieve(
            query=f"code implementation function class script {cleaned_query}".strip(),
            document_ids=document_ids,
            top_k=8,
        )

        code_chunks = retrieval_res.chunks
        if not code_chunks:
            retrieval_res = self.retriever.retrieve(
                query=cleaned_query or question,
                document_ids=document_ids,
                top_k=8,
            )
            code_chunks = retrieval_res.chunks

        if not code_chunks:
            def no_code_stream():
                yield (
                    "Based on the selected documents, I could not find code or implementation "
                    "passages to explain."
                )
            return no_code_stream(), []

        citations = build_citations(code_chunks)
        user_prompt = build_code_explanation_prompt(question, code_chunks)
        messages = [
            ChatMessage(role="system", content=CODE_EXPLANATION_SYSTEM_PROMPT),
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
            planner = QueryPlanner()
            intent = planner.classify_intent(self._question, has_documents=bool(self._doc_ids))
            page_range = planner.extract_page_range(self._question)

            if self._is_summary or intent == INTENT_DOCUMENT_SUMMARY:
                if self._doc_ids and len(self._doc_ids) > 1:
                    stream, citations = self._service.summarize_multiple_documents(
                        self._doc_ids,
                        cancel_flag=self._cancel,
                    )
                elif self._summary_doc_id is not None:
                    stream, citations = self._service.summarize_document_hierarchical(
                        self._summary_doc_id,
                        page_range=page_range,
                        cancel_flag=self._cancel,
                    )
                elif self._doc_ids and len(self._doc_ids) == 1:
                    stream, citations = self._service.summarize_document_hierarchical(
                        self._doc_ids[0],
                        page_range=page_range,
                        cancel_flag=self._cancel,
                    )
                else:
                    stream, citations = self._service.answer_question(
                        self._question,
                        document_ids=self._doc_ids,
                        cancel_flag=self._cancel,
                    )

            elif intent == INTENT_CODE_EXPLANATION:
                stream, citations = self._service.explain_code(
                    self._question,
                    document_ids=self._doc_ids,
                    cancel_flag=self._cancel,
                )

            elif self._is_verified_mode and not self._is_summary:
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

            else:
                stream, citations = self._service.answer_question(
                    self._question,
                    document_ids=self._doc_ids,
                    cancel_flag=self._cancel,
                )

            # Common streaming handler for summary, code_explanation, and standard Q&A
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

