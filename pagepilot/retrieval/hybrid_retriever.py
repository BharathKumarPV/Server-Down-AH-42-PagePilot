"""
PagePilot – Hybrid retrieval with Reciprocal Rank Fusion (Stage 4).

Combines dense vector semantic search (ChromaDB + nomic-embed-text)
with exact keyword search (SQLite FTS5) using Reciprocal Rank Fusion (RRF).

Formula:
  RRF_score(d) = SUM( 1.0 / (k + rank(d)) )   where k = 60
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pagepilot.config import RETRIEVAL_TOP_K
from pagepilot.ai.claim_models import Evidence, InformationRequirement
from pagepilot.pdf.models import Chunk, RetrievalResult
from pagepilot.retrieval.embedding_client import EmbeddingClient, EmbeddingError
from pagepilot.retrieval.keyword_search import KeywordSearch
from pagepilot.retrieval.vector_store import VectorStore


RRF_K = 60.0


class HybridRetriever:
    """
    Coordinates semantic vector search and exact keyword search, merging
    results via Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        vector_store: VectorStore,
        keyword_search: KeywordSearch,
        embedding_client: EmbeddingClient,
    ):
        self.vector_store = vector_store
        self.keyword_search = keyword_search
        self.embedding_client = embedding_client

    def retrieve(
        self,
        query: str,
        document_ids: list[int] | None = None,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> RetrievalResult:
        """
        Execute hybrid retrieval across indexed documents.

        Parameters
        ----------
        query:
            Natural language question or search query.
        document_ids:
            Optional filter to restrict retrieval to specific documents.
        top_k:
            Number of top results to return.

        Returns
        -------
        RetrievalResult containing merged, deduplicated, and ranked Chunks.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return RetrievalResult(chunks=[], method="empty")

        candidate_limit = max(top_k * 2, 10)
        chunks_by_id: dict[str, Chunk] = {}
        rrf_scores: dict[str, float] = {}

        # 1. Semantic Vector Search
        semantic_chunks: list[Chunk] = []
        try:
            query_embedding = self.embedding_client.embed_query(cleaned_query)
            if query_embedding and isinstance(query_embedding, list) and len(query_embedding) > 0:
                semantic_chunks = self.vector_store.query(
                    query_embedding=query_embedding,
                    top_k=candidate_limit,
                    document_ids=document_ids,
                )
        except (EmbeddingError, Exception):
            # Graceful degradation if embedding model is unavailable: fall back to keyword-only
            semantic_chunks = []

        for rank_idx, chunk in enumerate(semantic_chunks):
            rank = rank_idx + 1
            chunks_by_id[chunk.chunk_id] = chunk
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (
                1.0 / (RRF_K + rank)
            )

        # 2. Keyword Search (FTS5 / LIKE)
        keyword_chunks: list[Chunk] = []
        try:
            keyword_chunks = self.keyword_search.search(
                query=cleaned_query,
                document_ids=document_ids,
                top_k=candidate_limit,
            )
        except Exception:
            keyword_chunks = []

        for rank_idx, chunk in enumerate(keyword_chunks):
            rank = rank_idx + 1
            chunks_by_id[chunk.chunk_id] = chunk
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (
                1.0 / (RRF_K + rank)
            )

        # Determine method label
        if semantic_chunks and keyword_chunks:
            method = "hybrid"
        elif semantic_chunks:
            method = "semantic"
        elif keyword_chunks:
            method = "keyword"
        else:
            return RetrievalResult(chunks=[], method="none")

        # Sort candidate chunks by RRF score descending
        sorted_chunk_ids = sorted(
            rrf_scores.keys(),
            key=lambda cid: rrf_scores[cid],
            reverse=True,
        )

        final_chunks = [chunks_by_id[cid] for cid in sorted_chunk_ids[:top_k]]
        return RetrievalResult(chunks=final_chunks, method=method)

    def retrieve_cross_document(
        self,
        requirements: list[InformationRequirement],
        document_ids: list[int] | None = None,
        top_k_per_req: int = 4,
        max_total_evidence: int = 12,
        balance_diversity: bool = True,
    ) -> tuple[list[Evidence], list[InformationRequirement]]:
        """
        Execute multi-query cross-document retrieval.
        Gathers evidence for each information requirement, deduplicates identical
        chunks, balances document representation, and constructs Evidence objects.
        """
        if not requirements:
            return [], []

        # Map: chunk_id -> dict of (chunk, method, list_of_req_ids, min_rank)
        merged_chunks: dict[str, dict] = {}
        updated_reqs: list[InformationRequirement] = []

        for req in requirements:
            res = self.retrieve(
                query=req.subquery,
                document_ids=document_ids,
                top_k=top_k_per_req,
            )

            has_evidence = len(res.chunks) > 0
            supported_docs = sorted(list({c.document_id for c in res.chunks})) if has_evidence else []

            updated_reqs.append(
                InformationRequirement(
                    requirement_id=req.requirement_id,
                    subquery=req.subquery,
                    description=req.description,
                    supported_doc_ids=supported_docs,
                    has_sufficient_evidence=has_evidence,
                )
            )

            for rank_idx, chunk in enumerate(res.chunks):
                if chunk.chunk_id not in merged_chunks:
                    merged_chunks[chunk.chunk_id] = {
                        "chunk": chunk,
                        "method": res.method,
                        "req_ids": [req.requirement_id],
                        "best_rank": rank_idx + 1,
                    }
                else:
                    if req.requirement_id not in merged_chunks[chunk.chunk_id]["req_ids"]:
                        merged_chunks[chunk.chunk_id]["req_ids"].append(req.requirement_id)
                    merged_chunks[chunk.chunk_id]["best_rank"] = min(
                        merged_chunks[chunk.chunk_id]["best_rank"],
                        rank_idx + 1,
                    )

        if not merged_chunks:
            return [], updated_reqs

        # Diversity balancing: avoid single document crowding out others
        if balance_diversity and document_ids and len(document_ids) > 1:
            # Group by document_id
            by_doc: dict[int, list[dict]] = {}
            for item in merged_chunks.values():
                doc_id = item["chunk"].document_id
                by_doc.setdefault(doc_id, []).append(item)

            for d_items in by_doc.values():
                d_items.sort(key=lambda x: (len(x["req_ids"]) * -1, x["best_rank"]))

            selected_items: list[dict] = []
            # Round-robin pick from each contributing document
            max_rounds = max(len(items) for items in by_doc.values())
            for round_idx in range(max_rounds):
                for doc_id, items in by_doc.items():
                    if round_idx < len(items) and len(selected_items) < max_total_evidence:
                        if items[round_idx] not in selected_items:
                            selected_items.append(items[round_idx])

            candidate_items = selected_items
        else:
            # Sort by number of requirements satisfied desc, then best rank asc
            candidate_items = sorted(
                merged_chunks.values(),
                key=lambda x: (len(x["req_ids"]) * -1, x["best_rank"]),
            )[:max_total_evidence]

        # Convert to Evidence objects with clean E1, E2, ... IDs
        evidence_list: list[Evidence] = []
        for idx, item in enumerate(candidate_items, 1):
            ch: Chunk = item["chunk"]
            evidence_list.append(
                Evidence(
                    evidence_id=f"E{idx}",
                    document_id=ch.document_id,
                    filename=ch.filename,
                    page_number=ch.page_number,
                    chunk_id=ch.chunk_id,
                    chunk_index=ch.chunk_index,
                    original_text=ch.text,
                    retrieval_method=item["method"],
                    retrieval_rank=idx,
                    query_requirement_id=",".join(item["req_ids"]),
                )
            )

        return evidence_list, updated_reqs

