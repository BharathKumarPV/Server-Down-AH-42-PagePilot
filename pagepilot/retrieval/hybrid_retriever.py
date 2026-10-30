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

