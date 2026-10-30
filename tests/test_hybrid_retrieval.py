"""
Tests for Hybrid Retriever with Reciprocal Rank Fusion (Stage 4).
"""

from unittest.mock import MagicMock
from pagepilot.pdf.models import Chunk
from pagepilot.retrieval.hybrid_retriever import HybridRetriever, RRF_K


def test_rrf_scoring_combines_ranks():
    vector_store = MagicMock()
    keyword_search = MagicMock()
    embedding_client = MagicMock()
    embedding_client.embed_query.return_value = [0.1] * 768

    c1 = Chunk("c1", 1, "doc1.pdf", 1, 0, "Chunk 1 text", 10)
    c2 = Chunk("c2", 1, "doc1.pdf", 2, 0, "Chunk 2 text", 10)
    c3 = Chunk("c3", 1, "doc1.pdf", 3, 0, "Chunk 3 text", 10)

    # Semantic returns: c1 (rank 1), c2 (rank 2)
    vector_store.query.return_value = [c1, c2]

    # Keyword returns: c2 (rank 1), c3 (rank 2)
    keyword_search.search.return_value = [c2, c3]

    retriever = HybridRetriever(
        vector_store=vector_store,
        keyword_search=keyword_search,
        embedding_client=embedding_client,
    )

    res = retriever.retrieve("test query", top_k=3)
    assert res.method == "hybrid"
    assert len(res.chunks) == 3

    # c2 was rank 2 in semantic + rank 1 in keyword
    # score(c2) = 1/(60+2) + 1/(60+1) = 1/62 + 1/61 ~= 0.01613 + 0.01639 = 0.03252
    # score(c1) = 1/(60+1) = 1/61 ~= 0.01639
    # score(c3) = 1/(60+2) = 1/62 ~= 0.01613
    # Therefore, c2 MUST be ranked 1st!
    assert res.chunks[0].chunk_id == "c2"
    assert res.chunks[1].chunk_id == "c1"
    assert res.chunks[2].chunk_id == "c3"


def test_retriever_degrades_when_embedding_fails():
    vector_store = MagicMock()
    keyword_search = MagicMock()
    embedding_client = MagicMock()
    # Embedding client raises error (e.g. Ollama embed model unavailable)
    embedding_client.embed_query.side_effect = Exception("Ollama embed offline")

    c = Chunk("c1", 1, "doc.pdf", 1, 0, "Fallback keyword result", 5)
    keyword_search.search.return_value = [c]

    retriever = HybridRetriever(vector_store, keyword_search, embedding_client)
    res = retriever.retrieve("query", top_k=5)

    assert res.method == "keyword"
    assert len(res.chunks) == 1
    assert res.chunks[0].chunk_id == "c1"
    # Ensure vector store was NOT queried since no embedding was produced
    vector_store.query.assert_not_called()


def test_retriever_insufficient_evidence_when_both_empty():
    vector_store = MagicMock()
    keyword_search = MagicMock()
    embedding_client = MagicMock()
    # Embedding client fails
    embedding_client.embed_query.side_effect = Exception("Ollama embed offline")
    # Keyword search finds nothing
    keyword_search.search.return_value = []

    retriever = HybridRetriever(vector_store, keyword_search, embedding_client)
    res = retriever.retrieve("unknown query", top_k=5)

    assert res.method == "none"
    assert res.chunks == []
    vector_store.query.assert_not_called()


def test_no_dense_query_without_embedding():
    vector_store = MagicMock()
    keyword_search = MagicMock()
    embedding_client = MagicMock()
    # Embedding client returns None or empty
    embedding_client.embed_query.return_value = []

    retriever = HybridRetriever(vector_store, keyword_search, embedding_client)
    retriever.retrieve("query")

    vector_store.query.assert_not_called()


def test_retriever_empty_query():
    retriever = HybridRetriever(MagicMock(), MagicMock(), MagicMock())
    res = retriever.retrieve("")
    assert res.chunks == []
    assert res.method == "empty"
