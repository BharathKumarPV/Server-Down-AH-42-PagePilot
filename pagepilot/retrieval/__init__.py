"""
PagePilot – Retrieval package (Stage 4).
"""

from pagepilot.retrieval.embedding_client import EmbeddingClient, EmbeddingError
from pagepilot.retrieval.vector_store import VectorStore
from pagepilot.retrieval.keyword_search import KeywordSearch
from pagepilot.retrieval.hybrid_retriever import HybridRetriever

__all__ = [
    "EmbeddingClient",
    "EmbeddingError",
    "VectorStore",
    "KeywordSearch",
    "HybridRetriever",
]
