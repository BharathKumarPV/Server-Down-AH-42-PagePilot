"""
PagePilot – AI package.
"""

from pagepilot.ai.ollama_client import ChatMessage, OllamaClient, OllamaError, OllamaStatus
from pagepilot.ai.worker import InferenceWorker
from pagepilot.ai.rag_service import RAGService, RAGWorker
from pagepilot.ai.prompts import (
    RAG_SYSTEM_PROMPT,
    SUMMARIZE_SYSTEM_PROMPT,
    build_rag_user_prompt,
    build_summarize_doc_prompt,
)

__all__ = [
    "ChatMessage",
    "OllamaClient",
    "OllamaError",
    "OllamaStatus",
    "InferenceWorker",
    "RAGService",
    "RAGWorker",
    "RAG_SYSTEM_PROMPT",
    "SUMMARIZE_SYSTEM_PROMPT",
    "build_rag_user_prompt",
    "build_summarize_doc_prompt",
]
