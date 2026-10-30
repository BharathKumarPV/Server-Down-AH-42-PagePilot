"""
Stage 4 Offline Verification and Network Traffic Audit.

Monitors all socket connections made during:
1. PDF ingestion
2. Vector embedding
3. Vector storage
4. Hybrid retrieval
5. Qwen inference
6. Document summarization
7. Restart persistence check
"""

import os
import socket
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf as fitz

from pagepilot.db.database import DocumentDB
from pagepilot.pdf.ingestion_service import IngestionService
from pagepilot.retrieval.embedding_client import EmbeddingClient
from pagepilot.retrieval.hybrid_retriever import HybridRetriever
from pagepilot.retrieval.keyword_search import KeywordSearch
from pagepilot.retrieval.vector_store import VectorStore
from pagepilot.ai.rag_service import RAGService
from pagepilot.ai.ollama_client import OllamaClient


# ──────────────────────────────────────────────────────────────────────────────
# Network Traffic Auditor
# ──────────────────────────────────────────────────────────────────────────────

ORIGINAL_CONNECT = socket.socket.connect
OUTBOUND_CONNECTIONS = []


def audited_connect(self, address):
    host, port = address[0], address[1]
    OUTBOUND_CONNECTIONS.append((host, port))
    return ORIGINAL_CONNECT(self, address)


socket.socket.connect = audited_connect


def run_offline_audit():
    print("=" * 60)
    print("PAGEPILOT STAGE 4 — OFFLINE NETWORK AUDIT")
    print("=" * 60)

    audit_dir = Path(os.environ.get("TEMP", ".")) / "pagepilot_offline_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = audit_dir / "offline_sample.pdf"

    # Create test document
    doc = fitz.open()
    rect = fitz.Rect(50, 50, 550, 750)
    p = doc.new_page()
    p.insert_textbox(rect, "Offline Operations Policy\n\nAll cryptographic keys are stored exclusively in local secure memory without external sync.")
    doc.save(str(pdf_path))
    doc.close()

    db_path = audit_dir / "offline_docs.db"
    chroma_path = audit_dir / "offline_chroma"

    # Initial session
    print("[1] Initializing local offline services...")
    db = DocumentDB(db_path)
    vector_store = VectorStore(persist_dir=chroma_path)
    vector_store.reset()

    embedding_client = EmbeddingClient(model="nomic-embed-text")
    keyword_search = KeywordSearch(db)
    hybrid_retriever = HybridRetriever(vector_store, keyword_search, embedding_client)
    ollama_client = OllamaClient(model="qwen2.5:3b")
    rag_service = RAGService(ollama_client, hybrid_retriever, db)
    ingestion_service = IngestionService(db, vector_store, embedding_client)

    # 1. Ingest PDF
    print("[2] Ingesting PDF locally...")
    doc_rec = ingestion_service.ingest_pdf(str(pdf_path))
    assert doc_rec.status == "ready"

    # 2. Hybrid Search
    print("[3] Performing hybrid retrieval...")
    ret_res = hybrid_retriever.retrieve("Where are cryptographic keys stored?")
    assert len(ret_res.chunks) > 0

    # 3. Answer Generation
    print("[4] Generating answer with local Qwen...")
    stream, citations = rag_service.answer_question("Where are cryptographic keys stored?")
    ans = "".join(list(stream))
    print(f"    Answer: {ans.strip()}")
    assert len(citations) > 0

    # 4. Summarization
    print("[5] Generating document summary...")
    sum_stream, sum_cits = rag_service.summarize_document(doc_rec.id)
    summary = "".join(list(sum_stream))
    print(f"    Summary: {summary.strip()[:100]}...")

    # 5. Simulate App Restart (re-instantiate from persistent files)
    print("[6] Simulating application restart and verifying persistent storage...")
    del rag_service
    del hybrid_retriever
    del vector_store
    del db

    db_reopened = DocumentDB(db_path)
    vector_store_reopened = VectorStore(persist_dir=chroma_path)
    assert db_reopened.count_documents() == 1
    assert db_reopened.count_chunks() == 1
    assert vector_store_reopened.count_vectors() == 1
    print("[+] Persistence verified: SQLite DB and ChromaDB vectors intact across restart.")

    # 6. Audit Network Connections
    print("\n[7] Network Connection Audit:")
    external_calls = []
    local_calls = []

    for host, port in OUTBOUND_CONNECTIONS:
        if str(host) in ("127.0.0.1", "localhost", "::1"):
            local_calls.append((host, port))
        else:
            external_calls.append((host, port))

    print(f"    • Local loopback calls (127.0.0.1:11434): {len(local_calls)}")
    print(f"    • External Internet calls: {len(external_calls)}")

    if external_calls:
        print(f"[-] VIOLATION: External calls detected: {external_calls}")
        sys.exit(1)
    else:
        print("[+] ZERO external network calls made. System operates 100% offline.")

    print("\n" + "=" * 60)
    print("OFFLINE VERIFICATION SUCCESSFUL")
    print("=" * 60)


if __name__ == "__main__":
    run_offline_audit()

