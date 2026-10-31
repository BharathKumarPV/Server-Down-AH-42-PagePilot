"""
Stage 4 System Memory and Resource Benchmarking.

Measures:
1. Application process RSS memory.
2. Ollama memory during nomic-embed-text embedding.
3. Ollama memory during Qwen2.5 3B inference.
4. Chroma storage footprint on disk.
5. Peak memory during multi-page PDF ingestion.
"""

import os
import sys
import time
import tracemalloc
from pathlib import Path

# Ensure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psutil
import pymupdf as fitz

from pagepilot.config import CHROMA_DIR, DOCS_DB_PATH
from pagepilot.db.database import DocumentDB
from pagepilot.pdf.ingestion_service import IngestionService
from pagepilot.retrieval.embedding_client import EmbeddingClient
from pagepilot.retrieval.hybrid_retriever import HybridRetriever
from pagepilot.retrieval.keyword_search import KeywordSearch
from pagepilot.retrieval.vector_store import VectorStore
from pagepilot.ai.rag_service import RAGService
from pagepilot.ai.ollama_client import OllamaClient


def get_ollama_memory_mb() -> float:
    """Return total RSS memory of all Ollama related processes in MB."""
    total_rss = 0
    for p in psutil.process_iter(['name', 'memory_info']):
        try:
            name = (p.info['name'] or '').lower()
            if 'ollama' in name:
                total_rss += p.info['memory_info'].rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total_rss / (1024 * 1024)


def get_dir_size_mb(path: Path) -> float:
    """Return total size of directory in MB."""
    if not path.exists():
        return 0.0
    total = 0
    for f in path.rglob('*'):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 * 1024)


def run_benchmark():
    print("=" * 60)
    print("PAGEPILOT STAGE 4 — HARDWARE MEMORY BENCHMARK")
    print(f"OS: {psutil.os.name} | Total RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    print("=" * 60)

    proc = psutil.Process()

    # 1. Base Application Memory
    base_app_rss = proc.memory_info().rss / (1024 * 1024)
    print(f"[1] Base Python Process Memory: {base_app_rss:.2f} MB")

    # 2. Large PDF Ingestion & Peak Memory
    print("\n[2] Generating 50-page test PDF for stress measurement...")
    bench_dir = Path(os.environ.get("TEMP", ".")) / "pagepilot_bench"
    bench_dir.mkdir(parents=True, exist_ok=True)
    large_pdf_path = bench_dir / "large_benchmark_50p.pdf"

    doc = fitz.open()
    page_rect = fitz.Rect(50, 50, 550, 750)
    for p_idx in range(1, 51):
        p = doc.new_page()
        text = (
            f"Chapter {p_idx}: Automated Intelligence Analysis\n\n"
            + f"Section {p_idx}.1 Technical specifications and distributed indexing.\n"
            + ("Data stream processing occurs locally within system limits without cloud dependence. " * 15)
        )
        p.insert_textbox(page_rect, text)
    doc.save(str(large_pdf_path))
    doc.close()
    pdf_size_kb = large_pdf_path.stat().st_size / 1024
    print(f"[+] Created 50-page PDF ({pdf_size_kb:.1f} KB): {large_pdf_path}")

    # Setup isolated benchmark DB & Chroma
    b_db_path = bench_dir / "bench_docs.db"
    b_chroma_path = bench_dir / "bench_chroma"
    if b_db_path.exists():
        try:
            b_db_path.unlink()
        except Exception:
            pass

    db = DocumentDB(b_db_path)
    vector_store = VectorStore(persist_dir=b_chroma_path)
    vector_store.reset()
    embedding_client = EmbeddingClient(model="nomic-embed-text")
    ingestion_service = IngestionService(db, vector_store, embedding_client)

    # Measure during ingestion
    tracemalloc.start()
    rss_before_ingest = proc.memory_info().rss / (1024 * 1024)
    ollama_before_ingest = get_ollama_memory_mb()

    print("[*] Ingesting 50-page PDF with local nomic-embed-text...")
    t0 = time.time()
    ingested_doc = ingestion_service.ingest_pdf(str(large_pdf_path))
    ingest_time = time.time() - t0

    current_alloc, peak_alloc = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    rss_after_ingest = proc.memory_info().rss / (1024 * 1024)
    ollama_during_embed = get_ollama_memory_mb()

    print(f"[+] Ingestion completed in {ingest_time:.2f}s")
    print(f"[+] Ingested Chunks: {ingested_doc.chunk_count}")
    print(f"[+] Python Ingestion Memory Delta (RSS): {rss_after_ingest - rss_before_ingest:+.2f} MB")
    print(f"[+] Python Peak Tracemalloc Allocation: {peak_alloc / (1024 * 1024):.2f} MB")
    print(f"[+] Ollama Process Memory during Embedding: {ollama_during_embed:.2f} MB")

    # 3. Chroma Storage Size
    chroma_size_mb = get_dir_size_mb(b_chroma_path)
    print(f"\n[3] Chroma Storage on Disk (50 pages / {ingested_doc.chunk_count} vectors): {chroma_size_mb:.2f} MB")

    # 4. Ollama Memory during Qwen Inference
    keyword_search = KeywordSearch(db)
    hybrid_retriever = HybridRetriever(vector_store, keyword_search, embedding_client)
    ollama_client = OllamaClient(model="qwen2.5:3b")
    rag_service = RAGService(ollama_client, hybrid_retriever, db)

    print("\n[4] Measuring Ollama memory during Qwen2.5:3b inference...")
    stream, citations = rag_service.answer_question(
        "Summarize technical specifications in chapter 15",
        document_ids=[ingested_doc.id]
    )
    # Collect stream
    tokens = []
    for token in stream:
        tokens.append(token)

    ollama_during_qwen = get_ollama_memory_mb()
    print(f"[+] Qwen generated {len(tokens)} tokens")
    print(f"[+] Ollama Process Memory during Qwen Inference: {ollama_during_qwen:.2f} MB")

    print("\n" + "=" * 60)
    print("SUMMARY OF EMPIRICAL MEASUREMENTS:")
    print(f"• Application Base RSS:            {base_app_rss:.2f} MB")
    print(f"• Ingestion Peak Python Memory:    {peak_alloc / (1024*1024):.2f} MB (Tracemalloc)")
    print(f"• Ollama RAM (nomic-embed-text):   {ollama_during_embed:.2f} MB")
    print(f"• Ollama RAM (qwen2.5:3b):         {ollama_during_qwen:.2f} MB")
    print(f"• ChromaDB Disk Usage (50 pages):  {chroma_size_mb:.2f} MB ({ingested_doc.chunk_count} vectors)")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()

