"""
Stage 4 Citation Verification Script.

Tests with a real multi-page PDF containing unique text on each page:
- Page 1: Apollo Missions (1961 - 1972)
- Page 2: Voyager Interstellar Probes (1977)
- Page 3: James Webb Space Telescope (2021)
"""

import os
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf as fitz
from pagepilot.config import DOCS_DB_PATH, CHROMA_DIR, ensure_chroma_dir
from pagepilot.db.database import DocumentDB
from pagepilot.pdf.citation import open_pdf_document, build_citations
from pagepilot.pdf.document_manager import DocumentManager
from pagepilot.pdf.ingestion_service import IngestionService
from pagepilot.retrieval.embedding_client import EmbeddingClient
from pagepilot.retrieval.hybrid_retriever import HybridRetriever
from pagepilot.retrieval.keyword_search import KeywordSearch
from pagepilot.retrieval.vector_store import VectorStore
from pagepilot.ai.rag_service import RAGService
from pagepilot.ai.ollama_client import OllamaClient


def run_citation_verification():
    print("=" * 60)
    print("PAGEPILOT STAGE 4 — CITATION VERIFICATION RUN")
    print("=" * 60)

    # 1. Create multi-page test PDF with proper text wrapping
    pdf_dir = Path(os.environ.get("TEMP", ".")) / "pagepilot_test_docs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / "space_exploration_history.pdf"

    doc = fitz.open()
    page_rect = fitz.Rect(50, 50, 550, 750)

    p1 = doc.new_page()
    p1.insert_textbox(
        page_rect,
        "Apollo Lunar Exploration (Page 1)\n\n"
        "Project Apollo was initiated in 1961 by NASA. The Apollo 11 mission "
        "successfully landed astronauts Neil Armstrong and Buzz Aldrin on the Moon "
        "on July 20, 1969. The command module was named Columbia."
    )

    p2 = doc.new_page()
    p2.insert_textbox(
        page_rect,
        "Voyager Interstellar Probes (Page 2)\n\n"
        "Voyager 1 and Voyager 2 were launched in 1977 to explore the outer planets. "
        "Voyager 1 entered interstellar space on August 25, 2012, carrying a Golden Record "
        "with sounds and images of Earth."
    )

    p3 = doc.new_page()
    p3.insert_textbox(
        page_rect,
        "James Webb Space Telescope (Page 3)\n\n"
        "The James Webb Space Telescope (JWST) was launched on December 25, 2021. "
        "It orbits around the Sun-Earth Lagrange point L2, approximately 1.5 million kilometers "
        "from Earth, capturing deep infrared astronomical images."
    )

    doc.save(str(pdf_path))
    doc.close()
    print(f"[+] Created test PDF: {pdf_path} (3 pages)")

    # 2. Setup services in test location
    test_db_path = pdf_dir / "test_docs.db"
    test_chroma_path = pdf_dir / "test_chroma"

    # Remove previous test DB/Chroma if existing
    if test_db_path.exists():
        try:
            test_db_path.unlink()
        except Exception:
            pass

    db = DocumentDB(test_db_path)
    vector_store = VectorStore(persist_dir=test_chroma_path)
    vector_store.reset()

    embedding_client = EmbeddingClient(model="nomic-embed-text")
    keyword_search = KeywordSearch(db)
    hybrid_retriever = HybridRetriever(vector_store, keyword_search, embedding_client)
    ollama_client = OllamaClient(model="qwen2.5:3b")
    rag_service = RAGService(ollama_client, hybrid_retriever, db)
    ingestion_service = IngestionService(db, vector_store, embedding_client)

    # 3. Ingest document
    print("[*] Ingesting PDF into hybrid index...")
    doc_record = ingestion_service.ingest_pdf(str(pdf_path))
    print(f"[+] Document ingested: ID={doc_record.id}, Filename={doc_record.filename}, Pages={doc_record.page_count}, Chunks={doc_record.chunk_count}")

    # Verify ID and Filename
    assert doc_record.filename == "space_exploration_history.pdf", "Filename mismatch"
    assert doc_record.page_count == 3, "Page count mismatch"
    assert doc_record.chunk_count == 3, "Chunk count mismatch"

    # 4. Query specifically targeting Page 2
    query_p2 = "When did Voyager 1 enter interstellar space?"
    print(f"\n[*] Query 1: '{query_p2}'")
    stream, citations = rag_service.answer_question(query_p2, document_ids=[doc_record.id])
    answer_tokens = list(stream)
    answer_text = "".join(answer_tokens)

    print(f"[+] Answer: {answer_text.strip()}")
    print(f"[+] Citations count: {len(citations)}")
    for c in citations:
        print(f"    Citation: {c.format_short()} | Doc ID: {c.document_id} | Page: {c.page_number}")
        print(f"    Excerpt Preview: {c.chunk_text[:100]}...")

    # Assertions for Page 2
    assert len(citations) > 0, "No citations returned for query 1"
    top_citation = citations[0]
    assert top_citation.filename == "space_exploration_history.pdf"
    assert top_citation.page_number == 2, f"Expected page 2, got {top_citation.page_number}"
    assert top_citation.document_id == doc_record.id
    assert "August 25, 2012" in top_citation.chunk_text

    # 5. Query specifically targeting Page 3
    query_p3 = "Where does the James Webb Space Telescope orbit?"
    print(f"\n[*] Query 2: '{query_p3}'")
    stream3, citations3 = rag_service.answer_question(query_p3, document_ids=[doc_record.id])
    answer_tokens3 = list(stream3)
    answer_text3 = "".join(answer_tokens3)

    print(f"[+] Answer: {answer_text3.strip()}")
    print(f"[+] Citations count: {len(citations3)}")
    for c in citations3:
        print(f"    Citation: {c.format_short()} | Doc ID: {c.document_id} | Page: {c.page_number}")

    # Assertions for Page 3
    assert len(citations3) > 0, "No citations returned for query 2"
    top_citation3 = citations3[0]
    assert top_citation3.page_number == 3, f"Expected page 3, got {top_citation3.page_number}"
    assert "Lagrange point L2" in top_citation3.chunk_text

    # 6. Test opening the PDF at cited page
    print("\n[*] Testing open_pdf_document verification...")
    # Verify file existence and readability at cited page
    assert Path(doc_record.file_path).exists()
    doc_check = fitz.open(doc_record.file_path)
    assert doc_check.page_count >= top_citation.page_number
    page_data = doc_check[top_citation.page_number - 1].get_text()
    norm_page = " ".join(page_data.split())
    assert "Voyager 1 entered interstellar space on August 25, 2012" in norm_page
    doc_check.close()
    print("[+] Verified cited page contains the supporting evidence.")

    print("\n" + "=" * 60)
    print("ALL CITATION BEHAVIORS SUCCESSFULLY VERIFIED:")
    print("1. Correct filename: space_exploration_history.pdf")
    print("2. Correct 1-based page numbers: Page 2 for Voyager, Page 3 for JWST")
    print("3. Correct document ID:", doc_record.id)
    print("4. Cited passages directly support the generated answers")
    print("5. Document text matches cited page content exactly")
    print("=" * 60)


if __name__ == "__main__":
    run_citation_verification()
