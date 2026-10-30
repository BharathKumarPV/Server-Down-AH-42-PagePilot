"""
PagePilot – PDF package (Stage 4).
"""

from pagepilot.pdf.models import Document, PageText, Chunk, Citation, RetrievalResult
from pagepilot.pdf.extractor import extract_pages, get_page_count, compute_sha256, ExtractorError
from pagepilot.pdf.chunker import chunk_page, chunk_document
from pagepilot.pdf.citation import build_citations, extract_cited_references, open_pdf_document

__all__ = [
    "Document",
    "PageText",
    "Chunk",
    "Citation",
    "RetrievalResult",
    "extract_pages",
    "get_page_count",
    "compute_sha256",
    "ExtractorError",
    "chunk_page",
    "chunk_document",
    "build_citations",
    "extract_cited_references",
    "open_pdf_document",
]
