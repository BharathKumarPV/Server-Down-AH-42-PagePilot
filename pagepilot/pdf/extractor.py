"""
PagePilot – PDF text extractor using PyMuPDF (Stage 4).

Design rules:
- Pages are yielded one at a time (generator) to avoid holding all text in RAM.
- Page numbers are always 1-based in all public APIs.
- SHA-256 is computed on the raw file bytes before opening the document.
- Corrupted / password-protected files raise ExtractorError.
- Empty pages are flagged is_scanned=True (no OCR is attempted).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Generator

from pagepilot.pdf.models import PageText


class ExtractorError(Exception):
    """Raised when a PDF cannot be opened or is unreadable."""


def compute_sha256(file_path: str) -> str:
    """Compute SHA-256 hex digest of a file without loading it all into RAM."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_fitz():
    try:
        import pymupdf as fitz
        return fitz
    except ImportError:
        try:
            import fitz
            return fitz
        except ImportError as exc:
            raise ExtractorError(
                "PyMuPDF is not installed. Run: pip install pymupdf"
            ) from exc


def extract_pages(
    file_path: str,
    document_id: int,
) -> Generator[PageText, None, None]:
    """
    Yield PageText objects for each page in the PDF.

    Parameters
    ----------
    file_path:
        Absolute path to the PDF file.
    document_id:
        Database ID of the document record (already inserted).

    Yields
    ------
    PageText — one per page, 1-based page numbers, is_scanned=True for empty pages.

    Raises
    ------
    ExtractorError — if the file cannot be opened or is encrypted.
    """
    fitz = _get_fitz()

    path = Path(file_path)
    if not path.exists():
        raise ExtractorError(f"File not found: {file_path}")

    try:
        doc = fitz.open(str(path))
    except fitz.FileDataError as exc:
        raise ExtractorError(f"Cannot open PDF (corrupted or unsupported): {exc}") from exc
    except Exception as exc:
        raise ExtractorError(f"Unexpected error opening PDF: {exc}") from exc

    try:
        if doc.is_encrypted and not doc.authenticate(""):
            raise ExtractorError(
                f"PDF is password-protected and cannot be opened: {path.name}"
            )

        for page in doc:
            # Page numbers are 0-based in PyMuPDF; convert to 1-based
            page_number = page.number + 1
            text = page.get_text("text")  # plain text extraction
            text = text or ""
            stripped = text.strip()
            is_empty = len(stripped) == 0

            yield PageText(
                document_id=document_id,
                page_number=page_number,
                text=text,
                char_count=len(text),
                is_empty=is_empty,
                is_scanned=is_empty,   # flag potentially scanned pages
            )
    finally:
        doc.close()


def get_page_count(file_path: str) -> int:
    """Return the total number of pages in a PDF without full extraction."""
    fitz = _get_fitz()
    try:
        doc = fitz.open(str(file_path))
        count = doc.page_count
        doc.close()
        return count
    except Exception as exc:
        raise ExtractorError(f"Cannot determine page count: {exc}") from exc
