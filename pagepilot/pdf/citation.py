"""
PagePilot – Citation management and document referencing (Stage 4).

Citations are generated strictly from retrieved chunk metadata — never invented
by LLM text generation.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple

from pagepilot.pdf.models import Chunk, Citation


def build_citations(chunks: list[Chunk]) -> list[Citation]:
    """
    Construct deduplicated Citation objects from a list of retrieved Chunks.
    Preserves retrieval rank and 1-based page numbers.
    """
    citations: list[Citation] = []
    seen: set[Tuple[int, int]] = set()

    for idx, chunk in enumerate(chunks):
        key = (chunk.document_id, chunk.page_number)
        if key not in seen:
            seen.add(key)
            citations.append(
                Citation(
                    document_id=chunk.document_id,
                    filename=chunk.filename,
                    page_number=chunk.page_number,
                    chunk_text=chunk.text,
                    relevance_score=1.0 / (idx + 1),
                )
            )

    return citations


def extract_cited_references(text: str) -> list[str]:
    """
    Find citation markers like '[document.pdf, p. 3]' inside LLM response text.
    """
    pattern = r"\[([a-zA-Z0-9_\-\. ]+),\s*(?:p\.|page)\s*([0-9]+)\]"
    matches = re.findall(pattern, text, re.IGNORECASE)
    return [f"[{filename}, p. {page}]" for filename, page in matches]


def open_pdf_document(file_path: str, page_number: int = 1) -> bool:
    """
    Open the given PDF in the system default PDF viewer on Windows.
    Returns True if launched, False otherwise.
    """
    path = Path(file_path)
    if not path.exists():
        return False

    try:
        if os.name == "nt":
            os.startfile(str(path))
            return True
        else:
            import subprocess
            subprocess.Popen(["xdg-open", str(path)])
            return True
    except Exception:
        return False

