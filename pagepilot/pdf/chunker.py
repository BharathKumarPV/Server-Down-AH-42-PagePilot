"""
PagePilot – PDF text chunker with page-level provenance (Stage 4).

Rules:
- Never cross page boundaries: every chunk belongs strictly to one page (1-based).
- Chunks preserve original document_id, filename, and page_number.
- Chunk IDs are deterministic hashes.
- Chunk size is bounded (~400-500 tokens / ~1600-2000 chars) to stay safely inside
  nomic-embed-text's 2048 token window and Qwen's context.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, List

from pagepilot.pdf.models import Chunk, PageText


def estimate_tokens(text: str) -> int:
    """Approximate token count (~4 characters per token for English text)."""
    return max(1, len(text) // 4)


def make_chunk_id(document_id: int, page_number: int, chunk_index: int, text_sample: str) -> str:
    """Generate a deterministic unique chunk ID."""
    raw = f"{document_id}:{page_number}:{chunk_index}:{text_sample[:40]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def chunk_page(
    page: PageText,
    filename: str,
    target_tokens: int = 500,
    overlap_tokens: int = 50,
) -> list[Chunk]:
    """
    Split a single page's text into one or more Chunk objects.

    Parameters
    ----------
    page:
        The PageText object containing page text and 1-based page number.
    filename:
        Document filename for reference.
    target_tokens:
        Target token size per chunk (~500 tokens).
    overlap_tokens:
        Token overlap between consecutive chunks from the same page (~50 tokens).

    Returns
    -------
    list[Chunk] — Empty if page has no printable text.
    """
    raw_text = page.text.strip()
    if not raw_text:
        return []

    target_chars = target_tokens * 4
    overlap_chars = overlap_tokens * 4

    # If the whole page fits comfortably in one chunk
    if len(raw_text) <= target_chars:
        cid = make_chunk_id(page.document_id, page.page_number, 0, raw_text)
        return [
            Chunk(
                chunk_id=cid,
                document_id=page.document_id,
                filename=filename,
                page_number=page.page_number,
                chunk_index=0,
                text=raw_text,
                token_count=estimate_tokens(raw_text),
                section_heading=_detect_heading(raw_text),
            )
        ]

    # Split into paragraphs first to avoid cutting sentences when possible
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw_text) if p.strip()]
    if not paragraphs:
        paragraphs = [raw_text]

    chunks: list[Chunk] = []
    current_text = ""
    chunk_index = 0

    for para in paragraphs:
        # If single paragraph exceeds target, split by sentences or sliding window
        if len(para) > target_chars:
            if current_text:
                cid = make_chunk_id(page.document_id, page.page_number, chunk_index, current_text)
                chunks.append(
                    Chunk(
                        chunk_id=cid,
                        document_id=page.document_id,
                        filename=filename,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                        text=current_text,
                        token_count=estimate_tokens(current_text),
                        section_heading=_detect_heading(current_text),
                    )
                )
                chunk_index += 1
                current_text = ""

            sub_chunks = _split_long_paragraph(para, target_chars, overlap_chars)
            for sub in sub_chunks:
                cid = make_chunk_id(page.document_id, page.page_number, chunk_index, sub)
                chunks.append(
                    Chunk(
                        chunk_id=cid,
                        document_id=page.document_id,
                        filename=filename,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                        text=sub,
                        token_count=estimate_tokens(sub),
                        section_heading=_detect_heading(sub),
                    )
                )
                chunk_index += 1
            continue

        candidate = (current_text + "\n\n" + para).strip() if current_text else para
        if len(candidate) <= target_chars:
            current_text = candidate
        else:
            if current_text:
                cid = make_chunk_id(page.document_id, page.page_number, chunk_index, current_text)
                chunks.append(
                    Chunk(
                        chunk_id=cid,
                        document_id=page.document_id,
                        filename=filename,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                        text=current_text,
                        token_count=estimate_tokens(current_text),
                        section_heading=_detect_heading(current_text),
                    )
                )
                chunk_index += 1
            # Add overlap from tail of current_text if available
            tail = current_text[-overlap_chars:] if len(current_text) > overlap_chars else ""
            current_text = (tail + "\n\n" + para).strip() if tail else para

    if current_text:
        cid = make_chunk_id(page.document_id, page.page_number, chunk_index, current_text)
        chunks.append(
            Chunk(
                chunk_id=cid,
                document_id=page.document_id,
                filename=filename,
                page_number=page.page_number,
                chunk_index=chunk_index,
                text=current_text,
                token_count=estimate_tokens(current_text),
                section_heading=_detect_heading(current_text),
            )
        )

    return chunks


def _split_long_paragraph(text: str, target_chars: int, overlap_chars: int) -> list[str]:
    """Split a single large paragraph using sliding window with sentence awareness."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) <= 1:
        # Fall back to word sliding window
        words = text.split()
        res = []
        step = max(1, (target_chars - overlap_chars) // 6)
        window = max(1, target_chars // 6)
        for i in range(0, len(words), step):
            segment = " ".join(words[i : i + window])
            if segment:
                res.append(segment)
            if i + window >= len(words):
                break
        return res or [text]

    res = []
    cur = ""
    for s in sentences:
        if cur and len(cur) + len(s) + 1 > target_chars:
            res.append(cur)
            # overlap
            tail = cur[-overlap_chars:] if len(cur) > overlap_chars else ""
            cur = (tail + " " + s).strip()
        else:
            cur = (cur + " " + s).strip() if cur else s

    if cur:
        res.append(cur)
    return res


def _detect_heading(text: str) -> str:
    """Extract a likely heading from the first non-empty line of chunk text."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if lines:
        first = lines[0]
        if len(first) < 80 and not first.endswith("."):
            return first
    return ""


def chunk_document(
    pages: Iterable[PageText],
    filename: str,
    target_tokens: int = 500,
    overlap_tokens: int = 50,
) -> list[Chunk]:
    """
    Process all pages of a document and return a flat list of Chunks.
    Page-level provenance is strictly preserved.
    """
    all_chunks: list[Chunk] = []
    for page in pages:
        page_chunks = chunk_page(
            page=page,
            filename=filename,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
        )
        all_chunks.extend(page_chunks)
    return all_chunks

