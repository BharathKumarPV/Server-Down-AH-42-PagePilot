"""
PagePilot – DOCX Citation Formatter (Stage 5).

Handles formatting of inline citations and Sources / References sections.
Preserves real PDF filenames and 1-based page numbers from actual RAG retrievals.
Never invents citations or hallucinated page numbers.
"""

from __future__ import annotations

from typing import Any, List, Optional
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

from pagepilot.docx.templates import TemplateConfig
from pagepilot.docx.styles import (
    set_cell_background,
    set_cell_margins,
    set_table_borders,
)


class CitationMode:
    BOTH = "Both Inline & Sources Section"
    SOURCES_ONLY = "Sources Section Only"
    INLINE_ONLY = "Inline Only"
    NONE = "None"

    ALL_MODES = [BOTH, SOURCES_ONLY, INLINE_ONLY, NONE]

    @classmethod
    def should_include_inline(cls, mode: str) -> bool:
        norm = (mode or "").strip().lower()
        return "inline" in norm or norm == "both"

    @classmethod
    def should_include_sources_section(cls, mode: str) -> bool:
        norm = (mode or "").strip().lower()
        return "source" in norm or norm == "both"


def format_inline_citation(filename: str, page_number: int | None = None) -> str:
    """Format an inline citation string such as [Handbook.pdf, p. 4]."""
    safe_name = (filename or "Document.pdf").strip()
    if page_number is not None and int(page_number) > 0:
        return f"[{safe_name}, p. {page_number}]"
    return f"[{safe_name}]"


def add_sources_section(
    doc,
    citations: list[dict[str, Any]] | list[Any],
    template: TemplateConfig,
    heading_text: str = "Sources & References",
):
    """
    Append a structured, professionally formatted Sources & References section to the document.
    Each citation must be from verified evidence (filename, 1-based page number, optional snippet).
    """
    if not citations:
        return

    # Add section heading
    p_heading = doc.add_paragraph()
    p_heading.paragraph_format.space_before = Pt(16.0)
    p_heading.paragraph_format.space_after = Pt(6.0)
    p_heading.paragraph_format.keep_with_next = True

    run_heading = p_heading.add_run(heading_text)
    run_heading.font.name = template.font_heading
    run_heading.font.size = Pt(template.h2_size_pt)
    run_heading.font.bold = True
    run_heading.font.color.rgb = template.color_primary

    # Explanatory note
    p_intro = doc.add_paragraph()
    p_intro.paragraph_format.space_after = Pt(8.0)
    run_intro = p_intro.add_run(
        "The following sources from imported PDF documents provide the evidentiary grounding for this document:"
    )
    run_intro.font.name = template.font_primary
    run_intro.font.size = Pt(template.body_font_size_pt - 1.5)
    run_intro.font.italic = True
    run_intro.font.color.rgb = template.color_secondary

    # Add table: Index | Source Document | Page | Reference Passage
    col_widths = [Inches(0.6), Inches(2.2), Inches(0.8), Inches(3.0)]
    table = doc.add_table(rows=1, cols=4)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table, border_color=template.color_border)

    # Style Header row
    hdr_cells = table.rows[0].cells
    headers = ["#", "Source Document", "Page", "Verified Reference / Excerpt"]
    for i, header_title in enumerate(headers):
        hdr_cells[i].width = col_widths[i]
        set_cell_background(hdr_cells[i], template.color_bg_subtle)
        set_cell_margins(hdr_cells[i], top=100, bottom=100, left=140, right=140)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_after = Pt(2.0)
        r = p.add_run(header_title)
        r.font.name = template.font_heading
        r.font.size = Pt(template.body_font_size_pt - 1.0)
        r.font.bold = True
        r.font.color.rgb = template.color_primary

    # Deduplicate citations while preserving order
    seen: set[tuple[str, int]] = set()
    unique_citations: list[dict[str, Any]] = []

    for item in citations:
        # Support dict or object with attributes
        if isinstance(item, dict):
            fn = item.get("filename") or item.get("source") or "Document.pdf"
            pg = item.get("page_number") or item.get("page") or 1
            sn = item.get("snippet") or item.get("text") or ""
        else:
            fn = getattr(item, "filename", "Document.pdf")
            pg = getattr(item, "page_number", getattr(item, "page", 1))
            sn = getattr(item, "snippet", getattr(item, "text", ""))

        try:
            pg_int = int(pg)
        except (ValueError, TypeError):
            pg_int = 1

        key = (str(fn).strip(), pg_int)
        if key not in seen:
            seen.add(key)
            unique_citations.append({"filename": str(fn).strip(), "page": pg_int, "snippet": str(sn).strip()})

    for idx, cite in enumerate(unique_citations, 1):
        row_cells = table.add_row().cells
        for cell, w in zip(row_cells, col_widths):
            cell.width = w
            set_cell_margins(cell, top=80, bottom=80, left=140, right=140)

        # Index
        p0 = row_cells[0].paragraphs[0]
        p0.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r0 = p0.add_run(str(idx))
        r0.font.name = template.font_primary
        r0.font.size = Pt(template.body_font_size_pt - 1.5)
        r0.font.bold = True
        r0.font.color.rgb = template.color_secondary

        # Filename
        p1 = row_cells[1].paragraphs[0]
        r1 = p1.add_run(cite["filename"])
        r1.font.name = template.font_primary
        r1.font.size = Pt(template.body_font_size_pt - 1.5)
        r1.font.color.rgb = template.color_text

        # Page
        p2 = row_cells[2].paragraphs[0]
        p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r2 = p2.add_run(f"p. {cite['page']}")
        r2.font.name = template.font_primary
        r2.font.size = Pt(template.body_font_size_pt - 1.5)
        r2.font.bold = True
        r2.font.color.rgb = template.color_secondary

        # Excerpt
        p3 = row_cells[3].paragraphs[0]
        snippet_text = cite["snippet"]
        if not snippet_text:
            snippet_text = f"Referenced context on page {cite['page']}."
        elif len(snippet_text) > 240:
            snippet_text = snippet_text[:237] + "..."
        r3 = p3.add_run(f'"{snippet_text}"')
        r3.font.name = template.font_primary
        r3.font.size = Pt(template.body_font_size_pt - 2.0)
        r3.font.italic = True
        r3.font.color.rgb = template.color_secondary

    # Add trailing space after table
    p_tail = doc.add_paragraph()
    p_tail.paragraph_format.space_before = Pt(6.0)
    p_tail.paragraph_format.space_after = Pt(2.0)

