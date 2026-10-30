"""
PagePilot – DOCX Document Service (Stage 5).

High-level service coordinating document types, PDF content extraction,
citations, and background worker threads for PySide6 UI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Union
from PySide6.QtCore import QObject, QThread, Signal

from pagepilot.config import DOCX_OUTPUT_DIR, ensure_docx_dir
from pagepilot.db.database import DocumentDB
from pagepilot.docx.citation_formatter import CitationMode, format_inline_citation
from pagepilot.docx.generator import DocxGenerator
from pagepilot.docx.templates import TEMPLATES, get_template
from pagepilot.docx.validators import sanitize_filename, validate_output_path
from pagepilot.pdf.document_manager import DocumentManager
from pagepilot.retrieval.vector_store import VectorStore


class DocxWorker(QThread):
    """Background worker for non-blocking DOCX file generation in PySide6."""
    progress = Signal(str)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, task_fn, *args, **kwargs):
        super().__init__()
        self.task_fn = task_fn
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            self.progress.emit("Assembling document structure...")
            result_path = self.task_fn(*self.args, **self.kwargs)
            self.finished.emit(str(result_path))
        except Exception as exc:
            self.error.emit(str(exc))


class DocxService:
    """Coordinates generation of all 5 document types with real citations and templates."""

    def __init__(
        self,
        db: Optional[DocumentDB] = None,
        generator: Optional[DocxGenerator] = None,
    ):
        self.db = db or DocumentDB()
        self.generator = generator or DocxGenerator()

    # ──────────────────────────────────────────────────────────────────────────
    # Document Type A: PDF Summary
    # ──────────────────────────────────────────────────────────────────────────

    def generate_pdf_summary(
        self,
        doc_id: int,
        template_name: str = "Academic Summary",
        citation_mode: str = CitationMode.BOTH,
        output_path: Optional[Union[str, Path]] = None,
        custom_title: Optional[str] = None,
    ) -> Path:
        """Generate a structured summary document from an imported PDF with verified citations."""
        doc = self.db.get_document(doc_id)
        if not doc:
            raise ValueError(f"Document ID {doc_id} not found in database.")

        chunks = self.db.get_chunks_for_document(doc_id)
        if not chunks:
            raise ValueError(f"Document '{doc.filename}' has no indexed chunks.")

        title = custom_title or f"Executive Summary: {Path(doc.filename).stem}"
        citations = []

        # Build structured content sections
        sections = [
            f"# {title}",
            "",
            "> **Document Scope**: This structured summary synthesizes the primary concepts, "
            f"findings, and topics from **{doc.filename}** ({doc.page_count} pages).",
            "",
            "## 1. Executive Overview",
            "",
        ]

        # Use early chunks for overview
        overview_chunks = chunks[:min(3, len(chunks))]
        overview_text_parts = []
        for ch in overview_chunks:
            citation_tag = ""
            if CitationMode.should_include_inline(citation_mode):
                citation_tag = f" {format_inline_citation(ch.filename, ch.page_number)}"
            overview_text_parts.append(f"{ch.text.strip()}{citation_tag}")
            citations.append({
                "filename": ch.filename,
                "page_number": ch.page_number,
                "snippet": ch.text[:180],
            })

        sections.append(" ".join(overview_text_parts))
        sections.append("")

        # Section 2: Key Concepts & Definitions
        sections.append("## 2. Key Concepts & Definitions")
        sections.append("")
        sections.append("| Concept / Term | Context & Source | Page Ref |")
        sections.append("| --- | --- | --- |")

        for idx, ch in enumerate(chunks[min(3, len(chunks)):min(7, len(chunks))], 1):
            term = ch.section_heading or f"Key Subject #{idx}"
            snippet = ch.text[:90].replace("|", "-").replace("\n", " ").strip()
            sections.append(f"| **{term}** | {snippet}... | Page {ch.page_number} |")
            citations.append({
                "filename": ch.filename,
                "page_number": ch.page_number,
                "snippet": ch.text[:180],
            })

        sections.append("")

        # Section 3: Detailed Findings
        sections.append("## 3. Findings & Core Principles")
        sections.append("")
        for ch in chunks[min(7, len(chunks)):min(12, len(chunks))]:
            heading = ch.section_heading or f"Analysis (Page {ch.page_number})"
            sections.append(f"### {heading}")
            sections.append("")
            citation_tag = ""
            if CitationMode.should_include_inline(citation_mode):
                citation_tag = f" {format_inline_citation(ch.filename, ch.page_number)}"
            sections.append(f"{ch.text.strip()}{citation_tag}")
            sections.append("")
            citations.append({
                "filename": ch.filename,
                "page_number": ch.page_number,
                "snippet": ch.text[:180],
            })

        markdown_body = "\n".join(sections)
        metadata = {"source_doc": doc.filename, "page_count": doc.page_count}

        return self.generator.generate(
            title=title,
            content=markdown_body,
            subtitle=f"Summary Report for {doc.filename}",
            template_name=template_name,
            citations=citations,
            citation_mode=citation_mode,
            output_path=output_path,
            metadata=metadata,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Document Type B: Study Notes
    # ──────────────────────────────────────────────────────────────────────────

    def generate_study_notes(
        self,
        doc_id: int,
        topic: Optional[str] = None,
        template_name: str = "Simple Study Notes",
        citation_mode: str = CitationMode.BOTH,
        output_path: Optional[Union[str, Path]] = None,
        custom_title: Optional[str] = None,
    ) -> Path:
        """Generate structured revision notes with headings, callouts, and bullet points."""
        doc = self.db.get_document(doc_id)
        if not doc:
            raise ValueError(f"Document ID {doc_id} not found in database.")

        chunks = self.db.get_chunks_for_document(doc_id)
        if not chunks:
            raise ValueError(f"Document '{doc.filename}' has no indexed chunks.")

        title = custom_title or f"Study Notes: {topic or Path(doc.filename).stem}"
        citations = []

        sections = [
            f"# {title}",
            "",
            f"> **Study Objective**: Master core concepts from `{doc.filename}`. "
            "Review key terms, explanations, and verifiable evidence from the source material.",
            "",
            "## 1. Quick Revision Points",
            "",
        ]

        # Bullet points from chunks
        for ch in chunks[:min(5, len(chunks))]:
            citation_tag = ""
            if CitationMode.should_include_inline(citation_mode):
                citation_tag = f" {format_inline_citation(ch.filename, ch.page_number)}"
            first_sentence = ch.text.split(".")[0].strip()
            if first_sentence:
                sections.append(f"- **Core Point**: {first_sentence}.{citation_tag}")
                citations.append({
                    "filename": ch.filename,
                    "page_number": ch.page_number,
                    "snippet": first_sentence,
                })

        sections.append("")
        sections.append("## 2. In-Depth Concepts & Explanations")
        sections.append("")

        for idx, ch in enumerate(chunks[min(5, len(chunks)):min(10, len(chunks))], 1):
            heading = ch.section_heading or f"Concept #{idx}: Context on Page {ch.page_number}"
            sections.append(f"### {heading}")
            sections.append("")
            citation_tag = ""
            if CitationMode.should_include_inline(citation_mode):
                citation_tag = f" {format_inline_citation(ch.filename, ch.page_number)}"
            sections.append(f"{ch.text.strip()}{citation_tag}")
            sections.append("")

            # Callout box for important definition
            sections.append(f"> **Important Takeaway**: High-yield exam/revision note from page {ch.page_number}.")
            sections.append("")
            citations.append({
                "filename": ch.filename,
                "page_number": ch.page_number,
                "snippet": ch.text[:180],
            })

        markdown_body = "\n".join(sections)
        metadata = {"source_doc": doc.filename, "topic": topic or "General"}

        return self.generator.generate(
            title=title,
            content=markdown_body,
            subtitle="Revision Notes & Key Study Concepts",
            template_name=template_name,
            citations=citations,
            citation_mode=citation_mode,
            output_path=output_path,
            metadata=metadata,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Document Type C: Question & Answer Notes
    # ──────────────────────────────────────────────────────────────────────────

    def generate_qa_document(
        self,
        qa_pairs: list[dict[str, Any]],
        doc_id: Optional[int] = None,
        template_name: str = "Question and Answer Notes",
        citation_mode: str = CitationMode.BOTH,
        output_path: Optional[Union[str, Path]] = None,
        custom_title: Optional[str] = None,
    ) -> Path:
        """Format selected questions and answers into a structured Q&A document."""
        doc_filename = "Verified Document"
        if doc_id:
            doc = self.db.get_document(doc_id)
            if doc:
                doc_filename = doc.filename

        title = custom_title or f"Q&A Reference: {Path(doc_filename).stem}"
        citations = []

        sections = [
            f"# {title}",
            "",
            "> **Q&A Compendium**: Verified answers grounded in source documentation.",
            "",
            "## Questions & Answers",
            "",
        ]

        for idx, qa in enumerate(qa_pairs, 1):
            q_text = qa.get("question", f"Question {idx}")
            a_text = qa.get("answer", "No answer recorded.")
            q_citations = qa.get("citations", [])

            sections.append(f"### Question {idx}: {q_text}")
            sections.append("")

            citation_tags = []
            if CitationMode.should_include_inline(citation_mode):
                for c in q_citations:
                    fn = c.get("filename", doc_filename) if isinstance(c, dict) else getattr(c, "filename", doc_filename)
                    pg = c.get("page_number", 1) if isinstance(c, dict) else getattr(c, "page_number", 1)
                    citation_tags.append(format_inline_citation(fn, pg))

            tag_str = f" {' '.join(citation_tags)}" if citation_tags else ""
            sections.append(f"**Answer**: {a_text.strip()}{tag_str}")
            sections.append("")

            # Record citations for sources table
            for c in q_citations:
                citations.append(c)

        markdown_body = "\n".join(sections)
        metadata = {"source_doc": doc_filename, "total_questions": len(qa_pairs)}

        return self.generator.generate(
            title=title,
            content=markdown_body,
            subtitle="Verified Questions and Answers",
            template_name=template_name,
            citations=citations,
            citation_mode=citation_mode,
            output_path=output_path,
            metadata=metadata,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Document Type D: Custom Document
    # ──────────────────────────────────────────────────────────────────────────

    def generate_custom_document(
        self,
        title: str,
        content: str,
        template_name: str = "Custom Document",
        subtitle: Optional[str] = None,
        citations: Optional[list[Any]] = None,
        citation_mode: str = CitationMode.NONE,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Create a custom document from user-provided Markdown text and options."""
        return self.generator.generate(
            title=title or "Custom Document",
            content=content or "No content provided.",
            subtitle=subtitle,
            template_name=template_name,
            citations=citations,
            citation_mode=citation_mode,
            output_path=output_path,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Document Type E: Selected PDF Content
    # ──────────────────────────────────────────────────────────────────────────

    def generate_selected_pages(
        self,
        doc_id: int,
        page_numbers: list[int],
        template_name: str = "Simple Study Notes",
        citation_mode: str = CitationMode.BOTH,
        output_path: Optional[Union[str, Path]] = None,
        custom_title: Optional[str] = None,
    ) -> Path:
        """Export extracted content from specific selected pages of a PDF document."""
        doc = self.db.get_document(doc_id)
        if not doc:
            raise ValueError(f"Document ID {doc_id} not found in database.")

        chunks = self.db.get_chunks_for_document(doc_id)
        target_pages = set(page_numbers)
        matching_chunks = [ch for ch in chunks if ch.page_number in target_pages]

        if not matching_chunks:
            raise ValueError(f"No indexed content found for pages {sorted(target_pages)} in '{doc.filename}'.")

        title = custom_title or f"Selected Excerpts: {Path(doc.filename).stem}"
        citations = []

        pages_str = ", ".join(str(p) for p in sorted(target_pages))
        sections = [
            f"# {title}",
            "",
            f"> **Selected Content**: Verified excerpts from `{doc.filename}` (Pages: {pages_str}).",
            "",
        ]

        # Group by page number
        current_page = None
        for ch in matching_chunks:
            if ch.page_number != current_page:
                current_page = ch.page_number
                sections.append(f"## Page {current_page}")
                sections.append("")

            citation_tag = ""
            if CitationMode.should_include_inline(citation_mode):
                citation_tag = f" {format_inline_citation(ch.filename, ch.page_number)}"

            heading_prefix = f"### {ch.section_heading}\n\n" if ch.section_heading else ""
            sections.append(f"{heading_prefix}{ch.text.strip()}{citation_tag}\n")

            citations.append({
                "filename": ch.filename,
                "page_number": ch.page_number,
                "snippet": ch.text[:180],
            })

        markdown_body = "\n".join(sections)
        metadata = {"source_doc": doc.filename, "selected_pages": pages_str}

        return self.generator.generate(
            title=title,
            content=markdown_body,
            subtitle=f"Extracted Sections from {doc.filename}",
            template_name=template_name,
            citations=citations,
            citation_mode=citation_mode,
            output_path=output_path,
            metadata=metadata,
        )

