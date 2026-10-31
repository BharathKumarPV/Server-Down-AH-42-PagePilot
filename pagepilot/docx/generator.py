"""
PagePilot – Professional DOCX Generator (Stage 5).

Assembles complete Microsoft Word (.docx) documents from structured content,
templates, metadata, and citation sources.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

from pagepilot.config import DOCX_OUTPUT_DIR, ensure_docx_dir
from pagepilot.docx.citation_formatter import (
    CitationMode,
    add_sources_section,
    format_inline_citation,
)
from pagepilot.docx.content_formatter import ContentFormatter
from pagepilot.docx.styles import Palettes
from pagepilot.docx.templates import TemplateConfig, get_template
from pagepilot.docx.validators import (
    sanitize_filename,
    validate_docx_file,
    validate_output_path,
)


def _add_page_number_field(paragraph):
    """Add a dynamic Word page number XML field to a footer paragraph."""
    run = paragraph.add_run("Page ")
    fld = parse_xml(f'<w:fldSimple {nsdecls("w")} w:instr="PAGE"/>')
    paragraph.add_run()._r.append(fld)
    run_of = paragraph.add_run(" of ")
    fld_numpages = parse_xml(f'<w:fldSimple {nsdecls("w")} w:instr="NUMPAGES"/>')
    paragraph.add_run()._r.append(fld_numpages)


class DocxGenerator:
    """
    Core engine for generating styled DOCX documents from structured Markdown and citations.
    """

    def __init__(self, template: Union[str, TemplateConfig] = "Simple Study Notes"):
        if isinstance(template, str):
            self.template = get_template(template)
        else:
            self.template = template

    def generate(
        self,
        title: str,
        content: str,
        subtitle: Optional[str] = None,
        template_name: Optional[str] = None,
        citations: Optional[list[Any]] = None,
        citation_mode: str = CitationMode.BOTH,
        output_path: Optional[Union[str, Path]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Path:
        """
        Generate, style, format, and save a .docx document.
        Returns the resolved Path of the created file.
        """
        # Resolve template
        if template_name:
            template = get_template(template_name)
        else:
            template = self.template

        # Determine target file path
        if output_path is None:
            ensure_docx_dir()
            safe_name = sanitize_filename(title or "Document")
            target_path = DOCX_OUTPUT_DIR / safe_name
        else:
            target_path = Path(output_path)

        target_path = validate_output_path(target_path, allow_overwrite=True)

        # Initialize python-docx Document
        doc = docx.Document()

        # Apply page margins
        section = doc.sections[0]
        section.top_margin = Inches(template.margin_top_inches)
        section.bottom_margin = Inches(template.margin_bottom_inches)
        section.left_margin = Inches(template.margin_left_inches)
        section.right_margin = Inches(template.margin_right_inches)

        # Apply header
        if template.header_text:
            header = section.header
            hp = header.paragraphs[0]
            hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            hrun = hp.add_run(template.header_text)
            hrun.font.name = template.font_primary
            hrun.font.size = Pt(8.5)
            hrun.font.italic = True
            hrun.font.color.rgb = template.color_secondary

        # Apply footer
        if template.show_page_numbers:
            footer = section.footer
            fp = footer.paragraphs[0]
            fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            _add_page_number_field(fp)
            for r in fp.runs:
                r.font.name = template.font_primary
                r.font.size = Pt(8.5)
                r.font.color.rgb = template.color_secondary

        # Build Title / Cover Section
        if template.has_cover_page:
            self._render_cover_page(doc, template, title, subtitle, metadata, citations)
        else:
            self._render_title_block(doc, template, title, subtitle, metadata)

        # Render Main Content
        formatter = ContentFormatter(doc, template)
        formatter.render_markdown(content)

        # Render Sources / References Section if requested
        if citations and CitationMode.should_include_sources_section(citation_mode):
            add_sources_section(doc, citations, template)

        # Save document
        doc.save(str(target_path))

        # Verification check
        validate_docx_file(target_path)

        return target_path

    def _render_title_block(
        self,
        doc,
        template: TemplateConfig,
        title: str,
        subtitle: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """Render an elegant top-of-page title and metadata block."""
        p_title = doc.add_paragraph()
        p_title.paragraph_format.space_before = Pt(4.0)
        p_title.paragraph_format.space_after = Pt(2.0)
        p_title.paragraph_format.keep_with_next = True

        r_title = p_title.add_run(title or "Document")
        r_title.font.name = template.font_heading
        r_title.font.size = Pt(template.h1_size_pt + 4.0)
        r_title.font.bold = True
        r_title.font.color.rgb = template.color_primary

        # Subtitle or metadata line
        meta_items = []
        if subtitle:
            meta_items.append(subtitle)
        else:
            meta_items.append(f"Template: {template.name}")

        now_str = datetime.now().strftime("%B %d, %Y")
        meta_items.append(now_str)

        if metadata:
            if "source_doc" in metadata:
                meta_items.append(f"Source: {metadata['source_doc']}")
            if "author" in metadata:
                meta_items.append(f"Author: {metadata['author']}")

        p_meta = doc.add_paragraph()
        p_meta.paragraph_format.space_before = Pt(0.0)
        p_meta.paragraph_format.space_after = Pt(12.0)
        r_meta = p_meta.add_run("  •  ".join(meta_items))
        r_meta.font.name = template.font_primary
        r_meta.font.size = Pt(template.body_font_size_pt - 1.5)
        r_meta.font.color.rgb = template.color_secondary

    def _render_cover_page(
        self,
        doc,
        template: TemplateConfig,
        title: str,
        subtitle: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        citations: Optional[list[Any]] = None,
    ):
        """Render a formal cover page for Project Reports, followed by a page break."""
        # Vertical space
        for _ in range(4):
            doc.add_paragraph()

        # Formal Title
        p_title = doc.add_paragraph()
        p_title.paragraph_format.space_before = Pt(24.0)
        p_title.paragraph_format.space_after = Pt(8.0)
        r_title = p_title.add_run(title or "Project Report")
        r_title.font.name = template.font_heading
        r_title.font.size = Pt(28.0)
        r_title.font.bold = True
        r_title.font.color.rgb = template.color_primary

        # Subtitle
        if subtitle:
            p_sub = doc.add_paragraph()
            p_sub.paragraph_format.space_after = Pt(16.0)
            r_sub = p_sub.add_run(subtitle)
            r_sub.font.name = template.font_heading
            r_sub.font.size = Pt(15.0)
            r_sub.font.color.rgb = template.color_accent

        # Divider
        p_div = doc.add_paragraph()
        p_div.paragraph_format.space_after = Pt(24.0)
        r_div = p_div.add_run("―" * 40)
        r_div.font.color.rgb = template.color_secondary

        # Metadata box
        p_meta = doc.add_paragraph()
        p_meta.paragraph_format.space_before = Pt(24.0)
        p_meta.paragraph_format.space_after = Pt(4.0)

        r_lbl = p_meta.add_run("Prepared with: ")
        r_lbl.font.name = template.font_primary
        r_lbl.font.size = Pt(10.5)
        r_lbl.font.bold = True
        r_lbl.font.color.rgb = template.color_primary

        r_val = p_meta.add_run("PagePilot Document Engine (Offline AI)\n")
        r_val.font.name = template.font_primary
        r_val.font.size = Pt(10.5)
        r_val.font.color.rgb = template.color_secondary

        now_str = datetime.now().strftime("%B %d, %Y")
        p_date = doc.add_paragraph()
        p_date.paragraph_format.space_after = Pt(4.0)
        r_date_lbl = p_date.add_run("Date: ")
        r_date_lbl.font.bold = True
        r_date_lbl.font.size = Pt(10.5)
        r_date_lbl.font.color.rgb = template.color_primary
        r_date_val = p_date.add_run(now_str)
        r_date_val.font.size = Pt(10.5)
        r_date_val.font.color.rgb = template.color_secondary

        if metadata and "source_doc" in metadata:
            p_src = doc.add_paragraph()
            p_src.paragraph_format.space_after = Pt(4.0)
            r_src_lbl = p_src.add_run("Source Document: ")
            r_src_lbl.font.bold = True
            r_src_lbl.font.size = Pt(10.5)
            r_src_lbl.font.color.rgb = template.color_primary
            r_src_val = p_src.add_run(str(metadata["source_doc"]))
            r_src_val.font.size = Pt(10.5)
            r_src_val.font.color.rgb = template.color_secondary

        # Page break after cover page
        doc.add_page_break()

