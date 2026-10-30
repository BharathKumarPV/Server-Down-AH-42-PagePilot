"""
PagePilot – DOCX Content Formatter (Stage 5).

Converts markdown-like structured text (headings, bold/italic, lists, tables,
callouts, code blocks, dividers) into native python-docx paragraphs, tables, and runs.
"""

from __future__ import annotations

import re
from typing import List, Tuple
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

from pagepilot.docx.templates import TemplateConfig
from pagepilot.docx.styles import (
    set_cell_background,
    set_cell_left_border,
    set_cell_margins,
    set_table_borders,
)


# Token regex for inline styles: bold-italic, bold, italic, inline code
_INLINE_TOKEN_RE = re.compile(
    r"(\*\*\*[^\*]+?\*\*\*|___[^_]+?___|"
    r"\*\*[^\*]+?\*\*|__[^_]+?__|"
    r"(?<!\*)\*[^\*]+?\*(?!\*)|(?<!_)_[^_]+?_(?!_)|"
    r"`[^`]+?`)"
)


def append_formatted_runs(
    paragraph,
    text: str,
    font_name: str,
    font_size_pt: float,
    color: RGBColor,
    code_font: str = "Consolas",
):
    """Parse inline markdown tokens (**bold**, *italic*, `code`, ***both***) and add runs."""
    if not text:
        return

    tokens = _INLINE_TOKEN_RE.split(text)
    for token in tokens:
        if not token:
            continue

        bold = False
        italic = False
        is_code = False
        run_text = token

        if (token.startswith("***") and token.endswith("***")) or (token.startswith("___") and token.endswith("___")):
            bold = True
            italic = True
            run_text = token[3:-3]
        elif (token.startswith("**") and token.endswith("**")) or (token.startswith("__") and token.endswith("__")):
            bold = True
            run_text = token[2:-2]
        elif (token.startswith("*") and token.endswith("*")) or (token.startswith("_") and token.endswith("_")):
            italic = True
            run_text = token[1:-1]
        elif token.startswith("`") and token.endswith("`"):
            is_code = True
            run_text = token[1:-1]

        run = paragraph.add_run(run_text)
        run.font.name = code_font if is_code else font_name
        run.font.size = Pt(font_size_pt - 1.0 if is_code else font_size_pt)
        run.bold = bold
        run.italic = italic
        run.font.color.rgb = color


class ContentFormatter:
    """Renders structured Markdown content into a python-docx Document."""

    def __init__(self, doc, template: TemplateConfig):
        self.doc = doc
        self.template = template

    def render_markdown(self, markdown_text: str):
        """Parse lines and blocks from markdown text and emit docx elements."""
        if not markdown_text:
            return

        lines = markdown_text.splitlines()
        i = 0
        total_lines = len(lines)

        while i < total_lines:
            line = lines[i].rstrip()
            stripped = line.strip()

            # Empty line
            if not stripped:
                i += 1
                continue

            # Code block ```
            if stripped.startswith("```"):
                code_lines = []
                i += 1
                while i < total_lines and not lines[i].strip().startswith("```"):
                    code_lines.append(lines[i])
                    i += 1
                if i < total_lines and lines[i].strip().startswith("```"):
                    i += 1
                self._render_code_block("\n".join(code_lines))
                continue

            # Markdown Table | col 1 | col 2 |
            if stripped.startswith("|") and stripped.endswith("|"):
                table_lines = []
                while i < total_lines and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                    table_lines.append(lines[i].strip())
                    i += 1
                self._render_table(table_lines)
                continue

            # Blockquote / Callout > text
            if stripped.startswith(">"):
                quote_lines = []
                while i < total_lines and lines[i].strip().startswith(">"):
                    quote_lines.append(re.sub(r"^>\s*", "", lines[i].strip()))
                    i += 1
                self._render_callout("\n".join(quote_lines))
                continue

            # Horizontal Rule --- or ***
            if stripped in ("---", "***", "___", "- - -", "* * *"):
                self._render_horizontal_rule()
                i += 1
                continue

            # Headings
            if stripped.startswith("#"):
                m = re.match(r"^(#{1,6})\s*(.*)$", stripped)
                if m:
                    level = len(m.group(1))
                    heading_text = m.group(2).strip()
                    self._render_heading(heading_text, level)
                    i += 1
                    continue

            # Bullet List - or *
            if re.match(r"^[-*]\s+", stripped):
                item_text = re.sub(r"^[-*]\s+", "", stripped)
                self._render_bullet_item(item_text)
                i += 1
                continue

            # Numbered List 1. or 2.
            num_match = re.match(r"^\d+\.\s+(.*)$", stripped)
            if num_match:
                item_text = num_match.group(1)
                self._render_numbered_item(item_text)
                i += 1
                continue

            # Standard Paragraph
            self._render_paragraph(stripped)
            i += 1

    def _render_heading(self, text: str, level: int):
        p = self.doc.add_paragraph()
        p.paragraph_format.keep_with_next = True

        if level == 1:
            p.paragraph_format.space_before = Pt(18.0)
            p.paragraph_format.space_after = Pt(6.0)
            font_size = self.template.h1_size_pt
            color = self.template.color_primary
        elif level == 2:
            p.paragraph_format.space_before = Pt(14.0)
            p.paragraph_format.space_after = Pt(4.0)
            font_size = self.template.h2_size_pt
            color = self.template.color_primary
        elif level == 3:
            p.paragraph_format.space_before = Pt(10.0)
            p.paragraph_format.space_after = Pt(3.0)
            font_size = self.template.h3_size_pt
            color = self.template.color_accent
        else:
            p.paragraph_format.space_before = Pt(8.0)
            p.paragraph_format.space_after = Pt(2.0)
            font_size = self.template.body_font_size_pt
            color = self.template.color_secondary

        append_formatted_runs(
            p,
            text,
            font_name=self.template.font_heading,
            font_size_pt=font_size,
            color=color,
            code_font=self.template.font_code,
        )
        # Bold all headings
        for run in p.runs:
            run.bold = True

    def _render_paragraph(self, text: str):
        p = self.doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0.0)
        p.paragraph_format.space_after = Pt(self.template.space_after_p_pt)
        p.paragraph_format.line_spacing = self.template.line_spacing
        append_formatted_runs(
            p,
            text,
            font_name=self.template.font_primary,
            font_size_pt=self.template.body_font_size_pt,
            color=self.template.color_text,
            code_font=self.template.font_code,
        )

    def _render_bullet_item(self, text: str):
        p = self.doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_before = Pt(0.0)
        p.paragraph_format.space_after = Pt(2.5)
        p.paragraph_format.line_spacing = self.template.line_spacing
        append_formatted_runs(
            p,
            text,
            font_name=self.template.font_primary,
            font_size_pt=self.template.body_font_size_pt,
            color=self.template.color_text,
            code_font=self.template.font_code,
        )

    def _render_numbered_item(self, text: str):
        p = self.doc.add_paragraph(style="List Number")
        p.paragraph_format.space_before = Pt(0.0)
        p.paragraph_format.space_after = Pt(2.5)
        p.paragraph_format.line_spacing = self.template.line_spacing
        append_formatted_runs(
            p,
            text,
            font_name=self.template.font_primary,
            font_size_pt=self.template.body_font_size_pt,
            color=self.template.color_text,
            code_font=self.template.font_code,
        )

    def _render_callout(self, text: str):
        """Render a blockquote / callout box inside a single-cell table with left accent border."""
        table = self.doc.add_table(rows=1, cols=1)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = table.rows[0].cells[0]
        set_cell_background(cell, self.template.color_bg_subtle)
        set_cell_left_border(cell, hex_color=self.template.color_border, sz=24)
        set_cell_margins(cell, top=100, bottom=100, left=180, right=180)

        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(2.0)
        p.paragraph_format.space_after = Pt(2.0)
        p.paragraph_format.line_spacing = self.template.line_spacing
        append_formatted_runs(
            p,
            text,
            font_name=self.template.font_primary,
            font_size_pt=self.template.body_font_size_pt,
            color=self.template.color_secondary,
            code_font=self.template.font_code,
        )
        for r in p.runs:
            r.italic = True

        # Spacer paragraph after table
        sp = self.doc.add_paragraph()
        sp.paragraph_format.space_before = Pt(2.0)
        sp.paragraph_format.space_after = Pt(4.0)

    def _render_code_block(self, code_text: str):
        """Render code block inside a shaded box with monospace font."""
        table = self.doc.add_table(rows=1, cols=1)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = table.rows[0].cells[0]
        set_cell_background(cell, self.template.color_bg_subtle)
        set_cell_margins(cell, top=100, bottom=100, left=140, right=140)

        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(2.0)
        p.paragraph_format.space_after = Pt(2.0)
        run = p.add_run(code_text)
        run.font.name = self.template.font_code
        run.font.size = Pt(self.template.body_font_size_pt - 1.5)
        run.font.color.rgb = self.template.color_primary

        sp = self.doc.add_paragraph()
        sp.paragraph_format.space_before = Pt(2.0)
        sp.paragraph_format.space_after = Pt(4.0)

    def _render_table(self, table_lines: list[str]):
        """Parse markdown table rows and render into a styled docx table."""
        parsed_rows: list[list[str]] = []
        for line in table_lines:
            # Strip outer pipes and split by pipe
            cells = [c.strip() for c in line.strip("|").split("|")]
            # Skip separator line like | --- | --- |
            if all(re.match(r"^:?-+:?$", c) for c in cells):
                continue
            parsed_rows.append(cells)

        if not parsed_rows:
            return

        col_count = max(len(r) for r in parsed_rows)
        # Pad shorter rows
        for r in parsed_rows:
            while len(r) < col_count:
                r.append("")

        table = self.doc.add_table(rows=len(parsed_rows), cols=col_count)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        set_table_borders(table, border_color=self.template.color_border)

        for row_idx, row_data in enumerate(parsed_rows):
            is_header = (row_idx == 0)
            row_cells = table.rows[row_idx].cells
            for col_idx, cell_value in enumerate(row_data):
                cell = row_cells[col_idx]
                set_cell_margins(cell, top=90, bottom=90, left=130, right=130)
                if is_header:
                    set_cell_background(cell, self.template.color_bg_subtle)

                p = cell.paragraphs[0]
                p.paragraph_format.space_before = Pt(1.0)
                p.paragraph_format.space_after = Pt(1.0)

                append_formatted_runs(
                    p,
                    cell_value,
                    font_name=self.template.font_heading if is_header else self.template.font_primary,
                    font_size_pt=self.template.body_font_size_pt - 1.0,
                    color=self.template.color_primary if is_header else self.template.color_text,
                    code_font=self.template.font_code,
                )
                if is_header:
                    for r in p.runs:
                        r.bold = True

        sp = self.doc.add_paragraph()
        sp.paragraph_format.space_before = Pt(2.0)
        sp.paragraph_format.space_after = Pt(4.0)

    def _render_horizontal_rule(self):
        """Render a subtle horizontal divider line."""
        p = self.doc.add_paragraph()
        p.paragraph_format.space_before = Pt(8.0)
        p.paragraph_format.space_after = Pt(8.0)
        run = p.add_run("―" * 48)
        run.font.name = self.template.font_primary
        run.font.size = Pt(8.0)
        run.font.color.rgb = self.template.color_secondary

