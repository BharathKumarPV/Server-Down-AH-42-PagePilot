"""
PagePilot – Professional DOCX Generation Engine (Stage 5).

Package public exports for document generators, templates, styles, citations,
validators, and services.
"""

from pagepilot.docx.citation_formatter import (
    CitationMode,
    add_sources_section,
    format_inline_citation,
)
from pagepilot.docx.content_formatter import (
    ContentFormatter,
    append_formatted_runs,
)
from pagepilot.docx.document_service import (
    DocxService,
    DocxWorker,
)
from pagepilot.docx.generator import (
    DocxGenerator,
)
from pagepilot.docx.styles import (
    Palettes,
    clean_hex,
    hex_to_rgb,
    set_cell_background,
    set_cell_left_border,
    set_cell_margins,
    set_table_borders,
)
from pagepilot.docx.templates import (
    ACADEMIC_SUMMARY,
    CUSTOM_DOCUMENT,
    PROJECT_REPORT,
    QA_NOTES,
    SIMPLE_STUDY_NOTES,
    TEMPLATE_NAMES,
    TEMPLATES,
    TemplateConfig,
    get_template,
)
from pagepilot.docx.validators import (
    sanitize_filename,
    validate_docx_file,
    validate_output_path,
)

__all__ = [
    "DocxGenerator",
    "DocxService",
    "DocxWorker",
    "TemplateConfig",
    "TEMPLATES",
    "TEMPLATE_NAMES",
    "get_template",
    "SIMPLE_STUDY_NOTES",
    "ACADEMIC_SUMMARY",
    "QA_NOTES",
    "PROJECT_REPORT",
    "CUSTOM_DOCUMENT",
    "CitationMode",
    "format_inline_citation",
    "add_sources_section",
    "ContentFormatter",
    "append_formatted_runs",
    "Palettes",
    "hex_to_rgb",
    "clean_hex",
    "set_cell_background",
    "set_cell_margins",
    "set_cell_left_border",
    "set_table_borders",
    "sanitize_filename",
    "validate_output_path",
    "validate_docx_file",
]

