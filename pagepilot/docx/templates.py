"""
PagePilot – DOCX Templates Configuration (Stage 5).

Defines the 5 professional document templates:
1. Simple Study Notes
2. Academic Summary
3. Question and Answer Notes
4. Project Report
5. Custom Document
"""

from __future__ import annotations

from dataclasses import dataclass
from docx.shared import RGBColor
from pagepilot.docx.styles import Palettes, hex_to_rgb


@dataclass(frozen=True)
class TemplateConfig:
    name: str
    description: str
    font_primary: str
    font_heading: str
    font_code: str
    color_primary: RGBColor
    color_secondary: RGBColor
    color_accent: RGBColor
    color_text: RGBColor
    color_bg_subtle: str
    color_border: str
    margin_top_inches: float
    margin_bottom_inches: float
    margin_left_inches: float
    margin_right_inches: float
    body_font_size_pt: float
    h1_size_pt: float
    h2_size_pt: float
    h3_size_pt: float
    line_spacing: float
    space_after_p_pt: float
    has_cover_page: bool
    header_text: str
    show_page_numbers: bool


# ──────────────────────────────────────────────────────────────────────────────
# 1. Simple Study Notes Template
# ──────────────────────────────────────────────────────────────────────────────
SIMPLE_STUDY_NOTES = TemplateConfig(
    name="Simple Study Notes",
    description="Clean headings, bullet points, definition callout boxes, and compact margins for easy revision.",
    font_primary="Calibri",
    font_heading="Calibri",
    font_code="Consolas",
    color_primary=Palettes.SLATE_PRIMARY,
    color_secondary=Palettes.SLATE_SECONDARY,
    color_accent=Palettes.SLATE_ACCENT,
    color_text=Palettes.SLATE_PRIMARY,
    color_bg_subtle=Palettes.SLATE_SUBTLE_HEX,
    color_border=Palettes.SLATE_BORDER_HEX,
    margin_top_inches=0.8,
    margin_bottom_inches=0.8,
    margin_left_inches=0.8,
    margin_right_inches=0.8,
    body_font_size_pt=11.0,
    h1_size_pt=18.0,
    h2_size_pt=14.0,
    h3_size_pt=12.0,
    line_spacing=1.15,
    space_after_p_pt=4.0,
    has_cover_page=False,
    header_text="PagePilot • Study Notes",
    show_page_numbers=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Academic Summary Template
# ──────────────────────────────────────────────────────────────────────────────
ACADEMIC_SUMMARY = TemplateConfig(
    name="Academic Summary",
    description="Formal serif / classic styling, abstract callout, numbered sections, and structured reference tables.",
    font_primary="Georgia",
    font_heading="Georgia",
    font_code="Consolas",
    color_primary=Palettes.NAVY_PRIMARY,
    color_secondary=Palettes.NAVY_SECONDARY,
    color_accent=Palettes.NAVY_ACCENT,
    color_text=Palettes.NAVY_PRIMARY,
    color_bg_subtle=Palettes.NAVY_SUBTLE_HEX,
    color_border=Palettes.NAVY_BORDER_HEX,
    margin_top_inches=1.0,
    margin_bottom_inches=1.0,
    margin_left_inches=1.0,
    margin_right_inches=1.0,
    body_font_size_pt=11.5,
    h1_size_pt=20.0,
    h2_size_pt=15.0,
    h3_size_pt=12.5,
    line_spacing=1.25,
    space_after_p_pt=6.0,
    has_cover_page=False,
    header_text="PagePilot • Academic Summary",
    show_page_numbers=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# 3. Question and Answer Notes Template
# ──────────────────────────────────────────────────────────────────────────────
QA_NOTES = TemplateConfig(
    name="Question and Answer Notes",
    description="Distinct green-accented question styling, indented structured answers, and page citations.",
    font_primary="Arial",
    font_heading="Arial",
    font_code="Consolas",
    color_primary=Palettes.EMERALD_PRIMARY,
    color_secondary=Palettes.EMERALD_SECONDARY,
    color_accent=Palettes.EMERALD_ACCENT,
    color_text=Palettes.EMERALD_SECONDARY,
    color_bg_subtle=Palettes.EMERALD_SUBTLE_HEX,
    color_border=Palettes.EMERALD_BORDER_HEX,
    margin_top_inches=0.9,
    margin_bottom_inches=0.9,
    margin_left_inches=0.9,
    margin_right_inches=0.9,
    body_font_size_pt=10.5,
    h1_size_pt=18.0,
    h2_size_pt=13.5,
    h3_size_pt=11.5,
    line_spacing=1.15,
    space_after_p_pt=5.0,
    has_cover_page=False,
    header_text="PagePilot • Q&A Notes",
    show_page_numbers=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# 4. Project Report Template
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_REPORT = TemplateConfig(
    name="Project Report",
    description="Executive presentation layout with a formal title block, executive summary, tables, and structured appendix.",
    font_primary="Calibri",
    font_heading="Calibri",
    font_code="Consolas",
    color_primary=Palettes.CORPORATE_PRIMARY,
    color_secondary=Palettes.CORPORATE_SECONDARY,
    color_accent=Palettes.CORPORATE_ACCENT,
    color_text=Palettes.CORPORATE_PRIMARY,
    color_bg_subtle=Palettes.CORPORATE_SUBTLE_HEX,
    color_border=Palettes.CORPORATE_BORDER_HEX,
    margin_top_inches=1.0,
    margin_bottom_inches=1.0,
    margin_left_inches=1.0,
    margin_right_inches=1.0,
    body_font_size_pt=11.0,
    h1_size_pt=22.0,
    h2_size_pt=16.0,
    h3_size_pt=13.0,
    line_spacing=1.2,
    space_after_p_pt=6.0,
    has_cover_page=True,
    header_text="PagePilot • Project Report",
    show_page_numbers=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# 5. Custom Document Template
# ──────────────────────────────────────────────────────────────────────────────
CUSTOM_DOCUMENT = TemplateConfig(
    name="Custom Document",
    description="Versatile modern layout suitable for custom prompts, freeform text, and user-defined documents.",
    font_primary="Segoe UI",
    font_heading="Segoe UI",
    font_code="Consolas",
    color_primary=Palettes.MINIMAL_PRIMARY,
    color_secondary=Palettes.MINIMAL_SECONDARY,
    color_accent=Palettes.MINIMAL_ACCENT,
    color_text=Palettes.MINIMAL_PRIMARY,
    color_bg_subtle=Palettes.MINIMAL_SUBTLE_HEX,
    color_border=Palettes.MINIMAL_BORDER_HEX,
    margin_top_inches=1.0,
    margin_bottom_inches=1.0,
    margin_left_inches=1.0,
    margin_right_inches=1.0,
    body_font_size_pt=11.0,
    h1_size_pt=19.0,
    h2_size_pt=14.0,
    h3_size_pt=12.0,
    line_spacing=1.15,
    space_after_p_pt=5.0,
    has_cover_page=False,
    header_text="PagePilot • Document Engine",
    show_page_numbers=True,
)

TEMPLATES: dict[str, TemplateConfig] = {
    SIMPLE_STUDY_NOTES.name: SIMPLE_STUDY_NOTES,
    ACADEMIC_SUMMARY.name: ACADEMIC_SUMMARY,
    QA_NOTES.name: QA_NOTES,
    PROJECT_REPORT.name: PROJECT_REPORT,
    CUSTOM_DOCUMENT.name: CUSTOM_DOCUMENT,
}

TEMPLATE_NAMES: list[str] = list(TEMPLATES.keys())


def get_template(name: str | None) -> TemplateConfig:
    """Retrieve template by name, defaulting to Simple Study Notes if not found."""
    if not name:
        return SIMPLE_STUDY_NOTES
    return TEMPLATES.get(name.strip(), SIMPLE_STUDY_NOTES)

