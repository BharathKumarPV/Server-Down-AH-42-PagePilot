"""
PagePilot – DOCX Styles & Typography (Stage 5).

Provides color palettes, measurement conversions, and OpenXML styling helpers
for python-docx tables, cells, callouts, and borders.
"""

from __future__ import annotations

import re
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
from docx.shared import Inches, Pt, RGBColor


def clean_hex(hex_code: str) -> str:
    """Return uppercase 6-character hex string without leading hash."""
    clean = hex_code.strip().lstrip("#").upper()
    if len(clean) == 3 and all(c in "0123456789ABCDEF" for c in clean):
        clean = "".join(c * 2 for c in clean)
    if len(clean) != 6 or not all(c in "0123456789ABCDEF" for c in clean):
        clean = "1E293B"
    return clean


def hex_to_rgb(hex_code: str) -> RGBColor:
    """Convert hex color string (e.g. '#2563EB' or '2563EB') to RGBColor."""
    clean = clean_hex(hex_code)
    r = int(clean[0:2], 16)
    g = int(clean[2:4], 16)
    b = int(clean[4:6], 16)
    return RGBColor(r, g, b)


# ──────────────────────────────────────────────────────────────────────────────
# Palette Constants
# ──────────────────────────────────────────────────────────────────────────────

class Palettes:
    # 1. Slate / Study Notes (Modern, accessible, highly readable)
    SLATE_PRIMARY = hex_to_rgb("1E293B")      # Dark slate
    SLATE_SECONDARY = hex_to_rgb("475569")    # Slate body
    SLATE_ACCENT = hex_to_rgb("2563EB")       # Royal blue
    SLATE_SUBTLE_HEX = "F1F5F9"               # Very light slate background
    SLATE_BORDER_HEX = "CBD5E1"               # Subtle border

    # 2. Academic / Formal (Deep navy, classical and authoritative)
    NAVY_PRIMARY = hex_to_rgb("0F172A")       # Very dark navy
    NAVY_SECONDARY = hex_to_rgb("334155")     # Muted navy text
    NAVY_ACCENT = hex_to_rgb("1D4ED8")        # Deep academic blue
    NAVY_SUBTLE_HEX = "EFF6FF"                # Ice blue tint
    NAVY_BORDER_HEX = "BFDBFE"

    # 3. Emerald / Q&A (Vibrant teal/emerald for distinction)
    EMERALD_PRIMARY = hex_to_rgb("064E3B")    # Dark pine
    EMERALD_SECONDARY = hex_to_rgb("1F2937")  # Deep charcoal
    EMERALD_ACCENT = hex_to_rgb("059669")     # Emerald accent
    EMERALD_SUBTLE_HEX = "ECFDF5"             # Soft mint tint
    EMERALD_BORDER_HEX = "A7F3D0"

    # 4. Corporate / Project Report (Deep charcoal & burgundy/bronze)
    CORPORATE_PRIMARY = hex_to_rgb("18181B")  # Zinc dark
    CORPORATE_SECONDARY = hex_to_rgb("3F3F46")# Neutral grey
    CORPORATE_ACCENT = hex_to_rgb("991B1B")   # Crimson / Burgundy accent
    CORPORATE_SUBTLE_HEX = "F4F4F5"           # Light zinc tint
    CORPORATE_BORDER_HEX = "D4D4D8"

    # 5. Clean / Minimalist (Jet black & clean electric blue)
    MINIMAL_PRIMARY = hex_to_rgb("111827")
    MINIMAL_SECONDARY = hex_to_rgb("374151")
    MINIMAL_ACCENT = hex_to_rgb("3B82F6")
    MINIMAL_SUBTLE_HEX = "F9FAFB"
    MINIMAL_BORDER_HEX = "E5E7EB"


# ──────────────────────────────────────────────────────────────────────────────
# OpenXML Table & Cell Styling Helpers
# ──────────────────────────────────────────────────────────────────────────────

def set_cell_background(cell, hex_code: str):
    """Set the background fill color of a table cell."""
    color = clean_hex(hex_code)
    shading_xml = f'<w:shd {nsdecls("w")} w:fill="{color}"/>'
    cell._tc.get_or_add_tcPr().append(parse_xml(shading_xml))


def set_cell_margins(cell, top: int = 120, bottom: int = 120, left: int = 180, right: int = 180):
    """Set inner cell padding in twips (1 pt = 20 twips; 6 pt = 120 twips, 9 pt = 180 twips)."""
    tc_mar_xml = (
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    cell._tc.get_or_add_tcPr().append(parse_xml(tc_mar_xml))


def set_cell_left_border(cell, hex_color: str = "2563EB", sz: int = 24):
    """Set a thick left border on a cell (e.g. for callout / quote boxes) with other borders cleared."""
    color = clean_hex(hex_color)
    borders_xml = (
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:top w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:bottom w:val="none"/>'
        f'</w:tcBorders>'
    )
    cell._tc.get_or_add_tcPr().append(parse_xml(borders_xml))


def set_table_borders(
    table,
    border_color: str = "CBD5E1",
    inside_h_color: str = "E2E8F0",
    sz: int = 4,
):
    """Set subtle clean borders for a data table."""
    color = clean_hex(border_color)
    in_color = clean_hex(inside_h_color)
    tbl_borders_xml = (
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:bottom w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:insideH w:val="single" w:sz="{sz}" w:space="0" w:color="{in_color}"/>'
        f'<w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    table._tbl.tblPr.append(parse_xml(tbl_borders_xml))
