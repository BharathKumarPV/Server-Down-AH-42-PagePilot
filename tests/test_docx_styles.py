"""
Tests for PagePilot DOCX styles, palettes, and XML helpers (Stage 5).
"""

import pytest
import docx
from docx.shared import Inches, Pt, RGBColor

from pagepilot.docx.styles import (
    Palettes,
    clean_hex,
    hex_to_rgb,
    set_cell_background,
    set_cell_left_border,
    set_cell_margins,
    set_table_borders,
)


def test_hex_to_rgb():
    rgb = hex_to_rgb("#2563EB")
    assert rgb == RGBColor(0x25, 0x63, 0xEB)

    # Without hash
    rgb2 = hex_to_rgb("064E3B")
    assert rgb2 == RGBColor(0x06, 0x4E, 0x3B)

    # 3-char hex shorthand
    rgb3 = hex_to_rgb("#FFF")
    assert rgb3 == RGBColor(0xFF, 0xFF, 0xFF)

    # Invalid input fallback
    fallback = hex_to_rgb("invalid")
    assert isinstance(fallback, RGBColor)


def test_clean_hex():
    assert clean_hex("#1E293B") == "1E293B"
    assert clean_hex("  #abc  ") == "AABBCC"
    assert clean_hex("bad") == "BBAADD"  # Valid 3-char hex shorthand
    assert clean_hex("xyz") == "1E293B"  # Non-hex characters fallback
    assert clean_hex("too_long_color") == "1E293B"


def test_palettes_exist():
    assert Palettes.SLATE_PRIMARY == RGBColor(0x1E, 0x29, 0x3B)
    assert Palettes.NAVY_PRIMARY == RGBColor(0x0F, 0x17, 0x2A)
    assert Palettes.EMERALD_PRIMARY == RGBColor(0x06, 0x4E, 0x3B)
    assert Palettes.CORPORATE_PRIMARY == RGBColor(0x18, 0x18, 0x1B)
    assert Palettes.MINIMAL_PRIMARY == RGBColor(0x11, 0x18, 0x27)


def test_cell_and_table_xml_styling():
    doc = docx.Document()
    table = doc.add_table(rows=2, cols=2)
    cell = table.rows[0].cells[0]

    # Verify background fill
    set_cell_background(cell, "2563EB")
    tc_xml = cell._tc.xml
    assert 'w:fill="2563EB"' in tc_xml

    # Verify cell margins
    set_cell_margins(cell, top=120, bottom=120, left=180, right=180)
    tc_xml = cell._tc.xml
    assert 'w:top w:w="120"' in tc_xml
    assert 'w:left w:w="180"' in tc_xml

    # Verify left border callout
    set_cell_left_border(cell, hex_color="0D9488", sz=24)
    tc_xml = cell._tc.xml
    assert 'w:left w:val="single"' in tc_xml
    assert 'w:color="0D9488"' in tc_xml

    # Verify table borders
    set_table_borders(table, border_color="CBD5E1")
    tbl_xml = table._tbl.tblPr.xml
    assert 'w:tblBorders' in tbl_xml
    assert 'w:color="CBD5E1"' in tbl_xml
