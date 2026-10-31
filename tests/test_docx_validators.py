"""
Tests for PagePilot DOCX path validation, filename sanitization, and verification (Stage 5).
"""

from pathlib import Path
import pytest
import docx

from pagepilot.docx.validators import (
    sanitize_filename,
    validate_docx_file,
    validate_output_path,
)


def test_sanitize_filename_windows_illegal_chars():
    # Contains : / \ ? * < > |
    raw = 'Report: 2026/10/30 <Final>? *Draft* | Part "1"'
    clean = sanitize_filename(raw)
    assert clean.endswith(".docx")
    for bad_char in ':/?*<>"|\\':
        assert bad_char not in clean
    assert "Report" in clean


def test_sanitize_filename_reserved_dos_names():
    # Reserved DOS device names
    assert sanitize_filename("CON") == "Doc_CON.docx"
    assert sanitize_filename("PRN") == "Doc_PRN.docx"
    assert sanitize_filename("AUX") == "Doc_AUX.docx"
    assert sanitize_filename("NUL") == "Doc_NUL.docx"
    assert sanitize_filename("COM1") == "Doc_COM1.docx"


def test_sanitize_filename_empty_or_whitespace():
    assert sanitize_filename("") == "PagePilot_Document.docx"
    assert sanitize_filename("    ") == "PagePilot_Document.docx"
    assert sanitize_filename("   ...   ") == "PagePilot_Document.docx"


def test_validate_output_path(tmp_path):
    target = tmp_path / "exports" / "test_doc"
    resolved = validate_output_path(target)
    assert resolved.suffix == ".docx"
    assert resolved.parent.exists()

    # When allow_overwrite is False and file exists, increment suffix
    resolved.touch()
    resolved_no_overwrite = validate_output_path(target, allow_overwrite=False)
    assert resolved_no_overwrite.stem.endswith("_1")


def test_validate_docx_file(tmp_path):
    # 1. Valid file
    valid_file = tmp_path / "valid.docx"
    doc = docx.Document()
    doc.add_paragraph("Testing validation")
    doc.save(str(valid_file))
    assert validate_docx_file(valid_file) is True

    # 2. Non-existent file
    with pytest.raises(FileNotFoundError):
        validate_docx_file(tmp_path / "missing.docx")

    # 3. 0-byte file
    zero_file = tmp_path / "zero.docx"
    zero_file.touch()
    with pytest.raises(ValueError, match="0 bytes"):
        validate_docx_file(zero_file)

    # 4. Corrupt non-docx file
    corrupt_file = tmp_path / "corrupt.docx"
    corrupt_file.write_text("This is not a zip/docx file")
    with pytest.raises(ValueError, match="Corrupt or invalid DOCX"):
        validate_docx_file(corrupt_file)

