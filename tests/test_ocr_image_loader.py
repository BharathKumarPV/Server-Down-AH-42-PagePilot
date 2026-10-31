"""
Tests for Stage 6B Image Loader and PDF Page Renderer.
"""

from pathlib import Path
import pytest

from pagepilot.ocr.image_loader import (
    ImageLoaderError,
    compute_sha256,
    is_image_file,
    is_pdf_file,
    load_and_validate_image,
)


def test_image_and_pdf_extension_detection():
    assert is_image_file("note.png") is True
    assert is_image_file("photo.JPG") is True
    assert is_image_file("scan.jpeg") is True
    assert is_image_file("diagram.webp") is True
    assert is_image_file("doc.pdf") is False
    assert is_image_file("malicious.exe") is False

    assert is_pdf_file("doc.pdf") is True
    assert is_pdf_file("doc.PDF") is True
    assert is_pdf_file("note.png") is False


def test_load_and_validate_image(tmp_path):
    img_file = tmp_path / "valid_note.png"
    # Create minimal 1x1 PNG file bytes
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05"
        b"\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    img_file.write_bytes(png_bytes)

    meta = load_and_validate_image(str(img_file))
    assert meta["filename"] == "valid_note.png"
    assert len(meta["sha256"]) == 64
    assert meta["file_size"] > 0

    # Test missing file error
    with pytest.raises(ImageLoaderError, match="does not exist"):
        load_and_validate_image(str(tmp_path / "non_existent.png"))

    # Test invalid extension error
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("Hello")
    with pytest.raises(ImageLoaderError, match="Unsupported image format"):
        load_and_validate_image(str(txt_file))

