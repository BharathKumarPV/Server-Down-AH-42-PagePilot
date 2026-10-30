"""
PagePilot – DOCX Validators & Sanitization (Stage 5).

Validates output file paths, sanitizes document filenames for Windows 11 filesystem
rules, and verifies that generated .docx files are structurally valid OpenXML packages.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Union
import docx


# Windows reserved characters: < > : " / \ | ? * and ASCII 0-31
INVALID_WIN_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
}


def sanitize_filename(name: str, default: str = "PagePilot_Document") -> str:
    """
    Sanitize a filename for safe creation on Windows 11 filesystems.
    Removes illegal characters, reserved words, trailing dots/spaces, and enforces length limits.
    """
    if not name or not name.strip():
        name = default

    # Remove illegal chars
    cleaned = INVALID_WIN_CHARS.sub("_", name.strip())

    # Collapse consecutive underscores and whitespace
    cleaned = re.sub(r"[\s_]+", "_", cleaned)

    # Windows filenames cannot end in a period or space
    cleaned = cleaned.strip(". ")

    # Check for reserved DOS device names
    base_stem = Path(cleaned).stem.upper()
    if base_stem in RESERVED_NAMES:
        cleaned = f"Doc_{cleaned}"

    # Limit stem length to 100 characters to prevent path overflows
    if len(cleaned) > 100:
        cleaned = cleaned[:100].rstrip("._ ")

    if not cleaned:
        cleaned = default

    if not cleaned.lower().endswith(".docx"):
        cleaned = f"{cleaned}.docx"

    return cleaned


def validate_output_path(
    path: Union[str, Path],
    allow_overwrite: bool = True,
) -> Path:
    """
    Validate that an output path is writable and has a valid directory.
    If allow_overwrite is False and the file exists, appends an incrementing counter.
    """
    resolved = Path(path).resolve()

    # Ensure parent directory exists
    parent = resolved.parent
    parent.mkdir(parents=True, exist_ok=True)

    # Ensure extension is .docx
    if resolved.suffix.lower() != ".docx":
        resolved = resolved.with_suffix(".docx")

    if not allow_overwrite and resolved.exists():
        counter = 1
        stem = resolved.stem
        while resolved.exists():
            resolved = parent / f"{stem}_{counter}.docx"
            counter += 1

    return resolved


def validate_docx_file(file_path: Union[str, Path]) -> bool:
    """
    Verify that the generated .docx file is a valid, readable OpenXML package.
    Returns True if successfully parsed, or raises an error / returns False.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"DOCX file does not exist: {path}")

    if path.stat().st_size == 0:
        raise ValueError(f"DOCX file is 0 bytes: {path}")

    try:
        doc = docx.Document(str(path))
        # Ensure paragraphs and sections can be iterated
        _ = len(doc.paragraphs)
        _ = len(doc.sections)
        return True
    except Exception as exc:
        raise ValueError(f"Corrupt or invalid DOCX document at {path}: {exc}") from exc

