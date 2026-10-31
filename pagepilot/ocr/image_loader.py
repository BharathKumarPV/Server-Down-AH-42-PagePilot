"""
PagePilot – Stage 6B Image Loader and PDF Page Renderer.

Handles importing student note photographs (PNG, JPG, JPEG, WebP) and
rendering scanned PDF pages one-by-one via PyMuPDF without memory blowup.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Tuple

from pagepilot.config import OCR_CACHE_DIR, ensure_ocr_dirs
from pagepilot.ocr.models import BoundingBox


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


class ImageLoaderError(Exception):
    """Raised when an image or PDF page cannot be loaded or rendered."""


def is_image_file(file_path: str) -> bool:
    """Return True if the file path has a supported image extension."""
    return Path(file_path).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def is_pdf_file(file_path: str) -> bool:
    """Return True if the file is a PDF."""
    return Path(file_path).suffix.lower() == ".pdf"


def compute_sha256(file_path: str) -> str:
    """Compute SHA-256 hex digest of any file in 64 KB blocks."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_and_validate_image(file_path: str) -> dict:
    """
    Validate that a photographed note exists, is readable, and extract dimensions.
    Returns metadata dictionary with stable hash and dimensions.
    """
    path = Path(file_path)
    if not path.exists():
        raise ImageLoaderError(f"Image file does not exist: {file_path}")

    if not is_image_file(str(path)):
        raise ImageLoaderError(
            f"Unsupported image format: {path.suffix}. Supported: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}"
        )

    file_size = path.stat().st_size
    if file_size == 0:
        raise ImageLoaderError(f"Image file is empty (0 bytes): {file_path}")

    # Max image size limit: 50 MB to prevent OOM
    if file_size > 50 * 1024 * 1024:
        raise ImageLoaderError(f"Image file exceeds maximum 50 MB limit: {file_size / (1024*1024):.1f} MB")

    sha256 = compute_sha256(str(path))

    # Read dimensions safely via PyMuPDF or PySide6
    width, height = 0, 0
    try:
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz
        pix = fitz.Pixmap(str(path))
        width, height = pix.width, pix.height
        pix = None  # release immediately
    except Exception:
        # Fallback dimension check via PySide6 QImage
        try:
            from PySide6.QtGui import QImage
            img = QImage(str(path))
            if not img.isNull():
                width, height = img.width(), img.height()
        except Exception:
            pass

    return {
        "file_path": str(path.resolve()),
        "filename": path.name,
        "sha256": sha256,
        "width": width,
        "height": height,
        "file_size": file_size,
    }


def render_pdf_page_to_image(
    pdf_path: str,
    page_number: int,
    dpi: int = 150,
    out_dir: Optional[Path] = None,
) -> str:
    """
    Render a single PDF page into a PNG image using PyMuPDF.
    Processes exactly one page at a time and frees the pixmap buffer immediately.

    Parameters
    ----------
    pdf_path: Absolute path to the PDF file.
    page_number: 1-based physical page number.
    dpi: Rendering resolution (default 150 DPI balances OCR clarity and RAM).
    out_dir: Output directory for cached images. Defaults to OCR_CACHE_DIR.

    Returns
    -------
    Absolute path to the rendered PNG image.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise ImageLoaderError(f"PDF file not found: {pdf_path}")

    ensure_ocr_dirs()
    target_dir = out_dir or OCR_CACHE_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz
    except ImportError as exc:
        raise ImageLoaderError("PyMuPDF is required to render scanned PDF pages.") from exc

    try:
        doc = fitz.open(str(path))
        if page_number < 1 or page_number > len(doc):
            doc.close()
            raise ImageLoaderError(f"Page number {page_number} out of range (1..{len(doc)})")

        page = doc[page_number - 1]  # 0-based
        pix = page.get_pixmap(dpi=dpi)

        stem = path.stem
        out_file = target_dir / f"{stem}_page_{page_number}.png"
        pix.save(str(out_file))

        # Explicit cleanup for 8 GB RAM profile
        pix = None
        doc.close()

        return str(out_file.resolve())

    except Exception as exc:
        raise ImageLoaderError(f"Failed to render PDF page {page_number}: {exc}") from exc


def crop_region_image(
    image_path: str,
    box: BoundingBox,
    out_path: str,
) -> str:
    """
    Crop a specific bounding box region from an image and save it.
    Useful for formula previews and isolated equation recognition.
    """
    in_p = Path(image_path)
    if not in_p.exists():
        raise ImageLoaderError(f"Base image not found: {image_path}")

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    try:
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz
        pix = fitz.Pixmap(str(in_p))
        # Clamp coordinates to image dimensions
        x0 = max(0, min(int(box.x0), pix.width - 1))
        y0 = max(0, min(int(box.y0), pix.height - 1))
        x1 = max(x0 + 1, min(int(box.x1), pix.width))
        y1 = max(y0 + 1, min(int(box.y1), pix.height))

        rect = fitz.IRect(x0, y0, x1, y1)
        cropped = fitz.Pixmap(pix, rect)
        cropped.save(str(out_p))

        cropped = None
        pix = None
        return str(out_p.resolve())

    except Exception:
        # Fallback via PySide6
        try:
            from PySide6.QtGui import QImage
            img = QImage(str(in_p))
            x0 = max(0, int(box.x0))
            y0 = max(0, int(box.y0))
            w = max(1, int(box.width))
            h = max(1, int(box.height))
            cropped = img.copy(x0, y0, w, h)
            cropped.save(str(out_p))
            return str(out_p.resolve())
        except Exception as exc:
            raise ImageLoaderError(f"Failed to crop region image: {exc}") from exc
