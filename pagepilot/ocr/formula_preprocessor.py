"""
PagePilot – Stage 6B Mathematical Formula Preprocessor and Canonicalizer.

Provides deterministic, high-fidelity preprocessing for mathematical formula images
prior to Pix2Text MFR 1.5 inference:
1. Foreground bounding box detection to eliminate excessive whitespace.
2. Aspect-ratio preserving scale normalization for DeiT/TrOCR patch resolution.
3. Clean border padding to prevent edge character truncation.
4. Contrast optimization for scanned and photographed paper notes.
5. Non-destructive LaTeX post-processing clean-up (anti-hallucination compliant).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple, Union

try:
    from PIL import Image, ImageEnhance, ImageOps
except ImportError:
    Image = None  # type: ignore
    ImageEnhance = None  # type: ignore
    ImageOps = None  # type: ignore


def detect_content_bbox(
    img: Image.Image,
    threshold_margin: int = 40,
    pad: int = 4,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect the foreground content bounding box of an equation image.
    Supports both dark-on-light (standard) and inverted light-on-dark images.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    if w < 2 or h < 2:
        return (0, 0, w, h)

    # Sample corners to determine background luminance
    corners = [
        img.getpixel((0, 0)),
        img.getpixel((w - 1, 0)),
        img.getpixel((0, h - 1)),
        img.getpixel((w - 1, h - 1)),
    ]
    avg_bg = sum(sum(c[:3]) for c in corners) / (len(corners) * 3)

    gray = img.convert("L")
    if avg_bg > 128:
        # Dark text on light background: invert so text is foreground (bright)
        cutoff = max(30, int(avg_bg - threshold_margin))
        mask = gray.point(lambda p: 255 if p < cutoff else 0)
    else:
        # Light text on dark background
        cutoff = min(220, int(avg_bg + threshold_margin))
        mask = gray.point(lambda p: 255 if p > cutoff else 0)

    bbox = mask.getbbox()
    if not bbox:
        return None

    # Apply small safety padding
    x0 = max(0, bbox[0] - pad)
    y0 = max(0, bbox[1] - pad)
    x1 = min(w, bbox[2] + pad)
    y1 = min(h, bbox[3] + pad)
    return (x0, y0, x1, y1)


def preprocess_formula_image(
    image: Union[str, Path, Image.Image],
    min_height: int = 70,
    max_height: int = 240,
    border_padding: int = 30,
    contrast_factor: float = 1.35,
) -> Image.Image:
    """
    Prepare a mathematical formula crop for optimal TrOCR / DeiT patch recognition:
    - Crops excessive empty borders that compress formula height when resized.
    - Scales tiny formula strokes so character features span multiple patches.
    - Enhances contrast to clarify faint ink strokes or shadows.
    - Adds aspect-ratio preserving clean white margins around the expression.
    """
    if isinstance(image, (str, Path)):
        img = Image.open(str(image)).convert("RGB")
    elif isinstance(image, Image.Image):
        img = image.convert("RGB")
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    # 1. Detect tight foreground content bounds
    bbox = detect_content_bbox(img)
    if bbox:
        cropped = img.crop(bbox)
    else:
        cropped = img

    # 2. Aspect-ratio preserving scale normalization
    cw, ch = cropped.size
    if ch > 0:
        if ch < min_height:
            scale = min_height / float(ch)
            new_w = max(int(cw * scale), 1)
            new_h = min_height
            cropped = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
        elif ch > max_height:
            scale = max_height / float(ch)
            new_w = max(int(cw * scale), 1)
            new_h = max_height
            cropped = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 3. Contrast enhancement
    if contrast_factor != 1.0:
        enhancer = ImageEnhance.Contrast(cropped)
        cropped = enhancer.enhance(contrast_factor)

    # 4. Aspect-ratio preserving clean white border padding
    if border_padding > 0:
        padded = ImageOps.expand(cropped, border=border_padding, fill=(255, 255, 255))
    else:
        padded = cropped

    return padded


def clean_formula_latex(raw_latex: str) -> str:
    """
    Clean up formatting artifacts from formula OCR output without altering mathematical semantics.
    Anti-hallucination compliant: NEVER guesses or substitutes mathematical expressions.
    """
    if not raw_latex:
        return ""

    text = raw_latex.strip()

    # 1. Remove empty or degenerate \begin{array} ... \end{array} blocks
    text = re.sub(r"\\begin\{array\}\s*\{[lcr]*\}\s*(\{\s*\}\s*\\*)*\s*\\end\{array\}", "", text)

    # 2. Unwrap redundant outer \mathrm{...} or \mathbf{...} if it surrounds the whole expression
    outer_math = re.match(r"^\\(?:mathrm|mathbf|text|sf)\s*\{(.*)\}$", text, re.DOTALL)
    if outer_math:
        inner = outer_math.group(1).strip()
        # Ensure braces are balanced before unwrapping
        if inner.count("{") == inner.count("}"):
            text = inner

    # 3. Clean spurious \bf~ and \sf~ tokens produced by font-token decoders
    text = re.sub(r"\\(?:bf|sf)\s*~?\s*", "", text)

    # 4. Clean empty command brackets: e.g. \mathbf{} or \mathrm{}
    text = re.sub(r"\\(?:mathbf|mathrm|text|sf)\s*\{\s*\}", "", text)

    # 5. Normalize excessive whitespace inside curly braces
    text = re.sub(r"\{\s+", "{", text)
    text = re.sub(r"\s+\}", "}", text)

    # 6. Normalize multiple spaces
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()
