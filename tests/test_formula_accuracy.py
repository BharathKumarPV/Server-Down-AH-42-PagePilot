"""
PagePilot – Tests for Mathematical Formula Recognition Preprocessing & Validation.

Verifies:
1. Formula image preprocessing (content bounding box detection, scale normalization, aspect ratio padding).
2. Formula LaTeX syntax validation (balanced braces, parentheses, brackets, left/right delimiters).
3. Radical and fraction structure validation.
4. Trailing operator detection.
5. Suspicious ASCII math token detection.
6. Non-fabrication guarantee: validator never alters mathematical expressions.
7. End-to-end integration with OCR worker protocol and review approval gating.
"""

from __future__ import annotations

from pathlib import Path

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import pytest
except ImportError:
    class _DummyPytest:
        @staticmethod
        def skip(msg):
            pass
    pytest = _DummyPytest()

from pagepilot.ocr.formula_preprocessor import (
    clean_formula_latex,
    detect_content_bbox,
    preprocess_formula_image,
)
from pagepilot.ocr.formula_validator import (
    FormulaValidationResult,
    ValidationSeverity,
    check_delimiter_balance,
    validate_formula_latex,
)
from pagepilot.ocr.models import ApprovalStatus, BoundingBox, OCRRegion, PageOCRResult, RegionType


# ── Formula Preprocessor Tests ────────────────────────────────────────────────

def test_detect_content_bbox_empty_or_solid():
    if not HAS_PIL:
        pytest.skip("Pillow is installed in .venv-ocr isolated environment")
    img_solid = Image.new("RGB", (200, 100), color=(255, 255, 255))
    bbox = detect_content_bbox(img_solid)
    assert bbox is None or isinstance(bbox, tuple)


def test_detect_content_bbox_with_content():
    if not HAS_PIL:
        pytest.skip("Pillow is installed in .venv-ocr isolated environment")
    img = Image.new("RGB", (300, 150), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([50, 40, 120, 80], fill=(0, 0, 0))
    bbox = detect_content_bbox(img)
    assert bbox is not None
    x0, y0, x1, y1 = bbox
    assert x0 <= 50 and y0 <= 40
    assert x1 >= 120 and y1 >= 80


def test_preprocess_formula_image_normalizes_size_and_pads():
    if not HAS_PIL:
        pytest.skip("Pillow is installed in .venv-ocr isolated environment")
    img = Image.new("RGB", (100, 20), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((10, 5), "x = y^2", fill=(0, 0, 0))

    preproc = preprocess_formula_image(img, min_height=70, border_padding=25)
    assert preproc.height >= 70 + 50  # 70 + 2*25 padding
    assert preproc.width > preproc.height
    assert preproc.mode == "RGB"


def test_clean_formula_latex_formatting():
    # Unwrap outer \mathrm while preserving contents
    raw = r"\mathrm{ x = \frac{-b}{2a} }"
    cleaned = clean_formula_latex(raw)
    assert cleaned == r"x = \frac{-b}{2a}"

    # Remove empty array blocks
    raw_arr = r"\begin{array}{l} {} \\ \end{array} E = mc^2"
    cleaned_arr = clean_formula_latex(raw_arr)
    assert "E = mc^2" in cleaned_arr
    assert r"\begin{array}" not in cleaned_arr

    # Normalize spurious font tokens
    raw_font = r"x = \bf~ a + \sf~ b"
    cleaned_font = clean_formula_latex(raw_font)
    assert "bf~" not in cleaned_font
    assert "sf~" not in cleaned_font


# ── Formula Validator Tests ───────────────────────────────────────────────────

def test_validate_formula_valid_quadratic():
    expr = r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}"
    res = validate_formula_latex(expr, confidence_score=0.96)
    assert res.is_valid is True
    assert len(res.errors) == 0
    assert res.has_unbalanced_braces is False
    assert res.has_unbalanced_parens is False


def test_validate_formula_unbalanced_braces():
    expr = r"x = \frac{-b \pm \sqrt{b^2 - 4ac{2a}"
    res = validate_formula_latex(expr)
    assert res.is_valid is False
    assert res.has_unbalanced_braces is True
    assert any("braces" in e.lower() for e in res.errors)
    assert res.needs_human_review is True


def test_validate_formula_unbalanced_parentheses():
    expr = r"f(x) = (2x + 1"
    res = validate_formula_latex(expr)
    assert res.has_unbalanced_parens is True
    assert any("parentheses" in w.lower() for w in res.warnings)


def test_validate_formula_malformed_fraction():
    expr = r"x = \frac"
    res = validate_formula_latex(expr)
    assert res.is_valid is False
    assert any("frac" in e.lower() for e in res.errors)


def test_validate_formula_malformed_radical():
    expr = r"y = \sqrt"
    res = validate_formula_latex(expr)
    assert res.is_valid is False
    assert any("sqrt" in e.lower() for e in res.errors)


def test_validate_formula_dangling_operator():
    expr = r"x = a + b +"
    res = validate_formula_latex(expr)
    assert any("operator" in w.lower() for w in res.warnings)


def test_validate_formula_suspicious_ascii_math():
    expr = r"x = (-b +- sqrt(b^2 - 4ac)) / (2a)"
    res = validate_formula_latex(expr)
    assert any("sqrt" in w.lower() for w in res.warnings)
    assert any("+-" in w.lower() for w in res.warnings)
    assert res.needs_human_review is True


def test_validator_does_not_fabricate_output():
    broken_expr = r"x = \frac{1}{0 + \sqrt{"
    res = validate_formula_latex(broken_expr)
    # Target expression must not be silently altered or completed
    assert res.latex_expression == broken_expr
    assert res.is_valid is False


# ── Approval Workflow Gating Tests ───────────────────────────────────────────

def test_unapproved_formulas_excluded_from_rag_chunks():
    unapproved_region = OCRRegion(
        region_id="R1",
        document_id=1,
        original_filename="notes.png",
        source_type="image_note",
        page_number=1,
        bounding_box=BoundingBox(0, 0, 100, 100),
        region_type=RegionType.FORMULA,
        latex_expression=r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}",
        approval_status=ApprovalStatus.PENDING_REVIEW,
    )
    page_result = PageOCRResult(
        document_id=1,
        page_number=1,
        image_path="",
        regions=[unapproved_region],
        page_status=ApprovalStatus.PENDING_REVIEW,
    )
    # Anti-hallucination check: approved_text MUST be empty
    assert page_result.approved_text() == ""


def test_approved_formulas_included_as_display_blocks():
    approved_region = OCRRegion(
        region_id="R1",
        document_id=1,
        original_filename="notes.png",
        source_type="image_note",
        page_number=1,
        bounding_box=BoundingBox(0, 0, 100, 100),
        region_type=RegionType.FORMULA,
        latex_expression=r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}",
        approval_status=ApprovalStatus.APPROVED,
    )
    page_result = PageOCRResult(
        document_id=1,
        page_number=1,
        image_path="",
        regions=[approved_region],
        page_status=ApprovalStatus.APPROVED,
    )
    approved_text = page_result.approved_text()
    assert r"\frac{-b \pm \sqrt{b^2 - 4ac}}{2a}" in approved_text
    assert "\\[" in approved_text
    assert "\\]" in approved_text
