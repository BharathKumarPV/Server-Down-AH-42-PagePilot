"""
Tests for Stage 6B OCR Data Models: BoundingBox, OCRRegion, PageOCRResult.
"""

from pagepilot.ocr.models import (
    ApprovalStatus,
    BoundingBox,
    OCRRegion,
    PageOCRResult,
    RegionType,
)


def test_bounding_box_geometry_and_iou():
    box1 = BoundingBox(10.0, 10.0, 110.0, 110.0)
    assert box1.width == 100.0
    assert box1.height == 100.0
    assert box1.area == 10000.0

    # Overlapping box
    box2 = BoundingBox(60.0, 60.0, 160.0, 160.0)
    assert box1.intersects(box2)
    assert box2.intersects(box1)

    # Disjoint box
    box3 = BoundingBox(200.0, 200.0, 300.0, 300.0)
    assert not box1.intersects(box3)

    # IoU calculation
    iou = box1.iou(box2)
    assert 0.14 < iou < 0.16  # Intersection = 50*50=2500, Union = 10000+10000-2500 = 17500 -> 2500/17500 ~ 0.1428

    # Containment
    inner = BoundingBox(20.0, 20.0, 50.0, 50.0)
    assert box1.contains(inner)
    assert not inner.contains(box1)


def test_ocr_region_effective_text_and_latex():
    region = OCRRegion(
        region_id="R1",
        document_id=1,
        original_filename="notes.png",
        source_type="image_note",
        page_number=1,
        bounding_box=BoundingBox(0.0, 0.0, 100.0, 50.0),
        region_type=RegionType.FORMULA,
        recognized_text="x = -b / 2a",
        corrected_text=None,
        latex_expression=r"x = \frac{-b}{2a}",
        confidence_score=0.92,
        approval_status=ApprovalStatus.PENDING_REVIEW,
    )

    assert region.effective_text == "x = -b / 2a"
    assert region.effective_latex == r"x = \frac{-b}{2a}"
    assert not region.is_approved

    # User correction
    region.corrected_text = "x = -b / (2a)"
    region.latex_expression = r"x = -\frac{b}{2a}"
    assert region.effective_text == "x = -b / (2a)"
    assert region.effective_latex == r"x = -\frac{b}{2a}"

    region.approval_status = ApprovalStatus.APPROVED
    assert region.is_approved


def test_page_ocr_result_reading_order_synthesis():
    # Two regions in scrambled order
    r_bottom = OCRRegion(
        region_id="R2",
        document_id=1,
        original_filename="calc.pdf",
        source_type="scanned_pdf",
        page_number=2,
        bounding_box=BoundingBox(40.0, 200.0, 500.0, 250.0),
        region_type=RegionType.TEXT,
        recognized_text="Therefore the integral converges.",
        approval_status=ApprovalStatus.APPROVED,
    )
    r_top = OCRRegion(
        region_id="R1",
        document_id=1,
        original_filename="calc.pdf",
        source_type="scanned_pdf",
        page_number=2,
        bounding_box=BoundingBox(40.0, 50.0, 450.0, 120.0),
        region_type=RegionType.FORMULA,
        recognized_text="int e^(-x^2) dx",
        latex_expression=r"\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}",
        approval_status=ApprovalStatus.APPROVED,
    )
    r_unapproved = OCRRegion(
        region_id="R3",
        document_id=1,
        original_filename="calc.pdf",
        source_type="scanned_pdf",
        page_number=2,
        bounding_box=BoundingBox(40.0, 300.0, 500.0, 350.0),
        region_type=RegionType.TEXT,
        recognized_text="Hallucinated unapproved line.",
        approval_status=ApprovalStatus.PENDING_REVIEW,
    )

    page_res = PageOCRResult(
        document_id=1,
        page_number=2,
        image_path="page_2.png",
        regions=[r_bottom, r_top, r_unapproved],
    )

    # Reading order must put top formula first, bottom text second, and completely exclude unapproved R3
    synth = page_res.approved_text()
    assert r"\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}" in synth
    assert "Therefore the integral converges." in synth
    assert "Hallucinated unapproved line." not in synth
    # Ensure formula appears before bottom text
    assert synth.index(r"\int") < synth.index("Therefore")

