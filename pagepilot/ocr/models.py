"""
PagePilot – Stage 6B OCR and Formula Recognition Data Models.

Defines the core representations for:
- BoundingBox: Geometric bounds with intersection and containment helpers
- RegionType: Classifies region as printed text, handwriting, formula, heading, etc.
- ApprovalStatus: Explicit human-in-the-loop review workflow states
- OCRRegion: Individual recognized or corrected region with complete provenance
- PageOCRResult: Aggregated page results with reading-order synthesis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class RegionType(str, Enum):
    """Classifies the functional nature of an extracted region."""
    TEXT = "text"
    HANDWRITING = "handwriting"
    FORMULA = "formula"
    HEADING = "heading"
    TABLE = "table"
    DIAGRAM = "diagram"
    UNKNOWN = "unknown"


class ApprovalStatus(str, Enum):
    """Explicit human-in-the-loop verification status."""
    PENDING_REVIEW = "pending_review"
    NEEDS_CORRECTION = "needs_correction"
    APPROVED = "approved"
    REJECTED = "rejected"
    PROCESSING_FAILED = "processing_failed"


@dataclass(frozen=True)
class BoundingBox:
    """
    Normalized or pixel rectangular bounding coordinates.
    (x0, y0) is top-left, (x1, y1) is bottom-right.
    """
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    def to_dict(self) -> Dict[str, float]:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> BoundingBox:
        return cls(
            x0=float(d.get("x0", 0.0)),
            y0=float(d.get("y0", 0.0)),
            x1=float(d.get("x1", 0.0)),
            y1=float(d.get("y1", 0.0)),
        )

    def intersects(self, other: BoundingBox) -> bool:
        """Return True if this bounding box overlaps with another."""
        return not (
            self.x1 <= other.x0 or
            self.x0 >= other.x1 or
            self.y1 <= other.y0 or
            self.y0 >= other.y1
        )

    def contains(self, other: BoundingBox) -> bool:
        """Return True if this box completely encloses other."""
        return (
            self.x0 <= other.x0 and
            self.y0 <= other.y0 and
            self.x1 >= other.x1 and
            self.y1 >= other.y1
        )

    def iou(self, other: BoundingBox) -> float:
        """Calculate Intersection over Union (IoU) with another box."""
        inter_x0 = max(self.x0, other.x0)
        inter_y0 = max(self.y0, other.y0)
        inter_x1 = min(self.x1, other.x1)
        inter_y1 = min(self.y1, other.y1)

        inter_w = max(0.0, inter_x1 - inter_x0)
        inter_h = max(0.0, inter_y1 - inter_y0)
        inter_area = inter_w * inter_h

        if inter_area <= 0.0:
            return 0.0

        union_area = self.area + other.area - inter_area
        return inter_area / union_area if union_area > 0.0 else 0.0


@dataclass
class OCRRegion:
    """
    Represents an atomic extracted region on a scanned PDF or photograph.
    Retains immutable provenance back to the source file, 1-based page, and bounding box.
    """
    region_id: str                      # e.g., "R1", "R2"
    document_id: int
    original_filename: str
    source_type: str                    # "scanned_pdf" | "image_note"
    page_number: int                    # 1-based physical page
    bounding_box: BoundingBox
    region_type: RegionType = RegionType.TEXT
    recognized_text: str = ""
    corrected_text: Optional[str] = None
    latex_expression: Optional[str] = None
    confidence_score: float = 0.0
    recognition_engine: str = "paddleocr_ppocrv5"
    model_version: str = "5.0.0"
    approval_status: ApprovalStatus = ApprovalStatus.PENDING_REVIEW
    original_image_path: str = ""
    image_crop_path: Optional[str] = None
    original_file_hash: str = ""

    @property
    def effective_text(self) -> str:
        """Return user-corrected text if present, otherwise raw recognized text."""
        if self.corrected_text is not None and self.corrected_text.strip():
            return self.corrected_text.strip()
        return self.recognized_text.strip()

    @property
    def effective_latex(self) -> Optional[str]:
        """Return LaTeX expression for formula regions."""
        if self.region_type == RegionType.FORMULA and self.latex_expression:
            return self.latex_expression.strip()
        return None

    @property
    def is_approved(self) -> bool:
        return self.approval_status == ApprovalStatus.APPROVED

    @property
    def is_rejected(self) -> bool:
        return self.approval_status == ApprovalStatus.REJECTED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "document_id": self.document_id,
            "original_filename": self.original_filename,
            "source_type": self.source_type,
            "page_number": self.page_number,
            "box_x0": self.bounding_box.x0,
            "box_y0": self.bounding_box.y0,
            "box_x1": self.bounding_box.x1,
            "box_y1": self.bounding_box.y1,
            "region_type": self.region_type.value,
            "recognized_text": self.recognized_text,
            "corrected_text": self.corrected_text,
            "latex_expression": self.latex_expression,
            "confidence_score": self.confidence_score,
            "recognition_engine": self.recognition_engine,
            "model_version": self.model_version,
            "approval_status": self.approval_status.value,
            "original_image_path": self.original_image_path,
            "image_crop_path": self.image_crop_path or "",
            "original_file_hash": self.original_file_hash,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> OCRRegion:
        bbox = BoundingBox(
            x0=float(d.get("box_x0", 0.0)),
            y0=float(d.get("box_y0", 0.0)),
            x1=float(d.get("box_x1", 0.0)),
            y1=float(d.get("box_y1", 0.0)),
        )
        return cls(
            region_id=str(d.get("region_id", "")),
            document_id=int(d.get("document_id", 0)),
            original_filename=str(d.get("original_filename", "")),
            source_type=str(d.get("source_type", "image_note")),
            page_number=int(d.get("page_number", 1)),
            bounding_box=bbox,
            region_type=RegionType(d.get("region_type", "text")),
            recognized_text=str(d.get("recognized_text", "")),
            corrected_text=d.get("corrected_text"),
            latex_expression=d.get("latex_expression"),
            confidence_score=float(d.get("confidence_score", 0.0)),
            recognition_engine=str(d.get("recognition_engine", "paddleocr_ppocrv5")),
            model_version=str(d.get("model_version", "5.0.0")),
            approval_status=ApprovalStatus(d.get("approval_status", "pending_review")),
            original_image_path=str(d.get("original_image_path", "")),
            image_crop_path=d.get("image_crop_path") or None,
            original_file_hash=str(d.get("original_file_hash", "")),
        )


@dataclass
class PageOCRResult:
    """
    Complete OCR outcome for a single page.
    Synthesizes approved text in reading order for trusted RAG indexing.
    """
    document_id: int
    page_number: int                    # 1-based
    image_path: str
    regions: List[OCRRegion] = field(default_factory=list)
    page_status: ApprovalStatus = ApprovalStatus.PENDING_REVIEW
    error_message: str = ""

    @property
    def is_fully_approved(self) -> bool:
        return bool(self.regions) and all(r.is_approved for r in self.regions)

    @property
    def has_formulas(self) -> bool:
        return any(r.region_type == RegionType.FORMULA for r in self.regions)

    @property
    def has_handwriting(self) -> bool:
        return any(r.region_type == RegionType.HANDWRITING for r in self.regions)

    @property
    def approved_regions(self) -> List[OCRRegion]:
        return [r for r in self.regions if r.is_approved]

    def approved_text(self) -> str:
        """
        Concatenate all approved regions in top-to-bottom, left-to-right reading order.
        Formulas are represented as formatted display LaTeX blocks.
        """
        approved = self.approved_regions
        if not approved:
            return ""

        # Sort roughly top-to-bottom with 20px vertical quantization band, then left-to-right
        sorted_regions = sorted(
            approved,
            key=lambda r: (round(r.bounding_box.y0 / 20.0), r.bounding_box.x0),
        )

        parts = []
        for r in sorted_regions:
            if r.region_type == RegionType.FORMULA and r.effective_latex:
                latex_block = f"\\[\n{r.effective_latex}\n\\]"
                parts.append(latex_block)
            else:
                txt = r.effective_text
                if txt:
                    parts.append(txt)

        return "\n\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "page_number": self.page_number,
            "image_path": self.image_path,
            "page_status": self.page_status.value,
            "error_message": self.error_message,
            "regions": [r.to_dict() for r in self.regions],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PageOCRResult:
        regions = [OCRRegion.from_dict(r) for r in d.get("regions", [])]
        return cls(
            document_id=int(d.get("document_id", 0)),
            page_number=int(d.get("page_number", 1)),
            image_path=str(d.get("image_path", "")),
            page_status=ApprovalStatus(d.get("page_status", "pending_review")),
            error_message=str(d.get("error_message", "")),
            regions=regions,
        )

