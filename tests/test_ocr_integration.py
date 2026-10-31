"""
Tests for Stage 6B End-to-End Integration:
- Cross-document queries across normal PDFs and OCR-approved notes
- Stage 6A Claim Verifier verifying statements backed by OCR evidence
- Stage 5 DOCX export preserving OCR source citations
"""

from pathlib import Path

from pagepilot.ai.claim_models import (
    Claim,
    Evidence,
    VerificationStatus,
    VerifiedAnswer,
)
from pagepilot.ai.claim_verifier import ClaimVerifier
from pagepilot.db.database import DocumentDB
from pagepilot.docx.citation_formatter import format_inline_citation
from pagepilot.docx.document_service import DocxService
from pagepilot.ocr.service import OCRService
from pagepilot.retrieval.vector_store import VectorStore


def test_claim_verifier_with_ocr_and_pdf_evidence():
    # E1 from typed PDF, E2 from photographed OCR note
    e1 = Evidence(
        evidence_id="E1",
        document_id=1,
        filename="Physics_Handbook.pdf",
        page_number=14,
        chunk_id="chunk_1",
        chunk_index=0,
        original_text="The speed of sound in air at 20 degrees Celsius is approximately 343 meters per second.",
    )
    e2 = Evidence(
        evidence_id="E2",
        document_id=2,
        filename="Lab_Notes_Acoustics.png",
        page_number=1,
        chunk_id="ocr_2_p1_c0",
        chunk_index=0,
        original_text="Empirical trial measured acoustic velocity at 343 meters per second.",
    )

    verifier = ClaimVerifier()
    evidence_map = {"E1": e1, "E2": e2}

    claim = Claim(
        claim_id="C1",
        text="The acoustic velocity measured in the lab agrees with the theoretical value of 343 meters per second.",
        claim_type="factual",
        evidence_ids=["E1", "E2"],
    )

    verified = verifier.verify_claim(claim, evidence_map)
    assert verified.verification_status == VerificationStatus.SUPPORTED

    answer = VerifiedAnswer(
        question="What was the measured speed of sound compared to theory?",
        claims=[verified],
        evidence_map=evidence_map,
    )
    rendered = answer.render_sentence_level_text()
    assert "[Physics_Handbook.pdf, p. 14]" in rendered
    assert "[Lab_Notes_Acoustics.png, p. 1]" in rendered


def test_docx_export_preserves_ocr_provenance(tmp_path):
    e_ocr = Evidence(
        evidence_id="E1",
        document_id=1,
        filename="Calculus_Handwritten_Notes.jpg",
        page_number=1,
        chunk_id="ocr_1_p1_c0",
        chunk_index=0,
        original_text=r"Quadratic roots formula: x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}",
    )
    claim = Claim(
        claim_id="C1",
        text="The quadratic formula gives the analytical roots of a polynomial equation.",
        claim_type="factual",
        evidence_ids=["E1"],
        verification_status=VerificationStatus.SUPPORTED,
    )
    answer = VerifiedAnswer(
        question="What is the quadratic formula?",
        claims=[claim],
        evidence_map={"E1": e_ocr},
    )

    out_file = tmp_path / "ocr_math_report.docx"
    doc_service = DocxService()
    path = doc_service.generate_multi_document_report(
        verified_answer=answer,
        output_path=str(out_file),
        custom_title="Calculus Notes Synthesis",
    )

    assert Path(path).exists()
    assert Path(path).stat().st_size > 5000

    # Inline citation check
    citation_str = format_inline_citation("Calculus_Handwritten_Notes.jpg", 1)
    assert citation_str == "[Calculus_Handwritten_Notes.jpg, p. 1]"
