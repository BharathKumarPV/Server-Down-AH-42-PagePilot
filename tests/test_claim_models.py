"""
Tests for PagePilot Evidence and Claim Data Models (Stage 6A).
"""

import pytest
from pagepilot.ai.claim_models import (
    Claim,
    Contradiction,
    Evidence,
    InformationRequirement,
    VerificationStatus,
    VerifiedAnswer,
)


def test_evidence_model_immutability_and_tags():
    ev = Evidence(
        evidence_id="E1",
        document_id=42,
        filename="Security_Architecture.pdf",
        page_number=3,
        chunk_id="chk_abc",
        chunk_index=0,
        original_text="TLS 1.3 is mandated for all internal communication channels across clusters.",
        retrieval_method="hybrid",
        retrieval_rank=1,
    )

    assert ev.evidence_id == "E1"
    assert ev.citation_tag == "[Security_Architecture.pdf, p. 3]"
    assert "TLS 1.3 is mandated" in ev.snippet(50)
    assert ev.page_number == 3

    # Immutable dataclass check
    with pytest.raises(Exception):
        ev.page_number = 4  # type: ignore


def test_claim_model_properties():
    c1 = Claim(
        claim_id="C1",
        text="Symmetric encryption uses the same key.",
        evidence_ids=["E1"],
        verification_status=VerificationStatus.SUPPORTED,
    )
    assert c1.is_verified is True
    assert c1.is_contradicted is False

    c2 = Claim(
        claim_id="C2",
        text="Timeout is 500s.",
        evidence_ids=["E2"],
        verification_status=VerificationStatus.CONTRADICTED,
    )
    assert c2.is_verified is False
    assert c2.is_contradicted is True


def test_verified_answer_render_sentence_level():
    ev1 = Evidence(
        evidence_id="E1",
        document_id=1,
        filename="DocA.pdf",
        page_number=2,
        chunk_id="c1",
        chunk_index=0,
        original_text="AES-256 is used for database encryption.",
    )
    ev2 = Evidence(
        evidence_id="E2",
        document_id=2,
        filename="DocB.pdf",
        page_number=5,
        chunk_id="c2",
        chunk_index=0,
        original_text="RSA-4096 is used for identity tokens.",
    )

    claims = [
        Claim(
            claim_id="C1",
            text="AES-256 encrypts databases",
            evidence_ids=["E1"],
            verification_status=VerificationStatus.SUPPORTED,
        ),
        Claim(
            claim_id="C2",
            text="RSA-4096 is used for authentication",
            evidence_ids=["E2"],
            verification_status=VerificationStatus.SUPPORTED,
        ),
        Claim(
            claim_id="C3",
            text="Unsupported rumor about servers",
            evidence_ids=[],
            verification_status=VerificationStatus.INSUFFICIENT_EVIDENCE,
        ),
    ]

    answer = VerifiedAnswer(
        question="What encryption is used?",
        claims=claims,
        evidence_map={"E1": ev1, "E2": ev2},
        contributing_documents=["DocA.pdf", "DocB.pdf"],
    )

    rendered = answer.render_sentence_level_text()

    # Supported claims must be present with their individual citations
    assert "AES-256 encrypts databases. [DocA.pdf, p. 2]" in rendered
    assert "RSA-4096 is used for authentication. [DocB.pdf, p. 5]" in rendered

    # Unsupported claim must NOT appear in verified narrative text
    assert "Unsupported rumor" not in rendered

