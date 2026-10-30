"""
Tests for PagePilot Sentence-Level Citation Rendering (Stage 6A).
"""

import pytest
from pagepilot.ai.claim_models import (
    Claim,
    Evidence,
    VerificationStatus,
    VerifiedAnswer,
)


def test_sentence_level_citations_multi_source():
    ev_a = Evidence(
        evidence_id="E1",
        document_id=1,
        filename="Cryptography.pdf",
        page_number=24,
        chunk_id="c1",
        chunk_index=0,
        original_text="RSA is an asymmetric cryptographic algorithm.",
    )
    ev_b = Evidence(
        evidence_id="E2",
        document_id=2,
        filename="Networks.pdf",
        page_number=51,
        chunk_id="c2",
        chunk_index=0,
        original_text="Public-key cryptography uses a public and private key pair.",
    )

    claims = [
        Claim(
            claim_id="C1",
            text="RSA is an asymmetric cryptographic algorithm",
            evidence_ids=["E1"],
            verification_status=VerificationStatus.SUPPORTED,
        ),
        Claim(
            claim_id="C2",
            text="Public-key cryptography uses a public and private key pair",
            evidence_ids=["E2"],
            verification_status=VerificationStatus.SUPPORTED,
        ),
    ]

    answer = VerifiedAnswer(
        question="Describe RSA and public keys.",
        claims=claims,
        evidence_map={"E1": ev_a, "E2": ev_b},
        contributing_documents=["Cryptography.pdf", "Networks.pdf"],
    )

    rendered = answer.render_sentence_level_text()

    assert "RSA is an asymmetric cryptographic algorithm. [Cryptography.pdf, p. 24]" in rendered
    assert "Public-key cryptography uses a public and private key pair. [Networks.pdf, p. 51]" in rendered


def test_multiple_citations_on_single_claim():
    ev_a = Evidence(
        evidence_id="E1",
        document_id=1,
        filename="DocA.pdf",
        page_number=3,
        chunk_id="c1",
        chunk_index=0,
        original_text="Part 1 of claim.",
    )
    ev_b = Evidence(
        evidence_id="E2",
        document_id=2,
        filename="DocB.pdf",
        page_number=7,
        chunk_id="c2",
        chunk_index=0,
        original_text="Part 2 of claim.",
    )

    claim = Claim(
        claim_id="C1",
        text="A unified conclusion synthesizing both sources",
        evidence_ids=["E1", "E2"],
        verification_status=VerificationStatus.SUPPORTED,
    )

    answer = VerifiedAnswer(
        question="Test question",
        claims=[claim],
        evidence_map={"E1": ev_a, "E2": ev_b},
        contributing_documents=["DocA.pdf", "DocB.pdf"],
    )

    rendered = answer.render_sentence_level_text()
    assert "[DocA.pdf, p. 3]" in rendered
    assert "[DocB.pdf, p. 7]" in rendered

