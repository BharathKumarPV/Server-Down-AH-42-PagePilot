"""
Tests for PagePilot ClaimVerifier (Stage 6A).
"""

import pytest
from pagepilot.ai.claim_models import Claim, Evidence, VerificationStatus
from pagepilot.ai.claim_verifier import ClaimVerifier


@pytest.fixture
def evidence_pool():
    e1 = Evidence(
        evidence_id="E1",
        document_id=1,
        filename="Doc1.pdf",
        page_number=3,
        chunk_id="c1",
        chunk_index=0,
        original_text="Symmetric encryption uses a single shared secret key for encryption and decryption.",
    )
    e2 = Evidence(
        evidence_id="E2",
        document_id=2,
        filename="Doc2.pdf",
        page_number=5,
        chunk_id="c2",
        chunk_index=0,
        original_text="Maximum allowed worker threads is 8.",
    )
    return {"E1": e1, "E2": e2}


def test_verify_supported_claim(evidence_pool):
    verifier = ClaimVerifier()
    claim = Claim(
        claim_id="C1",
        text="Symmetric encryption uses a single shared secret key.",
        evidence_ids=["E1"],
    )
    verified = verifier.verify_claim(claim, evidence_pool)

    assert verified.verification_status == VerificationStatus.SUPPORTED
    assert verified.is_verified is True


def test_verify_invalid_citation_id(evidence_pool):
    verifier = ClaimVerifier()
    # E99 does not exist in pool
    claim = Claim(
        claim_id="C2",
        text="Some factual assertion.",
        evidence_ids=["E99"],
    )
    verified = verifier.verify_claim(claim, evidence_pool)

    assert verified.verification_status == VerificationStatus.INVALID_CITATION
    assert "E99" in verified.verification_notes


def test_verify_empty_evidence_ids(evidence_pool):
    verifier = ClaimVerifier()
    claim = Claim(
        claim_id="C3",
        text="An ungrounded assertion without citations.",
        evidence_ids=[],
    )
    verified = verifier.verify_claim(claim, evidence_pool)

    assert verified.verification_status == VerificationStatus.INSUFFICIENT_EVIDENCE


def test_verify_numerical_contradiction(evidence_pool):
    verifier = ClaimVerifier()
    # Evidence says 8, claim asserts 64
    claim = Claim(
        claim_id="C4",
        text="Maximum allowed worker threads is 64.",
        evidence_ids=["E2"],
    )
    verified = verifier.verify_claim(claim, evidence_pool)

    assert verified.verification_status == VerificationStatus.CONTRADICTED
    assert "64" in verified.verification_notes


def test_verify_insufficient_text_grounding(evidence_pool):
    verifier = ClaimVerifier()
    # Claim cites E1, but discusses completely unrelated quantum teleportation
    claim = Claim(
        claim_id="C5",
        text="Quantum entanglement facilitates instantaneous teleportation across optical relays.",
        evidence_ids=["E1"],
    )
    verified = verifier.verify_claim(claim, evidence_pool)

    assert verified.verification_status == VerificationStatus.INSUFFICIENT_EVIDENCE

