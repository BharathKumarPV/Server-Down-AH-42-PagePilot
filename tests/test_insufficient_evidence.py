"""
Tests for PagePilot Insufficient Evidence & Abstention Behavior (Stage 6A).
"""

import pytest
from pagepilot.ai.claim_models import (
    Claim,
    Evidence,
    VerificationStatus,
    VerifiedAnswer,
)


def test_partial_evidence_explicit_reporting():
    ev1 = Evidence(
        evidence_id="E1",
        document_id=7,
        filename="Service_Level.pdf",
        page_number=1,
        chunk_id="c1",
        chunk_index=0,
        original_text="Maximum client timeout is 30 seconds. Retry limit is 3 attempts.",
    )

    claims = [
        Claim(
            claim_id="C1",
            text="Maximum client timeout is 30 seconds",
            evidence_ids=["E1"],
            verification_status=VerificationStatus.SUPPORTED,
        ),
        Claim(
            claim_id="C2",
            text="Operation retry limit is 3 attempts",
            evidence_ids=["E1"],
            verification_status=VerificationStatus.SUPPORTED,
        ),
    ]

    answer = VerifiedAnswer(
        question="What are the maximum timeout, retry limit, and backup server IP?",
        claims=claims,
        evidence_map={"E1": ev1},
        unanswered_requirements=["backup server IP"],
        contributing_documents=["Service_Level.pdf"],
    )

    rendered = answer.render_sentence_level_text()

    # Supported facts present with citations
    assert "Maximum client timeout is 30 seconds. [Service_Level.pdf, p. 1]" in rendered
    assert "Operation retry limit is 3 attempts. [Service_Level.pdf, p. 1]" in rendered

    # Missing evidence explicitly noted
    assert "Missing Evidence for Sub-Questions:" in rendered
    assert "backup server IP" in rendered


def test_complete_absence_of_evidence():
    answer = VerifiedAnswer(
        question="What is the CEO's personal email?",
        claims=[],
        evidence_map={},
        unanswered_requirements=["CEO's personal email"],
        contributing_documents=[],
    )

    rendered = answer.render_sentence_level_text()
    assert "could not find sufficient evidence" in rendered.lower()

