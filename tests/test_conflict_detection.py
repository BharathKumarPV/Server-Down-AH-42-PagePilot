"""
Tests for PagePilot Cross-Document Conflict & Contradiction Detection (Stage 6A).
"""

import pytest
from pagepilot.ai.claim_models import Evidence
from pagepilot.ai.claim_verifier import ClaimVerifier


def test_detect_numerical_contradiction():
    verifier = ClaimVerifier()
    ev_a = Evidence(
        evidence_id="E1",
        document_id=1,
        filename="Spec_v1.pdf",
        page_number=2,
        chunk_id="c1",
        chunk_index=0,
        original_text="The connection limit is 500.",
    )
    ev_b = Evidence(
        evidence_id="E2",
        document_id=2,
        filename="Spec_v2.pdf",
        page_number=5,
        chunk_id="c2",
        chunk_index=0,
        original_text="The connection limit is 1000.",
    )

    conflicts = verifier.detect_cross_document_contradictions([ev_a, ev_b])

    assert len(conflicts) == 1
    assert "Connection Limit" in conflicts[0].topic
    assert conflicts[0].conflict_type == "value_mismatch"
    assert conflicts[0].source_a == "[Spec_v1.pdf, p. 2]"
    assert conflicts[0].source_b == "[Spec_v2.pdf, p. 5]"


def test_detect_status_negation_contradiction():
    verifier = ClaimVerifier()
    ev_a = Evidence(
        evidence_id="E1",
        document_id=10,
        filename="Doc2025.pdf",
        page_number=1,
        chunk_id="c1",
        chunk_index=0,
        original_text="Legacy XML protocol is supported.",
    )
    ev_b = Evidence(
        evidence_id="E2",
        document_id=11,
        filename="Doc2026.pdf",
        page_number=3,
        chunk_id="c2",
        chunk_index=0,
        original_text="Legacy XML protocol is deprecated.",
    )

    conflicts = verifier.detect_cross_document_contradictions([ev_a, ev_b])

    assert len(conflicts) == 1
    assert conflicts[0].conflict_type == "direct_negation"


def test_no_false_conflict_on_different_units():
    verifier = ClaimVerifier()
    ev_a = Evidence(
        evidence_id="E1",
        document_id=1,
        filename="DocA.pdf",
        page_number=1,
        chunk_id="c1",
        chunk_index=0,
        original_text="Buffer memory is 512 KB.",
    )
    ev_b = Evidence(
        evidence_id="E2",
        document_id=2,
        filename="DocB.pdf",
        page_number=1,
        chunk_id="c2",
        chunk_index=0,
        original_text="Buffer memory is 16 MB.",
    )

    conflicts = verifier.detect_cross_document_contradictions([ev_a, ev_b])
    # Different units (KB vs MB) should not be flagged as direct numerical value collision
    assert len(conflicts) == 0

