"""
PagePilot – Evidence and Claim-Level Verification Models (Stage 6A).

Implements the core data models for multi-document reasoning:
- Evidence: Verifiable retrieved passages with immutable document provenance
- Claim: Atomic factual assertions linked to evidence IDs
- Contradiction: Explicit conflict detection between sources
- VerifiedAnswer: Complete structured response with sentence-level citations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class VerificationStatus(str, Enum):
    """Explicit verification outcomes for factual claims."""
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    INVALID_CITATION = "INVALID_CITATION"
    NEEDS_REVIEW = "NEEDS_REVIEW"


@dataclass(frozen=True)
class Evidence:
    """
    Immutable representation of an atomic retrieved text passage.
    Evidence IDs (e.g. E1, E2) are assigned deterministically by application code.
    Qwen must never invent evidence IDs or document provenance.
    """
    evidence_id: str                    # e.g. "E1", "E2"
    document_id: int
    filename: str
    page_number: int                    # 1-based physical PDF page
    chunk_id: str
    chunk_index: int
    original_text: str
    retrieval_method: str = "hybrid"    # 'hybrid' | 'semantic' | 'keyword'
    retrieval_rank: int = 1
    query_requirement_id: Optional[str] = None

    @property
    def citation_tag(self) -> str:
        """Standard citation string: [filename.pdf, p. N]."""
        return f"[{self.filename}, p. {self.page_number}]"

    def snippet(self, max_length: int = 160) -> str:
        """Clean single-line excerpt for previews."""
        clean = " ".join(self.original_text.split())
        if len(clean) > max_length:
            return clean[:max_length - 1] + "…"
        return clean


@dataclass
class Claim:
    """
    An atomic factual claim extracted from model reasoning.
    Every factual claim must reference one or more verified evidence IDs.
    """
    claim_id: str                       # e.g. "C1", "C2"
    text: str
    claim_type: str = "factual"         # 'factual' | 'contextual' | 'disclaimer'
    evidence_ids: list[str] = field(default_factory=list)
    verification_status: VerificationStatus = VerificationStatus.NEEDS_REVIEW
    verification_notes: str = ""
    contradicting_evidence_ids: list[str] = field(default_factory=list)

    @property
    def is_verified(self) -> bool:
        return self.verification_status == VerificationStatus.SUPPORTED

    @property
    def is_contradicted(self) -> bool:
        return self.verification_status == VerificationStatus.CONTRADICTED


@dataclass
class Contradiction:
    """
    Represents an identified cross-document or intra-corpus contradiction.
    """
    topic: str
    statement_a: str
    source_a: str                       # e.g. "[Architecture_v1.pdf, p. 12]"
    doc_id_a: int
    page_a: int
    statement_b: str
    source_b: str                       # e.g. "[Architecture_v2.pdf, p. 14]"
    doc_id_b: int
    page_b: int
    conflict_type: str = "value_mismatch" # 'value_mismatch' | 'direct_negation' | 'definition_conflict'
    explanation: str = ""


@dataclass
class InformationRequirement:
    """
    A discrete sub-question or information need identified during query planning.
    """
    requirement_id: str                 # e.g. "R1", "R2"
    subquery: str
    description: str = ""
    supported_doc_ids: list[int] = field(default_factory=list)
    has_sufficient_evidence: bool = False


@dataclass
class VerifiedAnswer:
    """
    Final composite answer constructed from validated atomic claims.
    Faithful to every sentence: every factual statement has its own citations.
    """
    question: str
    claims: list[Claim]
    evidence_map: dict[str, Evidence]
    contradictions: list[Contradiction] = field(default_factory=list)
    unanswered_requirements: list[str] = field(default_factory=list)
    contributing_documents: list[str] = field(default_factory=list)
    raw_response: str = ""

    def get_supported_claims(self) -> list[Claim]:
        return [c for c in self.claims if c.verification_status == VerificationStatus.SUPPORTED]

    def get_rejected_claims(self) -> list[Claim]:
        return [c for c in self.claims if c.verification_status in (
            VerificationStatus.INSUFFICIENT_EVIDENCE,
            VerificationStatus.INVALID_CITATION,
            VerificationStatus.CONTRADICTED,
        )]

    def render_sentence_level_text(self) -> str:
        """
        Produce narrative text where every verified factual sentence is
        appended with its exact source citations. Unsupported claims are omitted
        from verified text.
        """
        lines: list[str] = []

        supported = self.get_supported_claims()
        if supported:
            for claim in supported:
                # Gather citations for this specific claim
                citations: list[str] = []
                seen_tags: set[str] = set()
                for eid in claim.evidence_ids:
                    ev = self.evidence_map.get(eid)
                    if ev and ev.citation_tag not in seen_tags:
                        seen_tags.add(ev.citation_tag)
                        citations.append(ev.citation_tag)

                cite_str = f" {' '.join(citations)}" if citations else ""
                clean_text = claim.text.strip().rstrip(".")
                lines.append(f"{clean_text}.{cite_str}")
        elif not self.contradictions and not self.unanswered_requirements:
            lines.append(
                "Based on the selected documents, I could not find sufficient "
                "evidence to verify an answer to this question."
            )

        # Append Contradiction alerts if detected
        if self.contradictions:
            lines.append("\n**Conflicting Information Across Documents:**")
            for c in self.contradictions:
                lines.append(
                    f"- **{c.topic}**: Source {c.source_a} states \"{c.statement_a}\", "
                    f"whereas source {c.source_b} states \"{c.statement_b}\". ({c.explanation})"
                )

        # Append Unanswered Requirements if any part could not be supported
        if self.unanswered_requirements:
            lines.append("\n**Missing Evidence for Sub-Questions:**")
            for req in self.unanswered_requirements:
                lines.append(f"- Based on the selected documents, I could not find sufficient evidence to verify: *{req}*")

        return "\n\n".join(lines)

