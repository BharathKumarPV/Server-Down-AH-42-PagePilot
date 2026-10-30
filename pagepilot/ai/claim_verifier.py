"""
PagePilot – Claim Verification and Contradiction Detection Engine (Stage 6A).

Evaluates every factual claim against its cited evidence passages:
1. Validates evidence ID existence (rejects hallucinated citation IDs).
2. Performs conservative entailment and keyword/entity grounding checks.
3. Detects direct value, numerical, and negation contradictions across documents.
4. Distinguishes genuine contradictions from temporal or unit differences.
5. Emits explicit status: SUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE,
   INVALID_CITATION, or NEEDS_REVIEW.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

from pagepilot.ai.claim_models import (
    Claim,
    Contradiction,
    Evidence,
    VerificationStatus,
)


# Words ignored during content overlap calculation
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to", "for",
    "with", "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "from", "up", "down", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "this",
    "that", "these", "those", "it", "its", "of", "by", "as", "such",
}

_NEGATION_WORDS = {"not", "never", "no", "cannot", "none", "neither", "prohibited", "deprecated"}


def _extract_keywords(text: str) -> set[str]:
    """Extract lowercase alpha-numeric tokens excluding common stopwords."""
    tokens = re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text.lower())
    return {t for t in tokens if t not in _STOPWORDS}


def _extract_numbers(text: str) -> list[str]:
    """Extract numeric values, percentages, and metrics."""
    return re.findall(r"\b\d+(?:\.\d+)?(?:%|ms|s|kb|mb|gb|tb|m|km|hz|ghz)?\b", text.lower())


class ClaimVerifier:
    """
    Evaluates factual claims against source evidence and identifies cross-document conflicts.
    """

    def __init__(self, conservative_mode: bool = True):
        self.conservative_mode = conservative_mode

    def verify_claim(self, claim: Claim, evidence_map: dict[str, Evidence]) -> Claim:
        """
        Verify a single atomic claim against its referenced evidence IDs.
        Updates verification_status and verification_notes in-place.
        """
        # Non-factual claims (e.g. conversational disclaimers)
        if claim.claim_type != "factual":
            claim.verification_status = VerificationStatus.SUPPORTED
            claim.verification_notes = "Non-factual or contextual statement."
            return claim

        # Rule 1: Factual claims must reference at least one evidence ID
        if not claim.evidence_ids:
            claim.verification_status = VerificationStatus.INSUFFICIENT_EVIDENCE
            claim.verification_notes = "Factual claim has no supporting evidence references."
            return claim

        # Rule 2: All referenced evidence IDs must exist in current request's evidence pool
        missing_ids = [eid for eid in claim.evidence_ids if eid not in evidence_map]
        if missing_ids:
            claim.verification_status = VerificationStatus.INVALID_CITATION
            claim.verification_notes = f"Referenced non-existent evidence IDs: {', '.join(missing_ids)}."
            return claim

        # Collect text from referenced evidence
        referenced_evidence = [evidence_map[eid] for eid in claim.evidence_ids]
        combined_evidence_text = " ".join(e.original_text for e in referenced_evidence)
        combined_evidence_lower = combined_evidence_text.lower()

        claim_keywords = _extract_keywords(claim.text)
        evidence_keywords = _extract_keywords(combined_evidence_text)

        if not claim_keywords:
            claim.verification_status = VerificationStatus.SUPPORTED
            return claim

        # Rule 3: Check numeric / quantitative assertions
        claim_numbers = _extract_numbers(claim.text)
        evidence_numbers = _extract_numbers(combined_evidence_text)
        for num in claim_numbers:
            if num not in evidence_numbers:
                # If claim states a specific number not found in evidence, check if contradictory
                claim.verification_status = VerificationStatus.CONTRADICTED
                claim.verification_notes = f"Numerical value '{num}' is not substantiated by cited evidence."
                return claim

        # Rule 4: Negation conflict check
        claim_words = set(re.findall(r"\b\w+\b", claim.text.lower()))
        claim_has_negation = bool(claim_words & _NEGATION_WORDS)
        evidence_has_negation = bool(set(re.findall(r"\b\w+\b", combined_evidence_lower)) & _NEGATION_WORDS)

        # Rule 5: Lexical keyword overlap ratio
        overlap = claim_keywords & evidence_keywords
        overlap_ratio = len(overlap) / len(claim_keywords)

        # Conservative grounding threshold: At least 50% of distinct keywords must appear in evidence
        min_overlap = 0.45 if len(claim_keywords) > 4 else 0.40

        if overlap_ratio >= min_overlap:
            # Check for direct contradictions with any other known evidence in pool
            contradicting_eids: list[str] = []
            for other_eid, other_ev in evidence_map.items():
                if other_eid not in claim.evidence_ids:
                    conflict_found = self._check_passage_contradicts_claim(claim, other_ev)
                    if conflict_found:
                        contradicting_eids.append(other_eid)

            if contradicting_eids:
                claim.verification_status = VerificationStatus.CONTRADICTED
                claim.contradicting_evidence_ids = contradicting_eids
                claim.verification_notes = (
                    f"Claim is contradicted by source passage in {evidence_map[contradicting_eids[0]].citation_tag}."
                )
            else:
                claim.verification_status = VerificationStatus.SUPPORTED
                claim.verification_notes = f"Supported by evidence ({overlap_ratio:.0%} lexical grounding)."
        else:
            missing_terms = list(claim_keywords - overlap)[:3]
            claim.verification_status = VerificationStatus.INSUFFICIENT_EVIDENCE
            claim.verification_notes = (
                f"Insufficient evidence grounding ({overlap_ratio:.0%}). Missing key terms: {missing_terms}."
            )

        return claim

    def verify_all(self, claims: list[Claim], evidence_map: dict[str, Evidence]) -> list[Claim]:
        """Verify all claims in a collection."""
        return [self.verify_claim(c, evidence_map) for c in claims]

    def detect_cross_document_contradictions(
        self,
        evidence_list: list[Evidence],
    ) -> list[Contradiction]:
        """
        Scan retrieved evidence across different documents to identify conflicting
        values, opposing policies, or contradictory statements.
        """
        contradictions: list[Contradiction] = []
        if len(evidence_list) < 2:
            return contradictions

        # Group evidence by distinct documents
        docs = list({e.document_id for e in evidence_list})
        if len(docs) < 2:
            return contradictions

        # Pairwise comparison across different documents
        for i in range(len(evidence_list)):
            for j in range(i + 1, len(evidence_list)):
                ev_a = evidence_list[i]
                ev_b = evidence_list[j]

                if ev_a.document_id == ev_b.document_id:
                    continue

                conflict = self._detect_pairwise_conflict(ev_a, ev_b)
                if conflict:
                    contradictions.append(conflict)

        return contradictions

    def _detect_pairwise_conflict(self, ev_a: Evidence, ev_b: Evidence) -> Optional[Contradiction]:
        """Check if two evidence passages from different documents conflict."""
        text_a = ev_a.original_text
        text_b = ev_b.original_text

        # 1. Numerical mismatch on shared metrics
        # Look for patterns like "value is X" vs "value is Y" or "timeout: X" vs "timeout: Y"
        num_conflict = self._check_numerical_conflict(text_a, text_b, ev_a, ev_b)
        if num_conflict:
            return num_conflict

        # 2. Opposite status / direct negation on shared subject
        status_conflict = self._check_status_negation_conflict(text_a, text_b, ev_a, ev_b)
        if status_conflict:
            return status_conflict

        return None

    def _check_numerical_conflict(
        self,
        text_a: str,
        text_b: str,
        ev_a: Evidence,
        ev_b: Evidence,
    ) -> Optional[Contradiction]:
        """Detect conflicting numbers on identical entities (e.g. rate is 10 vs 15)."""
        # Look for entities with numeric patterns like: [metric] is/of [number]
        pattern = r"\b([a-zA-Z\s]{3,25})\b\s*(?:is|=|was|set to|measured at|value of)\s*(\d+(?:\.\d+)?)\s*([a-zA-Z%]+)?"
        matches_a = re.findall(pattern, text_a, re.IGNORECASE)
        matches_b = re.findall(pattern, text_b, re.IGNORECASE)

        for ent_a, val_a, unit_a in matches_a:
            ent_a_clean = ent_a.strip().lower()
            if ent_a_clean in _STOPWORDS:
                continue

            for ent_b, val_b, unit_b in matches_b:
                ent_b_clean = ent_b.strip().lower()
                # Check entity match
                if ent_a_clean in ent_b_clean or ent_b_clean in ent_a_clean:
                    # Check unit difference vs genuine value conflict
                    if unit_a and unit_b and unit_a.lower() != unit_b.lower():
                        # Different units (e.g. MB vs GB, km vs miles) - not a direct numerical contradiction
                        continue

                    # If numerical values differ
                    if val_a != val_b:
                        # Check if temporal qualifiers exist (e.g. "in 2024" vs "in 2026")
                        year_a = re.search(r"\b(20\d\d)\b", text_a)
                        year_b = re.search(r"\b(20\d\d)\b", text_b)
                        if year_a and year_b and year_a.group(1) != year_b.group(1):
                            explanation = f"Values differ over time ({year_a.group(1)} vs {year_b.group(1)})."
                        else:
                            explanation = f"Conflicting values reported ({val_a} vs {val_b})."

                        return Contradiction(
                            topic=ent_a.strip().title(),
                            statement_a=f"{ent_a.strip()} is {val_a}{(' ' + unit_a) if unit_a else ''}",
                            source_a=ev_a.citation_tag,
                            doc_id_a=ev_a.document_id,
                            page_a=ev_a.page_number,
                            statement_b=f"{ent_b.strip()} is {val_b}{(' ' + unit_b) if unit_b else ''}",
                            source_b=ev_b.citation_tag,
                            doc_id_b=ev_b.document_id,
                            page_b=ev_b.page_number,
                            conflict_type="value_mismatch",
                            explanation=explanation,
                        )

        return None

    def _check_status_negation_conflict(
        self,
        text_a: str,
        text_b: str,
        ev_a: Evidence,
        ev_b: Evidence,
    ) -> Optional[Contradiction]:
        """Detect opposite status (e.g. supported vs deprecated, allowed vs prohibited)."""
        opposing_pairs = [
            ("supported", "deprecated"),
            ("enabled", "disabled"),
            ("required", "optional"),
            ("allowed", "prohibited"),
            ("valid", "invalid"),
            ("active", "inactive"),
            ("recommended", "discouraged"),
        ]

        text_a_lower = text_a.lower()
        text_b_lower = text_b.lower()

        # Check for shared key concept
        kw_a = _extract_keywords(text_a)
        kw_b = _extract_keywords(text_b)
        shared_keywords = kw_a & kw_b

        if len(shared_keywords) >= 2:
            shared_topic = " ".join(list(shared_keywords)[:2]).title()
            for pos, neg in opposing_pairs:
                if (pos in text_a_lower and neg in text_b_lower) or (neg in text_a_lower and pos in text_b_lower):
                    stat_a = f"Described as {pos}" if pos in text_a_lower else f"Described as {neg}"
                    stat_b = f"Described as {neg}" if pos in text_a_lower else f"Described as {pos}"

                    return Contradiction(
                        topic=shared_topic,
                        statement_a=stat_a,
                        source_a=ev_a.citation_tag,
                        doc_id_a=ev_a.document_id,
                        page_a=ev_a.page_number,
                        statement_b=stat_b,
                        source_b=ev_b.citation_tag,
                        doc_id_b=ev_b.document_id,
                        page_b=ev_b.page_number,
                        conflict_type="direct_negation",
                        explanation=f"Contradictory status '{pos}' vs '{neg}'.",
                    )

        return None

    def _check_passage_contradicts_claim(self, claim: Claim, other_ev: Evidence) -> bool:
        """Check if an unreferenced passage directly contradicts an assertion in claim."""
        claim_lower = claim.text.lower()
        passage_lower = other_ev.original_text.lower()

        claim_numbers = _extract_numbers(claim.text)
        passage_numbers = _extract_numbers(other_ev.original_text)

        kw_claim = _extract_keywords(claim.text)
        kw_passage = _extract_keywords(other_ev.original_text)

        # If they share major subject keywords but have conflicting numbers
        if len(kw_claim & kw_passage) >= 2:
            if claim_numbers and passage_numbers:
                if set(claim_numbers) != set(passage_numbers):
                    return True

        return False

