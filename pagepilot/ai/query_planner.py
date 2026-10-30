"""
PagePilot – Query Planning & Decomposition Engine (Stage 6A).

Analyzes user questions to determine whether multi-document or multi-aspect
reasoning is required. Decomposes comparative and multi-part questions into
bounded, verifiable information requirements.
"""

from __future__ import annotations

import re
from typing import List, Optional

from pagepilot.ai.claim_models import InformationRequirement
from pagepilot.ai.ollama_client import ChatMessage, OllamaClient


_COMPARISON_PATTERNS = [
    re.compile(r"\bcompare\s+(.+?)\s+(?:and|with|to)\s+(.+)", re.IGNORECASE),
    re.compile(r"\bdifference(?:s)?\s+between\s+(.+?)\s+and\s+(.+)", re.IGNORECASE),
    re.compile(r"\bhow\s+does\s+(.+?)\s+differ\s+from\s+(.+)", re.IGNORECASE),
    re.compile(r"(.+?)\s+\b(?:versus|vs\.?)\s+(.+)", re.IGNORECASE),
    re.compile(r"\bboth\s+(.+?)\s+and\s+(.+)", re.IGNORECASE),
]


class QueryPlanner:
    """
    Decomposes multi-part and comparative questions into atomic subqueries.
    Uses deterministic parsing first, with bounded LLM decomposition fallback.
    """

    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        self.client = ollama_client

    def plan(self, question: str) -> list[InformationRequirement]:
        """
        Produce a bounded list of 1 to 4 InformationRequirement objects.
        Simple questions return a single requirement containing the original question.
        """
        clean_q = question.strip()
        if not clean_q:
            return []

        # 1. Deterministic Rule-Based Decomposition
        deterministic_requirements = self._decompose_deterministic(clean_q)
        if deterministic_requirements:
            return deterministic_requirements[:4]

        # 2. Check for multi-sentence or multiple question clauses
        sentence_splits = re.split(r"(?<=[.?!])\s+", clean_q)
        if len(sentence_splits) > 1:
            reqs = []
            for idx, s in enumerate(sentence_splits[:3], 1):
                s_clean = s.strip()
                if len(s_clean) > 8:
                    reqs.append(
                        InformationRequirement(
                            requirement_id=f"R{idx}",
                            subquery=s_clean,
                            description=f"Aspect {idx}",
                        )
                    )
            if len(reqs) > 1:
                return reqs

        # 3. Default: Single requirement with original question
        return [
            InformationRequirement(
                requirement_id="R1",
                subquery=clean_q,
                description="Primary question",
            )
        ]

    def _decompose_deterministic(self, question: str) -> Optional[list[InformationRequirement]]:
        """Identify comparison and conjunction patterns deterministically."""
        # Check comparison patterns
        for pattern in _COMPARISON_PATTERNS:
            match = pattern.search(question)
            if match:
                part_a = self._clean_concept(match.group(1))
                part_b = self._clean_concept(match.group(2))

                # Extract secondary requirement like key usage if present
                trailing_aspect = ""
                aspect_match = re.search(r"(?:including|regarding|and|such as|specifically)\s+(?:how\s+)?(.+)", part_b, re.IGNORECASE)
                if aspect_match:
                    trailing_aspect = aspect_match.group(1).strip().rstrip("?.")
                    part_b = re.sub(r"(?:including|regarding|such as|specifically)\s+(?:how\s+)?(.+)", "", part_b, flags=re.IGNORECASE).strip()

                reqs = [
                    InformationRequirement(
                        requirement_id="R1",
                        subquery=f"What is {part_a} and its characteristics?",
                        description=f"Evidence regarding {part_a}",
                    ),
                    InformationRequirement(
                        requirement_id="R2",
                        subquery=f"What is {part_b} and its characteristics?",
                        description=f"Evidence regarding {part_b}",
                    ),
                ]

                if trailing_aspect:
                    reqs.append(
                        InformationRequirement(
                            requirement_id="R3",
                            subquery=f"{trailing_aspect} for {part_a} and {part_b}",
                            description=f"Evidence regarding {trailing_aspect}",
                        )
                    )

                return reqs

        return None

    def _clean_concept(self, text: str) -> str:
        """Strip trailing punctuation, conjunctions, and filler words."""
        cleaned = re.sub(r"[?!.,;]+$", "", text.strip())
        cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

