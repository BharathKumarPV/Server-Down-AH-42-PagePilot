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


# Intent Classifications
INTENT_DOCUMENT_SUMMARY          = "DOCUMENT_SUMMARY"
INTENT_DOCUMENT_QUESTION         = "DOCUMENT_QUESTION"
INTENT_MULTI_DOCUMENT_COMPARISON = "MULTI_DOCUMENT_COMPARISON"
INTENT_CODE_EXPLANATION          = "CODE_EXPLANATION"
INTENT_TARGETED_LOOKUP           = "TARGETED_LOOKUP"
INTENT_GENERAL_CHAT              = "GENERAL_CHAT"

_COMPARISON_PATTERNS = [
    re.compile(r"\bcompare\s+(?:the\s+)?(.+?)\s+(?:and|with|to)\s+(.+)", re.IGNORECASE),
    re.compile(r"\bcompare\s+(?:the\s+)?(?:two|both|all)?\s*(?:uploaded\s+)?(?:documents|pdfs|files)\b", re.IGNORECASE),
    re.compile(r"\bdifference(?:s)?\s+between\s+(.+?)\s+and\s+(.+)", re.IGNORECASE),
    re.compile(r"\bhow\s+does\s+(.+?)\s+differ\s+from\s+(.+)", re.IGNORECASE),
    re.compile(r"(.+?)\s+\b(?:versus|vs\.?)\s+(.+)", re.IGNORECASE),
    re.compile(r"\bboth\s+(.+?)\s+and\s+(.+)", re.IGNORECASE),
]

_SUMMARY_PATTERNS = [
    re.compile(r"\b(?:summarize|summary|overview|breakdown|gist|tldr|tl;dr|recap|abstract)\b", re.IGNORECASE),
    re.compile(r"\b(?:explain|tell\s+me|show\s+me)\s+(?:what(?:'s|\s+is)\s+in|what\s+.+\s+contains?|what\s+this\s+(?:pdf|document|file)\s+is\s+about)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(?:does\s+)?(?:this|the)\s+(?:pdf|document|file)\s+(?:contain|discuss|cover)\b", re.IGNORECASE),
    re.compile(r"\bcontents\s+of\s+(?:this|the)\s+(?:pdf|document|file)\b", re.IGNORECASE),
    re.compile(r"\bexplain\s+(?:this|the)\s+(?:pdf|document|file)\b", re.IGNORECASE),
    re.compile(r"\b(?:main\s+concepts|main\s+topics|key\s+takeaways|core\s+ideas)\s+in\s+this\b", re.IGNORECASE),
    re.compile(r"\b(?:summarize|explain)\s+pages?\s+(\d+)\s*(?:to|-|through)\s*(\d+)\b", re.IGNORECASE),
]

_CODE_PATTERNS = [
    re.compile(r"\b(?:explain|walk\s*through|break\s*down|show|analyze|describe)\s+(?:the\s+)?code\b", re.IGNORECASE),
    re.compile(r"\bcode\s+(?:in|from|of)\s+(?:the\s+)?(?:uploaded\s+)?(?:pdf|document|file)\b", re.IGNORECASE),
    re.compile(r"\b(?:code\s+in\s+detail|python\s+code|implementation\s+of|script\s+in|functions?\s+in)\b", re.IGNORECASE),
    re.compile(r"\b(?:how\s+is\s+.+\s+implemented|walkthrough\s+of\s+code)\b", re.IGNORECASE),
]

_LOOKUP_PATTERNS = [
    re.compile(r"\b(?:define|definition\s+of|lookup|formula\s+for|equation\s+for|table\s+\d+|figure\s+\d+)\b", re.IGNORECASE),
]


class QueryPlanner:
    """
    Decomposes multi-part and comparative questions into atomic subqueries.
    Uses deterministic parsing first, with bounded LLM decomposition fallback.
    """

    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        self.client = ollama_client

    def classify_intent(self, question: str, has_documents: bool = True) -> str:
        """
        Classifies user prompt into one of the 6 canonical intents without expensive LLM calls.
        """
        clean_q = question.strip()
        if not clean_q or not has_documents:
            return INTENT_GENERAL_CHAT

        # 1. Multi-document comparison check
        for pat in _COMPARISON_PATTERNS:
            if pat.search(clean_q):
                return INTENT_MULTI_DOCUMENT_COMPARISON

        # 2. Document summary check
        for pat in _SUMMARY_PATTERNS:
            if pat.search(clean_q):
                return INTENT_DOCUMENT_SUMMARY

        # 3. Code explanation check
        for pat in _CODE_PATTERNS:
            if pat.search(clean_q):
                return INTENT_CODE_EXPLANATION

        # 4. Targeted lookup check
        for pat in _LOOKUP_PATTERNS:
            if pat.search(clean_q):
                return INTENT_TARGETED_LOOKUP

        # 5. Default document question
        return INTENT_DOCUMENT_QUESTION

    def extract_page_range(self, question: str) -> Optional[tuple[int, int]]:
        """Extract explicit page span like 'pages 20 to 30' or 'page 10-15'."""
        match = re.search(r"\bpages?\s+(\d+)\s*(?:to|-|through)\s*(\d+)\b", question, re.IGNORECASE)
        if match:
            start_p = int(match.group(1))
            end_p = int(match.group(2))
            if start_p > end_p:
                start_p, end_p = end_p, start_p
            return (start_p, end_p)
        return None

    def clean_retrieval_query(self, question: str) -> str:
        """
        Strips meta-conversational words and filler phrases that damage keyword retrieval in FTS5.
        """
        cleaned = question.strip()
        cleaned = re.sub(r"\b(?:in|from|of)\s+(?:the\s+)?(?:uploaded\s+)?(?:pdf|pdfs|document|documents|file|files|textbook)\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(?:uploaded\s+)?(?:pdf|pdfs|document|documents|file|files)\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(?:please\s+)?(?:explain|tell\s+me|show\s+me|summarize)\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"[?!.,;:]+$", "", cleaned.strip())
        return cleaned.strip() or question.strip()

    def plan(self, question: str) -> list[InformationRequirement]:
        """
        Produce a bounded list of 1 to 4 InformationRequirement objects.
        Simple questions return a single requirement containing the original question.
        """
        clean_q = question.strip()
        if not clean_q:
            return []

        intent = self.classify_intent(clean_q, has_documents=True)
        if intent == INTENT_DOCUMENT_SUMMARY:
            page_range = self.extract_page_range(clean_q)
            if page_range:
                return [
                    InformationRequirement(
                        requirement_id="R1",
                        subquery=f"Overview of content from pages {page_range[0]} to {page_range[1]}",
                        description=f"Pages {page_range[0]}-{page_range[1]}",
                    )
                ]
            return [
                InformationRequirement(
                    requirement_id="R1",
                    subquery="Executive overview and main themes of the document",
                    description="Overview",
                ),
                InformationRequirement(
                    requirement_id="R2",
                    subquery="Key concepts, architecture, and major findings",
                    description="Core Concepts",
                ),
            ]

        if intent == INTENT_CODE_EXPLANATION:
            cleaned = self.clean_retrieval_query(clean_q)
            return [
                InformationRequirement(
                    requirement_id="R1",
                    subquery=f"code implementation algorithms functions {cleaned}".strip(),
                    description="Code Structure & Walkthrough",
                )
            ]

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
                            description=f"Aspect {idx}: {s_clean[:40]}",
                        )
                    )
            if len(reqs) > 1:
                return reqs

        # 3. Default: Single requirement with original question
        return [
            InformationRequirement(
                requirement_id="R1",
                subquery=clean_q,
                description=clean_q[:60],
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

