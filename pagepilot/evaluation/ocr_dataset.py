"""
PagePilot – Stage 6B OCR Evaluation Dataset & Metrics.

Provides 8 benchmark test categories covering:
A. Printed text
B. Handwritten English notes
C. Rotated or skewed notes
D. Low-light photographs
E. Mathematical formulas (printed)
F. Handwritten mathematical formulas
G. Mixed text and formulas
H. Scanned textbook pages

Includes CER (Character Error Rate) and LaTeX structural comparison algorithms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class OCREvalCase:
    """Benchmark test fixture for OCR and formula recognition."""
    case_id: str
    category: str                       # 'A' through 'H'
    category_name: str
    name: str
    description: str
    source_type: str                    # "image_note" | "scanned_pdf"
    image_fixture_filename: str
    ground_truth_text: str
    ground_truth_latex: Optional[str] = None
    has_formulas: bool = False
    has_handwriting: bool = False


def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Compute Character Error Rate (CER) via Levenshtein distance:
    CER = (Substitutions + Insertions + Deletions) / len(reference)
    Returns float from 0.0 (perfect) to 1.0+ (heavily degraded).
    """
    ref = reference.strip()
    hyp = hypothesis.strip()
    if not ref:
        return 0.0 if not hyp else 1.0

    r_len, h_len = len(ref), len(hyp)
    # Dynamic programming matrix (2 rows optimization)
    dp = [list(range(h_len + 1)), [0] * (h_len + 1)]

    for i in range(1, r_len + 1):
        cur = i % 2
        prev = 1 - cur
        dp[cur][0] = i
        for j in range(1, h_len + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[cur][j] = min(
                dp[prev][j] + 1,       # deletion
                dp[cur][j - 1] + 1,    # insertion
                dp[prev][j - 1] + cost # substitution
            )

    dist = dp[r_len % 2][h_len]
    return float(dist) / float(r_len)


def normalize_latex(latex: str) -> str:
    """
    Normalize whitespace, curly braces, and common LaTeX commands for structural comparison.
    """
    if not latex:
        return ""
    clean = latex.strip()
    # Remove leading/trailing delimiters \\[ ... \\] or $ ... $
    clean = re.sub(r"^\\\[\s*", "", clean)
    clean = re.sub(r"\s*\\\]$", "", clean)
    clean = re.sub(r"^\$\$\s*", "", clean)
    clean = re.sub(r"\s*\$\$$", "", clean)
    clean = re.sub(r"^\$\s*", "", clean)
    clean = re.sub(r"\s*\$$", "", clean)

    # Normalize whitespace around operators
    clean = re.sub(r"\s+", " ", clean)
    clean = re.sub(r"\s*([=+\-*^_\\])\s*", r"\1", clean)
    clean = re.sub(r"\s*([{}(),])\s*", r"\1", clean)

    # Normalize common fraction spacing \frac {a} {b} -> \frac{a}{b}
    clean = re.sub(r"\\frac\s*\{", r"\\frac{", clean)
    clean = re.sub(r"\}\s*\{", r"}{", clean)

    return clean.strip()


def compare_latex_structures(reference: str, hypothesis: str) -> Tuple[bool, bool]:
    """
    Compare reference and hypothesis LaTeX expressions.
    Returns (exact_match, structural_match).
    """
    if not reference or not hypothesis:
        return (reference == hypothesis, reference == hypothesis)

    norm_ref = normalize_latex(reference)
    norm_hyp = normalize_latex(hypothesis)

    exact = (norm_ref == norm_hyp)

    # Structural match: checks presence of identical core operators and operands
    def extract_structure(s: str) -> set:
        tokens = re.findall(r"\\[a-zA-Z]+|[a-zA-Z0-9]+|[=+\-*/^_]", s)
        return set(tokens)

    ref_tokens = extract_structure(norm_ref)
    hyp_tokens = extract_structure(norm_hyp)
    structural = (ref_tokens == hyp_tokens) or exact

    return (exact, structural)


OCR_EVALUATION_DATASET: List[OCREvalCase] = [
    # Category A: Printed text
    OCREvalCase(
        case_id="case_a_printed_text",
        category="A",
        category_name="Printed Text",
        name="Operating System Virtual Memory Architecture",
        description="Clean high-contrast printed textbook excerpt on page tables and TLBs.",
        source_type="scanned_pdf",
        image_fixture_filename="printed_os_memory.png",
        ground_truth_text="The translation lookaside buffer (TLB) caches recent virtual-to-physical address mappings. When a TLB miss occurs, the hardware page table walker traverses the multi-level page table in main memory.",
        ground_truth_latex=None,
        has_formulas=False,
        has_handwriting=False,
    ),

    # Category B: Handwritten English notes
    OCREvalCase(
        case_id="case_b_handwritten_notes",
        category="B",
        category_name="Handwritten English Notes",
        name="Machine Learning Lecture Notes",
        description="Student handwritten lecture notes on gradient descent optimization.",
        source_type="image_note",
        image_fixture_filename="handwritten_ml_notes.png",
        ground_truth_text="Gradient descent iteratively updates model weights in the direction opposite to the gradient of the loss function. The learning rate hyperparameter controls step size.",
        ground_truth_latex=None,
        has_formulas=False,
        has_handwriting=True,
    ),

    # Category C: Rotated or skewed notes
    OCREvalCase(
        case_id="case_c_skewed_notes",
        category="C",
        category_name="Rotated or Skewed Notes",
        name="Rotated Calculus Review Sheet",
        description="Student photograph captured with a 12-degree clockwise camera angle skew.",
        source_type="image_note",
        image_fixture_filename="skewed_calculus_note.jpg",
        ground_truth_text="L'Hopital's rule applies when evaluating limits resulting in indeterminate forms such as zero over zero or infinity over infinity.",
        ground_truth_latex=None,
        has_formulas=False,
        has_handwriting=True,
    ),

    # Category D: Low-light photographs
    OCREvalCase(
        case_id="case_d_low_light_photo",
        category="D",
        category_name="Low-Light Photographs",
        name="Evening Study Desk Photograph",
        description="Photograph taken under dim desk lamp illumination with shadow gradients.",
        source_type="image_note",
        image_fixture_filename="low_light_study_photo.jpg",
        ground_truth_text="The critical path in project scheduling defines the longest sequence of dependent activities required for project completion.",
        ground_truth_latex=None,
        has_formulas=False,
        has_handwriting=True,
    ),

    # Category E: Mathematical formulas (printed)
    OCREvalCase(
        case_id="case_e_printed_formulas",
        category="E",
        category_name="Mathematical Formulas (Printed)",
        name="Quadratic Formula and Normal Distribution",
        description="Typeset mathematical formulas from standard algebra and statistics textbooks.",
        source_type="scanned_pdf",
        image_fixture_filename="printed_quadratic_formula.png",
        ground_truth_text="Quadratic formula for finding roots of ax^2 + bx + c = 0:",
        ground_truth_latex=r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}",
        has_formulas=True,
        has_handwriting=False,
    ),

    # Category F: Handwritten mathematical formulas
    OCREvalCase(
        case_id="case_f_handwritten_formulas",
        category="F",
        category_name="Handwritten Mathematical Formulas",
        name="Calculus Product Rule Derivative",
        description="Handwritten student equation showing differentiation with exponential and trigonometric terms.",
        source_type="image_note",
        image_fixture_filename="handwritten_derivative.jpg",
        ground_truth_text="Derivative of product:",
        ground_truth_latex=r"\frac{d}{dx}[e^{2x}\sin(x)] = 2e^{2x}\sin(x) + e^{2x}\cos(x)",
        has_formulas=True,
        has_handwriting=True,
    ),

    # Category G: Mixed text and formulas
    OCREvalCase(
        case_id="case_g_mixed_text_formula",
        category="G",
        category_name="Mixed Text and Formulas",
        name="Thermodynamics State Equation Problem",
        description="Physics exercise problem combining explanatory prose and the ideal gas equation.",
        source_type="image_note",
        image_fixture_filename="thermodynamics_problem.png",
        ground_truth_text="For an ideal gas undergoing an isothermal expansion, pressure and volume satisfy PV = nRT with constant temperature T.",
        ground_truth_latex=r"PV = nRT",
        has_formulas=True,
        has_handwriting=True,
    ),

    # Category H: Scanned textbook pages
    OCREvalCase(
        case_id="case_h_scanned_textbook",
        category="H",
        category_name="Scanned Textbook Pages",
        name="Taylor Series Expansion Theorem",
        description="Scanned page from advanced calculus textbook containing definition, theorem, and series formula.",
        source_type="scanned_pdf",
        image_fixture_filename="scanned_taylor_series.pdf",
        ground_truth_text="Taylor's Theorem states that any infinitely differentiable function can be represented as a power series around point a:",
        ground_truth_latex=r"f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!} (x - a)^n",
        has_formulas=True,
        has_handwriting=False,
    ),
]

