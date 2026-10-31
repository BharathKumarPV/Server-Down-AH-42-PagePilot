"""
PagePilot – Stage 6B Mathematical Formula Validator.

Performs robust syntactic and structural validation on LaTeX expressions recognized
from scanned pages and photographed student notes.

CRITICAL ANTI-HALLUCINATION ENFORCEMENT:
The validator detects syntax flaws, unbalanced delimiters, malformed fractions,
and missing operands. It NEVER silently fabricates, guesses, or substitutes equations.
Unsound formulas are flagged for human review.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ValidationSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class FormulaIssue:
    code: str
    message: str
    severity: ValidationSeverity = ValidationSeverity.WARNING

    def to_dict(self) -> Dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
        }


@dataclass
class FormulaValidationResult:
    is_valid: bool
    latex_expression: str
    confidence_score: float = 1.0
    issues: List[FormulaIssue] = field(default_factory=list)
    bracket_counts: Dict[str, int] = field(default_factory=dict)
    has_unbalanced_braces: bool = False
    has_unbalanced_parens: bool = False
    has_unbalanced_brackets: bool = False
    has_unbalanced_left_right: bool = False
    needs_human_review: bool = False

    @property
    def errors(self) -> List[str]:
        return [i.message for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[str]:
        return [i.message for i in self.issues if i.severity == ValidationSeverity.WARNING]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "latex_expression": self.latex_expression,
            "confidence_score": round(self.confidence_score, 4),
            "errors": self.errors,
            "warnings": self.warnings,
            "has_unbalanced_braces": self.has_unbalanced_braces,
            "has_unbalanced_parens": self.has_unbalanced_parens,
            "has_unbalanced_brackets": self.has_unbalanced_brackets,
            "has_unbalanced_left_right": self.has_unbalanced_left_right,
            "needs_human_review": self.needs_human_review,
            "issues": [i.to_dict() for i in self.issues],
        }


def check_delimiter_balance(latex: str) -> Dict[str, Any]:
    """Check balancing of {}, (), [], and \\left/\\right pairs."""
    counts = {
        "open_brace": latex.count("{"),
        "close_brace": latex.count("}"),
        "open_paren": latex.count("("),
        "close_paren": latex.count(")"),
        "open_bracket": latex.count("["),
        "close_bracket": latex.count("]"),
        "left_delim": len(re.findall(r"\\left(?![a-zA-Z])", latex)),
        "right_delim": len(re.findall(r"\\right(?![a-zA-Z])", latex)),
    }

    # Sequential stack check for curly braces
    brace_stack = 0
    brace_order_valid = True
    for ch in latex:
        if ch == "{":
            brace_stack += 1
        elif ch == "}":
            brace_stack -= 1
            if brace_stack < 0:
                brace_order_valid = False
                break
    if brace_stack != 0:
        brace_order_valid = False

    return {
        "counts": counts,
        "braces_balanced": (counts["open_brace"] == counts["close_brace"]) and brace_order_valid,
        "parens_balanced": counts["open_paren"] == counts["close_paren"],
        "brackets_balanced": counts["open_bracket"] == counts["close_bracket"],
        "left_right_balanced": counts["left_delim"] == counts["right_delim"],
    }


def validate_formula_latex(
    latex: str,
    confidence_score: float = 1.0,
    confidence_threshold: float = 0.88,
) -> FormulaValidationResult:
    """
    Validate LaTeX formula syntax, delimiter balance, and operator integrity.
    Never alters or hallucinates output; flags defects for user verification.
    """
    expr = latex.strip() if latex else ""
    issues: List[FormulaIssue] = []

    if not expr:
        issues.append(FormulaIssue(
            code="EMPTY_EXPRESSION",
            message="Formula expression is empty.",
            severity=ValidationSeverity.ERROR,
        ))
        return FormulaValidationResult(
            is_valid=False,
            latex_expression="",
            confidence_score=confidence_score,
            issues=issues,
            needs_human_review=True,
        )

    # 1. Delimiter Balance
    delim = check_delimiter_balance(expr)
    counts = delim["counts"]
    unbalanced_braces = not delim["braces_balanced"]
    unbalanced_parens = not delim["parens_balanced"]
    unbalanced_brackets = not delim["brackets_balanced"]
    unbalanced_lr = not delim["left_right_balanced"]

    if unbalanced_braces:
        issues.append(FormulaIssue(
            code="UNBALANCED_BRACES",
            message=f"Unbalanced curly braces: {counts['open_brace']} open vs {counts['close_brace']} close.",
            severity=ValidationSeverity.ERROR,
        ))

    if unbalanced_parens:
        issues.append(FormulaIssue(
            code="UNBALANCED_PARENS",
            message=f"Unbalanced parentheses: {counts['open_paren']} open vs {counts['close_paren']} close.",
            severity=ValidationSeverity.WARNING,
        ))

    if unbalanced_brackets:
        issues.append(FormulaIssue(
            code="UNBALANCED_BRACKETS",
            message=f"Unbalanced square brackets: {counts['open_bracket']} open vs {counts['close_bracket']} close.",
            severity=ValidationSeverity.WARNING,
        ))

    if unbalanced_lr:
        issues.append(FormulaIssue(
            code="UNBALANCED_LEFT_RIGHT",
            message=f"Unbalanced \\left / \\right tags: {counts['left_delim']} \\left vs {counts['right_delim']} \\right.",
            severity=ValidationSeverity.WARNING,
        ))

    # 2. Fraction syntax check: \frac followed by groups
    frac_matches = re.finditer(r"\\frac(?![a-zA-Z])", expr)
    for fm in frac_matches:
        idx = fm.end()
        # Lookahead for {
        sub = expr[idx:].lstrip()
        if not sub.startswith("{"):
            issues.append(FormulaIssue(
                code="MALFORMED_FRACTION",
                message="\\frac operator is missing numerator argument group '{...}'.",
                severity=ValidationSeverity.ERROR,
            ))

    # 3. Radical syntax check: \sqrt
    sqrt_matches = re.finditer(r"\\sqrt(?![a-zA-Z])", expr)
    for sm in sqrt_matches:
        idx = sm.end()
        sub = expr[idx:].lstrip()
        if sub.startswith("["):
            # Optional index: find closing ]
            close_brk = sub.find("]")
            if close_brk != -1:
                sub = sub[close_brk + 1:].lstrip()
        if not sub.startswith("{") and not (sub and sub[0].isalnum()):
            issues.append(FormulaIssue(
                code="MALFORMED_RADICAL",
                message="\\sqrt operator is missing radicand argument '{...}'.",
                severity=ValidationSeverity.ERROR,
            ))

    # 4. Trailing / Dangling Operator check
    dangling_match = re.search(r"(?:[+\-*/=^_\\]|\\times|\\cdot|\\pm)\s*$", expr)
    if dangling_match:
        issues.append(FormulaIssue(
            code="DANGLING_OPERATOR",
            message=f"Expression ends with an incomplete operator: '{dangling_match.group(0).strip()}'.",
            severity=ValidationSeverity.WARNING,
        ))

    # 5. Suspicious ASCII Math Token check (bare literal letters without backslash)
    if re.search(r"(?<!\\)s\s*q\s*r\s*t\b", expr, re.IGNORECASE) or re.search(r"(?<!\\)s\s*q\s*c\s*t\b", expr, re.IGNORECASE):
        issues.append(FormulaIssue(
            code="SUSPICIOUS_ASCII_SQRT",
            message="Formula contains literal 'sqrt' characters rather than standard LaTeX '\\sqrt'. Human verification advised.",
            severity=ValidationSeverity.WARNING,
        ))

    if "+-" in expr or "+ -" in expr:
        issues.append(FormulaIssue(
            code="SUSPICIOUS_ASCII_PLUSMINUS",
            message="Formula contains ASCII '+-' instead of standard LaTeX '\\pm'. Human verification advised.",
            severity=ValidationSeverity.WARNING,
        ))

    # 6. Confidence Score threshold check
    if confidence_score < confidence_threshold:
        issues.append(FormulaIssue(
            code="LOW_CONFIDENCE",
            message=f"Model confidence score ({confidence_score:.2f}) is below threshold ({confidence_threshold:.2f}).",
            severity=ValidationSeverity.INFO,
        ))

    # Determination
    has_errors = any(i.severity == ValidationSeverity.ERROR for i in issues)
    needs_review = has_errors or bool(issues) or (confidence_score < confidence_threshold)

    return FormulaValidationResult(
        is_valid=not has_errors,
        latex_expression=expr,
        confidence_score=confidence_score,
        issues=issues,
        bracket_counts=counts,
        has_unbalanced_braces=unbalanced_braces,
        has_unbalanced_parens=unbalanced_parens,
        has_unbalanced_brackets=unbalanced_brackets,
        has_unbalanced_left_right=unbalanced_lr,
        needs_human_review=needs_review,
    )
