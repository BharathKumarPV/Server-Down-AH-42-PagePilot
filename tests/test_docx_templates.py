"""
Tests for PagePilot DOCX templates (Stage 5).
"""

import pytest
from pagepilot.docx.templates import (
    ACADEMIC_SUMMARY,
    CUSTOM_DOCUMENT,
    PROJECT_REPORT,
    QA_NOTES,
    SIMPLE_STUDY_NOTES,
    TEMPLATE_NAMES,
    TEMPLATES,
    TemplateConfig,
    get_template,
)


def test_template_registry():
    assert len(TEMPLATES) == 5
    expected_names = [
        "Simple Study Notes",
        "Academic Summary",
        "Question and Answer Notes",
        "Project Report",
        "Custom Document",
    ]
    for name in expected_names:
        assert name in TEMPLATES
        assert name in TEMPLATE_NAMES


def test_template_properties():
    for name, tmpl in TEMPLATES.items():
        assert isinstance(tmpl, TemplateConfig)
        assert tmpl.name == name
        assert len(tmpl.description) > 10
        assert tmpl.font_primary in ("Calibri", "Georgia", "Arial", "Segoe UI")
        assert tmpl.margin_top_inches > 0
        assert tmpl.margin_left_inches > 0
        assert tmpl.h1_size_pt > tmpl.h2_size_pt > tmpl.body_font_size_pt


def test_project_report_cover_page():
    assert PROJECT_REPORT.has_cover_page is True
    assert SIMPLE_STUDY_NOTES.has_cover_page is False
    assert ACADEMIC_SUMMARY.has_cover_page is False


def test_get_template():
    # Exact lookup
    tmpl = get_template("Academic Summary")
    assert tmpl.name == "Academic Summary"

    # Fallback for unknown or empty
    assert get_template("Nonexistent").name == "Simple Study Notes"
    assert get_template(None).name == "Simple Study Notes"
    assert get_template("").name == "Simple Study Notes"

