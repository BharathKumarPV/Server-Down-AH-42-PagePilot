"""
Tests for PagePilot Query Planning & Decomposition Engine (Stage 6A).
"""

import pytest
from pagepilot.ai.query_planner import QueryPlanner


def test_simple_question_no_decomposition():
    planner = QueryPlanner()
    reqs = planner.plan("What is the maximum token limit for Qwen2.5?")
    assert len(reqs) == 1
    assert reqs[0].requirement_id == "R1"
    assert "maximum token limit" in reqs[0].subquery


def test_comparative_question_decomposition():
    planner = QueryPlanner()
    q = "Compare symmetric encryption and asymmetric encryption, including how their keys are used."
    reqs = planner.plan(q)

    assert len(reqs) >= 2
    subqueries = [r.subquery.lower() for r in reqs]
    # Check that both symmetric and asymmetric are represented
    assert any("symmetric" in s for s in subqueries)
    assert any("asymmetric" in s for s in subqueries)


def test_difference_between_decomposition():
    planner = QueryPlanner()
    q = "What is the difference between TCP and UDP?"
    reqs = planner.plan(q)

    assert len(reqs) == 2
    subqueries = [r.subquery.lower() for r in reqs]
    assert any("tcp" in s for s in subqueries)
    assert any("udp" in s for s in subqueries)


def test_versus_decomposition():
    planner = QueryPlanner()
    q = "REST versus GraphQL for API design"
    reqs = planner.plan(q)

    assert len(reqs) == 2
    subqueries = [r.subquery.lower() for r in reqs]
    assert any("rest" in s for s in subqueries)
    assert any("graphql" in s for s in subqueries)


def test_multi_sentence_question():
    planner = QueryPlanner()
    q = "What is the port number for HTTPS? What cipher suites are allowed?"
    reqs = planner.plan(q)

    assert len(reqs) == 2
    assert "https" in reqs[0].subquery.lower()
    assert "cipher" in reqs[1].subquery.lower()

