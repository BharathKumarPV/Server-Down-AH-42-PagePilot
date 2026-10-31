"""
Pytest configuration for PagePilot tests.
Provides fine-grained OCR test isolation:
- Synthetic OCR tests receive scoped simulation mode during their test run
- Unrelated application tests (chat, docx, rag, voice, etc.) run in clean default environment
- Production-mode OCR tests run with simulation strictly disabled
"""

import os
import pytest


@pytest.fixture
def ocr_simulation():
    """Explicit fixture enabling OCR simulation mode for synthetic unit/integration tests."""
    old_val = os.environ.get("PAGEPILOT_OCR_SIMULATION_MODE")
    os.environ["PAGEPILOT_OCR_SIMULATION_MODE"] = "1"
    yield
    if old_val is None:
        os.environ.pop("PAGEPILOT_OCR_SIMULATION_MODE", None)
    else:
        os.environ["PAGEPILOT_OCR_SIMULATION_MODE"] = old_val


@pytest.fixture
def ocr_production_mode():
    """Explicit fixture disabling OCR simulation mode for production verification."""
    old_val = os.environ.get("PAGEPILOT_OCR_SIMULATION_MODE")
    os.environ["PAGEPILOT_OCR_SIMULATION_MODE"] = "0"
    yield
    if old_val is None:
        os.environ.pop("PAGEPILOT_OCR_SIMULATION_MODE", None)
    else:
        os.environ["PAGEPILOT_OCR_SIMULATION_MODE"] = old_val


def pytest_runtest_setup(item):
    """
    Isolate OCR simulation strictly to legacy synthetic OCR tests.
    Unrelated application tests (chat, docx, rag, voice, etc.) and
    explicit production tests run with simulation disabled.
    """
    if "test_ocr" in item.nodeid and "production" not in item.name.lower():
        item._orig_ocr_sim = os.environ.get("PAGEPILOT_OCR_SIMULATION_MODE")
        os.environ["PAGEPILOT_OCR_SIMULATION_MODE"] = "1"


def pytest_runtest_teardown(item):
    """Restore original environment after test execution."""
    if hasattr(item, "_orig_ocr_sim"):
        if item._orig_ocr_sim is None:
            os.environ.pop("PAGEPILOT_OCR_SIMULATION_MODE", None)
        else:
            os.environ["PAGEPILOT_OCR_SIMULATION_MODE"] = item._orig_ocr_sim
