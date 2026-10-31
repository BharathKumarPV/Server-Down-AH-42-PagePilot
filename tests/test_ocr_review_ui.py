"""
Tests for Stage 6B OCR Review UI and Model Manager.
"""

from PySide6.QtWidgets import QApplication
import pytest

from pagepilot.db.database import DocumentDB
from pagepilot.ocr.model_manager import OCRModelManager, OCR_MODEL_REGISTRY
from pagepilot.ocr.models import ApprovalStatus
from pagepilot.ocr.service import OCRService
from pagepilot.retrieval.vector_store import VectorStore
from pagepilot.screens.ocr_review_dialog import OCRReviewDialog


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_ocr_model_manager_specs_and_status(tmp_path):
    mgr = OCRModelManager(base_dir=tmp_path / "models" / "ocr")
    specs = mgr.list_specs()
    assert len(specs) >= 5

    # Check spec keys
    assert "ppocrv5_det" in OCR_MODEL_REGISTRY
    assert "ppocrv5_server_rec" in OCR_MODEL_REGISTRY
    assert "pix2text_mfr" in OCR_MODEL_REGISTRY

    # Check initial status (uninstalled in empty temp directory)
    status = mgr.get_status()
    assert status["ppocrv5_det"]["installed"] is False

    # Simulate local weights installation
    mgr.install_placeholder_weights("ppocrv5_det")
    assert mgr.is_installed("ppocrv5_det") is True

    summary = mgr.get_storage_summary()
    assert summary["installed_models"] == 1


def test_ocr_review_dialog_instantiates(qapp, tmp_path):
    db = DocumentDB(db_path=tmp_path / "test_docs.db")
    vstore = VectorStore(persist_dir=tmp_path / "vstore")
    service = OCRService(db=db, vector_store=vstore)

    img_file = tmp_path / "printed_quadratic.png"
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05"
        b"\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    img_file.write_bytes(png_bytes)

    doc = service.import_image_note(str(img_file))
    dialog = OCRReviewDialog(ocr_service=service, document_id=doc.id, page_number=1)

    assert dialog.document_id == doc.id
    assert dialog.page_number == 1
    assert dialog.status_badge.text() in ("Pending Review", "Approved")
    assert dialog.region_combo.count() >= 1

    dialog.close()
