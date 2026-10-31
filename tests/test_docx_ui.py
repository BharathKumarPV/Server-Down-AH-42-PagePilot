"""
Tests for PagePilot DocxScreen UI component (Stage 5).
"""

from pathlib import Path
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from pagepilot.db.database import DocumentDB
from pagepilot.pdf.models import Chunk
from pagepilot.screens.docx_creator import DocxScreen


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "test_docs_ui.db"
    db = DocumentDB(db_path=db_file)
    doc = db.create_document(
        filename="Handbook_2026.pdf",
        file_path=str(tmp_path / "Handbook_2026.pdf"),
        sha256_hash="hb123hash",
        page_count=3,
        embed_model="nomic-embed-text",
    )
    doc_id = doc.id
    chunks_data = [
        Chunk(
            chunk_id=f"hb_c{i}",
            document_id=doc_id,
            filename="Handbook_2026.pdf",
            page_number=i,
            chunk_index=i - 1,
            text=f"Handbook page {i} essential documentation rules and guidelines.",
            token_count=12,
            section_heading=f"Chapter {i}",
        )
        for i in range(1, 4)
    ]
    db.insert_chunks(chunks_data)
    return db, doc_id


def test_docx_screen_init(qapp, test_db):
    db, doc_id = test_db
    screen = DocxScreen(db=db)

    # Check header and initial controls
    assert screen.objectName() == "DocxScreen"
    assert screen._doc_combo.count() >= 2  # (No PDF / Manual Input) + Handbook_2026.pdf
    assert screen._template_combo.count() == 5
    assert screen._type_combo.count() == 6
    assert screen._editor is not None
    assert screen._export_btn.isEnabled() is True


def test_docx_screen_doc_type_change(qapp, test_db):
    db, doc_id = test_db
    screen = DocxScreen(db=db)
    screen.show()

    # Change to "Selected PDF Content"
    idx = screen._type_combo.findText("Selected PDF Content")
    screen._type_combo.setCurrentIndex(idx)
    assert not screen._pages_input.isHidden()

    # Change back to "Study Notes"
    idx2 = screen._type_combo.findText("Study Notes")
    screen._type_combo.setCurrentIndex(idx2)
    assert screen._pages_input.isHidden()
    assert screen._template_combo.currentText() == "Simple Study Notes"


def test_docx_screen_draft_and_stats(qapp, test_db):
    db, doc_id = test_db
    screen = DocxScreen(db=db)

    # Select document in dropdown
    doc_idx = screen._doc_combo.findData(doc_id)
    assert doc_idx >= 0
    screen._doc_combo.setCurrentIndex(doc_idx)

    # Select "Study Notes"
    screen._type_combo.setCurrentIndex(screen._type_combo.findText("Study Notes"))

    # Draft content
    screen._on_draft_content()

    editor_text = screen._editor.toPlainText()
    assert "Study Notes: Handbook_2026" in editor_text
    assert "Chapter" in editor_text or "Handbook page" in editor_text
    assert "Words:" in screen._stats_label.text()


def test_docx_screen_export_flow(qapp, test_db, tmp_path):
    db, doc_id = test_db
    screen = DocxScreen(db=db)
    screen._output_dir = tmp_path

    # Set document and draft content
    doc_idx = screen._doc_combo.findData(doc_id)
    screen._doc_combo.setCurrentIndex(doc_idx)
    screen._type_combo.setCurrentIndex(screen._type_combo.findText("PDF Summary"))
    screen._on_draft_content()

    # Trigger export
    screen._on_export_docx()
    # Wait for background worker to complete
    if screen._worker:
        screen._worker.wait(5000)
        qapp.processEvents()

    assert screen._last_exported_path is not None
    assert Path(screen._last_exported_path).exists()
    assert screen._open_btn.isEnabled() is True
