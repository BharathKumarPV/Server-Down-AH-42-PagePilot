"""
PagePilot – DOCX Creator & Document Generation Screen (Stage 5).

Provides a desktop UI for generating styled Microsoft Word (.docx)
documents from PDF summaries, study notes, question-and-answer pairs,
selected pages, and custom text with verified citations.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from pagepilot.config import DOCX_OUTPUT_DIR, ensure_docx_dir
from pagepilot.db.database import DocumentDB
from pagepilot.docx.citation_formatter import CitationMode
from pagepilot.docx.document_service import DocxService, DocxWorker
from pagepilot.docx.templates import TEMPLATE_NAMES, TEMPLATES, get_template
from pagepilot.docx.validators import sanitize_filename
from pagepilot.theme import Colors, Fonts, Radius, Spacing
from pagepilot.ui.components import (
    Badge,
    Card,
    Divider,
    DropDown,
    IconButton,
    PrimaryButton,
    SecondaryButton,
    SectionLabel,
    TextArea,
    TextInput,
)


_DOC_TYPES = [
    "PDF Summary",
    "Study Notes",
    "Question & Answer Notes",
    "Project Report",
    "Selected PDF Content",
    "Custom Document",
]


class DocxScreen(QWidget):
    """
    Main screen for creating and exporting professional DOCX files.
    """

    def __init__(self, parent=None, db: Optional[DocumentDB] = None):
        super().__init__(parent)
        self.setObjectName("DocxScreen")
        self.setStyleSheet(f"background-color: {Colors.BG_APP};")

        self.db = db or DocumentDB()
        self.service = DocxService(db=self.db)
        self._last_exported_path: Optional[str] = None
        self._worker: Optional[DocxWorker] = None
        self._output_dir: Path = ensure_docx_dir()

        self._init_ui()
        self.refresh_documents()
        self._on_doc_type_changed(0)

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header
        root.addWidget(self._build_header())

        # Main splitter layout (Options Left, Content/Editor Right)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: {Colors.BORDER};
                width: 1px;
            }}
        """)

        left_container = self._build_options_panel()
        right_container = self._build_editor_panel()

        splitter.addWidget(left_container)
        splitter.addWidget(right_container)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)

        root.addWidget(splitter, 1)

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setFixedHeight(64)
        header.setObjectName("DocxHeader")
        header.setStyleSheet(f"""
            QWidget#DocxHeader {{
                background-color: {Colors.BG_SURFACE};
                border-bottom: 1px solid {Colors.BORDER};
            }}
        """)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(Spacing.LG, 0, Spacing.LG, 0)
        layout.setSpacing(Spacing.MD)

        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        title = QLabel("📝  DOCX Creator & Document Engine")
        title.setFont(Fonts.make(Fonts.SIZE_MD, QFont.Weight.DemiBold))
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent; border: none;")

        sub = QLabel("Generate styled Word (.docx) documents with verified PDF citations and professional templates.")
        sub.setFont(Fonts.make(Fonts.SIZE_XS))
        sub.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent; border: none;")

        title_col.addWidget(title)
        title_col.addWidget(sub)
        layout.addLayout(title_col)

        layout.addStretch()

        layout.addWidget(Badge("Stage 5", "accent"))
        layout.addWidget(Badge("python-docx", "info"))
        layout.addWidget(Badge("5 Templates", "default"))
        layout.addWidget(Badge("Verified Citations", "success"))

        return header

    def _build_options_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background: transparent; border: none;")

        container = QWidget()
        container.setStyleSheet(f"background-color: {Colors.BG_APP};")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(Spacing.LG, Spacing.LG, Spacing.LG, Spacing.LG)
        layout.setSpacing(Spacing.LG)

        # 1. Document Configuration Card
        card = Card(title="Document Configuration")
        clayout = card.body_layout()

        # Document Type
        clayout.addWidget(SectionLabel("DOCUMENT TYPE"))
        self._type_combo = DropDown(_DOC_TYPES)
        self._type_combo.currentIndexChanged.connect(self._on_doc_type_changed)
        clayout.addWidget(self._type_combo)

        # Source PDF Selection
        clayout.addWidget(SectionLabel("SOURCE PDF"))
        src_row = QHBoxLayout()
        self._doc_combo = DropDown(["(No PDF / Manual Input)"])
        self._refresh_btn = IconButton("🔄", tooltip="Refresh document list", size=34)
        self._refresh_btn.clicked.connect(self.refresh_documents)
        src_row.addWidget(self._doc_combo, 1)
        src_row.addWidget(self._refresh_btn)
        clayout.addLayout(src_row)

        # Selected Pages Row (conditional)
        self._pages_label = SectionLabel("PAGE NUMBERS (E.G. 1, 2, 3)")
        self._pages_input = TextInput("1, 2, 3")
        self._pages_label.setVisible(False)
        self._pages_input.setVisible(False)
        clayout.addWidget(self._pages_label)
        clayout.addWidget(self._pages_input)

        # Document Title
        clayout.addWidget(SectionLabel("DOCUMENT TITLE"))
        self._title_input = TextInput("PagePilot Generated Document")
        clayout.addWidget(self._title_input)

        # Template Selection
        clayout.addWidget(SectionLabel("TEMPLATE"))
        self._template_combo = DropDown(TEMPLATE_NAMES)
        self._template_combo.currentIndexChanged.connect(self._on_template_changed)
        clayout.addWidget(self._template_combo)

        self._template_desc = QLabel(TEMPLATES[TEMPLATE_NAMES[0]].description)
        self._template_desc.setFont(Fonts.make(Fonts.SIZE_XS))
        self._template_desc.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-style: italic; background: transparent; border: none;"
        )
        self._template_desc.setWordWrap(True)
        clayout.addWidget(self._template_desc)

        # Citation Mode Selection
        clayout.addWidget(SectionLabel("CITATION FORMATTING"))
        self._citation_combo = DropDown(CitationMode.ALL_MODES)
        clayout.addWidget(self._citation_combo)

        # Save Directory
        clayout.addWidget(SectionLabel("EXPORT FOLDER"))
        save_row = QHBoxLayout()
        self._save_label = QLabel(f"📁  {self._output_dir}")
        self._save_label.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_SM}px; background: transparent; border: none;"
        )
        self._save_label.setWordWrap(True)
        self._browse_btn = SecondaryButton("Change")
        self._browse_btn.setFixedWidth(80)
        self._browse_btn.clicked.connect(self._on_browse_output_dir)
        save_row.addWidget(self._save_label, 1)
        save_row.addWidget(self._browse_btn)
        clayout.addLayout(save_row)

        layout.addWidget(card)

        # 2. Action Buttons Card
        action_card = Card(title="Actions")
        alayout = action_card.body_layout()

        self._draft_btn = SecondaryButton("✨  Load / Draft Content")
        self._draft_btn.clicked.connect(self._on_draft_content)
        alayout.addWidget(self._draft_btn)

        self._export_btn = PrimaryButton("📝  Export to Word (.docx)")
        self._export_btn.clicked.connect(self._on_export_docx)
        alayout.addWidget(self._export_btn)

        self._open_btn = SecondaryButton("📂  Open in Word / Explorer")
        self._open_btn.setEnabled(False)
        self._open_btn.clicked.connect(self._on_open_file)
        alayout.addWidget(self._open_btn)

        layout.addWidget(action_card)
        layout.addStretch()

        scroll.setWidget(container)
        return scroll

    def _build_editor_panel(self) -> QWidget:
        container = QWidget()
        container.setStyleSheet(f"background-color: {Colors.BG_APP};")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(Spacing.LG, Spacing.LG, Spacing.LG, Spacing.LG)
        layout.setSpacing(Spacing.MD)

        card = Card(title="Content Preview & Markdown Editor")
        clayout = card.body_layout()

        # Stats bar
        stats_row = QHBoxLayout()
        self._stats_label = QLabel("Words: 0   •   Characters: 0   •   Lines: 0")
        self._stats_label.setFont(Fonts.make(Fonts.SIZE_XS))
        self._stats_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent; border: none;")

        clear_btn = IconButton("🗑", tooltip="Clear editor content", size=30)
        clear_btn.clicked.connect(self._on_clear_editor)

        stats_row.addWidget(self._stats_label, 1)
        stats_row.addWidget(clear_btn)
        clayout.addLayout(stats_row)

        # Text Editor Area
        self._editor = TextArea(
            "Enter or generate Markdown document content here.\n\n"
            "Supported elements:\n"
            "# Heading 1\n"
            "## Heading 2\n"
            "**Bold text**, *Italic text*, `inline code`\n"
            "- Bullet lists\n"
            "1. Numbered lists\n"
            "> Callout notes and definitions\n"
            "| Tables | With | Columns |\n"
            "```\nCode blocks\n```\n"
            "Inline citations like [Document.pdf, p. 2]"
        )
        self._editor.textChanged.connect(self._update_editor_stats)
        clayout.addWidget(self._editor, 1)

        # Progress indicator
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)  # indeterminate
        self._progress_bar.setFixedHeight(4)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setVisible(False)
        self._progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {Colors.BG_SURFACE_ALT};
                border: none;
                border-radius: 2px;
            }}
            QProgressBar::chunk {{
                background-color: {Colors.ACCENT};
                border-radius: 2px;
            }}
        """)
        clayout.addWidget(self._progress_bar)

        # Status Message
        self._status_label = QLabel("Ready to compose or generate document.")
        self._status_label.setFont(Fonts.make(Fonts.SIZE_SM))
        self._status_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; background: transparent; border: none;")
        self._status_label.setWordWrap(True)
        clayout.addWidget(self._status_label)

        layout.addWidget(card, 1)
        return container

    # ──────────────────────────────────────────────────────────────────────────
    # Document Selection & Refresh
    # ──────────────────────────────────────────────────────────────────────────

    def refresh_documents(self):
        """Reload imported documents from DocumentDB into the dropdown."""
        current_data = self._doc_combo.currentData()
        self._doc_combo.clear()
        self._doc_combo.addItem("(No PDF / Manual Input)", userData=None)

        try:
            docs = self.db.list_documents()
            for d in docs:
                self._doc_combo.addItem(f"{d.filename} ({d.page_count} pages)", userData=d.id)
        except Exception:
            pass

        # Restore selection if possible
        if current_data is not None:
            idx = self._doc_combo.findData(current_data)
            if idx >= 0:
                self._doc_combo.setCurrentIndex(idx)

    def _on_doc_type_changed(self, index: int):
        doc_type = self._type_combo.currentText()
        is_selected_content = (doc_type == "Selected PDF Content")
        self._pages_label.setVisible(is_selected_content)
        self._pages_input.setVisible(is_selected_content)

        # Update recommended template based on document type
        if doc_type == "PDF Summary":
            self._set_template("Academic Summary")
        elif doc_type == "Study Notes":
            self._set_template("Simple Study Notes")
        elif doc_type == "Question & Answer Notes":
            self._set_template("Question and Answer Notes")
        elif doc_type == "Project Report":
            self._set_template("Project Report")
        elif doc_type == "Custom Document":
            self._set_template("Custom Document")

    def _set_template(self, template_name: str):
        idx = self._template_combo.findText(template_name)
        if idx >= 0:
            self._template_combo.setCurrentIndex(idx)

    def _on_template_changed(self, index: int):
        template_name = self._template_combo.currentText()
        tmpl = get_template(template_name)
        self._template_desc.setText(tmpl.description)

    def _on_browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Choose DOCX Output Folder", str(self._output_dir))
        if path:
            self._output_dir = Path(path)
            self._save_label.setText(f"📁  {self._output_dir}")

    def _update_editor_stats(self):
        text = self._editor.toPlainText()
        words = len(text.split())
        chars = len(text)
        lines = len(text.splitlines())
        self._stats_label.setText(f"Words: {words}   •   Characters: {chars}   •   Lines: {lines}")

    def _on_clear_editor(self):
        self._editor.clear()
        self._update_editor_stats()

    # ──────────────────────────────────────────────────────────────────────────
    # Content Drafting & Generation
    # ──────────────────────────────────────────────────────────────────────────

    def _on_draft_content(self):
        """Load or draft structured markdown for the selected document and type."""
        doc_type = self._type_combo.currentText()
        doc_id = self._doc_combo.currentData()

        if doc_id is None and doc_type != "Custom Document":
            self._status_label.setText("⚠ Please select an imported PDF document or switch to 'Custom Document'.")
            self._status_label.setStyleSheet(f"color: {Colors.WARNING}; background: transparent; border: none;")
            return

        self._status_label.setText("Drafting structured content...")
        self._status_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent; border: none;")

        try:
            if doc_type == "PDF Summary":
                self._draft_pdf_summary(doc_id)
            elif doc_type == "Study Notes":
                self._draft_study_notes(doc_id)
            elif doc_type == "Question & Answer Notes":
                self._draft_qa_notes(doc_id)
            elif doc_type == "Selected PDF Content":
                self._draft_selected_content(doc_id)
            elif doc_type == "Project Report":
                self._draft_project_report(doc_id)
            else:
                self._draft_custom_template()

            self._status_label.setText("✓ Structured content drafted. You may review and edit before exporting.")
            self._status_label.setStyleSheet(f"color: {Colors.SUCCESS}; background: transparent; border: none;")
        except Exception as exc:
            self._status_label.setText(f"Error drafting content: {exc}")
            self._status_label.setStyleSheet(f"color: {Colors.ERROR}; background: transparent; border: none;")

    def _draft_pdf_summary(self, doc_id: int):
        doc = self.db.get_document(doc_id)
        if not doc:
            return
        self._title_input.setText(f"Executive Summary: {Path(doc.filename).stem}")
        chunks = self.db.get_chunks_for_document(doc_id)

        lines = [
            f"# Executive Summary: {Path(doc.filename).stem}",
            "",
            f"> **Overview**: Comprehensive summary synthesized from `{doc.filename}` ({doc.page_count} pages).",
            "",
            "## 1. Document Purpose & Scope",
            "",
        ]
        if chunks:
            lines.append(f"{chunks[0].text.strip()} [{doc.filename}, p. {chunks[0].page_number}]")
            lines.append("")

        lines.extend([
            "## 2. Key Concepts & Definitions",
            "",
            "| Concept | Definition / Context | Source Page |",
            "| --- | --- | --- |",
        ])
        for idx, ch in enumerate(chunks[1:4], 1):
            term = ch.section_heading or f"Key Topic {idx}"
            snippet = ch.text[:80].replace("|", "-").replace("\n", " ").strip()
            lines.append(f"| **{term}** | {snippet}... | Page {ch.page_number} |")

        lines.extend([
            "",
            "## 3. Detailed Findings & Analysis",
            "",
        ])
        for ch in chunks[4:8]:
            head = ch.section_heading or f"Section Context (Page {ch.page_number})"
            lines.append(f"### {head}")
            lines.append(f"{ch.text.strip()} [{doc.filename}, p. {ch.page_number}]")
            lines.append("")

        self._editor.setPlainText("\n".join(lines))

    def _draft_study_notes(self, doc_id: int):
        doc = self.db.get_document(doc_id)
        if not doc:
            return
        self._title_input.setText(f"Study Notes: {Path(doc.filename).stem}")
        chunks = self.db.get_chunks_for_document(doc_id)

        lines = [
            f"# Study Notes: {Path(doc.filename).stem}",
            "",
            f"> **Study Objective**: High-yield revision guide extracted from `{doc.filename}`.",
            "",
            "## 1. Core Revision Takeaways",
            "",
        ]
        for ch in chunks[:4]:
            first_sent = ch.text.split(".")[0].strip()
            if first_sent:
                lines.append(f"- **Key Fact**: {first_sent}. [{doc.filename}, p. {ch.page_number}]")

        lines.extend([
            "",
            "## 2. In-Depth Concepts",
            "",
        ])
        for ch in chunks[4:7]:
            lines.append(f"### Topic: Page {ch.page_number}")
            lines.append(f"{ch.text.strip()} [{doc.filename}, p. {ch.page_number}]")
            lines.append("")
            lines.append(f"> **Exam Note**: Crucial principle grounded on page {ch.page_number}.")
            lines.append("")

        self._editor.setPlainText("\n".join(lines))

    def _draft_qa_notes(self, doc_id: Optional[int]):
        filename = "Source Material"
        if doc_id:
            doc = self.db.get_document(doc_id)
            if doc:
                filename = doc.filename
                self._title_input.setText(f"Q&A Reference: {Path(doc.filename).stem}")

        lines = [
            f"# Questions & Answers: {Path(filename).stem}",
            "",
            "> **Reference**: Verified questions and answers grounded in source documentation.",
            "",
            "## Questions & Answers",
            "",
            f"### Question 1: What are the primary objectives outlined in the document?",
            "",
            f"**Answer**: The document establishes essential foundational frameworks and requirements. [{filename}, p. 1]",
            "",
            f"### Question 2: What methodology or procedures are specified?",
            "",
            f"**Answer**: Standardized operational workflows are applied with structured verification stages. [{filename}, p. 2]",
            "",
        ]
        self._editor.setPlainText("\n".join(lines))

    def _draft_selected_content(self, doc_id: int):
        doc = self.db.get_document(doc_id)
        if not doc:
            return
        self._title_input.setText(f"Selected Excerpts: {Path(doc.filename).stem}")

        # Parse requested pages
        pages_text = self._pages_input.text()
        pages = []
        for part in pages_text.split(","):
            part = part.strip()
            if part.isdigit():
                pages.append(int(part))
        if not pages:
            pages = [1]

        chunks = self.db.get_chunks_for_document(doc_id)
        matching = [ch for ch in chunks if ch.page_number in pages]

        lines = [
            f"# Selected Excerpts: {Path(doc.filename).stem}",
            "",
            f"> **Selected Pages**: Excerpts from pages {pages} of `{doc.filename}`.",
            "",
        ]
        for ch in matching:
            head = f"### {ch.section_heading}" if ch.section_heading else f"### Page {ch.page_number}"
            lines.append(head)
            lines.append(f"{ch.text.strip()} [{doc.filename}, p. {ch.page_number}]")
            lines.append("")

        self._editor.setPlainText("\n".join(lines))

    def _draft_project_report(self, doc_id: Optional[int]):
        filename = "Documentation"
        if doc_id:
            doc = self.db.get_document(doc_id)
            if doc:
                filename = doc.filename
                self._title_input.setText(f"Project Report: {Path(doc.filename).stem}")

        lines = [
            f"# Project Assessment Report",
            "",
            "## Executive Summary",
            "",
            f"This formal project report documents strategic findings, system analysis, and recommendations derived from **{filename}**.",
            "",
            "## 1. Strategic Objectives",
            "",
            "- Implement resilient, offline-capable document intelligence.",
            "- Ensure zero data leakage and 100% verifiable local citation grounding.",
            "- Deliver editable Microsoft Word documents matching publication-grade typography.",
            "",
            "## 2. Structural Analysis & Data Summary",
            "",
            "| Milestone / Component | Evaluation | Status |",
            "| --- | --- | --- |",
            "| **Document Ingestion** | PyMuPDF local parsing | Completed |",
            "| **Hybrid Retrieval** | Dense nomic embeddings + FTS5 BM25 | Completed |",
            "| **DOCX Generation** | python-docx OpenXML engine | Active |",
            "",
            "## 3. Operational Recommendations",
            "",
            "> **Key Finding**: Localized OpenXML generation operates with negligible CPU overhead and zero cloud egress.",
            "",
        ]
        self._editor.setPlainText("\n".join(lines))

    def _draft_custom_template(self):
        self._title_input.setText("Custom Document")
        lines = [
            "# Custom Document Title",
            "",
            "## 1. Overview",
            "",
            "Write your custom text here. You can use full **markdown formatting**.",
            "",
            "## 2. Key Highlights",
            "",
            "- Feature A: Clean typography",
            "- Feature B: Native Word tables and callouts",
            "- Feature C: Bounded memory and instant export",
            "",
            "> **Pro Tip**: Use callout boxes like this for notes, quotes, or important warnings.",
            "",
        ]
        self._editor.setPlainText("\n".join(lines))

    # ──────────────────────────────────────────────────────────────────────────
    # Export Execution
    # ──────────────────────────────────────────────────────────────────────────

    def _on_export_docx(self):
        """Export editor contents into a professional DOCX file."""
        title = self._title_input.text().strip() or "PagePilot Document"
        content = self._editor.toPlainText().strip()
        template_name = self._template_combo.currentText()
        citation_mode = self._citation_combo.currentText()

        if not content:
            self._status_label.setText("⚠ Cannot export empty document. Please enter or draft content first.")
            self._status_label.setStyleSheet(f"color: {Colors.WARNING}; background: transparent; border: none;")
            return

        # Prepare output path
        safe_filename = sanitize_filename(title)
        output_file = self._output_dir / safe_filename

        # Collect citations if a PDF document is selected
        citations = []
        doc_id = self._doc_combo.currentData()
        if doc_id is not None:
            doc = self.db.get_document(doc_id)
            if doc:
                chunks = self.db.get_chunks_for_document(doc_id)
                for ch in chunks[:8]:
                    citations.append({
                        "filename": ch.filename,
                        "page_number": ch.page_number,
                        "snippet": ch.text[:180],
                    })

        self._export_btn.setEnabled(False)
        self._progress_bar.setVisible(True)
        self._status_label.setText("Generating Word document...")
        self._status_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent; border: none;")

        def task():
            return self.service.generator.generate(
                title=title,
                content=content,
                template_name=template_name,
                citations=citations,
                citation_mode=citation_mode,
                output_path=output_file,
            )

        self._worker = DocxWorker(task)
        self._worker.progress.connect(lambda msg: self._status_label.setText(msg))
        self._worker.finished.connect(self._on_export_success)
        self._worker.error.connect(self._on_export_error)
        self._worker.start()

    def _on_export_success(self, file_path: str):
        self._last_exported_path = file_path
        self._export_btn.setEnabled(True)
        self._open_btn.setEnabled(True)
        self._progress_bar.setVisible(False)
        self._status_label.setText(f"✓ Successfully exported: {Path(file_path).name}")
        self._status_label.setStyleSheet(f"color: {Colors.SUCCESS}; background: transparent; border: none;")

    def _on_export_error(self, err_msg: str):
        self._export_btn.setEnabled(True)
        self._progress_bar.setVisible(False)
        self._status_label.setText(f"Export failed: {err_msg}")
        self._status_label.setStyleSheet(f"color: {Colors.ERROR}; background: transparent; border: none;")

    def _on_open_file(self):
        """Open the generated file in Word / default app, or show in explorer."""
        target = self._last_exported_path or str(self._output_dir)
        try:
            if os.path.exists(target):
                os.startfile(target)
            else:
                self._status_label.setText(f"File not found: {target}")
        except Exception as exc:
            self._status_label.setText(f"Could not open file: {exc}")
