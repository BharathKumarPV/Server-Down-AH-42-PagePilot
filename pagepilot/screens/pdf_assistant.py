"""
PagePilot – PDF Assistant screen.

All PDF processing features are placeholders in Stage 1.
No PDF library is imported; no file is processed.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFileDialog, QFrame, QHBoxLayout, QLabel,
    QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from pagepilot.theme import Colors, Fonts, Radius, Spacing
from pagepilot.ui.components import (
    Card, Divider, PlannedBanner, PrimaryButton,
    SecondaryButton, TextArea, TextInput,
)


class PdfScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("PdfScreen")
        self.setStyleSheet(f"background-color: {Colors.BG_APP};")
        self._selected_path: str | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header ─────────────────────────────────────────────────
        header = self._build_header()
        root.addWidget(header)

        # ── Scrollable body ────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background: transparent; border: none;")

        body = QWidget()
        body.setStyleSheet(f"background-color: {Colors.BG_APP};")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(Spacing.XL, Spacing.XL, Spacing.XL, Spacing.XL)
        body_layout.setSpacing(Spacing.LG)

        # Not-implemented banner
        body_layout.addWidget(PlannedBanner(
            "PDF processing is not yet implemented. "
            "You can select a file, but no analysis will be performed in Stage 1."
        ))

        # Upload card
        body_layout.addWidget(self._build_upload_card())

        # Document info card
        self._doc_info_card = self._build_doc_info_card()
        body_layout.addWidget(self._doc_info_card)

        # Q&A card
        body_layout.addWidget(self._build_qa_card())

        body_layout.addStretch()
        scroll.setWidget(body)
        root.addWidget(scroll, 1)

    # ── Header ───────────────────────────────────────────────────────
    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setFixedHeight(56)
        header.setObjectName("PdfHeader")
        header.setStyleSheet(f"""
            QWidget#PdfHeader {{
                background-color: {Colors.BG_SURFACE};
                border-bottom: 1px solid {Colors.BORDER};
            }}
        """)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(Spacing.LG, 0, Spacing.LG, 0)
        title = QLabel("📄  PDF Assistant")
        title.setFont(Fonts.make(Fonts.SIZE_MD, QFont.Weight.DemiBold))
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent; border: none;")
        layout.addWidget(title)
        return header

    # ── Upload card ──────────────────────────────────────────────────
    def _build_upload_card(self) -> Card:
        card = Card(title="Upload Document")
        layout = card.body_layout()

        row = QWidget()
        row.setStyleSheet("background: transparent;")
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(Spacing.MD)

        self._file_label = QLabel("No file selected")
        self._file_label.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_SM}px;"
            f" background: transparent; border: none;"
        )
        self._file_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        browse_btn = SecondaryButton("📂  Choose PDF")
        browse_btn.setFixedWidth(160)
        browse_btn.clicked.connect(self._on_browse)

        row_layout.addWidget(self._file_label, 1)
        row_layout.addWidget(browse_btn)
        layout.addWidget(row)
        return card

    # ── Document info card ───────────────────────────────────────────
    def _build_doc_info_card(self) -> Card:
        card = Card(title="Document Information")
        layout = card.body_layout()

        self._doc_name_lbl = QLabel("—")
        self._doc_name_lbl.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_SM}px;"
            f" background: transparent; border: none;"
        )
        self._doc_name_lbl.setWordWrap(True)
        layout.addWidget(self._doc_name_lbl)

        layout.addWidget(Divider())

        # Summarise action
        self._summarise_btn = PrimaryButton("✨  Summarise Document")
        self._summarise_btn.setEnabled(False)
        self._summarise_btn.clicked.connect(self._on_summarise)
        layout.addWidget(self._summarise_btn)

        self._summary_area = TextArea(
            "Summary will appear here once PDF processing is implemented…",
            read_only=True,
        )
        self._summary_area.setMinimumHeight(120)
        layout.addWidget(self._summary_area)

        # Source reference
        src_label = QLabel("Source references:")
        src_label.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_SM}px;"
            f" font-weight: 600; background: transparent; border: none;"
        )
        self._source_area = TextArea(
            "Relevant pages and sections will be listed here…",
            read_only=True,
        )
        self._source_area.setMaximumHeight(80)
        layout.addWidget(src_label)
        layout.addWidget(self._source_area)
        return card

    # ── Q&A card ─────────────────────────────────────────────────────
    def _build_qa_card(self) -> Card:
        card = Card(title="Ask a Question")
        layout = card.body_layout()

        self._question_input = TextInput("Ask a question about the document…")
        self._ask_btn = PrimaryButton("Ask")
        self._ask_btn.setEnabled(False)
        self._ask_btn.setFixedWidth(100)

        row = QWidget()
        row.setStyleSheet("background: transparent;")
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(Spacing.SM)
        row_layout.addWidget(self._question_input, 1)
        row_layout.addWidget(self._ask_btn)
        layout.addWidget(row)

        self._answer_area = TextArea(
            "The answer will appear here once AI integration is implemented…",
            read_only=True,
        )
        self._answer_area.setMinimumHeight(100)
        layout.addWidget(self._answer_area)
        return card

    # ── Slots ─────────────────────────────────────────────────────────
    def _on_browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a PDF file", "", "PDF Files (*.pdf)"
        )
        if path:
            self._selected_path = path
            fname = path.split("/")[-1].split("\\")[-1]
            self._file_label.setText(f"📄  {fname}")
            self._file_label.setStyleSheet(
                f"color: {Colors.TEXT_PRIMARY}; font-size: {Fonts.SIZE_SM}px;"
                f" background: transparent; border: none;"
            )
            self._doc_name_lbl.setText(
                f"Selected: {fname}\n"
                f"(PDF analysis is not available in Stage 1)"
            )
            self._summarise_btn.setEnabled(False)   # stay disabled — no backend
            self._ask_btn.setEnabled(False)

    def _on_summarise(self):
        # Intentionally left inert — no backend in Stage 1
        pass

