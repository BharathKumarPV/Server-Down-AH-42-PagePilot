"""
PagePilot – DOCX Creator screen.

All document generation is a placeholder in Stage 1.
No python-docx or similar library is imported.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QScrollArea,
    QSizePolicy, QVBoxLayout, QWidget,
)

from pagepilot.theme import Colors, Fonts, Spacing
from pagepilot.ui.components import (
    Card, DropDown, PlannedBanner, PrimaryButton,
    SecondaryButton, TextArea, TextInput,
)


_DOC_TYPES = [
    "Business Letter",
    "Meeting Minutes",
    "Report",
    "Invoice",
    "Proposal",
    "Cover Letter",
    "Resume / CV",
    "Technical Document",
]


class DocxScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("DocxScreen")
        self.setStyleSheet(f"background-color: {Colors.BG_APP};")
        self._save_path: str | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header
        root.addWidget(self._build_header())

        # Scrollable body
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background: transparent; border: none;")

        body = QWidget()
        body.setStyleSheet(f"background-color: {Colors.BG_APP};")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(Spacing.XL, Spacing.XL, Spacing.XL, Spacing.XL)
        body_layout.setSpacing(Spacing.LG)

        body_layout.addWidget(PlannedBanner(
            "DOCX generation is not yet implemented. "
            "Document creation will be available in a future stage."
        ))

        body_layout.addWidget(self._build_input_card())
        body_layout.addWidget(self._build_options_card())
        body_layout.addStretch()

        scroll.setWidget(body)
        root.addWidget(scroll, 1)

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setFixedHeight(56)
        header.setObjectName("DocxHeader")
        header.setStyleSheet(f"""
            QWidget#DocxHeader {{
                background-color: {Colors.BG_SURFACE};
                border-bottom: 1px solid {Colors.BORDER};
            }}
        """)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(Spacing.LG, 0, Spacing.LG, 0)
        title = QLabel("📝  DOCX Creator")
        title.setFont(Fonts.make(Fonts.SIZE_MD, QFont.Weight.DemiBold))
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; background: transparent; border: none;")
        layout.addWidget(title)
        return header

    def _build_input_card(self) -> Card:
        card = Card(title="Document Content")
        layout = card.body_layout()

        # Document type
        type_row = QWidget()
        type_row.setStyleSheet("background: transparent;")
        tr_layout = QHBoxLayout(type_row)
        tr_layout.setContentsMargins(0, 0, 0, 0)
        tr_layout.setSpacing(Spacing.MD)

        type_label = QLabel("Document type:")
        type_label.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_SM}px;"
            f" background: transparent; border: none; min-width: 130px;"
        )
        self._type_dropdown = DropDown(_DOC_TYPES)
        tr_layout.addWidget(type_label)
        tr_layout.addWidget(self._type_dropdown, 1)
        layout.addWidget(type_row)

        # Prompt / description
        prompt_label = QLabel("Describe what you need:")
        prompt_label.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_SM}px;"
            f" font-weight: 600; background: transparent; border: none;"
        )
        self._prompt_area = TextArea(
            "Describe the document you want to create. "
            "For example: 'A business letter to a supplier requesting a price quote for 500 units of product X.'"
        )
        self._prompt_area.setMinimumHeight(140)
        layout.addWidget(prompt_label)
        layout.addWidget(self._prompt_area)
        return card

    def _build_options_card(self) -> Card:
        card = Card(title="Generate & Save")
        layout = card.body_layout()

        # Save location row
        save_row = QWidget()
        save_row.setStyleSheet("background: transparent;")
        sr_layout = QHBoxLayout(save_row)
        sr_layout.setContentsMargins(0, 0, 0, 0)
        sr_layout.setSpacing(Spacing.MD)

        self._save_label = QLabel("Save location: not selected")
        self._save_label.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_SM}px;"
            f" background: transparent; border: none;"
        )
        self._save_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        browse_btn = SecondaryButton("📁  Choose Folder")
        browse_btn.setFixedWidth(160)
        browse_btn.clicked.connect(self._on_browse_save)

        sr_layout.addWidget(self._save_label, 1)
        sr_layout.addWidget(browse_btn)
        layout.addWidget(save_row)

        # Generate button
        gen_btn = PrimaryButton("📝  Generate DOCX  (coming soon)")
        gen_btn.setEnabled(False)
        layout.addWidget(gen_btn)

        return card

    def _on_browse_save(self):
        path = QFileDialog.getExistingDirectory(self, "Choose save folder")
        if path:
            self._save_path = path
            self._save_label.setText(f"Save to: {path}")
            self._save_label.setStyleSheet(
                f"color: {Colors.TEXT_PRIMARY}; font-size: {Fonts.SIZE_SM}px;"
                f" background: transparent; border: none;"
            )

