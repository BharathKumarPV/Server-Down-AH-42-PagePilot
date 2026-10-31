"""
PagePilot – Stage 6B Human-in-the-Loop OCR Review & Correction Dialog.

Enforces strict anti-hallucination policy:
- Left pane displays original scanned page or photographed note
- Right pane provides editable text, editable LaTeX, and confidence scores
- Explicit user approval controls (Approve, Reject, Needs Correction)
- Only Approved regions are indexed into trusted RAG
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from pagepilot.ocr.formula_validator import validate_formula_latex
from pagepilot.ocr.models import ApprovalStatus, OCRRegion, PageOCRResult, RegionType
from pagepilot.ocr.service import OCRService
from pagepilot.theme import Colors, Fonts, Radius, Spacing
from pagepilot.ui.components import Badge, Card, Divider, PrimaryButton, SecondaryButton


class ImageCanvas(QLabel):
    """Displays the page image with overlaid color-coded bounding boxes."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(f"background-color: {Colors.BG_SURFACE}; border: 1px solid {Colors.BORDER};")
        self._pixmap: Optional[QPixmap] = None
        self._regions: list[OCRRegion] = []
        self._selected_region_id: Optional[str] = None

    def set_image(self, image_path: str, regions: list[OCRRegion], selected_id: Optional[str] = None):
        self._regions = regions
        self._selected_region_id = selected_id
        if image_path and Path(image_path).exists():
            self._pixmap = QPixmap(image_path)
            self._render_overlay()
        else:
            self._pixmap = None
            self.setText("Image not available")

    def _render_overlay(self):
        if not self._pixmap:
            return

        overlay = self._pixmap.copy()
        painter = QPainter(overlay)

        for r in self._regions:
            box = r.bounding_box
            is_selected = (r.region_id == self._selected_region_id)

            if r.approval_status == ApprovalStatus.APPROVED:
                pen_color = QColor("#10B981")  # Green
            elif r.approval_status == ApprovalStatus.REJECTED:
                pen_color = QColor("#EF4444")  # Red
            elif r.approval_status == ApprovalStatus.NEEDS_CORRECTION:
                pen_color = QColor("#F59E0B")  # Amber
            else:
                pen_color = QColor("#3B82F6")  # Blue / Pending

            pen = QPen(pen_color, 3 if is_selected else 2)
            painter.setPen(pen)
            painter.drawRect(int(box.x0), int(box.y0), int(box.width), int(box.height))

            # Label tag
            painter.fillRect(int(box.x0), max(0, int(box.y0) - 22), 80, 20, pen_color)
            painter.setPen(QColor("#FFFFFF"))
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.drawText(int(box.x0) + 4, max(0, int(box.y0) - 6), f"{r.region_id} [{r.region_type.value[:4]}]")

        painter.end()

        # Scale down if very large for smooth laptop viewing
        max_dim = 1200
        if overlay.width() > max_dim or overlay.height() > max_dim:
            scaled = overlay.scaled(max_dim, max_dim, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.setPixmap(scaled)
        else:
            self.setPixmap(overlay)


class OCRReviewDialog(QDialog):
    """Interactive split-view review interface for human-in-the-loop OCR verification."""

    indexing_completed = Signal(int)  # emits number of chunks indexed

    def __init__(
        self,
        ocr_service: OCRService,
        document_id: int,
        page_number: int = 1,
        parent=None,
    ):
        super().__init__(parent)
        self.ocr_service = ocr_service
        self.document_id = document_id
        self.page_number = page_number
        self._page_result: Optional[PageOCRResult] = None
        self._current_region_idx = 0

        self.setWindowTitle(f"PagePilot — OCR Review & Formula Verification (Page {page_number})")
        self.resize(1100, 750)
        self.setStyleSheet(f"background-color: {Colors.BG_APP}; color: {Colors.TEXT_PRIMARY};")

        self._build_ui()
        self._load_data()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(Spacing.LG, Spacing.LG, Spacing.LG, Spacing.LG)
        root.setSpacing(Spacing.MD)

        # Header bar
        header = QHBoxLayout()
        self.title_lbl = QLabel(f"Document #{self.document_id} — Page {self.page_number} OCR Review")
        self.title_lbl.setFont(Fonts.make(Fonts.SIZE_LG, QFont.Weight.Bold))
        self.status_badge = Badge("Pending Review", "warning")

        header.addWidget(self.title_lbl)
        header.addWidget(self.status_badge)
        header.addStretch()

        root.addLayout(header)

        # Main Splitter: Left Image Canvas, Right Editor
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(3)

        # ── Left: Image Viewer ────────────────────────────────────────────────
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_lbl = QLabel("📷 Original Source Page / Photograph")
        left_lbl.setFont(Fonts.make(Fonts.SIZE_SM, QFont.Weight.Bold))
        left_lbl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        left_layout.addWidget(left_lbl)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.canvas = ImageCanvas()
        scroll.setWidget(self.canvas)
        left_layout.addWidget(scroll, 1)

        splitter.addWidget(left_widget)

        # ── Right: Editable Content Panel ─────────────────────────────────────
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(Spacing.MD, 0, 0, 0)
        right_layout.setSpacing(Spacing.MD)

        right_lbl = QLabel("✍ Extracted Regions & Mathematical Formulas")
        right_lbl.setFont(Fonts.make(Fonts.SIZE_SM, QFont.Weight.Bold))
        right_lbl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        right_layout.addWidget(right_lbl)

        # Region selector row
        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("Select Region:"))
        self.region_combo = QComboBox()
        self.region_combo.currentIndexChanged.connect(self._on_region_selected)
        sel_row.addWidget(self.region_combo, 1)

        self.conf_badge = Badge("Score: --", "info")
        sel_row.addWidget(self.conf_badge)
        right_layout.addLayout(sel_row)

        # Region Type selector
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Type:"))
        self.type_combo = QComboBox()
        for t in RegionType:
            self.type_combo.addItem(t.value.capitalize(), t.value)
        self.type_combo.currentIndexChanged.connect(self._on_type_changed)
        type_row.addWidget(self.type_combo, 1)
        right_layout.addLayout(type_row)

        # Region Crop Preview (Zoomed inspectable view of selected region)
        self.crop_preview_container = QWidget()
        crop_layout = QVBoxLayout(self.crop_preview_container)
        crop_layout.setContentsMargins(0, 0, 0, 0)
        crop_layout.setSpacing(4)
        crop_layout.addWidget(QLabel("🔍 Region Crop Preview (Original):"))
        self.crop_preview_lbl = QLabel()
        self.crop_preview_lbl.setStyleSheet(
            f"background-color: {Colors.BG_SURFACE}; border: 1px solid {Colors.BORDER}; "
            f"padding: 6px; border-radius: 4px;"
        )
        self.crop_preview_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        crop_layout.addWidget(self.crop_preview_lbl)
        right_layout.addWidget(self.crop_preview_container)

        # Recognized / Corrected text editor
        right_layout.addWidget(QLabel("Recognized Text (Editable):"))
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Edit transcribed text here...")
        self.text_edit.textChanged.connect(self._on_text_edited)
        right_layout.addWidget(self.text_edit, 1)

        # Formula LaTeX editor
        self.formula_container = QWidget()
        form_layout = QVBoxLayout(self.formula_container)
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_header = QHBoxLayout()
        form_header.addWidget(QLabel("LaTeX Mathematical Formula:"))
        self.formula_val_badge = Badge("Syntax Check", "info")
        form_header.addWidget(self.formula_val_badge)
        form_header.addStretch()
        form_layout.addLayout(form_header)

        self.latex_edit = QTextEdit()
        self.latex_edit.setPlaceholderText(r"e.g. x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}")
        self.latex_edit.textChanged.connect(self._on_latex_edited)
        form_layout.addWidget(self.latex_edit)

        # Formula Preview label
        self.latex_preview = QLabel()
        self.latex_preview.setStyleSheet(
            f"background-color: {Colors.BG_SURFACE}; border: 1px dashed {Colors.BORDER}; "
            f"padding: 8px; font-family: monospace; color: {Colors.ACCENT};"
        )
        self.latex_preview.setWordWrap(True)
        form_layout.addWidget(self.latex_preview)
        right_layout.addWidget(self.formula_container)

        # Region Action Buttons
        btn_row = QHBoxLayout()
        self.approve_btn = QPushButton("✓ Approve Region")
        self.approve_btn.setStyleSheet("background-color: #059669; color: white; padding: 6px 12px; border-radius: 4px;")
        self.approve_btn.clicked.connect(self._approve_current_region)

        self.reject_btn = QPushButton("✗ Reject Region")
        self.reject_btn.setStyleSheet("background-color: #DC2626; color: white; padding: 6px 12px; border-radius: 4px;")
        self.reject_btn.clicked.connect(self._reject_current_region)

        self.correct_btn = QPushButton("Mark Needs Correction")
        self.correct_btn.setStyleSheet(f"background-color: {Colors.BG_SURFACE}; color: {Colors.TEXT_PRIMARY}; border: 1px solid {Colors.BORDER}; padding: 6px 12px; border-radius: 4px;")
        self.correct_btn.clicked.connect(self._mark_current_needs_correction)

        btn_row.addWidget(self.approve_btn)
        btn_row.addWidget(self.reject_btn)
        btn_row.addWidget(self.correct_btn)
        right_layout.addLayout(btn_row)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        root.addWidget(splitter, 1)

        # Bottom Bar: Page Actions
        bottom_row = QHBoxLayout()
        self.approve_all_btn = SecondaryButton("Approve All Page Regions")
        self.approve_all_btn.clicked.connect(self._approve_all_page)

        self.save_index_btn = PrimaryButton("Save & Index Approved Content")
        self.save_index_btn.clicked.connect(self._save_and_index)

        self.close_btn = SecondaryButton("Close")
        self.close_btn.clicked.connect(self.accept)

        bottom_row.addWidget(self.approve_all_btn)
        bottom_row.addStretch()
        bottom_row.addWidget(self.save_index_btn)
        bottom_row.addWidget(self.close_btn)

        root.addLayout(bottom_row)

    def _load_data(self):
        """Fetch OCR result from database or trigger initial recognition if needed."""
        page_res = self.ocr_service.get_page_ocr_result(self.document_id, self.page_number)
        if not page_res or not page_res.regions:
            # Run OCR now
            results = self.ocr_service.process_document_ocr(self.document_id)
            if results:
                page_res = results[0]

        self._page_result = page_res
        if not self._page_result:
            return

        self._refresh_region_list()
        if self._page_result.regions:
            self._select_region(0)

    def _refresh_region_list(self):
        if not self._page_result:
            return

        self.region_combo.blockSignals(True)
        self.region_combo.clear()
        for r in self._page_result.regions:
            status_icon = "✓" if r.is_approved else ("✗" if r.is_rejected else "⏳")
            self.region_combo.addItem(f"{status_icon} {r.region_id}: [{r.region_type.value}] {r.effective_text[:35]}")
        self.region_combo.blockSignals(False)

        # Update page status badge
        if self._page_result.is_fully_approved:
            self.status_badge.setText("Approved")
            self.status_badge.setStyleSheet("background-color: #10B981; color: white;")
        else:
            self.status_badge.setText("Pending Review")
            self.status_badge.setStyleSheet("background-color: #F59E0B; color: white;")

    def _on_region_selected(self, index: int):
        if index < 0 or not self._page_result or index >= len(self._page_result.regions):
            return
        self._select_region(index)

    def _select_region(self, index: int):
        self._current_region_idx = index
        region = self._page_result.regions[index]

        # Update Left Canvas
        self.canvas.set_image(self._page_result.image_path, self._page_result.regions, selected_id=region.region_id)

        # Update Zoomed Crop Preview
        if self._page_result.image_path and Path(self._page_result.image_path).exists():
            box = region.bounding_box
            if box.width > 0 and box.height > 0:
                pix = QPixmap(self._page_result.image_path)
                if not pix.isNull():
                    crop_w = max(1, int(box.width))
                    crop_h = max(1, int(box.height))
                    crop_pix = pix.copy(int(box.x0), int(box.y0), crop_w, crop_h)
                    if crop_pix.height() > 0:
                        scaled_h = min(110, max(50, crop_pix.height() * 2))
                        scaled_crop = crop_pix.scaledToHeight(scaled_h, Qt.TransformationMode.SmoothTransformation)
                        self.crop_preview_lbl.setPixmap(scaled_crop)
                        self.crop_preview_container.setVisible(True)
        else:
            self.crop_preview_container.setVisible(False)

        # Update Editor Inputs
        self.text_edit.blockSignals(True)
        self.text_edit.setText(region.effective_text)
        self.text_edit.blockSignals(False)

        # Confidence
        self.conf_badge.setText(f"Score: {region.confidence_score:.2f}")

        # Region Type
        idx = self.type_combo.findData(region.region_type.value)
        if idx >= 0:
            self.type_combo.blockSignals(True)
            self.type_combo.setCurrentIndex(idx)
            self.type_combo.blockSignals(False)

        # Formula UI visibility
        is_formula = (region.region_type == RegionType.FORMULA)
        self.formula_container.setVisible(is_formula)
        if is_formula:
            self.latex_edit.blockSignals(True)
            self.latex_edit.setText(region.latex_expression or "")
            self.latex_edit.blockSignals(False)
            self._update_latex_preview(region.latex_expression or "")

    def _on_type_changed(self, index: int):
        if not self._page_result or self._current_region_idx >= len(self._page_result.regions):
            return
        new_val = self.type_combo.currentData()
        region = self._page_result.regions[self._current_region_idx]
        region.region_type = RegionType(new_val)
        self.formula_container.setVisible(region.region_type == RegionType.FORMULA)
        if region.region_type == RegionType.FORMULA and not region.latex_expression:
            region.latex_expression = region.effective_text
            self.latex_edit.setText(region.latex_expression)
            self._update_latex_preview(region.latex_expression)
        self.ocr_service.update_region(region.region_id, region_type=new_val, latex_expression=region.latex_expression)
        self._refresh_region_list()

    def _on_text_edited(self):
        if not self._page_result or self._current_region_idx >= len(self._page_result.regions):
            return
        region = self._page_result.regions[self._current_region_idx]
        new_txt = self.text_edit.toPlainText().strip()
        region.corrected_text = new_txt
        self.ocr_service.update_region(region.region_id, corrected_text=new_txt)

    def _on_latex_edited(self):
        if not self._page_result or self._current_region_idx >= len(self._page_result.regions):
            return
        region = self._page_result.regions[self._current_region_idx]
        new_latex = self.latex_edit.toPlainText().strip()
        region.latex_expression = new_latex
        self._update_latex_preview(new_latex)
        self.ocr_service.update_region(region.region_id, latex_expression=new_latex)

    def _update_latex_preview(self, latex: str):
        if latex:
            conf = self._page_result.regions[self._current_region_idx].confidence_score if self._page_result else 1.0
            val = validate_formula_latex(latex, confidence_score=conf)
            if not val.is_valid:
                err_text = val.errors[0] if val.errors else "Syntax Error"
                self.formula_val_badge.setText(f"✗ {err_text[:28]}")
                self.formula_val_badge.setStyleSheet("background-color: #EF4444; color: white;")
            elif val.warnings:
                warn_text = val.warnings[0]
                self.formula_val_badge.setText(f"⚠ {warn_text[:28]}")
                self.formula_val_badge.setStyleSheet("background-color: #F59E0B; color: white;")
            else:
                self.formula_val_badge.setText("✓ Valid LaTeX Syntax")
                self.formula_val_badge.setStyleSheet("background-color: #10B981; color: white;")
            self.latex_preview.setText(f"Rendered Preview:\n\\[ {latex} \\]")
        else:
            self.formula_val_badge.setText("Empty Formula")
            self.formula_val_badge.setStyleSheet("background-color: #6B7280; color: white;")
            self.latex_preview.setText("(No formula expression)")

    def _approve_current_region(self):
        if not self._page_result or self._current_region_idx >= len(self._page_result.regions):
            return
        region = self._page_result.regions[self._current_region_idx]
        region.approval_status = ApprovalStatus.APPROVED
        self.ocr_service.update_region(region.region_id, approval_status=ApprovalStatus.APPROVED.value)
        self._refresh_region_list()
        self.canvas.set_image(self._page_result.image_path, self._page_result.regions, selected_id=region.region_id)

    def _reject_current_region(self):
        if not self._page_result or self._current_region_idx >= len(self._page_result.regions):
            return
        region = self._page_result.regions[self._current_region_idx]
        region.approval_status = ApprovalStatus.REJECTED
        self.ocr_service.update_region(region.region_id, approval_status=ApprovalStatus.REJECTED.value)
        self._refresh_region_list()
        self.canvas.set_image(self._page_result.image_path, self._page_result.regions, selected_id=region.region_id)

    def _mark_current_needs_correction(self):
        if not self._page_result or self._current_region_idx >= len(self._page_result.regions):
            return
        region = self._page_result.regions[self._current_region_idx]
        region.approval_status = ApprovalStatus.NEEDS_CORRECTION
        self.ocr_service.update_region(region.region_id, approval_status=ApprovalStatus.NEEDS_CORRECTION.value)
        self._refresh_region_list()
        self.canvas.set_image(self._page_result.image_path, self._page_result.regions, selected_id=region.region_id)

    def _approve_all_page(self):
        if not self._page_result:
            return
        self.ocr_service.approve_page(self.document_id, self.page_number)
        for r in self._page_result.regions:
            if not r.is_rejected:
                r.approval_status = ApprovalStatus.APPROVED
        self._refresh_region_list()
        self.canvas.set_image(self._page_result.image_path, self._page_result.regions)

    def _save_and_index(self):
        """Commit approved regions into SQLite FTS5 and ChromaDB."""
        if not self._page_result:
            return

        approved_count = len(self._page_result.approved_regions)
        if approved_count == 0:
            QMessageBox.warning(
                self,
                "No Approved Content",
                "Cannot index document: No regions have been approved.\n"
                "Please review and approve at least one region before indexing.",
            )
            return

        try:
            chunks_indexed = self.ocr_service.index_approved_content(self.document_id, self.page_number)
            QMessageBox.information(
                self,
                "Indexing Complete",
                f"Successfully indexed {chunks_indexed} approved chunks into trusted hybrid retrieval.\n"
                f"Approved content is now searchable and verifiable in PagePilot.",
            )
            self.indexing_completed.emit(chunks_indexed)
            self.accept()
        except Exception as exc:
            QMessageBox.critical(self, "Indexing Error", f"Failed to index approved OCR content: {exc}")

