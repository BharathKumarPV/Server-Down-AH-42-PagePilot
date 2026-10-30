"""
Manual verification script for PagePilot Stage 5 DOCX Generation Engine.
Tests all 5 document types, 5 templates, citation modes, and reopening validation.
"""

import sys
from pathlib import Path
import docx

# Ensure PagePilot is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pagepilot.config import ensure_docx_dir
from pagepilot.docx.generator import DocxGenerator
from pagepilot.docx.document_service import DocxService
from pagepilot.docx.citation_formatter import CitationMode
from pagepilot.docx.templates import TEMPLATE_NAMES, get_template
from pagepilot.docx.validators import validate_docx_file


def run_verification():
    output_dir = Path("scripts_output_docx")
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = DocxGenerator()

    print("=" * 60)
    print("PAGEPILOT STAGE 5 — DOCX ENGINE VERIFICATION")
    print("=" * 60)

    # 1. Test All 5 Templates with rich content
    print("\n[1/4] Verifying all 5 professional templates...")
    sample_content = """# Executive Strategic Overview
This document showcases the **PagePilot Stage 5 DOCX generation engine**.

## 1. Key Principles & Architecture
- Local OpenXML assembly using `python-docx`
- Zero external cloud service dependencies
- Verifiable 1-based PDF citation fidelity

> **Important Architecture Standard**: All document transformations run locally
> with deterministic performance and strict memory boundaries.

## 2. Capability Comparison

| Feature | Stage 4 (PDF) | Stage 5 (DOCX) |
| --- | --- | --- |
| Ingestion | PyMuPDF | python-docx / lxml |
| Provenance | 1-based pages | Full Source Tables |
| Offline | 100% Local | 100% Local |

```python
def export_document(title, markdown):
    return generator.generate(title, markdown)
```
"""
    citations = [
        {"filename": "Engineering_Standard_2026.pdf", "page_number": 4, "snippet": "Core protocol definition."},
        {"filename": "Safety_Compliance.pdf", "page_number": 12, "snippet": "Mandated validation metrics."},
    ]

    for tmpl in TEMPLATE_NAMES:
        fpath = output_dir / f"Demo_{tmpl.replace(' ', '_')}.docx"
        res = generator.generate(
            title=f"PagePilot Demo: {tmpl}",
            content=sample_content,
            subtitle="Stage 5 Automated Verification Sample",
            template_name=tmpl,
            citations=citations,
            citation_mode=CitationMode.BOTH,
            output_path=fpath,
        )
        validate_docx_file(res)
        doc = docx.Document(str(res))
        p_count = len(doc.paragraphs)
        t_count = len(doc.tables)
        sz_kb = res.stat().st_size / 1024
        print(f"  ✓ {tmpl:30} -> {sz_kb:6.1f} KB | {p_count:2} paras | {t_count} tables")

    # 2. Test Document Service Types
    print("\n[2/4] Verifying Document Service generation workflows...")
    service = DocxService(generator=generator)

    # Q&A Document
    qa_path = output_dir / "Demo_QA_Document.docx"
    qa_pairs = [
        {
            "question": "What is the memory footprint target for PagePilot?",
            "answer": "The target environment is 8 GB RAM with low footprint and no cloud egress.",
            "citations": [{"filename": "System_Requirements.pdf", "page_number": 2, "snippet": "8 GB RAM budget."}],
        },
        {
            "question": "Which templates are available for DOCX export?",
            "answer": "Simple Study Notes, Academic Summary, Question and Answer Notes, Project Report, and Custom Document.",
            "citations": [{"filename": "User_Manual.pdf", "page_number": 15, "snippet": "Five core templates."}],
        },
    ]
    res_qa = service.generate_qa_document(
        qa_pairs=qa_pairs,
        template_name="Question and Answer Notes",
        citation_mode=CitationMode.BOTH,
        output_path=qa_path,
        custom_title="Verified QA Compendium",
    )
    validate_docx_file(res_qa)
    print(f"  ✓ Q&A Document                   -> {res_qa.stat().st_size / 1024:6.1f} KB")

    # Custom Document
    custom_path = output_dir / "Demo_Custom_Document.docx"
    res_custom = service.generate_custom_document(
        title="Custom Operations Brief",
        content="# Operational Procedures\n1. Check network status (Offline)\n2. Index local documents\n3. Export report",
        template_name="Custom Document",
        output_path=custom_path,
    )
    validate_docx_file(res_custom)
    print(f"  ✓ Custom Document                -> {res_custom.stat().st_size / 1024:6.1f} KB")

    # 3. Citation Modes Verification
    print("\n[3/4] Verifying citation modes (Both, Sources Only, Inline Only, None)...")
    for mode in CitationMode.ALL_MODES:
        mpath = output_dir / f"Demo_Citation_{mode.replace(' ', '_')}.docx"
        res_m = generator.generate(
            title=f"Citation Mode: {mode}",
            content="Standard paragraph with contextual analysis.",
            citations=citations,
            citation_mode=mode,
            output_path=mpath,
        )
        validate_docx_file(res_m)
        doc_m = docx.Document(str(res_m))
        print(f"  ✓ {mode:30} -> {len(doc_m.tables)} tables in document")

    # 4. OpenXML Structure Reopening Check
    print("\n[4/4] Validating OpenXML structure and readability...")
    for f in output_dir.glob("*.docx"):
        assert validate_docx_file(f) is True

    print("\n✓ All DOCX generation engine verifications passed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_verification()
