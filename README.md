# PagePilot

**Offline-first personal AI assistant** for local chat, voice interaction, PDF document intelligence, DOCX creation, multi-document reasoning, handwritten notes OCR, and mathematical formula recognition.

---

## Current Status — Stage 6B: Handwritten Notes OCR & Mathematical Formula Recognition

PagePilot currently supports:
- **Handwritten Notes OCR & Formula Engine (Stage 6B):**
  - **Student Notes Photograph Ingestion:** Import photographed lecture notes in PNG, JPG, JPEG, and WebP formats with SHA-256 deduplication and page-level provenance.
  - **Scanned PDF Page Extraction:** Detects image-only/scanned PDF pages and renders them on demand one page at a time via PyMuPDF without memory bloat.
  - **Isolated Worker Architecture:** Runs PaddleOCR and Pix2Text in an isolated Python 3.12 subprocess worker (`.venv-ocr/`) over a robust JSON-RPC protocol, preserving the main Python 3.14.5 application runtime and keeping memory bounded for 8 GB RAM laptops.
  - **Local Text & Handwriting Recognition:** PaddleOCR PP-OCRv5 local text recognition for printed lecture materials and student handwriting with confidence scores and bounding boxes.
  - **Mathematical Formula Recognition:** Pix2Text Mathematical Formula Detection (MFD) and Recognition (MFR) converting formula image regions into structured, editable LaTeX (`\[ ... \]`).
  - **Human-in-the-Loop Review & Correction Interface:** Split-screen dialog showing the original photograph/page on the left with color-coded bounding boxes, and editable text, editable LaTeX, confidence badges, and approval controls on the right.
  - **Trusted RAG Indexing:** Strict anti-hallucination guarantee: only user-approved OCR regions are chunked, embedded via Nomic Embed, and indexed into SQLite FTS5 and ChromaDB. Unapproved and rejected regions are excluded.
  - **Provenance-Preserving Citations:** Verified answers cite handwritten notes and scanned pages with exact source filenames, 1-based page numbers, and bounding-box coordinates (`[Notes_Lecture1.jpg, p. 1]`).
  - **8-Category Benchmark Suite:** Evaluated across printed text, handwritten notes, skewed images, low-light photos, printed formulas, handwritten formulas, mixed text/equations, and scanned textbooks.
- **Faithful to Every Sentence (Stage 6A):**
  - **Multi-Document Reasoning & Synthesis:** Users can select 2 or more PDFs simultaneously to perform cross-document comparison, requirement synthesis, and cross-source analysis.
  - **Deterministic Query Planning:** Automatically decomposes complex multi-document questions into atomic information requirements (subqueries) to ensure targeted retrieval across disparate documents.
  - **Cross-Document Hybrid Retrieval:** Performs balanced, multi-query retrieval across ChromaDB and SQLite FTS5 with cross-document chunk deduplication and diversity balancing so no single document starves others.
  - **Claim-Level Evidence Verification:** Parses LLM responses into structured atomic factual claims and verifies each claim against application-managed evidence chunks (`[E1]`, `[E2]`). Claims receive deterministic statuses: `SUPPORTED`, `CONTRADICTED`, `INSUFFICIENT_EVIDENCE`, `INVALID_CITATION`, or `NEEDS_REVIEW`.
  - **Cross-Document Contradiction Engine:** Automatically scans retrieved evidence pairs for conflicting numerical metrics, opposing policy/status definitions (e.g. `supported` vs `deprecated`), and flags them as structured contradiction notices in the UI and DOCX reports.
  - **Sentence-Level Citations:** Formats answers with verifiable sentence-level citations (`[E1, E2]` mapping to `[Document.pdf, p. N]`). Unsupported assertions are highlighted with warning indicators rather than presented as verified facts.
  - **Explicit Missing Evidence Handling:** When information requirements lack grounding evidence, the assistant explicitly reports which information is missing and refrains from fabricating answers.
  - **Multi-Document DOCX Report:** Exports verified multi-document synthesis to Word documents with full claim verification tables, contradiction warnings, and citation bibliographies.
- **Professional DOCX Engine (Stage 5):**
  - **Local Document Automation:** Native Microsoft Word (.docx) generation powered by `python-docx` and `lxml`.
  - **5 Document Types:** PDF Summary, Study Notes, Question & Answer Notes, Project Report, Selected PDF Content, and Custom Documents.
  - **5 Professional Templates:** Simple Study Notes, Academic Summary, Question and Answer Notes, Project Report, and Custom Document with distinct typography, palettes, margins, and callouts.
  - **Strict Citation Preservation:** Preserves real PDF filenames, 1-based page numbers, and verified snippets. Supports inline citations, Sources section tables, both, or none.
- **Offline AI Chat (Stage 2):** Streaming conversations with local `qwen2.5:3b` via Ollama and persistent SQLite history.
- **Offline Voice Recognition (Stage 3):** High-precision speech-to-text powered by `whisper.cpp` (`pywhispercpp`) using the `ggml-base` multilingual model (or `base.en`). Audio recorded via `sounddevice` at 16 kHz mono.
- **Offline Speech Synthesis (Stage 3):** Natural local neural text-to-speech powered by Piper (`piper-tts`) with background playback and pause/stop controls.
- **Offline PDF Intelligence (Stage 4):** PyMuPDF page-level extraction, ChromaDB vector store, SQLite FTS5 keyword search, and hybrid Reciprocal Rank Fusion ($k=60$).

---

## Technology Stack

| Capability | Technology | Details |
|---|---|---|
| Desktop GUI | PySide6 (Qt 6.x) | Native Windows 11 Fluent-inspired theme |
| Local LLM | Ollama + Qwen2.5 3B | Streaming chat and structured claim generation at `127.0.0.1:11434` |
| Local Embeddings | Ollama + nomic-embed-text | 768-dimensional dense vectors via `/api/embed` |
| OCR Worker | Isolated Python 3.12 | Independent subprocess via JSON-RPC over stdin/stdout (`.venv-ocr/`) |
| Text & Handwriting OCR | PaddleOCR (PP-OCRv5) | Local CPU inference with line detection and recognition |
| Formula Recognition | Pix2Text (MFD + MFR) | Mathematical formula detection and LaTeX conversion |
| Vector Store | ChromaDB (Local persistent) | Stored in `%APPDATA%\PagePilot\vectordb\` |
| Keyword Search | SQLite FTS5 (unicode61) | Full-text search with LIKE fallback |
| Claim Verification | Python deterministic engine | Lexical overlap, entity consistency, contradiction detection |
| Document Storage | SQLite (`sqlite3`) | `%APPDATA%\PagePilot\documents.db` |
| Chat Persistence | SQLite (`sqlite3`) | `%APPDATA%\PagePilot\conversations.db` |
| PDF Extraction | PyMuPDF (`pymupdf`) | Fast streaming extraction, page rendering, page metadata |
| DOCX Generation | `python-docx` + `lxml` | OpenXML document assembly & validation |
| Audio Capture | `sounddevice` + PortAudio | 16 kHz mono 16-bit PCM |
| Speech Recognition | `whisper.cpp` (`pywhispercpp`) | `ggml-base.bin` multilingual model (~148 MB) |
| Speech Synthesis | Piper (`piper-tts`) | `en_US-lessac-medium` neural ONNX voice |

---

## Prerequisites

1. **Windows 11 / 10 (64-bit)** with at least 8 GB RAM.
2. **Python 3.10+** (Tested on Python 3.14).
3. **Ollama**:
   - Install from [ollama.com](https://ollama.com)
   - Pull the required models:
     ```bash
     ollama pull qwen2.5:3b
     ollama pull nomic-embed-text
     ```

---

## Installation & Setup

```bash
# 1. Clone repository and navigate to folder
cd PagePilot

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Local Storage Locations

All user data and models remain strictly on your local machine outside the Git repository:

* **Chat Database:** `%APPDATA%\PagePilot\conversations.db`
* **Documents Database:** `%APPDATA%\PagePilot\documents.db`
* **Chroma Vector Store:** `%APPDATA%\PagePilot\vectordb\`
* **DOCX Exports:** `%USERPROFILE%\Documents\PagePilot\`
* **Whisper Models:** `%APPDATA%\PagePilot\models\whisper\`
* **Piper Voices:** `%APPDATA%\PagePilot\models\piper\`
* **OCR & Formula Models:** `%APPDATA%\PagePilot\models\ocr\`
* **OCR Image Cache:** `%APPDATA%\PagePilot\cache\ocr\`

---

## OCR Worker Setup (Stage 6B)

The main PagePilot application runs on Python 3.14.5. To maintain compatibility with PaddlePaddle and Pix2Text on Windows, OCR runs in an isolated Python 3.12 worker environment (`.venv-ocr/`).

To configure the isolated worker:
```powershell
# 1. Install Python 3.12 if not already present
winget install Python.Python.3.12 --scope user

# 2. Run the automated worker setup script
python scripts/setup_ocr_worker.py
```
*Note: If Python 3.12 is not yet installed, PagePilot operates seamlessly with its deterministic simulation worker without failing or crashing.*

---

## Running the Application

```bash
python main.py
```

---

## Running Tests

Run the full automated test suite:

```bash
python -m pytest -q
```
