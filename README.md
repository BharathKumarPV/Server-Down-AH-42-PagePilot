# PagePilot

**Offline-first personal AI assistant** for local chat, voice interaction, PDF document intelligence, DOCX creation, multi-document reasoning, and offline navigation.

---

## Current Status — Stage 6A: Faithful to Every Sentence (Multi-Document Reasoning & Claim Verification)

PagePilot currently supports:
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
  - **Interactive Editor:** Markdown content preview and live editor allowing review and customization prior to export.
  - **Reliable Validation:** Reopens and verifies generated OpenXML packages before export confirmation.
- **Offline AI Chat (Stage 2):** Streaming conversations with local `qwen2.5:3b` via Ollama and persistent SQLite history.
- **Offline Voice Recognition (Stage 3):** High-precision speech-to-text powered by `whisper.cpp` (`pywhispercpp`) using the `ggml-base` multilingual model (or `base.en`). Audio recorded via `sounddevice` at 16 kHz mono.
- **Offline Speech Synthesis (Stage 3):** Natural local neural text-to-speech powered by Piper (`piper-tts`) with background playback and pause/stop controls.
- **Review Before Send:** Voice input populates the composer for review and manual editing before submission—never auto-sent.
- **Read Aloud:** Every completed assistant response features an interactive "🔊 Read Aloud" control.
- **Offline PDF Intelligence (Stage 4):**
  - **Local Ingestion & Deduplication:** Fast page-by-page text extraction via PyMuPDF with SHA-256 deduplication.
  - **Page-Level Provenance:** Chunks preserve strict 1-based page numbers; chunks never cross page boundaries.
  - **Hybrid Retrieval (RRF):** Blends dense vector search (`nomic-embed-text` in ChromaDB) with BM25-equivalent exact keyword search (SQLite FTS5) using Reciprocal Rank Fusion ($k=60$).
  - **Strict Evidence Grounding:** Anti-hallucination prompt architecture; answers cite sources as `[filename.pdf, p. N]`.
  - **Clickable Citations:** Source pills in the UI link directly to verified document page excerpts.
  - **Document Summarization:** One-click structured summaries for imported documents.
  - **Full Privacy & Zero Telemetry:** ChromaDB anonymized telemetry explicitly disabled; zero external network calls.

---

## Technology Stack

| Capability | Technology | Details |
|---|---|---|
| Desktop GUI | PySide6 (Qt 6.x) | Native Windows 11 Fluent-inspired theme |
| Local LLM | Ollama + Qwen2.5 3B | Streaming chat and structured claim generation at `127.0.0.1:11434` |
| Local Embeddings | Ollama + nomic-embed-text | 768-dimensional dense vectors via `/api/embed` |
| Vector Store | ChromaDB (Local persistent) | Stored in `%APPDATA%\PagePilot\vectordb\` |
| Keyword Search | SQLite FTS5 (unicode61) | Full-text search with LIKE fallback |
| Claim Verification | Python deterministic engine | Lexical overlap, entity consistency, contradiction detection |
| Document Storage | SQLite (`sqlite3`) | `%APPDATA%\PagePilot\documents.db` |
| Chat Persistence | SQLite (`sqlite3`) | `%APPDATA%\PagePilot\conversations.db` |
| PDF Extraction | PyMuPDF (`pymupdf`) | Fast streaming extraction, page metadata |
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
