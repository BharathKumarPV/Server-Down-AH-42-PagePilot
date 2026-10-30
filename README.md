# PagePilot

**Offline-first AI assistant** for local chat, PDF intelligence, DOCX generation, and offline navigation.

---

## Current Status — Stage 1: Windows Frontend (UI only)

This release delivers the complete **Windows desktop UI shell** built with Python 3 and PySide6.
No AI inference, PDF processing, DOCX generation, voice recognition, or map rendering is wired in yet.
All backend features are clearly marked as planned within the interface.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| GUI framework | PySide6 (Qt 6) |
| AI inference | _(Stage 2 — planned)_ |
| Voice recognition | _(Stage 2 — planned)_ |
| PDF processing | _(Stage 3 — planned)_ |
| DOCX generation | _(Stage 3 — planned)_ |
| Offline maps | _(Stage 4 — planned)_ |
| Android port | _(Stage 5 — planned)_ |

---

## Project Structure

```
PagePilot/
├── main.py                    # Application entry point
├── requirements.txt
├── README.md
├── .gitignore
└── pagepilot/
    ├── __init__.py
    ├── config.py              # App-wide constants (sizes, nav IDs, version)
    ├── theme.py               # Design system: colours, fonts, spacing, stylesheet
    ├── ui/
    │   ├── __init__.py
    │   ├── components.py      # Reusable widgets (buttons, cards, inputs, badges…)
    │   ├── sidebar.py         # Left navigation panel
    │   └── main_window.py     # QMainWindow shell + stacked screen router
    └── screens/
        ├── __init__.py
        ├── chat.py            # Chat screen (welcome, conversation, input bar)
        ├── history.py         # Chat history screen
        ├── pdf_assistant.py   # PDF Assistant screen
        ├── docx_creator.py    # DOCX Creator screen
        ├── maps.py            # Offline Maps screen
        └── settings.py        # Settings screen
```

---

## Installation

> Python 3.10 or later is required.

```bash
# 1. Clone the repository
git clone <repo-url>
cd PagePilot

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate      # Windows PowerShell
# or: .venv/Scripts/activate.bat  # Windows cmd

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the Application

```bash
python main.py
```

---

## Planned Features (future stages)

| Stage | Feature |
|---|---|
| Stage 2 | Offline AI chat (local LLM via llama.cpp / Ollama) |
| Stage 2 | Voice recognition and text-to-speech (offline) |
| Stage 3 | PDF summarisation and document Q&A |
| Stage 3 | DOCX document generation |
| Stage 4 | Offline maps and routing (OpenStreetMap tiles) |
| Stage 5 | Android application |

---

## License

To be determined.
