# PagePilot

**Offline-first personal AI assistant** for local chat, voice interaction, document intelligence, DOCX creation, and offline navigation.

---

## Current Status — Stage 3: Offline Voice Recognition & Speech Generation

PagePilot currently supports:
- **Offline AI Chat (Stage 2):** Streaming conversations with local `qwen2.5:3b` via Ollama and persistent SQLite history.
- **Offline Voice Recognition (Stage 3):** High-precision speech-to-text powered by `whisper.cpp` (`pywhispercpp`) using the `ggml-base` multilingual model (or `base.en`). Audio recorded via `sounddevice` at 16 kHz mono.
- **Offline Speech Synthesis (Stage 3):** Natural local neural text-to-speech powered by Piper (`piper-tts`) with background playback and pause/stop controls.
- **Review Before Send:** Voice input populates the composer for review and manual editing before submission—never auto-sent.
- **Read Aloud:** Every completed assistant response features an interactive "🔊 Read Aloud" control.
- **Voice Management:** Built-in settings card to download Whisper models and browse/configure Piper voices.

---

## Technology Stack

| Capability | Technology | Details |
|---|---|---|
| Desktop GUI | PySide6 (Qt 6.x) | Native Windows 11 Fluent-inspired theme |
| Local LLM | Ollama + Qwen2.5 3B | Streaming chat at `127.0.0.1:11434` |
| Local Persistence | SQLite (`sqlite3`) | `%APPDATA%\PagePilot\conversations.db` |
| Audio Capture | `sounddevice` + PortAudio | 16 kHz mono 16-bit PCM |
| Speech Recognition | `whisper.cpp` (`pywhispercpp`) | `ggml-base.bin` multilingual model (~148 MB) |
| Speech Synthesis | Piper (`piper-tts`) | `en_US-lessac-medium` neural ONNX voice |

---

## Prerequisites

1. **Windows 11 / 10 (64-bit)** with at least 8 GB RAM.
2. **Python 3.10+** (Tested on Python 3.14).
3. **Ollama**:
   - Install from [ollama.com](https://ollama.com)
   - Pull the model:
     ```bash
     ollama pull qwen2.5:3b
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

## Voice Models Setup (Offline)

### 1. Whisper Speech Recognition Model
Whisper models are stored in `%APPDATA%\PagePilot\models\whisper\`.

* **Option A (Automatic via UI):** Open PagePilot, go to **Settings → Speech Recognition**, and click **Download ggml-base model (~148 MB)**.
* **Option B (Manual):** Download `ggml-base.bin` from [Hugging Face](https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin) and place it into:
  ```
  %APPDATA%\PagePilot\models\whisper\ggml-base.bin
  ```

### 2. Piper Text-to-Speech Voice
Piper voice models are stored in `%APPDATA%\PagePilot\models\piper\`.

Download both the `.onnx` and `.onnx.json` files for your preferred voice from the [Piper Voices repository](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/lessac/medium) (e.g. `en_US-lessac-medium`):
* `en_US-lessac-medium.onnx`
* `en_US-lessac-medium.onnx.json`

Place both files into:
```
%APPDATA%\PagePilot\models\piper\
```

---

## Running the Application

```bash
python main.py
```

---

## Running Tests

Run the full automated test suite (80 unit and integration tests):

```bash
python -m pytest -q
```

Or run individual test modules:
```bash
python -m pytest tests/test_voice.py -v
python -m pytest tests/test_chat_integration.py -v
python -m pytest tests/test_database.py -v
python -m pytest tests/test_ollama_client.py -v
```

---

## Project Roadmap

| Stage | Feature | Status |
|---|---|---|
| Stage 1 | Windows Frontend Desktop UI | Completed |
| Stage 2 | Offline AI Chat (Qwen2.5 3B via Ollama) | Completed |
| Stage 3 | Offline Voice Input (Whisper.cpp) & Speech Output (Piper) | Completed |
| Stage 4 | PDF Intelligence & Question Answering | Planned |
| Stage 5 | DOCX Document Generation | Planned |
| Stage 6 | Android Application | Planned |
| Stage 7 | Offline Maps & Turn-by-Turn Navigation | Planned |

---

## Third-Party Licenses & Notices

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for full licensing documentation:
- **Piper TTS:** GPL-3.0-or-later
- **whisper.cpp & pywhispercpp:** MIT License
- **sounddevice & PortAudio:** MIT License
- **PySide6:** LGPL-3.0
