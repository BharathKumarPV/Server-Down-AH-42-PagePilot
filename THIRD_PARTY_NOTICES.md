# Third-Party Notices & Licensing Guide

PagePilot integrates open-source components for offline AI inference, speech recognition, and speech synthesis. This document describes the licenses of third-party libraries and runtime models used in Stage 3.

---

## 1. Piper Text-to-Speech (`piper-tts`)

* **Project:** [OHF-Voice/piper1-gpl](https://github.com/OHF-Voice/piper1-gpl)
* **License:** GNU General Public License v3.0 or later (**GPL-3.0-or-later**)
* **Copyright:** © 2023–2025 The Home Assistant Authors & Rhasspy Contributors

### Distribution & GPL-3.0 Obligations:
* **Personal & Internal Use:** You may freely run, inspect, and use PagePilot with Piper locally for personal, research, or private internal workflows without restriction.
* **Redistribution / Commercial Distribution:** Because `piper-tts` is licensed under GPL-3.0-or-later, if you distribute PagePilot as a compiled application or package that directly embeds or dynamically/statically links against `piper-tts`, **the entire combined work must be distributed under the terms of the GPL-3.0 license**, including making full corresponding source code available.
* **Commercial redistribution is NOT unrestricted.** You cannot convert PagePilot into a closed-source proprietary distribution if it bundles GPL-3.0 components.

### Individual Voice Model Licenses:
Piper voice models (`.onnx` and `.onnx.json`) are developed independently and hosted separately (e.g., on Hugging Face). They have varying licenses:
* Some voices are licensed under **Public Domain (CC0)**.
* Other voices are licensed under **Creative Commons Attribution (CC-BY 4.0)** or **MIT**.
* Certain voice models may carry non-commercial restrictions depending on the training corpus (e.g., LibriTTS, VCTK).
* **Important:** Check the individual license associated with each downloaded voice model before redistributing voice files.

---

## 2. whisper.cpp & pywhispercpp

* **Underlying Engine:** [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp)
  * **License:** **MIT License**
  * **Copyright:** © 2023–2025 Georgi Gerganov
* **Python Bindings:** [abdeladim-s/pywhispercpp](https://github.com/abdeladim-s/pywhispercpp)
  * **License:** **MIT License**
  * **Copyright:** © 2023–2025 Abdeladim Sadiki
* **Whisper Model Weights:**
  * Developed by OpenAI, converted to GGML format.
  * **License:** **MIT License**

### Distribution & Usage:
* Permissive MIT license allows commercial use, modification, distribution, and private use, provided that the copyright and permission notice are included.

---

## 3. sounddevice & PortAudio

* **Library:** `sounddevice` ([spatialaudio/python-sounddevice](https://github.com/spatialaudio/python-sounddevice))
  * **License:** **MIT License**
  * **Copyright:** © Matthias Geier
* **Underlying Audio Driver:** PortAudio ([portaudio.com](http://www.portaudio.com/))
  * **License:** **MIT-style License**

---

## 4. PySide6 & Qt 6

* **Library:** PySide6 (Qt for Python)
* **License:** **GNU Lesser General Public License v3 (LGPL-3.0)** / Commercial
* Dynamic linking to Qt libraries allows proprietary and open-source applications complying with LGPL-3.0 requirements.

---

## 5. Ollama & Qwen2.5 3B

* **Runtime:** Ollama ([ollama.com](https://ollama.com)) — MIT License
* **Model:** Qwen2.5 3B Instruct — Apache 2.0 License by Alibaba Cloud

