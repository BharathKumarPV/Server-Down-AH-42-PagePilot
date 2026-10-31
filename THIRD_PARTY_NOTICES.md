# Third-Party Notices & Licensing Guide

PagePilot integrates open-source components for offline AI inference, speech recognition, speech synthesis, and document intelligence. This document describes the licenses of third-party libraries and runtime models used in Stage 4.

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
* **Licensing Model:** Dual/Multi-Licensed (Community Edition vs Commercial Edition)

### A. Qt for Python Community Edition:
1. **LGPL-3.0 (GNU Lesser General Public License v3):**
   * The core modules used by PagePilot (`QtCore`, `QtGui`, `QtWidgets`) are available under LGPL-3.0.
   * **Obligations for redistribution:** Applications may dynamically link to PySide6/Qt libraries. If redistributed as a binary package, you must provide the LGPL-3.0 license text, include copyright notices, and allow users to replace or relink the shared Qt libraries (ensuring reverse engineering is permitted for debugging modifications to those libraries).
2. **GPL-3.0 (GNU General Public License v3):**
   * Certain optional modules, extensions, and build tools within Qt are available only under GPL-3.0. If any GPL-3.0-only Qt module is utilized, the entire combined software distribution must be licensed under GPL-3.0.

### B. Qt Commercial Edition:
* Developed and distributed by **The Qt Company** ([qt.io](https://www.qt.io)).
* Provides proprietary commercial licensing without open-source copyleft obligations, source code disclosure requirements, or relinking restrictions. Required for organizations unwilling or unable to comply with LGPLv3 or GPLv3 terms.

---

## 5. Ollama & Qwen2.5 3B

* **Runtime:** Ollama ([ollama.com](https://ollama.com)) — MIT License
* **Model:** Qwen2.5 3B Instruct — Apache 2.0 License by Alibaba Cloud

---

## 6. PyMuPDF (`pymupdf`) (Stage 4)

* **Project:** [Artifex Software / PyMuPDF](https://github.com/pymupdf/PyMuPDF)
* **License:** GNU Affero General Public License v3.0 or later (**AGPL-3.0-or-later**) / Commercial
* **Copyright:** © 2012–2025 Artifex Software, Inc.

### Distribution & AGPL-3.0 Obligations:
* **Personal & Internal Use:** You may freely run and inspect PagePilot with PyMuPDF locally for personal or internal workflows.
* **Redistribution / Commercial Distribution:** Any distribution bundling or linking PyMuPDF requires complying with the AGPL-3.0 license (releasing source code of the entire application under AGPL-3.0), or obtaining a commercial license from Artifex Software.

---

## 7. ChromaDB (`chromadb`) (Stage 4)

* **Project:** [chroma-core/chroma](https://github.com/chroma-core/chroma)
* **License:** **Apache License 2.0**
* **Copyright:** © 2023–2025 Chroma
* Permissive Apache 2.0 license. Note that PagePilot explicitly disables ChromaDB telemetry via `Settings(anonymized_telemetry=False)`.

---

## 8. nomic-embed-text (Stage 4)

* **Model:** `nomic-embed-text` by Nomic AI
* **License:** **Apache License 2.0**
* **Weights:** 768-dimensional open-weights embedding model optimized for long context (8192 tokens). Used strictly offline via Ollama.

---

## 9. python-docx (Stage 5)

* **Project:** [python-openxml/python-docx](https://github.com/python-openxml/python-docx)
* **License:** **MIT License**
* **Copyright:** © 2013 Steve Canny and contributors
* **Usage in PagePilot:** Generates native Microsoft Word (.docx) documents from structured Markdown, headings, tables, callout blocks, and verified PDF citations.
* Permissive MIT license allows commercial use, modification, distribution, and private use, provided that the copyright and permission notice are preserved.

---

## 10. lxml (Stage 5)

* **Project:** [lxml/lxml](https://github.com/lxml/lxml)
* **License:** **BSD-3-Clause** (with elements under ZPL 2.0 and GPL)
* **Copyright:** © 2004 Stefan Behnel, Martijn Faassen and contributors
* **Usage in PagePilot:** High-performance XML processing underlying python-docx for OpenXML document manipulation.
* Permissive BSD license allows distribution and use in source and binary forms with copyright notice retention.

---

## 11. PaddleOCR & PaddlePaddle (Stage 6B)

* **Project:** [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
* **Framework:** [PaddlePaddle/Paddle](https://github.com/PaddlePaddle/Paddle)
* **License:** **Apache License 2.0**
* **Copyright:** © 2020–2025 PaddlePaddle Authors
* **Usage in PagePilot:** Offline text detection and recognition for printed documents and handwritten student notes via PP-OCRv5 architectures. Runs in an isolated Python 3.12 subprocess worker (`.venv-ocr/`) to maintain compatibility with the primary Python 3.14 desktop runtime.
* **Model Assets:**
  - `ch_PP-OCRv5_det` (Text line detection): Apache-2.0
  - `en_PP-OCRv5_rec` (English text recognition): Apache-2.0
  - `ch_PP-OCRv5_server_rec` (Server handwriting recognition): Apache-2.0
* **Distribution Policy:** Large pre-trained weight files are downloaded on-demand by user action and stored in `%APPDATA%\PagePilot\models\ocr\paddle\`. Weights are never committed to the Git repository.

---

## 12. Pix2Text & Formula Recognition (Stage 6B)

* **Project:** [breezedeus/Pix2Text](https://github.com/breezedeus/Pix2Text)
* **Author:** © 2022–2025 Breezedeus
* **License:** **MIT License**
* **Usage in PagePilot:** Mathematical Formula Detection (MFD) and Mathematical Formula Recognition (MFR), converting formula image regions into structured, editable LaTeX expressions.
* **Component Model Licenses:**
  - Mathematical Formula Detection (`mfd-1.5` / `doclayout-yolo`): Apache-2.0 / GPL dual-consideration; runs as an isolated subprocess.
  - Mathematical Formula Recognition (`mfr-1.5` / `LaTeX-OCR`): MIT License (Lukas Blecher).
* **Distribution Policy:** Formula weights reside in `%APPDATA%\PagePilot\models\ocr\pix2text\` outside version control.

---

## 13. ONNX Runtime (Stage 6B)

* **Project:** [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)
* **License:** **MIT License**
* **Copyright:** © Microsoft Corporation
* **Usage in PagePilot:** High-efficiency CPU inference for formula detection and recognition models within the isolated OCR worker environment.
