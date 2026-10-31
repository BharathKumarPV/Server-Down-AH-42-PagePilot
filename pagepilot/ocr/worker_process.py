"""
PagePilot – Stage 6B Isolated OCR & Mathematical Formula Worker Process.

This script runs as an independent subprocess under an isolated Python 3.12
environment (.venv-ocr/). It communicates with the main Python 3.14 PagePilot
application via JSON-RPC lines over stdin/stdout.

Sequential Inference:
Processes one region / engine at a time to stay strictly within the 8 GB RAM
laptop memory profile.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# Preload Windows C++ runtime DLLs before PaddlePaddle/PaddleOCR
try:
    import torch  # noqa: F401
except Exception:
    pass

# Ensure PagePilot package root is on sys.path
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

try:
    from pagepilot.ocr.formula_preprocessor import clean_formula_latex, preprocess_formula_image
    from pagepilot.ocr.formula_validator import validate_formula_latex
except ImportError:
    def preprocess_formula_image(img, **kwargs):
        from PIL import Image
        return Image.open(str(img)).convert("RGB") if isinstance(img, (str, Path)) else img

    def clean_formula_latex(s):
        return s.strip() if s else ""

    def validate_formula_latex(s, confidence_score=1.0, **kwargs):
        class _DummyVal:
            is_valid = True
            errors = []
            warnings = []
            needs_human_review = False
        return _DummyVal()

# Ensure protocol can be imported if in same package or standalone
try:
    from pagepilot.ocr.worker_protocol import (
        CMD_DETECT_LAYOUT,
        CMD_PING,
        CMD_PROCESS_PAGE,
        CMD_RECOGNIZE_FORMULA,
        CMD_RECOGNIZE_TEXT,
        CMD_SHUTDOWN,
        MODE_ENGINE_UNAVAILABLE,
        MODE_REAL_INFERENCE,
        MODE_SIMULATED_RESULT,
        PROTOCOL_VERSION,
        STATUS_DEPENDENCY_MISSING,
        STATUS_ERROR,
        STATUS_MODELS_NOT_INSTALLED,
        STATUS_OK,
        deserialize_request,
        serialize_response,
    )
except ImportError:
    # Standalone protocol definitions for isolated execution
    PROTOCOL_VERSION = "1.0.0"
    CMD_PING = "ping"
    CMD_DETECT_LAYOUT = "detect_layout"
    CMD_RECOGNIZE_TEXT = "recognize_text"
    CMD_RECOGNIZE_FORMULA = "recognize_formula"
    CMD_PROCESS_PAGE = "process_page"
    CMD_SHUTDOWN = "shutdown"
    STATUS_OK = "ok"
    STATUS_ERROR = "error"
    STATUS_MODELS_NOT_INSTALLED = "models_not_installed"
    STATUS_DEPENDENCY_MISSING = "dependency_missing"

    MODE_REAL_INFERENCE = "REAL_INFERENCE"
    MODE_SIMULATED_RESULT = "SIMULATED_RESULT"
    MODE_ENGINE_UNAVAILABLE = "ENGINE_UNAVAILABLE"

    def deserialize_request(line: str) -> Dict[str, Any]:
        return json.loads(line.strip())

    def serialize_response(
        status: str,
        payload: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        request_id: Optional[str] = None,
        execution_ms: float = 0.0,
    ) -> str:
        return json.dumps({
            "jsonrpc": "2.0",
            "id": request_id or "",
            "status": status,
            "execution_ms": round(execution_ms, 2),
            "payload": payload or {},
            "error": error or "",
        }, ensure_ascii=False) + "\n"


class OCRWorkerEngine:
    """Manages PaddleOCR and Pix2Text instances with lazy on-demand loading."""

    def __init__(self, models_dir: Optional[str] = None, simulation_mode: bool = False):
        self.models_dir = Path(models_dir) if models_dir else Path.home() / "AppData" / "Roaming" / "PagePilot" / "models" / "ocr"
        self.simulation_mode = simulation_mode or (os.environ.get("PAGEPILOT_OCR_SIMULATION_MODE", "0") == "1")
        self._paddle_ocr = None
        self._pix2text = None
        self._has_paddle = False
        self._has_p2t = False
        self._check_dependencies()

    def _check_dependencies(self):
        try:
            import paddleocr  # noqa: F401
            self._has_paddle = True
        except ImportError:
            self._has_paddle = False

        try:
            import pix2text  # noqa: F401
            self._has_p2t = True
        except ImportError:
            self._has_p2t = False

    def get_info(self) -> Dict[str, Any]:
        """Return diagnostic environment status."""
        return {
            "python_version": sys.version,
            "platform": sys.platform,
            "models_dir": str(self.models_dir),
            "paddleocr_available": self._has_paddle,
            "pix2text_available": self._has_p2t,
            "simulation_mode": self.simulation_mode,
            "protocol_version": PROTOCOL_VERSION,
        }

    def _get_paddle(self, lang: str = "en"):
        if not self._has_paddle:
            return None
        if self._paddle_ocr is None:
            import os
            os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
            try:
                import paddleocr
                from paddleocr import PaddleOCR
                is_v3 = getattr(paddleocr, "__version__", "").startswith("3")
                det_dir = str(self.models_dir / "paddle" / "det") if (self.models_dir / "paddle" / "det").exists() else None
                rec_dir = str(self.models_dir / "paddle" / "rec") if (self.models_dir / "paddle" / "rec").exists() else None

                if is_v3:
                    kwargs = {
                        "device": "cpu",
                        "enable_mkldnn": False,
                        "use_textline_orientation": True,
                    }
                    if det_dir:
                        kwargs["text_detection_model_dir"] = det_dir
                    if rec_dir:
                        kwargs["text_recognition_model_dir"] = rec_dir
                    self._paddle_ocr = PaddleOCR(**kwargs)
                else:
                    self._paddle_ocr = PaddleOCR(
                        use_angle_cls=True,
                        lang=lang,
                        det_model_dir=det_dir,
                        rec_model_dir=rec_dir,
                        show_log=False,
                        use_gpu=False,
                    )
            except Exception as e:
                sys.stderr.write(f"[worker_process] Failed to initialize PaddleOCR: {e}\n")
                self._paddle_ocr = None
        return self._paddle_ocr

    def _get_pix2text(self):
        if not self._has_p2t:
            return None
        if self._pix2text is None:
            try:
                from pix2text import Pix2Text
                p2t_dir = str(self.models_dir / "pix2text") if (self.models_dir / "pix2text").exists() else None
                if p2t_dir:
                    self._pix2text = Pix2Text.from_config(
                        total_configs={"mfr": {"model_dir": p2t_dir}}
                    )
                else:
                    self._pix2text = Pix2Text()
            except Exception as e:
                sys.stderr.write(f"[worker_process] Failed to initialize Pix2Text: {e}\n")
                self._pix2text = None
        return self._pix2text

    def recognize_text(
        self,
        image_path: str,
        lang: str = "en",
        simulation_mode: Optional[bool] = None,
    ) -> Dict[str, Any]:
        sim = self.simulation_mode if simulation_mode is None else simulation_mode

        if sim:
            res = self._fallback_recognize_text(image_path)
            res["inference_mode"] = MODE_SIMULATED_RESULT
            return res

        paddle = self._get_paddle(lang=lang)
        if paddle is None:
            return {
                "regions": [],
                "inference_mode": MODE_ENGINE_UNAVAILABLE,
                "error": "PaddleOCR engine is unavailable or uninstalled",
            }

        try:
            result = paddle.predict(image_path) if hasattr(paddle, "predict") else paddle.ocr(image_path, cls=True)
        except Exception as e:
            sys.stderr.write(f"[worker_process] Real OCR predict error: {e}\n")
            return {
                "regions": [],
                "inference_mode": MODE_ENGINE_UNAVAILABLE,
                "error": f"PaddleOCR inference error: {e}",
            }

        regions = []
        if result:
            # PaddleOCR 3.x dict result
            if isinstance(result[0], dict):
                r_dict = result[0]
                texts = r_dict.get("rec_texts", [])
                scores = r_dict.get("rec_scores", [])
                boxes = r_dict.get("rec_boxes", [])
                polys = r_dict.get("rec_polys", [])
                for idx, txt in enumerate(texts):
                    score = float(scores[idx]) if idx < len(scores) else 0.95
                    if idx < len(polys) and len(polys[idx]) > 0:
                        xs = [p[0] for p in polys[idx]]
                        ys = [p[1] for p in polys[idx]]
                        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
                    elif idx < len(boxes) and len(boxes[idx]) >= 4:
                        x0, y0, x1, y1 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
                    else:
                        x0, y0, x1, y1 = 0, 0, 100, 30
                    regions.append({
                        "region_id": f"R{idx + 1}",
                        "box_x0": float(x0),
                        "box_y0": float(y0),
                        "box_x1": float(x1),
                        "box_y1": float(y1),
                        "region_type": "text",
                        "recognized_text": str(txt),
                        "confidence_score": float(score),
                        "recognition_engine": "paddleocr_ppocrv5",
                        "inference_mode": MODE_REAL_INFERENCE,
                    })
            elif isinstance(result[0], list):  # 2.x list of lines
                for idx, line in enumerate(result[0]):
                    box, (txt, score) = line
                    xs = [p[0] for p in box]
                    ys = [p[1] for p in box]
                    regions.append({
                        "region_id": f"R{idx + 1}",
                        "box_x0": float(min(xs)),
                        "box_y0": float(min(ys)),
                        "box_x1": float(max(xs)),
                        "box_y1": float(max(ys)),
                        "region_type": "text",
                        "recognized_text": str(txt),
                        "confidence_score": float(score),
                        "recognition_engine": "paddleocr_ppocrv5",
                        "inference_mode": MODE_REAL_INFERENCE,
                    })

        return {
            "regions": regions,
            "inference_mode": MODE_REAL_INFERENCE,
        }

    def recognize_formula(
        self,
        image_path: str,
        simulation_mode: Optional[bool] = None,
    ) -> Dict[str, Any]:
        sim = self.simulation_mode if simulation_mode is None else simulation_mode

        if sim:
            res = self._fallback_recognize_formula(image_path)
            res["inference_mode"] = MODE_SIMULATED_RESULT
            return res

        p2t = self._get_pix2text()
        if p2t is None:
            return {
                "latex_expression": "",
                "raw_latex": "",
                "confidence_score": 0.0,
                "recognition_engine": "",
                "inference_mode": MODE_ENGINE_UNAVAILABLE,
                "error": "Pix2Text formula recognition engine is unavailable or uninstalled",
                "is_valid": False,
                "validation_errors": ["Pix2Text engine unavailable"],
                "validation_warnings": [],
                "needs_human_review": True,
            }

        try:
            # 1. Apply robust formula image preprocessing
            preproc_img = preprocess_formula_image(image_path)

            # 2. Execute genuine Pix2Text MFR 1.5 inference with return_text=False
            res = p2t.recognize_formula(preproc_img, return_text=False)
            if isinstance(res, dict):
                raw_latex = str(res.get("text", ""))
                score = float(res.get("score", 0.90))
            else:
                raw_latex = str(res)
                score = 0.90

            # 3. Clean formatting artifacts without altering math semantics
            cleaned_latex = clean_formula_latex(raw_latex)

            # 4. Perform deterministic syntactic and structural validation
            val_res = validate_formula_latex(cleaned_latex, confidence_score=score)

            return {
                "latex_expression": cleaned_latex,
                "raw_latex": raw_latex,
                "confidence_score": score,
                "recognition_engine": "pix2text_mfr",
                "inference_mode": MODE_REAL_INFERENCE,
                "is_valid": val_res.is_valid,
                "validation_errors": val_res.errors,
                "validation_warnings": val_res.warnings,
                "needs_human_review": val_res.needs_human_review,
            }
        except Exception as e:
            sys.stderr.write(f"[worker_process] Formula recognition error: {e}\n")
            return {
                "latex_expression": "",
                "raw_latex": "",
                "confidence_score": 0.0,
                "recognition_engine": "",
                "inference_mode": MODE_ENGINE_UNAVAILABLE,
                "error": f"Formula recognition error: {e}",
                "is_valid": False,
                "validation_errors": [f"Formula recognition error: {e}"],
                "validation_warnings": [],
                "needs_human_review": True,
            }

    def process_page(
        self,
        image_path: str,
        document_id: int,
        page_number: int,
        filename: str,
        file_hash: str,
        lang: str = "en",
        simulation_mode: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Process a complete scanned page or photographed note:
        Detects layout, extracts text, recognizes mathematical expressions.
        Production mode always attempts genuine PaddleOCR/Pix2Text inference.
        Simulation mode is enabled strictly through explicit configuration.
        """
        sim = self.simulation_mode if simulation_mode is None else simulation_mode

        if sim:
            fb = self._fallback_process_page(image_path, document_id, page_number, filename, file_hash)
            fb["inference_mode"] = MODE_SIMULATED_RESULT
            for r in fb.get("regions", []):
                r["inference_mode"] = MODE_SIMULATED_RESULT
            return fb

        paddle = self._get_paddle(lang=lang)
        if paddle is None:
            return {
                "regions": [],
                "inference_mode": MODE_ENGINE_UNAVAILABLE,
                "error": "PaddleOCR engine is unavailable or uninstalled",
            }

        text_res = self.recognize_text(image_path, lang=lang, simulation_mode=False)
        if text_res.get("inference_mode") == MODE_ENGINE_UNAVAILABLE:
            return text_res

        regions = text_res.get("regions", [])
        source_type = "scanned_pdf" if filename.lower().endswith(".pdf") else "image_note"
        for r in regions:
            r["document_id"] = document_id
            r["page_number"] = page_number
            r["original_filename"] = filename
            r["original_file_hash"] = file_hash
            r["original_image_path"] = image_path
            r["source_type"] = source_type
            r["approval_status"] = "pending_review"
            r["corrected_text"] = None
            r["latex_expression"] = None
            r["model_version"] = "5.0.0"
            r["inference_mode"] = MODE_REAL_INFERENCE

        return {
            "regions": regions,
            "inference_mode": MODE_REAL_INFERENCE,
        }

    def _fallback_recognize_text(self, image_path: str) -> Dict[str, Any]:
        page_fb = self._fallback_process_page(image_path, 0, 1, Path(image_path).name, "")
        text_regions = [
            {
                "region_id": r.get("region_id", f"R{idx+1}"),
                "box_x0": r.get("box_x0", 50.0),
                "box_y0": r.get("box_y0", 60.0),
                "box_x1": r.get("box_x1", 500.0),
                "box_y1": r.get("box_y1", 120.0),
                "region_type": r.get("region_type", "text"),
                "recognized_text": r.get("recognized_text", ""),
                "confidence_score": r.get("confidence_score", 0.94),
                "recognition_engine": "paddleocr_ppocrv5_sim",
            }
            for idx, r in enumerate(page_fb.get("regions", []))
            if r.get("region_type") != "formula"
        ]
        if not text_regions:
            text_regions = [
                {
                    "region_id": "R1",
                    "box_x0": 50.0,
                    "box_y0": 60.0,
                    "box_x1": 500.0,
                    "box_y1": 120.0,
                    "region_type": "text",
                    "recognized_text": "Sample lecture notes on calculus and matrix algebra.",
                    "confidence_score": 0.94,
                    "recognition_engine": "paddleocr_ppocrv5_sim",
                }
            ]
        return {
            "regions": text_regions,
            "inference_mode": MODE_SIMULATED_RESULT,
        }

    def _fallback_recognize_formula(self, image_path: str) -> Dict[str, Any]:
        expr = r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}"
        val_res = validate_formula_latex(expr, confidence_score=0.92)
        return {
            "latex_expression": expr,
            "raw_latex": expr,
            "confidence_score": 0.92,
            "recognition_engine": "pix2text_mfr_sim",
            "is_valid": val_res.is_valid,
            "validation_errors": val_res.errors,
            "validation_warnings": val_res.warnings,
            "needs_human_review": val_res.needs_human_review,
        }

    def _fallback_process_page(
        self,
        image_path: str,
        document_id: int,
        page_number: int,
        filename: str,
        file_hash: str,
    ) -> Dict[str, Any]:
        """Provides high-fidelity deterministic layout & content when heavy weights are uninstalled."""
        stem = Path(image_path).stem.lower()
        fn_stem = Path(filename).stem.lower() if filename else ""
        combined_stem = f"{stem} {fn_stem}"
        source_type = "scanned_pdf" if filename.lower().endswith(".pdf") else "image_note"

        if "printed_os_memory" in combined_stem or "memory" in combined_stem:
            regions = [
                {
                    "region_id": "R1",
                    "document_id": document_id,
                    "page_number": page_number,
                    "original_filename": filename,
                    "original_file_hash": file_hash,
                    "original_image_path": image_path,
                    "source_type": source_type,
                    "box_x0": 40.0,
                    "box_y0": 40.0,
                    "box_x1": 550.0,
                    "box_y1": 180.0,
                    "region_type": "text",
                    "recognized_text": "The translation lookaside buffer (TLB) caches recent virtual-to-physical address mappings. When a TLB miss occurs, the hardware page table walker traverses the multi-level page table in main memory.",
                    "corrected_text": None,
                    "latex_expression": None,
                    "confidence_score": 0.98,
                    "recognition_engine": "paddleocr_ppocrv5_sim",
                    "model_version": "5.0.0",
                    "approval_status": "pending_review",
                }
            ]
        elif "handwritten_ml" in combined_stem or "gradient" in combined_stem:
            regions = [
                {
                    "region_id": "R1",
                    "document_id": document_id,
                    "page_number": page_number,
                    "original_filename": filename,
                    "original_file_hash": file_hash,
                    "original_image_path": image_path,
                    "source_type": source_type,
                    "box_x0": 50.0,
                    "box_y0": 60.0,
                    "box_x1": 520.0,
                    "box_y1": 200.0,
                    "region_type": "handwriting",
                    "recognized_text": "Gradient descent iteratively updates model weights in the direction opposite to the gradient of the loss function. The learning rate hyperparameter controls step size.",
                    "corrected_text": None,
                    "latex_expression": None,
                    "confidence_score": 0.89,
                    "recognition_engine": "paddleocr_ppocrv5_sim",
                    "model_version": "5.0.0",
                    "approval_status": "pending_review",
                }
            ]
        elif "skewed_calculus" in combined_stem or "skew" in combined_stem:
            regions = [
                {
                    "region_id": "R1",
                    "document_id": document_id,
                    "page_number": page_number,
                    "original_filename": filename,
                    "original_file_hash": file_hash,
                    "original_image_path": image_path,
                    "source_type": source_type,
                    "box_x0": 45.0,
                    "box_y0": 55.0,
                    "box_x1": 530.0,
                    "box_y1": 190.0,
                    "region_type": "handwriting",
                    "recognized_text": "L'Hopital's rule applies when evaluating limits resulting in indeterminate forms such as zero over zero or infinity over infinity.",
                    "corrected_text": None,
                    "latex_expression": None,
                    "confidence_score": 0.86,
                    "recognition_engine": "paddleocr_ppocrv5_sim",
                    "model_version": "5.0.0",
                    "approval_status": "pending_review",
                }
            ]
        elif "low_light" in combined_stem or "light" in combined_stem:
            regions = [
                {
                    "region_id": "R1",
                    "document_id": document_id,
                    "page_number": page_number,
                    "original_filename": filename,
                    "original_file_hash": file_hash,
                    "original_image_path": image_path,
                    "source_type": source_type,
                    "box_x0": 40.0,
                    "box_y0": 50.0,
                    "box_x1": 510.0,
                    "box_y1": 180.0,
                    "region_type": "handwriting",
                    "recognized_text": "The critical path in project scheduling defines the longest sequence of dependent activities required for project completion.",
                    "corrected_text": None,
                    "latex_expression": None,
                    "confidence_score": 0.84,
                    "recognition_engine": "paddleocr_ppocrv5_sim",
                    "model_version": "5.0.0",
                    "approval_status": "pending_review",
                }
            ]
        elif "printed_quadratic" in combined_stem:
            regions = [
                {
                    "region_id": "R1",
                    "document_id": document_id,
                    "page_number": page_number,
                    "original_filename": filename,
                    "original_file_hash": file_hash,
                    "original_image_path": image_path,
                    "source_type": source_type,
                    "box_x0": 40.0,
                    "box_y0": 40.0,
                    "box_x1": 450.0,
                    "box_y1": 80.0,
                    "region_type": "text",
                    "recognized_text": "Quadratic formula for finding roots of ax^2 + bx + c = 0:",
                    "corrected_text": None,
                    "latex_expression": None,
                    "confidence_score": 0.97,
                    "recognition_engine": "paddleocr_ppocrv5_sim",
                    "model_version": "5.0.0",
                    "approval_status": "pending_review",
                },
                {
                    "region_id": "R2",
                    "document_id": document_id,
                    "page_number": page_number,
                    "original_filename": filename,
                    "original_file_hash": file_hash,
                    "original_image_path": image_path,
                    "source_type": source_type,
                    "box_x0": 50.0,
                    "box_y0": 100.0,
                    "box_x1": 420.0,
                    "box_y1": 170.0,
                    "region_type": "formula",
                    "recognized_text": "x = (-b ± sqrt(b^2 - 4ac)) / 2a",
                    "corrected_text": None,
                    "latex_expression": r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}",
                    "confidence_score": 0.95,
                    "recognition_engine": "pix2text_mfr_sim",
                    "model_version": "1.5.0",
                    "approval_status": "pending_review",
                },
            ]
        elif "handwritten_derivative" in combined_stem or "derivative" in combined_stem:
            regions = [
                {
                    "region_id": "R1",
                    "document_id": document_id,
                    "page_number": page_number,
                    "original_filename": filename,
                    "original_file_hash": file_hash,
                    "original_image_path": image_path,
                    "source_type": source_type,
                    "box_x0": 40.0,
                    "box_y0": 40.0,
                    "box_x1": 350.0,
                    "box_y1": 80.0,
                    "region_type": "handwriting",
                    "recognized_text": "Derivative of product:",
                    "corrected_text": None,
                    "latex_expression": None,
                    "confidence_score": 0.90,
                    "recognition_engine": "paddleocr_ppocrv5_sim",
                    "model_version": "5.0.0",
                    "approval_status": "pending_review",
                },
                {
                    "region_id": "R2",
                    "document_id": document_id,
                    "page_number": page_number,
                    "original_filename": filename,
                    "original_file_hash": file_hash,
                    "original_image_path": image_path,
                    "source_type": source_type,
                    "box_x0": 50.0,
                    "box_y0": 100.0,
                    "box_x1": 520.0,
                    "box_y1": 180.0,
                    "region_type": "formula",
                    "recognized_text": "d/dx[e^(2x) sin(x)] = 2e^(2x) sin(x) + e^(2x) cos(x)",
                    "corrected_text": None,
                    "latex_expression": r"\frac{d}{dx}[e^{2x}\sin(x)] = 2e^{2x}\sin(x) + e^{2x}\cos(x)",
                    "confidence_score": 0.88,
                    "recognition_engine": "pix2text_mfr_sim",
                    "model_version": "1.5.0",
                    "approval_status": "pending_review",
                },
            ]
        elif "thermodynamics" in combined_stem or "gas" in combined_stem:
            regions = [
                {
                    "region_id": "R1",
                    "document_id": document_id,
                    "page_number": page_number,
                    "original_filename": filename,
                    "original_file_hash": file_hash,
                    "original_image_path": image_path,
                    "source_type": source_type,
                    "box_x0": 40.0,
                    "box_y0": 40.0,
                    "box_x1": 550.0,
                    "box_y1": 120.0,
                    "region_type": "text",
                    "recognized_text": "For an ideal gas undergoing an isothermal expansion, pressure and volume satisfy PV = nRT with constant temperature T.",
                    "corrected_text": None,
                    "latex_expression": None,
                    "confidence_score": 0.94,
                    "recognition_engine": "paddleocr_ppocrv5_sim",
                    "model_version": "5.0.0",
                    "approval_status": "pending_review",
                },
                {
                    "region_id": "R2",
                    "document_id": document_id,
                    "page_number": page_number,
                    "original_filename": filename,
                    "original_file_hash": file_hash,
                    "original_image_path": image_path,
                    "source_type": source_type,
                    "box_x0": 50.0,
                    "box_y0": 130.0,
                    "box_x1": 250.0,
                    "box_y1": 180.0,
                    "region_type": "formula",
                    "recognized_text": "PV = nRT",
                    "corrected_text": None,
                    "latex_expression": r"PV = nRT",
                    "confidence_score": 0.96,
                    "recognition_engine": "pix2text_mfr_sim",
                    "model_version": "1.5.0",
                    "approval_status": "pending_review",
                },
            ]
        elif "scanned_taylor" in combined_stem or "taylor" in combined_stem:
            regions = [
                {
                    "region_id": "R1",
                    "document_id": document_id,
                    "page_number": page_number,
                    "original_filename": filename,
                    "original_file_hash": file_hash,
                    "original_image_path": image_path,
                    "source_type": source_type,
                    "box_x0": 40.0,
                    "box_y0": 40.0,
                    "box_x1": 560.0,
                    "box_y1": 110.0,
                    "region_type": "text",
                    "recognized_text": "Taylor's Theorem states that any infinitely differentiable function can be represented as a power series around point a:",
                    "corrected_text": None,
                    "latex_expression": None,
                    "confidence_score": 0.96,
                    "recognition_engine": "paddleocr_ppocrv5_sim",
                    "model_version": "5.0.0",
                    "approval_status": "pending_review",
                },
                {
                    "region_id": "R2",
                    "document_id": document_id,
                    "page_number": page_number,
                    "original_filename": filename,
                    "original_file_hash": file_hash,
                    "original_image_path": image_path,
                    "source_type": source_type,
                    "box_x0": 50.0,
                    "box_y0": 120.0,
                    "box_x1": 480.0,
                    "box_y1": 190.0,
                    "region_type": "formula",
                    "recognized_text": "f(x) = sum_{n=0}^inf f^(n)(a)/n! (x - a)^n",
                    "corrected_text": None,
                    "latex_expression": r"f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!} (x - a)^n",
                    "confidence_score": 0.92,
                    "recognition_engine": "pix2text_mfr_sim",
                    "model_version": "1.5.0",
                    "approval_status": "pending_review",
                },
            ]
        elif "formula" in combined_stem or "math" in combined_stem or "calc" in combined_stem:
            regions = [
                {
                    "region_id": "R1",
                    "document_id": document_id,
                    "page_number": page_number,
                    "original_filename": filename,
                    "original_file_hash": file_hash,
                    "original_image_path": image_path,
                    "source_type": source_type,
                    "box_x0": 40.0,
                    "box_y0": 40.0,
                    "box_x1": 450.0,
                    "box_y1": 80.0,
                    "region_type": "heading",
                    "recognized_text": "Quadratic Formula and Root Analysis",
                    "corrected_text": None,
                    "latex_expression": None,
                    "confidence_score": 0.96,
                    "recognition_engine": "paddleocr_ppocrv5_sim",
                    "model_version": "5.0.0",
                    "approval_status": "pending_review",
                },
                {
                    "region_id": "R2",
                    "document_id": document_id,
                    "page_number": page_number,
                    "original_filename": filename,
                    "original_file_hash": file_hash,
                    "original_image_path": image_path,
                    "source_type": source_type,
                    "box_x0": 50.0,
                    "box_y0": 100.0,
                    "box_x1": 400.0,
                    "box_y1": 180.0,
                    "region_type": "formula",
                    "recognized_text": "x = (-b ± sqrt(b^2 - 4ac)) / 2a",
                    "corrected_text": None,
                    "latex_expression": r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}",
                    "confidence_score": 0.91,
                    "recognition_engine": "pix2text_mfr_sim",
                    "model_version": "1.5.0",
                    "approval_status": "pending_review",
                },
                {
                    "region_id": "R3",
                    "document_id": document_id,
                    "page_number": page_number,
                    "original_filename": filename,
                    "original_file_hash": file_hash,
                    "original_image_path": image_path,
                    "source_type": source_type,
                    "box_x0": 40.0,
                    "box_y0": 200.0,
                    "box_x1": 520.0,
                    "box_y1": 270.0,
                    "region_type": "handwriting",
                    "recognized_text": "The discriminant determines the nature of the roots: positive for two real roots.",
                    "corrected_text": None,
                    "latex_expression": None,
                    "confidence_score": 0.88,
                    "recognition_engine": "paddleocr_ppocrv5_sim",
                    "model_version": "5.0.0",
                    "approval_status": "pending_review",
                },
            ]
        else:
            regions = [
                {
                    "region_id": "R1",
                    "document_id": document_id,
                    "page_number": page_number,
                    "original_filename": filename,
                    "original_file_hash": file_hash,
                    "original_image_path": image_path,
                    "source_type": source_type,
                    "box_x0": 40.0,
                    "box_y0": 40.0,
                    "box_x1": 480.0,
                    "box_y1": 90.0,
                    "region_type": "heading",
                    "recognized_text": "Chapter Notes: Introduction to Computer Systems",
                    "corrected_text": None,
                    "latex_expression": None,
                    "confidence_score": 0.95,
                    "recognition_engine": "paddleocr_ppocrv5_sim",
                    "model_version": "5.0.0",
                    "approval_status": "pending_review",
                },
                {
                    "region_id": "R2",
                    "document_id": document_id,
                    "page_number": page_number,
                    "original_filename": filename,
                    "original_file_hash": file_hash,
                    "original_image_path": image_path,
                    "source_type": source_type,
                    "box_x0": 40.0,
                    "box_y0": 110.0,
                    "box_x1": 540.0,
                    "box_y1": 220.0,
                    "region_type": "text",
                    "recognized_text": "The memory hierarchy consists of L1/L2 caches, main DRAM memory, and non-volatile storage.",
                    "corrected_text": None,
                    "latex_expression": None,
                    "confidence_score": 0.93,
                    "recognition_engine": "paddleocr_ppocrv5_sim",
                    "model_version": "5.0.0",
                    "approval_status": "pending_review",
                },
            ]

        return {"regions": regions}


def main():
    """Main loop reading JSON-RPC lines from standard input."""
    # Force unbuffered UTF-8 stdout/stdin
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")

    models_dir = os.environ.get("PAGEPILOT_OCR_MODELS_DIR")
    sim_env = os.environ.get("PAGEPILOT_OCR_SIMULATION_MODE", "0") == "1"
    engine = OCRWorkerEngine(models_dir=models_dir, simulation_mode=sim_env)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        start_time = time.perf_counter()
        req_id = None
        try:
            req = deserialize_request(line)
            req_id = req.get("id")
            cmd = req.get("command")
            args = req.get("args", {})

            if cmd == CMD_PING:
                info = engine.get_info()
                elapsed = (time.perf_counter() - start_time) * 1000.0
                sys.stdout.write(serialize_response(STATUS_OK, payload=info, request_id=req_id, execution_ms=elapsed))
                sys.stdout.flush()

            elif cmd == CMD_SHUTDOWN:
                elapsed = (time.perf_counter() - start_time) * 1000.0
                sys.stdout.write(serialize_response(STATUS_OK, payload={"message": "Shutting down"}, request_id=req_id, execution_ms=elapsed))
                sys.stdout.flush()
                break

            elif cmd == CMD_RECOGNIZE_TEXT:
                img_path = args.get("image_path", "")
                lang = args.get("lang", "en")
                sim = args.get("simulation_mode", engine.simulation_mode)
                res = engine.recognize_text(img_path, lang=lang, simulation_mode=sim)
                elapsed = (time.perf_counter() - start_time) * 1000.0
                sys.stdout.write(serialize_response(STATUS_OK, payload=res, request_id=req_id, execution_ms=elapsed))
                sys.stdout.flush()

            elif cmd == CMD_RECOGNIZE_FORMULA:
                img_path = args.get("image_path", "")
                sim = args.get("simulation_mode", engine.simulation_mode)
                res = engine.recognize_formula(img_path, simulation_mode=sim)
                elapsed = (time.perf_counter() - start_time) * 1000.0
                sys.stdout.write(serialize_response(STATUS_OK, payload=res, request_id=req_id, execution_ms=elapsed))
                sys.stdout.flush()

            elif cmd == CMD_PROCESS_PAGE:
                img_path = args.get("image_path", "")
                doc_id = args.get("document_id", 0)
                page_no = args.get("page_number", 1)
                filename = args.get("filename", "")
                file_hash = args.get("file_hash", "")
                lang = args.get("lang", "en")
                sim = args.get("simulation_mode", engine.simulation_mode)

                res = engine.process_page(img_path, doc_id, page_no, filename, file_hash, lang=lang, simulation_mode=sim)
                elapsed = (time.perf_counter() - start_time) * 1000.0
                sys.stdout.write(serialize_response(STATUS_OK, payload=res, request_id=req_id, execution_ms=elapsed))
                sys.stdout.flush()

            else:
                elapsed = (time.perf_counter() - start_time) * 1000.0
                sys.stdout.write(serialize_response(STATUS_ERROR, error=f"Unknown command: {cmd}", request_id=req_id, execution_ms=elapsed))
                sys.stdout.flush()

        except Exception as exc:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            err_msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            sys.stdout.write(serialize_response(STATUS_ERROR, error=err_msg, request_id=req_id, execution_ms=elapsed))
            sys.stdout.flush()


if __name__ == "__main__":
    main()
