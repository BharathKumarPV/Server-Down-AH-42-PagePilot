"""
PagePilot – Stage 6B OCR Worker Subprocess Client.

Manages the lifecycle, health-checking, and JSON-RPC communication with the
isolated worker process. Handles timeouts, process crashes, and thread-safety.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from pagepilot.config import OCR_MODELS_DIR, OCR_WORKER_TIMEOUT
from pagepilot.ocr.models import BoundingBox, OCRRegion, RegionType, ApprovalStatus
from pagepilot.ocr.worker_protocol import (
    CMD_PING,
    CMD_PROCESS_PAGE,
    CMD_RECOGNIZE_FORMULA,
    CMD_RECOGNIZE_TEXT,
    CMD_SHUTDOWN,
    STATUS_OK,
    deserialize_response,
    serialize_request,
)


class OCRWorkerError(Exception):
    """Raised when an error occurs during OCR worker communication or execution."""


class OCRWorkerClient:
    """Thread-safe client managing an isolated OCR worker subprocess."""

    def __init__(
        self,
        python_path: Optional[str] = None,
        worker_script_path: Optional[str] = None,
        timeout: int = OCR_WORKER_TIMEOUT,
        simulation_mode: Optional[bool] = None,
        mock_engine: Optional[Any] = None,
    ):
        self._python_path = python_path or self._resolve_python_executable()
        self._script_path = worker_script_path or str(
            Path(__file__).parent / "worker_process.py"
        )
        self._timeout = timeout
        if simulation_mode is not None:
            self._simulation_mode = simulation_mode
        else:
            self._simulation_mode = os.environ.get("PAGEPILOT_OCR_SIMULATION_MODE", "0") == "1"
        self._mock_engine = mock_engine
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._worker_info: Dict[str, Any] = {}

    @property
    def simulation_mode(self) -> bool:
        return self._simulation_mode

    @simulation_mode.setter
    def simulation_mode(self, value: bool):
        self._simulation_mode = bool(value)

    @classmethod
    def _resolve_python_executable(cls) -> str:
        """
        Locate the best available Python interpreter for the OCR worker.
        Prefers isolated Python 3.12/.venv-ocr, then falls back to current python.
        """
        env_py = os.environ.get("PAGEPILOT_OCR_PYTHON")
        if env_py and Path(env_py).exists():
            return env_py

        # Check local virtual environment .venv-ocr
        local_venv_py = Path(".venv-ocr") / "Scripts" / "python.exe"
        if local_venv_py.exists():
            return str(local_venv_py.resolve())

        # Check common Windows Python 3.12 install paths
        candidates = [
            Path.home() / "AppData" / "Local" / "Programs" / "Python" / "Python312" / "python.exe",
            Path.home() / "AppData" / "Roaming" / "uv" / "python" / "cpython-3.12" / "python.exe",
        ]
        for c in candidates:
            if c.exists():
                return str(c.resolve())

        # Fallback to current running Python interpreter
        return sys.executable

    @property
    def python_path(self) -> str:
        return self._python_path

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.poll() is None

    def start(self) -> bool:
        """Start the worker subprocess and perform an initial handshake ping."""
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return True

            env = os.environ.copy()
            env["PAGEPILOT_OCR_MODELS_DIR"] = str(OCR_MODELS_DIR)
            env["PYTHONIOENCODING"] = "utf-8"
            if self._simulation_mode:
                env["PAGEPILOT_OCR_SIMULATION_MODE"] = "1"
            else:
                env["PAGEPILOT_OCR_SIMULATION_MODE"] = "0"

            try:
                self._process = subprocess.Popen(
                    [self._python_path, self._script_path],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    encoding="utf-8",
                    env=env,
                )
            except Exception as exc:
                raise OCRWorkerError(f"Failed to start OCR worker subprocess using {self._python_path}: {exc}") from exc

        # Test connection outside lock to allow normal messaging
        pong = self.ping()
        self._worker_info = pong.get("payload", {})
        return True

    def ping(self) -> Dict[str, Any]:
        """Send health-check ping to the worker."""
        return self._send_command(CMD_PING)

    def get_info(self) -> Dict[str, Any]:
        """Return diagnostic info returned by the worker."""
        if not self._worker_info:
            self.ping()
        return self._worker_info

    def process_page(
        self,
        image_path: str,
        document_id: int,
        page_number: int,
        filename: str,
        file_hash: str,
        lang: str = "en",
        timeout: Optional[int] = None,
    ) -> List[OCRRegion]:
        """
        Request complete page OCR and formula extraction from the worker.
        Returns a list of typed OCRRegion objects.
        """
        if self._mock_engine is not None:
            raw = self._mock_engine.process_page(
                image_path=image_path,
                document_id=document_id,
                page_number=page_number,
                filename=filename,
                file_hash=file_hash,
                lang=lang,
            )
            raw_regions = raw.get("regions", []) if isinstance(raw, dict) else raw
            return [r if isinstance(r, OCRRegion) else OCRRegion.from_dict(r) for r in raw_regions]

        args = {
            "image_path": str(Path(image_path).resolve()),
            "document_id": document_id,
            "page_number": page_number,
            "filename": filename,
            "file_hash": file_hash,
            "lang": lang,
            "simulation_mode": self._simulation_mode,
        }
        resp = self._send_command(CMD_PROCESS_PAGE, args, timeout=timeout)
        payload = resp.get("payload", {})
        raw_regions = payload.get("regions", [])

        results = []
        for r in raw_regions:
            results.append(OCRRegion.from_dict(r))
        return results

    def recognize_text(
        self,
        image_path: str,
        lang: str = "en",
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Recognize prose/handwriting on an image."""
        if self._mock_engine is not None and hasattr(self._mock_engine, "recognize_text"):
            return self._mock_engine.recognize_text(image_path=image_path, lang=lang)

        args = {
            "image_path": str(Path(image_path).resolve()),
            "lang": lang,
            "simulation_mode": self._simulation_mode,
        }
        resp = self._send_command(CMD_RECOGNIZE_TEXT, args, timeout=timeout)
        return resp.get("payload", {})

    def recognize_formula(
        self,
        image_path: str,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Recognize a mathematical formula and return LaTeX."""
        if self._mock_engine is not None and hasattr(self._mock_engine, "recognize_formula"):
            return self._mock_engine.recognize_formula(image_path=image_path)

        args = {
            "image_path": str(Path(image_path).resolve()),
            "simulation_mode": self._simulation_mode,
        }
        resp = self._send_command(CMD_RECOGNIZE_FORMULA, args, timeout=timeout)
        return resp.get("payload", {})

    def stop(self):
        """Send shutdown signal to worker and wait for termination."""
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                self._process = None
                return

            try:
                line = serialize_request(CMD_SHUTDOWN)
                self._process.stdin.write(line)
                self._process.stdin.flush()
                self._process.wait(timeout=2.0)
            except Exception:
                try:
                    self._process.terminate()
                    self._process.wait(timeout=1.0)
                except Exception:
                    pass
            finally:
                self._process = None

    def _send_command(
        self,
        command: str,
        args: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Send an NDJSON request to the worker and read the response line."""
        effective_timeout = timeout or self._timeout
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                # Restart if not running
                self._lock.release()
                try:
                    self.start()
                finally:
                    self._lock.acquire()

            req_line = serialize_request(command, args=args)
            try:
                self._process.stdin.write(req_line)
                self._process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                self._process = None
                raise OCRWorkerError(f"Worker process communication failed: {exc}") from exc

            # Read response line
            resp_line = self._process.stdout.readline()
            if not resp_line:
                exit_code = self._process.poll()
                stderr = self._process.stderr.read() if self._process.stderr else ""
                self._process = None
                raise OCRWorkerError(f"Worker process terminated unexpectedly (exit code {exit_code}). Stderr: {stderr}")

            resp = deserialize_response(resp_line)
            if resp.get("status") != STATUS_OK:
                err = resp.get("error") or "Unknown worker error"
                raise OCRWorkerError(f"Worker returned error: {err}")

            return resp

