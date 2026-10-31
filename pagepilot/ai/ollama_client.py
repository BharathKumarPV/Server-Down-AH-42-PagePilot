"""
PagePilot – Ollama API client.

Handles all HTTP communication with the local Ollama service.
This module is intentionally free of Qt dependencies so it can be
used from worker threads and unit tests.

Supported Ollama API: POST /api/chat  (streaming NDJSON)
Tested against Ollama 0.40.13.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Generator, Iterator

from pagepilot.config import OLLAMA_HOST, MODEL_NAME, NUM_CTX, REQUEST_TIMEOUT


# ──────────────────────────────────────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ChatMessage:
    role: str          # "system" | "user" | "assistant"
    content: str


@dataclass
class OllamaStatus:
    server_reachable: bool = False
    model_available: bool = False
    model_name: str = MODEL_NAME
    error: str = ""

    @property
    def ready(self) -> bool:
        return self.server_reachable and self.model_available


# ──────────────────────────────────────────────────────────────────────────────
# Ollama client
# ──────────────────────────────────────────────────────────────────────────────

class OllamaClient:
    """
    Thin wrapper around the Ollama REST API.

    Uses only stdlib urllib so we don't need the 'requests' package.
    All network calls are blocking — call from a background thread.
    """

    def __init__(
        self,
        host: str = OLLAMA_HOST,
        model: str = MODEL_NAME,
        num_ctx: int = NUM_CTX,
        timeout: int = REQUEST_TIMEOUT,
    ):
        self.host    = host.rstrip("/")
        self.model   = model
        self.num_ctx = num_ctx
        self.timeout = timeout

    # ── Connectivity ─────────────────────────────────────────────────

    def check_status(self) -> OllamaStatus:
        """
        Query /api/tags to test server reachability and model presence.
        Never raises — always returns an OllamaStatus.
        """
        status = OllamaStatus(model_name=self.model)
        try:
            url = f"{self.host}/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                status.server_reachable = True
                installed = [m.get("name", "") for m in data.get("models", [])]
                status.model_available = self.model in installed
                if not status.model_available:
                    status.error = (
                        f"Model '{self.model}' is not installed.\n"
                        f"Run:  ollama pull {self.model}"
                    )
        except urllib.error.URLError as exc:
            status.error = (
                f"Cannot reach Ollama at {self.host}.\n"
                f"Make sure Ollama is running (run: ollama serve).\n"
                f"Detail: {exc.reason}"
            )
        except Exception as exc:
            status.error = f"Unexpected error checking Ollama: {exc}"
        return status

    # ── Chat (streaming) ─────────────────────────────────────────────

    def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        cancel_flag: list[bool] | None = None,
    ) -> Generator[str, None, None]:
        """
        Send a chat request to Ollama and yield response tokens as they arrive.

        Parameters
        ----------
        messages:
            Full conversation history including the system prompt.
        cancel_flag:
            Mutable list[bool] – if cancel_flag[0] is set to True from another
            thread, streaming stops early and a GeneratorExit is raised.

        Yields
        ------
        str — individual text tokens from the model.

        Raises
        ------
        OllamaError   — on server-side or network error.
        StopIteration — on clean completion.
        """
        if cancel_flag is None:
            cancel_flag = [False]

        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "options": {
                "num_ctx": self.num_ctx,
            },
        }
        body = json.dumps(payload).encode("utf-8")
        url  = f"{self.host}/api/chat"
        req  = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for raw_line in resp:
                    if cancel_flag[0]:
                        return   # caller cancelled

                    line = raw_line.strip()
                    if not line:
                        continue

                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue   # skip malformed lines

                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token

                    if chunk.get("done"):
                        return

        except urllib.error.URLError as exc:
            raise OllamaError(f"Connection to Ollama lost: {exc.reason}") from exc
        except TimeoutError as exc:
            raise OllamaError(
                f"Ollama did not respond within {self.timeout}s. "
                "Try a shorter message or increase the timeout in Settings."
            ) from exc
        except Exception as exc:
            raise OllamaError(f"Unexpected error during streaming: {exc}") from exc


class OllamaError(Exception):
    """Raised when Ollama returns an error or the connection fails."""

