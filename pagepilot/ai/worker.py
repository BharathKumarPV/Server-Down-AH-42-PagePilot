"""
PagePilot – Inference worker (QThread-based).

Runs blocking Ollama streaming in a background thread and reports
progress back to the UI via Qt signals.

Thread safety rules:
- Never touch QWidget from this thread.
- Communicate only through signals.
- cancel_flag is a shared mutable list[bool]; the UI thread sets [0]=True.
"""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from pagepilot.ai.ollama_client import ChatMessage, OllamaClient, OllamaError


class InferenceWorker(QThread):
    """
    Background thread that streams a single Ollama response.

    Signals
    -------
    token_received(str)        — one streamed token
    finished(str)              — complete accumulated text on clean stop
    error_occurred(str)        — human-readable error message
    cancelled()                — generation was cancelled by the user
    """

    token_received = Signal(str)
    finished       = Signal(str)
    error_occurred = Signal(str)
    cancelled      = Signal()

    def __init__(
        self,
        client: OllamaClient,
        messages: list[ChatMessage],
        parent=None,
    ):
        super().__init__(parent)
        self._client   = client
        self._messages = messages
        self._cancel   = [False]   # shared flag; set via cancel()
        self._accumulated = ""

    def cancel(self):
        """Request cancellation. Safe to call from the UI thread."""
        self._cancel[0] = True

    def run(self):
        """Execute on the background thread."""
        try:
            for token in self._client.chat_stream(
                self._messages,
                cancel_flag=self._cancel,
            ):
                if self._cancel[0]:
                    self.cancelled.emit()
                    return
                self._accumulated += token
                self.token_received.emit(token)

            if self._cancel[0]:
                self.cancelled.emit()
            else:
                self.finished.emit(self._accumulated)

        except OllamaError as exc:
            self.error_occurred.emit(str(exc))
        except Exception as exc:
            self.error_occurred.emit(f"Unexpected inference error: {exc}")

