"""
tests/test_ollama_client.py

Unit tests for the Ollama API client.
These tests run against the live local Ollama server when available.
Tests that require Ollama are skipped gracefully when it is not running.

Run with:
    python -m pytest tests/test_ollama_client.py -v
"""

import json
import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pagepilot.ai.ollama_client import ChatMessage, OllamaClient, OllamaError, OllamaStatus


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _server_running() -> bool:
    """Return True if the Ollama server is reachable on localhost."""
    client = OllamaClient()
    status = client.check_status()
    return status.server_reachable


def _model_available() -> bool:
    """Return True if qwen2.5:3b is installed."""
    client = OllamaClient()
    status = client.check_status()
    return status.model_available


# ──────────────────────────────────────────────────────────────────────────────
# OllamaStatus dataclass
# ──────────────────────────────────────────────────────────────────────────────

class TestOllamaStatus(unittest.TestCase):
    def test_not_ready_when_both_false(self):
        s = OllamaStatus(server_reachable=False, model_available=False)
        self.assertFalse(s.ready)

    def test_not_ready_when_server_only(self):
        s = OllamaStatus(server_reachable=True, model_available=False)
        self.assertFalse(s.ready)

    def test_ready_when_both_true(self):
        s = OllamaStatus(server_reachable=True, model_available=True)
        self.assertTrue(s.ready)

    def test_default_model_name(self):
        from pagepilot.config import MODEL_NAME
        s = OllamaStatus()
        self.assertEqual(s.model_name, MODEL_NAME)


# ──────────────────────────────────────────────────────────────────────────────
# Live connectivity tests (skipped if Ollama not running)
# ──────────────────────────────────────────────────────────────────────────────

class TestOllamaClientLive(unittest.TestCase):

    def setUp(self):
        if not _server_running():
            self.skipTest("Ollama server not running — skipping live tests")
        self.client = OllamaClient()

    def test_server_reachable(self):
        status = self.client.check_status()
        self.assertTrue(status.server_reachable, "Ollama server should be reachable")

    def test_model_available(self):
        status = self.client.check_status()
        self.assertTrue(
            status.model_available,
            f"Model '{self.client.model}' should be installed. "
            f"Run: ollama pull {self.client.model}"
        )

    def test_status_ready_when_model_present(self):
        status = self.client.check_status()
        if status.model_available:
            self.assertTrue(status.ready)

    def test_wrong_model_not_available(self):
        client = OllamaClient(model="definitely-not-installed:99b")
        status = client.check_status()
        self.assertTrue(status.server_reachable)
        self.assertFalse(status.model_available)
        self.assertIn("not installed", status.error.lower())


# ──────────────────────────────────────────────────────────────────────────────
# Live streaming test
# ──────────────────────────────────────────────────────────────────────────────

class TestOllamaStreaming(unittest.TestCase):

    def setUp(self):
        if not _model_available():
            self.skipTest("qwen2.5:3b not available — skipping streaming tests")
        self.client = OllamaClient(num_ctx=512)

    def test_stream_returns_tokens(self):
        messages = [
            ChatMessage("system", "You are a test assistant. Be very brief."),
            ChatMessage("user", "Reply with exactly three words."),
        ]
        tokens = list(self.client.chat_stream(messages))
        self.assertGreater(len(tokens), 0, "Should receive at least one token")
        full = "".join(tokens)
        self.assertGreater(len(full.strip()), 0, "Response should not be empty")

    def test_stream_cancellation(self):
        messages = [
            ChatMessage("user", "Count from 1 to 1000 slowly."),
        ]
        cancel_flag = [False]
        received = []
        gen = self.client.chat_stream(messages, cancel_flag=cancel_flag)
        for i, token in enumerate(gen):
            received.append(token)
            if i >= 2:
                cancel_flag[0] = True
                break
        # We should have received some tokens before cancellation
        self.assertGreater(len(received), 0)

    def test_malformed_host_raises_ollama_error(self):
        bad_client = OllamaClient(host="http://127.0.0.1:9999", timeout=2)
        messages = [ChatMessage("user", "hi")]
        with self.assertRaises(OllamaError):
            list(bad_client.chat_stream(messages))


# ──────────────────────────────────────────────────────────────────────────────
# Mocked unit tests (no live server needed)
# ──────────────────────────────────────────────────────────────────────────────

class TestOllamaClientMocked(unittest.TestCase):

    def test_server_unreachable_returns_error_status(self):
        import urllib.error
        client = OllamaClient(host="http://127.0.0.1:9999")
        with patch("urllib.request.urlopen",
                   side_effect=urllib.error.URLError("connection refused")):
            status = client.check_status()
        self.assertFalse(status.server_reachable)
        self.assertFalse(status.model_available)
        self.assertIn("Cannot reach Ollama", status.error)

    def test_check_status_parses_model_list(self):
        fake_response = MagicMock()
        fake_response.read.return_value = json.dumps({
            "models": [{"name": "qwen2.5:3b"}, {"name": "llama3:8b"}]
        }).encode()
        fake_response.__enter__ = lambda s: s
        fake_response.__exit__ = MagicMock(return_value=False)

        client = OllamaClient(model="qwen2.5:3b")
        with patch("urllib.request.urlopen", return_value=fake_response):
            status = client.check_status()
        self.assertTrue(status.server_reachable)
        self.assertTrue(status.model_available)

    def test_check_status_model_missing(self):
        fake_response = MagicMock()
        fake_response.read.return_value = json.dumps({
            "models": [{"name": "llama3:8b"}]
        }).encode()
        fake_response.__enter__ = lambda s: s
        fake_response.__exit__ = MagicMock(return_value=False)

        client = OllamaClient(model="qwen2.5:3b")
        with patch("urllib.request.urlopen", return_value=fake_response):
            status = client.check_status()
        self.assertTrue(status.server_reachable)
        self.assertFalse(status.model_available)
        self.assertIn("not installed", status.error)

    def test_chat_stream_parses_ndjson(self):
        lines = [
            b'{"message":{"role":"assistant","content":"Hello"},"done":false}\n',
            b'{"message":{"role":"assistant","content":" world"},"done":false}\n',
            b'{"message":{"role":"assistant","content":""},"done":true}\n',
        ]
        fake_response = MagicMock()
        fake_response.__iter__ = MagicMock(return_value=iter(lines))
        fake_response.__enter__ = lambda s: s
        fake_response.__exit__ = MagicMock(return_value=False)

        client = OllamaClient()
        with patch("urllib.request.urlopen", return_value=fake_response):
            tokens = list(client.chat_stream([ChatMessage("user", "hi")]))
        self.assertEqual(tokens, ["Hello", " world"])

    def test_chat_stream_skips_malformed_lines(self):
        lines = [
            b'not-json\n',
            b'{"message":{"role":"assistant","content":"ok"},"done":true}\n',
        ]
        fake_response = MagicMock()
        fake_response.__iter__ = MagicMock(return_value=iter(lines))
        fake_response.__enter__ = lambda s: s
        fake_response.__exit__ = MagicMock(return_value=False)

        client = OllamaClient()
        with patch("urllib.request.urlopen", return_value=fake_response):
            tokens = list(client.chat_stream([ChatMessage("user", "hi")]))
        self.assertEqual(tokens, ["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

