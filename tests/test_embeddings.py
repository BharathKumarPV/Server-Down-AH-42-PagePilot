"""
Tests for Ollama embedding client (Stage 4).
"""

import json
from unittest.mock import MagicMock, patch
import urllib.error
import pytest

from pagepilot.retrieval.embedding_client import (
    EmbeddingClient,
    EmbeddingError,
    NOMIC_DOC_PREFIX,
    NOMIC_QUERY_PREFIX,
)


def test_check_model_available_success():
    client = EmbeddingClient(model="nomic-embed-text")
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "models": [{"name": "nomic-embed-text:latest"}, {"name": "qwen2.5:3b"}]
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        assert client.check_model_available() is True


def test_check_model_available_missing():
    client = EmbeddingClient(model="nonexistent-model")
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "models": [{"name": "qwen2.5:3b"}]
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        assert client.check_model_available() is False


def test_embed_batching_and_progress():
    client = EmbeddingClient(model="nomic-embed-text", batch_size=2)

    def fake_urlopen(req, timeout=None):
        data = json.loads(req.data.decode("utf-8"))
        inputs = data["input"]
        # Return dummy 768-dim embeddings for each input
        mock = MagicMock()
        mock.read.return_value = json.dumps({
            "embeddings": [[0.1] * 768 for _ in inputs]
        }).encode("utf-8")
        mock.__enter__.return_value = mock
        return mock

    progress_calls = []

    def progress_cb(cur, tot):
        progress_calls.append((cur, tot))

    texts = ["text1", "text2", "text3", "text4", "text5"]

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        embeddings = client.embed(texts, progress_callback=progress_cb)

    assert len(embeddings) == 5
    for emb in embeddings:
        assert len(emb) == 768

    # Progress called after each batch (batch size 2 -> 2, 4, 5)
    assert progress_calls == [(2, 5), (4, 5), (5, 5)]


def test_embed_single():
    client = EmbeddingClient(model="nomic-embed-text")
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "embeddings": [[0.5] * 768]
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        vec = client.embed_single("test query")

    assert len(vec) == 768
    assert vec[0] == 0.5


def test_embed_empty_list():
    client = EmbeddingClient()
    assert client.embed([]) == []


def test_embed_network_error():
    client = EmbeddingClient()
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
        with pytest.raises(EmbeddingError, match="Cannot reach Ollama"):
            client.embed(["test"])


def test_embed_mismatch_raises():
    client = EmbeddingClient()
    mock_resp = MagicMock()
    # Return 1 embedding when 2 texts were requested
    mock_resp.read.return_value = json.dumps({
        "embeddings": [[0.1] * 768]
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(EmbeddingError, match="count mismatch"):
            client.embed(["t1", "t2"])


def test_nomic_query_prefix():
    client = EmbeddingClient(model="nomic-embed-text")
    captured_payloads = []

    def fake_urlopen(req, timeout=None):
        payload = json.loads(req.data.decode("utf-8"))
        captured_payloads.append(payload)
        mock = MagicMock()
        mock.read.return_value = json.dumps({
            "embeddings": [[0.2] * 768]
        }).encode("utf-8")
        mock.__enter__.return_value = mock
        return mock

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        vec = client.embed_query("What is offline RAG?")

    assert len(vec) == 768
    assert len(captured_payloads) == 1
    sent_input = captured_payloads[0]["input"]
    assert sent_input == [f"{NOMIC_QUERY_PREFIX}What is offline RAG?"]


def test_nomic_document_prefix():
    client = EmbeddingClient(model="nomic-embed-text")
    captured_payloads = []

    def fake_urlopen(req, timeout=None):
        payload = json.loads(req.data.decode("utf-8"))
        captured_payloads.append(payload)
        mock = MagicMock()
        mock.read.return_value = json.dumps({
            "embeddings": [[0.3] * 768, [0.4] * 768]
        }).encode("utf-8")
        mock.__enter__.return_value = mock
        return mock

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        vecs = client.embed_documents(["Section A text.", "Section B text."])

    assert len(vecs) == 2
    assert len(captured_payloads) == 1
    sent_inputs = captured_payloads[0]["input"]
    assert sent_inputs == [
        f"{NOMIC_DOC_PREFIX}Section A text.",
        f"{NOMIC_DOC_PREFIX}Section B text.",
    ]
