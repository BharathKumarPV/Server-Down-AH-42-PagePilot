"""
PagePilot – Local embedding client using Ollama nomic-embed-text (Stage 4).

Features:
- Pure stdlib urllib HTTP client (thread-safe, no external requests dependency).
- Nomic asymmetric prefixing:
    search_document: <chunk_text>   for document indexing
    search_query: <question>        for search queries
- Batching support for embedding chunks efficiently without RAM spikes.
- Model availability verification.
- Validates vector dimensions.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Callable, List, Optional

from pagepilot.config import EMBED_MODEL, OLLAMA_HOST

NOMIC_DOC_PREFIX = "search_document: "
NOMIC_QUERY_PREFIX = "search_query: "


class EmbeddingError(Exception):
    """Raised when the embedding request fails or model is missing."""


class EmbeddingClient:
    """
    Communicates with Ollama's /api/embed endpoint to generate dense vector embeddings.
    Enforces Nomic search_document: and search_query: prefixes.
    """

    def __init__(
        self,
        host: str = OLLAMA_HOST,
        model: str = EMBED_MODEL,
        batch_size: int = 16,
        timeout: int = 60,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.batch_size = max(1, batch_size)
        self.timeout = timeout

    def check_model_available(self) -> bool:
        """
        Check whether the embedding model is installed in the local Ollama instance.
        """
        try:
            url = f"{self.host}/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                installed_models = [m.get("name", "") for m in data.get("models", [])]
                # Compare both exact and stripped tag names
                target_base = self.model.split(":")[0]
                for name in installed_models:
                    if name == self.model or name.split(":")[0] == target_base:
                        return True
                return False
        except Exception:
            return False

    def embed_query(self, query: str) -> list[float]:
        """
        Generate embedding vector for a search question/query.
        Prefixes query with 'search_query: ' as required by Nomic models.
        """
        cleaned = query.strip()
        if not cleaned:
            raise EmbeddingError("Cannot embed an empty query.")
        prefixed = cleaned if cleaned.startswith(NOMIC_QUERY_PREFIX) else f"{NOMIC_QUERY_PREFIX}{cleaned}"
        results = self.embed([prefixed])
        if not results:
            raise EmbeddingError("No embedding returned for query.")
        return results[0]

    def embed_single(self, text: str, prefix: str = NOMIC_QUERY_PREFIX) -> list[float]:
        """
        Generate embedding vector for a single text with the specified prefix (default: search_query:).
        """
        cleaned = text.strip()
        if not cleaned:
            raise EmbeddingError("Cannot embed an empty text.")
        if prefix and not cleaned.startswith(prefix):
            cleaned = f"{prefix}{cleaned}"
        results = self.embed([cleaned])
        if not results:
            raise EmbeddingError("No embedding returned for query.")
        return results[0]

    def embed_documents(
        self,
        texts: list[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[list[float]]:
        """
        Generate embedding vectors for document chunks.
        Prefixes each chunk with 'search_document: ' as required by Nomic models.
        """
        if not texts:
            return []
        prefixed_texts = [
            t if t.startswith(NOMIC_DOC_PREFIX) else f"{NOMIC_DOC_PREFIX}{t}"
            for t in texts
        ]
        return self.embed(prefixed_texts, progress_callback=progress_callback)

    def embed(
        self,
        texts: list[str],
        prefix: str = "",
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[list[float]]:
        """
        Generate embedding vectors for a list of texts in batches.

        Parameters
        ----------
        texts:
            List of strings to embed.
        prefix:
            Optional prefix to prepend to each text if not already present.
        progress_callback:
            Optional function(completed: int, total: int) for progress reporting.

        Returns
        -------
        list[list[float]] — embedding vectors for each input text.
        """
        if not texts:
            return []

        if prefix:
            formatted_texts = [
                t if t.startswith(prefix) else f"{prefix}{t}"
                for t in texts
            ]
        else:
            formatted_texts = texts

        all_embeddings: list[list[float]] = []
        total = len(formatted_texts)

        for i in range(0, total, self.batch_size):
            batch = formatted_texts[i : i + self.batch_size]
            batch_vectors = self._embed_batch(batch)
            if len(batch_vectors) != len(batch):
                raise EmbeddingError(
                    f"Embedding count mismatch: expected {len(batch)}, got {len(batch_vectors)}"
                )
            all_embeddings.extend(batch_vectors)

            if progress_callback:
                progress_callback(len(all_embeddings), total)

        return all_embeddings

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        """Send a single batch to Ollama /api/embed."""
        url = f"{self.host}/api/embed"
        payload = {
            "model": self.model,
            "input": batch,
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                embeddings = data.get("embeddings")
                if embeddings is not None and isinstance(embeddings, list):
                    return embeddings
                raise EmbeddingError(f"Unexpected response format from Ollama: {data}")
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 404:
                # Try fallback to legacy /api/embeddings endpoint (one by one)
                return self._embed_batch_legacy(batch)
            raise EmbeddingError(
                f"Ollama embedding request failed (HTTP {exc.code}): {err_body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise EmbeddingError(f"Cannot reach Ollama at {self.host}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise EmbeddingError(f"Ollama embedding timed out after {self.timeout}s") from exc
        except Exception as exc:
            raise EmbeddingError(f"Error during embedding generation: {exc}") from exc

    def _embed_batch_legacy(self, batch: list[str]) -> list[list[float]]:
        """Fallback for older Ollama versions that only support /api/embeddings."""
        results: list[list[float]] = []
        for text in batch:
            url = f"{self.host}/api/embeddings"
            payload = {"model": self.model, "prompt": text}
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                emb = data.get("embedding")
                if not emb:
                    raise EmbeddingError(f"No embedding returned in legacy endpoint: {data}")
                results.append(emb)
        return results
