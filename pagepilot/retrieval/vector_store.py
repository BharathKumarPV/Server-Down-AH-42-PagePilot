"""
PagePilot – ChromaDB local vector store (Stage 4).

Features:
- Pure local persistence in %APPDATA%/PagePilot/vectordb.
- Anonymized telemetry explicitly disabled.
- Page-level chunk metadata (1-based page numbers, document ID, filename).
- Query filtering by document ID(s).
- Full cleanup on document deletion.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.config import Settings

from pagepilot.config import CHROMA_DIR, ensure_chroma_dir
from pagepilot.pdf.models import Chunk


COLLECTION_NAME = "pagepilot_documents"


class VectorStore:
    """
    Manages embedding storage and similarity search using ChromaDB.
    """

    def __init__(self, persist_dir: Path | str = CHROMA_DIR):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]):
        """
        Store chunk embeddings, text, and metadata in ChromaDB.
        """
        if not chunks or not embeddings:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
            )

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "document_id": int(c.document_id),
                "filename": str(c.filename),
                "page_number": int(c.page_number),
                "chunk_index": int(c.chunk_index),
                "token_count": int(c.token_count),
                "section_heading": str(c.section_heading or ""),
            }
            for c in chunks
        ]

        # Upsert in batches to avoid payload limits
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            self._collection.upsert(
                ids=ids[i : i + batch_size],
                embeddings=embeddings[i : i + batch_size],
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        document_ids: list[int] | None = None,
    ) -> list[Chunk]:
        """
        Query the vector store for the most semantically relevant chunks.

        Parameters
        ----------
        query_embedding:
            Dense vector representation of the search query.
        top_k:
            Number of top results to return.
        document_ids:
            Optional list of document IDs to scope the search.

        Returns
        -------
        list[Chunk] — Ranked in descending similarity.
        """
        if self._collection.count() == 0:
            return []

        where_filter = None
        if document_ids:
            if len(document_ids) == 1:
                where_filter = {"document_id": int(document_ids[0])}
            else:
                where_filter = {"document_id": {"$in": [int(d) for d in document_ids]}}

        # Ensure n_results does not exceed total vector count
        n_results = min(top_k, self._collection.count())
        if n_results <= 0:
            return []

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas"],
        )

        res_ids = results.get("ids", [[]])[0]
        res_docs = results.get("documents", [[]])[0]
        res_metas = results.get("metadatas", [[]])[0]

        chunks: list[Chunk] = []
        for cid, doc, meta in zip(res_ids, res_docs, res_metas):
            meta = meta or {}
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    document_id=int(meta.get("document_id", 0)),
                    filename=str(meta.get("filename", "")),
                    page_number=int(meta.get("page_number", 1)),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    text=doc or "",
                    token_count=int(meta.get("token_count", 0)),
                    section_heading=str(meta.get("section_heading", "")),
                )
            )

        return chunks

    def delete_document(self, document_id: int):
        """Delete all vectors associated with a specific document ID."""
        try:
            self._collection.delete(where={"document_id": int(document_id)})
        except Exception:
            pass

    def count_vectors(self) -> int:
        """Return the number of stored chunk vectors."""
        try:
            return self._collection.count()
        except Exception:
            return 0

    def reset(self):
        """Clear all stored vectors."""
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

