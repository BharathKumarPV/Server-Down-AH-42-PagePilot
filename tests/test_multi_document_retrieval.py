"""
Tests for PagePilot Cross-Document Multi-Query Retrieval (Stage 6A).
"""

from unittest.mock import MagicMock
import pytest

from pagepilot.ai.claim_models import InformationRequirement
from pagepilot.pdf.models import Chunk
from pagepilot.retrieval.hybrid_retriever import HybridRetriever


@pytest.fixture
def mock_retriever():
    vs = MagicMock()
    ks = MagicMock()
    ec = MagicMock()
    return HybridRetriever(vector_store=vs, keyword_search=ks, embedding_client=ec)


def test_cross_document_retrieval_across_two_docs(mock_retriever):
    chunk_doc1 = Chunk(
        chunk_id="c1",
        document_id=1,
        filename="Doc1.pdf",
        page_number=2,
        chunk_index=0,
        text="Symmetric encryption uses a single secret key.",
        token_count=10,
    )
    chunk_doc2 = Chunk(
        chunk_id="c2",
        document_id=2,
        filename="Doc2.pdf",
        page_number=4,
        chunk_index=0,
        text="Asymmetric encryption uses public and private key pairs.",
        token_count=10,
    )

    def fake_retrieve(query, document_ids=None, top_k=4):
        from pagepilot.pdf.models import RetrievalResult
        if "asymmetric" in query.lower():
            return RetrievalResult(chunks=[chunk_doc2], method="hybrid")
        else:
            return RetrievalResult(chunks=[chunk_doc1], method="hybrid")

    mock_retriever.retrieve = fake_retrieve

    reqs = [
        InformationRequirement("R1", "symmetric encryption"),
        InformationRequirement("R2", "asymmetric encryption"),
    ]

    evidence_list, updated_reqs = mock_retriever.retrieve_cross_document(
        requirements=reqs,
        document_ids=[1, 2],
    )

    assert len(evidence_list) == 2
    # Verify provenance and evidence IDs
    assert evidence_list[0].evidence_id == "E1"
    assert evidence_list[1].evidence_id == "E2"
    assert {e.document_id for e in evidence_list} == {1, 2}
    assert {e.filename for e in evidence_list} == {"Doc1.pdf", "Doc2.pdf"}

    # Requirements coverage
    assert all(r.has_sufficient_evidence for r in updated_reqs)


def test_cross_document_deduplication(mock_retriever):
    shared_chunk = Chunk(
        chunk_id="shared_1",
        document_id=1,
        filename="Security.pdf",
        page_number=1,
        chunk_index=0,
        text="Key exchange and digital signatures are both vital.",
        token_count=10,
    )

    def fake_retrieve(query, document_ids=None, top_k=4):
        from pagepilot.pdf.models import RetrievalResult
        return RetrievalResult(chunks=[shared_chunk], method="semantic")

    mock_retriever.retrieve = fake_retrieve

    reqs = [
        InformationRequirement("R1", "key exchange"),
        InformationRequirement("R2", "digital signatures"),
    ]

    evidence_list, _ = mock_retriever.retrieve_cross_document(reqs)

    # Identical chunk retrieved by two requirements must be deduplicated to 1 Evidence object
    assert len(evidence_list) == 1
    assert evidence_list[0].chunk_id == "shared_1"
    assert "R1" in evidence_list[0].query_requirement_id
    assert "R2" in evidence_list[0].query_requirement_id


def test_scope_filtering_excludes_unselected_docs(mock_retriever):
    def fake_retrieve(query, document_ids=None, top_k=4):
        from pagepilot.pdf.models import RetrievalResult
        # Only return chunks matching document_ids filter
        all_chunks = [
            Chunk("c1", 1, "DocA.pdf", 1, 0, "Doc A text", 5),
            Chunk("c2", 2, "DocB.pdf", 1, 0, "Doc B text", 5),
            Chunk("c3", 3, "DocC.pdf", 1, 0, "Doc C text", 5),
        ]
        filtered = [c for c in all_chunks if document_ids is None or c.document_id in document_ids]
        return RetrievalResult(chunks=filtered, method="hybrid")

    mock_retriever.retrieve = fake_retrieve

    reqs = [InformationRequirement("R1", "any text")]
    # Scope restricted to doc 2 only
    evidence_list, _ = mock_retriever.retrieve_cross_document(reqs, document_ids=[2])

    assert len(evidence_list) == 1
    assert evidence_list[0].document_id == 2
    assert evidence_list[0].filename == "DocB.pdf"
