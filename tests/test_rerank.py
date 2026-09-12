"""Tests for cross-encoder rerankers (MockReranker and FlashRankReranker)."""

import pytest
from recall.core.interfaces import BaseReranker
from recall.core.models import ChunkMetadata, SearchResult
from recall.rerank import FlashRankReranker, MockReranker


def _make_candidate(chunk_id: str, text: str) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        score=0.5,
        text=text,
        metadata=ChunkMetadata(doc_id=f"doc_{chunk_id}", chunk_index=0),
        vector_name="fused",
    )


def test_mock_reranker_conformance_and_ranking():
    reranker = MockReranker()
    assert isinstance(reranker, BaseReranker)

    candidates = [
        _make_candidate("c1", "Unrelated discussion about gardening and botany."),
        _make_candidate("c2", "PostgreSQL database configuration and connection pooling with PgBouncer."),
        _make_candidate("c3", "PostgreSQL query optimization techniques."),
    ]

    results = reranker.rerank(
        query="PostgreSQL connection pooling PgBouncer",
        candidates=candidates,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk_id == "c2"
    assert results[0].rerank_score is not None and results[0].rerank_score > 0.5


def test_mock_reranker_threshold_gating():
    reranker = MockReranker()
    candidates = [
        _make_candidate("c1", "Kubernetes pod autoscaling policy."),
        _make_candidate("c2", "Random banana fruit salad recipe."),
    ]

    # Threshold 0.5 should drop the completely unrelated recipe
    results = reranker.rerank(
        query="Kubernetes pod autoscaler",
        candidates=candidates,
        top_k=5,
        score_threshold=0.3,
    )
    assert len(results) == 1
    assert results[0].chunk_id == "c1"


def test_flashrank_reranker_conformance_and_real_inference():
    reranker = FlashRankReranker()
    assert isinstance(reranker, BaseReranker)

    candidates = [
        _make_candidate("c1", "The capital and largest city of France is Paris, located on the Seine River."),
        _make_candidate("c2", "Quantum computing utilizes qubits for superposition and entanglement calculations."),
        _make_candidate("c3", "French cuisine is world-renowned for croissants, baguettes, and fine wine."),
    ]

    query = "What is the capital city of France?"
    results = reranker.rerank(query=query, candidates=candidates, top_k=2)

    assert len(results) == 2
    # Paris document should be top ranked
    assert results[0].chunk_id == "c1"
    assert results[0].rerank_score is not None
    assert results[0].rerank_score > 0.8

    # Threshold gating: high cutoff drops non-answers
    gated = reranker.rerank(query=query, candidates=candidates, top_k=5, score_threshold=0.5)
    assert len(gated) >= 1
    assert all(r.rerank_score >= 0.5 for r in gated)
