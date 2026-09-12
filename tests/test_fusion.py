"""Unit tests for Reciprocal Rank Fusion (RRF)."""

import pytest
from recall.core.models import ChunkMetadata, SearchResult
from recall.retrieval.fusion import reciprocal_rank_fusion


def _make_result(chunk_id: str, score: float, vector_name: str) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        score=score,
        text=f"Text for {chunk_id}",
        metadata=ChunkMetadata(doc_id=f"doc_{chunk_id}", chunk_index=0),
        vector_name=vector_name,
    )


def test_rrf_dual_channel_boost():
    # Dense results: A at rank 1, B at rank 2
    dense = [
        _make_result("A", 0.95, "dense"),
        _make_result("B", 0.85, "dense"),
    ]
    # Sparse results: B at rank 1, C at rank 2
    sparse = [
        _make_result("B", 12.5, "sparse"),
        _make_result("C", 8.2, "sparse"),
    ]

    # With k=60, equal weights:
    # A score = 1/(60+1) = 1/61 ~= 0.016393
    # B score = 1/(60+2) + 1/(60+1) = 1/62 + 1/61 ~= 0.016129 + 0.016393 = 0.032522
    # C score = 1/(60+2) = 1/62 ~= 0.016129
    # Expected order: B (fused from both), then A, then C
    fused = reciprocal_rank_fusion([dense, sparse], k=60)

    assert len(fused) == 3
    assert fused[0].chunk_id == "B"
    assert fused[0].vector_name == "dense+sparse"
    assert pytest.approx(fused[0].score, rel=1e-4) == (1 / 62 + 1 / 61)

    assert fused[1].chunk_id == "A"
    assert fused[1].vector_name == "dense"
    assert pytest.approx(fused[1].score, rel=1e-4) == (1 / 61)

    assert fused[2].chunk_id == "C"
    assert fused[2].vector_name == "sparse"
    assert pytest.approx(fused[2].score, rel=1e-4) == (1 / 62)


def test_rrf_with_custom_weights():
    dense = [_make_result("D", 0.9, "dense")]
    sparse = [_make_result("S", 10.0, "sparse")]

    # Heavy bias towards sparse (weights: [0.1, 0.9])
    fused = reciprocal_rank_fusion([dense, sparse], weights=[0.1, 0.9], k=60)
    assert fused[0].chunk_id == "S"
    assert fused[1].chunk_id == "D"
    assert pytest.approx(fused[0].score, rel=1e-4) == (0.9 / 61)
    assert pytest.approx(fused[1].score, rel=1e-4) == (0.1 / 61)


def test_rrf_empty_and_error_handling():
    # Empty lists
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []

    # Single list preserves ordering
    single = [_make_result("1", 0.9, "dense"), _make_result("2", 0.8, "dense")]
    fused_single = reciprocal_rank_fusion([single], k=60)
    assert [r.chunk_id for r in fused_single] == ["1", "2"]

    # Mismatched weights length raises ValueError
    with pytest.raises(ValueError, match="Length of weights"):
        reciprocal_rank_fusion([single], weights=[0.5, 0.5])
