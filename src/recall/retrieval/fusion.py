"""Reciprocal Rank Fusion (RRF) for combining multiple ranking channels."""

from __future__ import annotations

from collections import defaultdict
from recall.core.models import SearchResult


def reciprocal_rank_fusion(
    ranking_lists: list[list[SearchResult]],
    weights: list[float] | None = None,
    k: int = 60,
) -> list[SearchResult]:
    """Combines multiple ranked lists using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_score(d) = sum_m ( w_m / (k + rank_m(d)) )

    Args:
        ranking_lists: List of search result rankings (e.g. [dense_results, sparse_results]).
        weights: Optional weights for each ranking list. Defaults to 1.0 per list.
        k: Smoothing constant preventing high-rank saturation (standard default 60).

    Returns:
        List of SearchResult objects sorted descending by fused RRF score.
    """
    if not ranking_lists:
        return []

    # Default unit weights if not specified
    if weights is None:
        channel_weights = [1.0] * len(ranking_lists)
    else:
        if len(weights) != len(ranking_lists):
            raise ValueError(
                f"Length of weights ({len(weights)}) must match length of ranking_lists ({len(ranking_lists)})"
            )
        channel_weights = weights

    rrf_scores: dict[str, float] = defaultdict(float)
    canonical_results: dict[str, SearchResult] = {}
    contributing_vectors: dict[str, set[str]] = defaultdict(set)

    for list_idx, result_list in enumerate(ranking_lists):
        weight = channel_weights[list_idx]
        for rank_zero_idx, result in enumerate(result_list):
            rank = rank_zero_idx + 1  # 1-indexed rank
            chunk_id = result.chunk_id

            score_contribution = weight / (k + rank)
            rrf_scores[chunk_id] += score_contribution

            if chunk_id not in canonical_results:
                canonical_results[chunk_id] = result
            if result.vector_name:
                contributing_vectors[chunk_id].add(result.vector_name)

    # Sort descending by RRF score
    sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

    fused_results: list[SearchResult] = []
    for chunk_id in sorted_chunk_ids:
        base = canonical_results[chunk_id]
        vector_tag = "+".join(sorted(contributing_vectors[chunk_id])) or "fused"

        fused = SearchResult(
            chunk_id=base.chunk_id,
            score=rrf_scores[chunk_id],
            text=base.text,
            metadata=base.metadata,
            vector_name=vector_tag,
            rerank_score=base.rerank_score,
        )
        fused_results.append(fused)

    return fused_results
