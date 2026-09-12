"""Mock reranker for fast, deterministic unit and integration testing."""

from __future__ import annotations

import re
from recall.core.interfaces import BaseReranker
from recall.core.models import SearchResult


class MockReranker:
    """Deterministic, zero-dependency mock cross-encoder reranker
    scoring candidates by query keyword overlap ratio.
    """

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        if not candidates:
            return []

        query_terms = set(re.findall(r"\w+", query.lower()))
        if not query_terms:
            return candidates[:top_k]

        reranked: list[SearchResult] = []
        for candidate in candidates:
            cand_tokens = set(re.findall(r"\w+", candidate.text.lower()))
            overlap = len(query_terms.intersection(cand_tokens))
            score = overlap / len(query_terms)

            if score_threshold is not None and score < score_threshold:
                continue

            res = candidate.model_copy(
                update={"rerank_score": round(score, 4), "score": round(score, 4)}
            )
            reranked.append(res)

        # Sort descending by rerank score
        reranked.sort(key=lambda x: x.rerank_score or 0.0, reverse=True)
        return reranked[:top_k]
