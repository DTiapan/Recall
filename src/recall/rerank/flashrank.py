"""Production-grade ONNX Cross-Encoder Reranker using FlashRank."""

from __future__ import annotations

import logging
from typing import Any
from flashrank import Ranker, RerankRequest

from recall.core.interfaces import BaseReranker
from recall.core.models import SearchResult

logger = logging.getLogger(__name__)


class FlashRankReranker:
    """Local, high-speed ONNX Cross-Encoder reranker.

    Computes joint cross-attention over (query, document) pairs without PyTorch or GPU
    dependencies, outputting calibrated relevance probabilities [0, 1].
    """

    def __init__(
        self,
        model_name: str = "ms-marco-TinyBERT-L-2-v2",
        cache_dir: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._ranker: Ranker | None = None

    @property
    def ranker(self) -> Ranker:
        if self._ranker is None:
            logger.info("Initializing FlashRank cross-encoder: %s", self.model_name)
            kwargs: dict[str, Any] = {"model_name": self.model_name}
            if self.cache_dir:
                kwargs["cache_dir"] = self.cache_dir
            self._ranker = Ranker(**kwargs)
        return self._ranker

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """Reranks candidates using full cross-attention and applies score gating.

        Args:
            query: User search query.
            candidates: Retrieved candidates from hybrid search.
            top_k: Maximum number of top reranked results to return.
            score_threshold: Minimum cross-encoder score cutoff (e.g. 0.35) to filter distractors.

        Returns:
            List of SearchResult objects sorted descending by rerank score.
        """
        if not candidates:
            return []

        # Preserve lookup mapping by chunk_id
        candidate_map = {c.chunk_id: c for c in candidates}

        # Build passage dicts for FlashRank
        passages = [
            {
                "id": c.chunk_id,
                "text": c.text,
            }
            for c in candidates
        ]

        request = RerankRequest(query=query, passages=passages)
        ranked_passages = self.ranker.rerank(request)

        reranked_results: list[SearchResult] = []
        for p in ranked_passages:
            chunk_id = str(p["id"])
            score = float(p.get("score", 0.0))

            # Apply score threshold gating
            if score_threshold is not None and score < score_threshold:
                continue

            base = candidate_map.get(chunk_id)
            if not base:
                continue

            updated = base.model_copy(
                update={
                    "rerank_score": score,
                }
            )
            reranked_results.append(updated)

        return reranked_results[:top_k]
