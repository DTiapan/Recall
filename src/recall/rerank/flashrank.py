"""Production-grade ONNX Cross-Encoder Reranker using FlashRank."""

from __future__ import annotations

import logging
from typing import Any
from flashrank import Ranker, RerankRequest

from recall.core.interfaces import BaseReranker
from recall.core.models import SearchResult

logger = logging.getLogger(__name__)

FLASHRANK_MODEL_NAMES = frozenset(
    {
        "ms-marco-TinyBERT-L-2-v2",
        "ms-marco-MiniLM-L-12-v2",
        "rank-T5-flan",
    }
)

# HuggingFace / docs names that FlashRank does not load directly — map to best ONNX peer.
_FLASHRANK_MODEL_ALIASES: dict[str, str] = {
    "BAAI/bge-reranker-base": "ms-marco-MiniLM-L-12-v2",
    "bge-reranker-base": "ms-marco-MiniLM-L-12-v2",
    "BAAI/bge-reranker-large": "ms-marco-MiniLM-L-12-v2",
    "BAAI/bge-reranker-v2-m3": "ms-marco-MiniLM-L-12-v2",
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "ms-marco-MiniLM-L-12-v2",
    "cross-encoder/ms-marco-TinyBERT-L-6": "ms-marco-TinyBERT-L-2-v2",
}


def resolve_flashrank_model_name(local_model: str) -> str:
    """Map config ``reranking.local_model`` to a FlashRank ``Ranker`` model name."""
    normalized = local_model.strip()
    if normalized in FLASHRANK_MODEL_NAMES:
        return normalized
    if normalized in _FLASHRANK_MODEL_ALIASES:
        resolved = _FLASHRANK_MODEL_ALIASES[normalized]
        logger.info(
            "Reranker model %s is not a native FlashRank ONNX name; using %s",
            normalized,
            resolved,
        )
        return resolved
    tail = normalized.rsplit("/", 1)[-1]
    if tail in FLASHRANK_MODEL_NAMES:
        return tail
    if tail in _FLASHRANK_MODEL_ALIASES:
        resolved = _FLASHRANK_MODEL_ALIASES[tail]
        logger.info(
            "Reranker model %s is not a native FlashRank ONNX name; using %s",
            normalized,
            resolved,
        )
        return resolved
    logger.warning(
        "Unknown reranker model %s; falling back to ms-marco-MiniLM-L-12-v2",
        normalized,
    )
    return "ms-marco-MiniLM-L-12-v2"


class FlashRankReranker:
    """Local, high-speed ONNX Cross-Encoder reranker.

    Computes joint cross-attention over (query, document) pairs without PyTorch or GPU
    dependencies, outputting calibrated relevance probabilities [0, 1].
    """

    def __init__(
        self,
        model_name: str = "ms-marco-MiniLM-L-12-v2",
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
