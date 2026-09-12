"""Concurrent Hybrid Retriever combining Dense Vector and Sparse Lexical search with RRF and Circuit Breaker."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from recall.core.interfaces import BaseEmbeddingProvider, BaseHybridRetriever, BaseSparseIndex, BaseVectorStore
from recall.core.models import SearchResult
from recall.retrieval.fusion import reciprocal_rank_fusion

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Production concurrent hybrid retriever implementing two-fold retrieval
    (dense vector similarity via Qdrant + native sparse vectors) fused with Reciprocal
    Rank Fusion (RRF) and protected by a dense circuit breaker.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_provider: BaseEmbeddingProvider,
        sparse_index: BaseSparseIndex,
        collection_name: str,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        rrf_k: int = 60,
        dense_timeout_seconds: float = 0.5,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.sparse_index = sparse_index
        self.collection_name = collection_name
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k
        self.dense_timeout_seconds = dense_timeout_seconds

    async def _execute_dense_search(
        self,
        query: str,
        limit: int,
        filter_dict: dict[str, Any] | None,
    ) -> list[SearchResult]:
        """Runs embedding generation and vector search protected by timeout circuit breaker."""
        try:
            async def _dense_task() -> list[SearchResult]:
                query_vector = await asyncio.to_thread(
                    self.embedding_provider.embed_query, query
                )
                return await asyncio.to_thread(
                    self.vector_store.search,
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    filter_dict=filter_dict,
                )

            return await asyncio.wait_for(_dense_task(), timeout=self.dense_timeout_seconds)
        except asyncio.TimeoutError:
            logger.warning(
                "Dense vector retrieval timed out after %.3fs. Circuit breaker triggered; falling back to sparse results.",
                self.dense_timeout_seconds,
            )
            return []
        except Exception as exc:
            logger.warning(
                "Dense vector retrieval encountered error: %s. Circuit breaker triggered; falling back to sparse results.",
                exc,
            )
            return []

    async def _execute_sparse_search(
        self,
        query: str,
        limit: int,
        filter_dict: dict[str, Any] | None,
    ) -> list[SearchResult]:
        """Runs Qdrant native sparse vector search."""
        try:
            return await asyncio.to_thread(
                self.sparse_index.search,
                query=query,
                limit=limit,
                filter_dict=filter_dict,
            )
        except Exception as exc:
            logger.warning("Sparse BM25 retrieval encountered error: %s", exc)
            return []

    async def retrieve(
        self,
        query: str,
        limit: int = 20,
        filter_dict: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """Performs concurrent two-fold retrieval (Dense + Sparse) with Reciprocal Rank Fusion.

        Dense and sparse retrieval channels are dispatched simultaneously on the event loop.
        If the dense vector store is degraded or times out, the circuit breaker protects the request
        and serves available sparse matches.
        """
        # Concurrent fan-out
        dense_results, sparse_results = await asyncio.gather(
            self._execute_dense_search(query, limit=limit, filter_dict=filter_dict),
            self._execute_sparse_search(query, limit=limit, filter_dict=filter_dict),
        )

        # Fused rankings via Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion(
            ranking_lists=[dense_results, sparse_results],
            weights=[self.dense_weight, self.sparse_weight],
            k=self.rrf_k,
        )

        # Apply score threshold gating if specified
        if score_threshold is not None:
            fused = [r for r in fused if r.score >= score_threshold]

        return fused[:limit]
