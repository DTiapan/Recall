"""Deterministic mock sparse embedding provider for fast CI without model downloads."""

from __future__ import annotations

import xxhash

from recall.retrieval.bm25 import tokenize_lexical


class MockSparseEmbeddingProvider:
    """Builds reproducible sparse vectors from lexical tokens for in-memory Qdrant tests."""

    def embed_texts(self, texts: list[str]) -> list[dict[int, float]]:
        return [self._embed_text(text) for text in texts]

    def embed_query(self, query: str) -> dict[int, float]:
        return self._embed_text(query)

    def _embed_text(self, text: str) -> dict[int, float]:
        sparse: dict[int, float] = {}
        for token in tokenize_lexical(text):
            index = xxhash.xxh32(token.encode("utf-8")).intdigest() % 100_000
            sparse[index] = sparse.get(index, 0.0) + 1.0
        return sparse
