"""Deterministic mock embedding provider for high-speed unit testing."""

from __future__ import annotations

import numpy as np
import xxhash
from recall.core.interfaces import BaseEmbeddingProvider


class MockEmbeddingProvider:
    """Generates deterministic, unit-normalized pseudorandom vectors based on input text hashes.
    Guarantees that identical texts yield identical embeddings without downloading weights.
    """

    def __init__(self, dimensions: int = 384) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for text in texts:
            # Seed PRNG with text hash
            seed = xxhash.xxh32(text.encode("utf-8")).intdigest()
            rng = np.random.RandomState(seed)
            raw = rng.randn(self._dimensions).astype(np.float32)
            norm = np.linalg.norm(raw)
            unit = (raw / (norm + 1e-9)).tolist()
            results.append(unit)
        return results

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]
