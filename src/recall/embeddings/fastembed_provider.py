"""Local ONNX embedding provider powered by FastEmbed."""

from __future__ import annotations

from typing import Any
from fastembed import TextEmbedding
from recall.core.interfaces import BaseEmbeddingProvider


# Common FastEmbed model dimension registry
MODEL_DIMENSIONS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "nomic-ai/nomic-embed-text-v1.5": 768,
}


class FastEmbedProvider:
    """Lightweight, CPU/GPU ONNX embedding provider requiring zero external API keys."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        max_length: int = 512,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = TextEmbedding(model_name=model_name, max_length=max_length, **kwargs)
        self._dimensions = MODEL_DIMENSIONS.get(model_name, 384)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = list(self._model.embed(texts, batch_size=self.batch_size))
        return [e.tolist() for e in embeddings]

    def embed_query(self, query: str) -> list[float]:
        if not query.strip():
            return [0.0] * self._dimensions
        generator = self._model.query_embed(query)
        result = next(iter(generator))
        return result.tolist()
