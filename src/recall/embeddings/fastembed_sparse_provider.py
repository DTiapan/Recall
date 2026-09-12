"""Local ONNX sparse embedding provider powered by FastEmbed SPLADE models."""

from __future__ import annotations

from typing import Any

from fastembed import SparseTextEmbedding


DEFAULT_SPARSE_MODEL = "prithivida/Splade_PP_en_v1"


class FastEmbedSparseProvider:
    """Sparse lexical embedding provider for Qdrant native sparse vector indexing."""

    def __init__(
        self,
        model_name: str = DEFAULT_SPARSE_MODEL,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SparseTextEmbedding(model_name=model_name, **kwargs)

    def embed_texts(self, texts: list[str]) -> list[dict[int, float]]:
        if not texts:
            return []
        embeddings = list(self._model.embed(texts, batch_size=self.batch_size))
        return [embedding.as_dict() for embedding in embeddings]

    def embed_query(self, query: str) -> dict[int, float]:
        if not query.strip():
            return {}
        embedding = next(iter(self._model.query_embed(query)))
        return embedding.as_dict()
