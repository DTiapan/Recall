"""Qdrant-backed sparse index implementing BaseSparseIndex for hybrid retrieval."""

from __future__ import annotations

from typing import Any

from recall.core.interfaces import BaseSparseIndex
from recall.core.models import Chunk, SearchResult
from recall.core.interfaces import BaseSparseEmbeddingProvider
from recall.storage.qdrant import QdrantVectorStore


class QdrantSparseIndex:
    """Sparse retrieval channel backed by Qdrant native sparse vectors.

    Chunk indexing occurs during ``QdrantVectorStore.upsert`` when ``chunk.sparse_vector``
    is populated. This adapter handles query-time sparse embedding and search only.
    """

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        collection_name: str,
        sparse_embedder: BaseSparseEmbeddingProvider,
    ) -> None:
        self.vector_store = vector_store
        self.collection_name = collection_name
        self.sparse_embedder = sparse_embedder

    def index(self, chunks: list[Chunk]) -> int:
        """No-op: sparse vectors are persisted via ``QdrantVectorStore.upsert``."""
        return len(chunks)

    def search(
        self,
        query: str,
        limit: int = 10,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        sparse_vector = self.sparse_embedder.embed_query(query)
        return self.vector_store.search_sparse(
            collection_name=self.collection_name,
            sparse_vector=sparse_vector,
            limit=limit,
            filter_dict=filter_dict,
        )

    def count(self) -> int:
        return self.vector_store.count(self.collection_name)
